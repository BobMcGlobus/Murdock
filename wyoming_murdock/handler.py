"""Wyoming handler: verify speaker, then forward or block.

Latency-optimised pipeline:

    ┌── Satellite ──┐          ┌── Murdock proxy ──┐          ┌── STT ──┐
    │  AudioStart   │ ───────▶ │  open upstream     │ ───────▶ │          │
    │  AudioChunk … │ ───────▶ │  fwd live + buffer │ ───────▶ │ streaming│
    │  AudioStop    │ ───────▶ │  verify on buffer  │          │  decode  │
    │               │          │  wait for both     │ ◀─────── │          │
    │  Transcript   │ ◀─────── │  gate + respond    │          │          │

We open the upstream connection lazily on the first ``AudioChunk`` and
forward every chunk as it arrives, so the STT engine starts decoding in
parallel with the rest of the satellite's audio and with our own
speaker-verify work. By the time ``AudioStop`` arrives we usually only
wait for the last few chunks of STT output, not the whole pipeline.

If verification ultimately fails we still return an empty transcript to
the satellite (the STT side has wasted some CPU, but no user-visible
effect). The gate decides in *this* order:

    1. Upload shorter than ``min_verify_seconds``              → passthrough
    2. Zero enrolled speakers + passthrough_when_no_speakers   → passthrough
    3. Match                                                   → forward + HA event
    4. No match + not require_match                            → forward + log unknown
    5. No match + require_match                                → empty + log unknown
"""

from __future__ import annotations

import asyncio
import logging
import time
import uuid
from typing import Optional

import numpy as np
from wyoming.asr import Transcribe, Transcript
from wyoming.audio import AudioChunk, AudioStart, AudioStop
from wyoming.client import AsyncClient
from wyoming.event import Event
from wyoming.info import Describe
from wyoming.server import AsyncEventHandler

from murdock.core.context import AppContext
from murdock.core.extraction import extract_target_speaker
from murdock.core.info_cache import UpstreamInfoCache
from murdock.core.liveness import analyze_pcm as analyze_liveness
from murdock.core.recognition_log import (
    OUTCOME_BLOCKED_EMBED_FAILED,
    OUTCOME_BLOCKED_NO_MATCH,
    OUTCOME_BLOCKED_NO_SPEAKERS,
    OUTCOME_BLOCKED_TV_NOISE,
    OUTCOME_EMPTY,
    OUTCOME_MATCH,
    OUTCOME_PASSTHROUGH_NO_SPEAKERS,
    OUTCOME_PASSTHROUGH_SHORT,
    OUTCOME_UNKNOWN_FORWARDED,
)

_LOGGER = logging.getLogger("murdock.wyoming.handler")

# Embedder is not thread-safe when the underlying ORT session is shared,
# so serialise calls across connections.
_MODEL_LOCK = asyncio.Lock()


class MurdockHandler(AsyncEventHandler):
    """Gatekeeper: forward audio to STT while verifying the speaker."""

    def __init__(
        self,
        info_cache: UpstreamInfoCache,
        context: AppContext,
        *args,
        **kwargs,
    ) -> None:
        super().__init__(*args, **kwargs)
        self.info_cache = info_cache
        self.context = context

        # Per-connection state.
        self._audio_buffer = bytearray()
        self._audio_rate = 16000
        self._audio_width = 2
        self._audio_channels = 1
        self._language: Optional[str] = None
        self._responded = False
        self._stream_start: Optional[float] = None
        self._session_id = uuid.uuid4().hex[:8]
        self._satellite_id: Optional[str] = None

        # Upstream streaming state. The transcript future is (re)created at
        # each AudioStart so we never reuse a resolved future across sessions.
        self._upstream_client: Optional[AsyncClient] = None
        self._upstream_open: bool = False
        self._upstream_failed: bool = False
        self._upstream_reader_task: Optional[asyncio.Task] = None
        self._upstream_transcript: Optional[asyncio.Future[str]] = None

        # Streaming-Verify (early cutoff) state.
        self._early_probe_task: Optional[asyncio.Task] = None
        self._early_match: Optional[str] = None  # speaker name if early-matched
        self._early_match_id: Optional[int] = None
        self._early_distance: Optional[float] = None
        self._early_probed: bool = False  # True after first probe attempt

    @property
    def upstream_uri(self) -> str:
        """Always read the live value from context so the UI can update
        it without a service restart."""
        return self.context.get_upstream_uri()

    @property
    def _is_voxtral(self) -> bool:
        """True when the active STT backend is Voxtral (Mistral Cloud)."""
        return self.context.get_stt_backend() == "voxtral"

    # ------------------------------------------------------------------
    # Wyoming event dispatch
    # ------------------------------------------------------------------

    async def handle_event(self, event: Event) -> bool:
        sid = self._session_id

        if Describe.is_type(event.type):
            info = await self.info_cache.build_info()
            languages = [
                lang
                for asr in info.asr
                for model in asr.models
                for lang in model.languages
            ]
            _LOGGER.info(
                "[%s] Describe → advertising languages: %s",
                sid, ", ".join(languages) or "(none)",
            )
            await self.write_event(info.event())
            return True

        if Transcribe.is_type(event.type):
            transcribe = Transcribe.from_event(event)
            self._language = transcribe.language
            if transcribe.name:
                self._satellite_id = transcribe.name
            _LOGGER.info(
                "[%s] Transcribe from HA — language=%r name=%r",
                sid, transcribe.language, transcribe.name,
            )
            return True

        if AudioStart.is_type(event.type):
            self._reset_session_state()
            start = AudioStart.from_event(event)
            self._audio_rate = start.rate
            self._audio_width = start.width
            self._audio_channels = start.channels
            _LOGGER.info(
                "[%s] ── New audio session (%d Hz, %d-bit, %d ch) ──",
                sid, start.rate, start.width * 8, start.channels,
            )
            return True

        if AudioChunk.is_type(event.type):
            chunk = AudioChunk.from_event(event)
            self._audio_rate = chunk.rate
            self._audio_width = chunk.width
            self._audio_channels = chunk.channels
            self._audio_buffer.extend(chunk.audio)

            # Lazily open upstream on the first chunk so we know the format.
            # When using Voxtral cloud backend, skip the upstream connection
            # entirely — we'll transcribe the buffered audio on AudioStop.
            if not self._is_voxtral:
                if self._upstream_client is None and not self._upstream_failed:
                    await self._open_upstream(chunk.rate, chunk.width, chunk.channels)

                # Forward the chunk as-is to the upstream STT.
                if self._upstream_open:
                    try:
                        await self._upstream_client.write_event(  # type: ignore[union-attr]
                            AudioChunk(
                                audio=chunk.audio,
                                rate=chunk.rate,
                                width=chunk.width,
                                channels=chunk.channels,
                            ).event()
                        )
                    except Exception:
                        _LOGGER.exception("[%s] Upstream write failed", sid)
                        self._upstream_failed = True
                        self._upstream_open = False

            # --- Streaming-Verify: early probe ---
            # After ~1.5 s of 16 kHz/16-bit audio (48000 bytes) we have
            # enough material for a meaningful embedding. Fire off a
            # background probe; if it matches with high confidence the
            # HA events are sent immediately, shaving 2-4 s off the
            # perceived speaker-recognition latency.
            if not self._early_probed and len(self._audio_buffer) >= 48000:
                self._early_probed = True
                speaker_count = len(self.context.speakers.list_speakers())
                if speaker_count > 0:
                    snapshot = bytes(self._audio_buffer)
                    self._early_probe_task = asyncio.create_task(
                        self._run_early_probe(snapshot)
                    )
            return True

        if AudioStop.is_type(event.type):
            if not self._responded:
                await self._finish_session()
            return True

        return True

    # ------------------------------------------------------------------
    # Session helpers
    # ------------------------------------------------------------------

    def _reset_session_state(self) -> None:
        self._audio_buffer = bytearray()
        self._responded = False
        self._stream_start = time.monotonic()
        self._upstream_client = None
        self._upstream_open = False
        self._upstream_failed = False
        self._upstream_reader_task = None
        self._early_probe_task = None
        self._early_match = None
        self._early_match_id = None
        self._early_distance = None
        self._early_probed = False
        # Only create an upstream transcript future for Wyoming mode.
        # In Voxtral mode, transcription happens on-demand after AudioStop.
        if not self._is_voxtral:
            loop = asyncio.get_running_loop()
            self._upstream_transcript = loop.create_future()
        else:
            self._upstream_transcript = None

    async def _open_upstream(self, rate: int, width: int, channels: int) -> None:
        """Open the upstream ASR connection and start streaming."""
        sid = self._session_id
        t0 = time.monotonic()
        try:
            client = AsyncClient.from_uri(self.upstream_uri)
            await client.connect()
            await client.write_event(Transcribe(language=self._language).event())
            await client.write_event(
                AudioStart(rate=rate, width=width, channels=channels).event()
            )
            self._upstream_client = client
            self._upstream_open = True
            self._upstream_reader_task = asyncio.create_task(
                self._read_upstream_transcript()
            )
            connect_ms = (time.monotonic() - t0) * 1000
            _LOGGER.info(
                "[%s] Upstream opened → %s (%d Hz, lang=%r, %.0fms)",
                sid, self.upstream_uri, rate, self._language, connect_ms,
            )
        except Exception as exc:
            _LOGGER.error(
                "[%s] Could not open upstream %s: %s",
                sid, self.upstream_uri, exc,
            )
            self._upstream_failed = True
            self._upstream_open = False
            if (
                self._upstream_transcript is not None
                and not self._upstream_transcript.done()
            ):
                self._upstream_transcript.set_result("")

    async def _read_upstream_transcript(self) -> None:
        """Wait for the first Transcript event from upstream.

        Logs every event type we see from the upstream so that a broken
        STT (wrong model, language mismatch, half-open socket, …) is
        obvious in the Murdock logs instead of manifesting as a silent
        empty transcript.
        """
        sid = self._session_id
        assert self._upstream_client is not None
        fut = self._upstream_transcript
        event_count = 0
        try:
            while True:
                event = await self._upstream_client.read_event()
                event_count += 1
                if event is None:
                    _LOGGER.warning(
                        "[%s] Upstream closed without transcript "
                        "(after %d events)",
                        sid, event_count,
                    )
                    if fut is not None and not fut.done():
                        fut.set_result("")
                    return
                # Log every non-Transcript event at DEBUG so --log-level
                # debug shows the full upstream chatter, but keep the
                # Transcript itself on INFO so a normal operator can see
                # the STT output without enabling debug.
                if Transcript.is_type(event.type):
                    transcript = Transcript.from_event(event)
                    _LOGGER.info(
                        "[%s] Upstream transcript received: %r",
                        sid, transcript.text,
                    )
                    if fut is not None and not fut.done():
                        fut.set_result(transcript.text or "")
                    return
                _LOGGER.debug(
                    "[%s] Upstream event (ignored): %s", sid, event.type
                )
        except asyncio.CancelledError:
            # Normal on session teardown — re-raise so the task status
            # is "cancelled" rather than "exception".
            if fut is not None and not fut.done():
                fut.set_result("")
            raise
        except Exception as exc:
            _LOGGER.exception(
                "[%s] Upstream reader crashed after %d events: %s",
                sid, event_count, exc,
            )
            if fut is not None and not fut.done():
                fut.set_result("")

    async def _close_upstream(self, *, send_stop: bool) -> None:
        """Tear down the upstream connection cleanly."""
        sid = self._session_id
        client = self._upstream_client
        if client is None:
            return
        try:
            if send_stop and self._upstream_open:
                try:
                    await client.write_event(AudioStop().event())
                except Exception:
                    _LOGGER.debug("[%s] Failed to send AudioStop upstream", sid)
        finally:
            try:
                await client.disconnect()
            except Exception:
                pass
            self._upstream_client = None
            self._upstream_open = False

    async def _run_early_probe(self, audio_snapshot: bytes) -> None:
        """Background task: embed a snapshot of the buffer and check for
        a high-confidence match. If found, fire HA events immediately
        so downstream automations can react before AudioStop."""
        sid = self._session_id
        try:
            from murdock.core.audio import to_mono_16k_pcm
            audio_16k = to_mono_16k_pcm(
                audio_snapshot, self._audio_rate,
                self._audio_width, self._audio_channels,
            )
            # Trim leading chime the same way _finish_session does.
            skip_seconds = self.context.get_skip_leading_seconds()
            if skip_seconds > 0:
                skip_bytes = int(skip_seconds * 16000 * 2)
                skip_bytes -= skip_bytes % 2
                if skip_bytes < len(audio_16k):
                    audio_16k = audio_16k[skip_bytes:]

            threshold = self.context.get_verify_threshold(self._satellite_id)
            # Only declare early if clearly under threshold (extra margin).
            early_threshold = threshold * 0.75

            async with _MODEL_LOCK:
                loop = asyncio.get_running_loop()
                embedding = await loop.run_in_executor(
                    None, self.context.embedder.embed_pcm, audio_16k,
                )
                result = await loop.run_in_executor(
                    None,
                    self.context.speakers.verify_embedding,
                    embedding,
                    threshold,
                )

            if result is not None and result.is_match and result.distance <= early_threshold:
                self._early_match = result.matched_speaker
                self._early_match_id = result.matched_speaker_id
                self._early_distance = result.distance
                _LOGGER.info(
                    "[%s] EARLY MATCH: %s (d=%.4f, early_th=%.3f)",
                    sid, result.matched_speaker, result.distance, early_threshold,
                )
                # Push all recognition data to HA + MQTT immediately.
                spk = self.context.speakers.get_speaker(result.matched_speaker_id)
                self.context.publish_recognition(
                    speaker=result.matched_speaker,
                    confidence=self.context.confidence_for(result.distance),
                    satellite_id=self._satellite_id,
                    is_known=True,
                    distance=result.distance,
                    threshold=result.threshold,
                    nearest_speaker=result.matched_speaker,
                    role=spk.role if spk else None,
                )
            else:
                _LOGGER.debug(
                    "[%s] Early probe: no confident match (d=%.4f, early_th=%.3f)",
                    sid, result.distance if result else 2.0, early_threshold,
                )
        except asyncio.CancelledError:
            return
        except Exception:
            _LOGGER.debug("[%s] Early probe failed", sid, exc_info=True)

    async def _maybe_extract(self, verify_audio: bytes) -> bytes:
        """Run adaptive speaker extraction, returning the audio to verify.

        Returns ``verify_audio`` unchanged on the fast-path (single speech
        region), when extraction is disabled, when no speakers are
        enrolled, or when extraction can't confidently improve the clip.
        Never raises — any failure falls back to the original audio.
        """
        sid = self._session_id
        if not self.context.get_enable_extraction():
            return verify_audio
        if self.context.vad is None:
            return verify_audio
        if len(self.context.speakers.list_speakers()) == 0:
            return verify_audio

        ext_threshold = self.context.get_extraction_threshold(self._satellite_id)
        min_region = self.context.get_extraction_min_region_sec()
        try:
            async with _MODEL_LOCK:
                loop = asyncio.get_running_loop()
                result = await loop.run_in_executor(
                    None,
                    lambda: extract_target_speaker(
                        verify_audio,
                        vad=self.context.vad,
                        embedder=self.context.embedder,
                        speakers=self.context.speakers,
                        extraction_threshold=ext_threshold,
                        min_region_sec=min_region,
                    ),
                )
        except Exception:
            _LOGGER.debug("[%s] Extraction failed; using full audio", sid, exc_info=True)
            return verify_audio

        if result.applied:
            _LOGGER.info(
                "[%s] EXTRACTION: %d regions → kept %d for %s "
                "(dropped %.2fs of interfering audio)",
                sid, result.n_regions, result.n_kept,
                result.target_speaker, result.dropped_seconds,
            )
            return result.audio
        return verify_audio

    async def _wait_for_upstream_transcript(self, timeout: float = 15.0) -> str:
        """Block until the upstream reader resolves the transcript.

        15 s is well below HA's pipeline STT timeout (~30 s) so on a
        broken upstream we still return SOMETHING to the satellite
        instead of letting the whole pipeline hang.
        """
        sid = self._session_id
        if self._upstream_transcript is None:
            _LOGGER.debug("[%s] No upstream transcript future to await", sid)
            return ""
        t0 = time.monotonic()
        try:
            text = await asyncio.wait_for(
                asyncio.shield(self._upstream_transcript), timeout=timeout
            )
            wait_ms = (time.monotonic() - t0) * 1000
            _LOGGER.info(
                "[%s] Transcript ready after %.0fms: %r", sid, wait_ms, text
            )
            return text
        except asyncio.TimeoutError:
            _LOGGER.warning(
                "[%s] Upstream transcript timeout after %.0fs", sid, timeout
            )
            return ""

    async def _transcribe_voxtral(self, audio_16k: bytes) -> str:
        """Transcribe audio via the Voxtral (Mistral Cloud) API.

        Called instead of _wait_for_upstream_transcript when
        stt_backend == 'voxtral'. Falls back to empty string on error.
        """
        sid = self._session_id
        backend = self.context.get_voxtral_backend()
        if backend is None:
            _LOGGER.warning(
                "[%s] Voxtral backend selected but no API key configured", sid
            )
            return ""
        _LOGGER.info("[%s] Transcribing via Voxtral (%s)…", sid, backend.model)
        return await backend.transcribe(
            audio_16k,
            rate=16000,
            width=2,
            channels=1,
            language=self._language,
        )

    async def _get_transcript(self, audio_16k: bytes) -> str:
        """Obtain the transcript from whichever backend is active."""
        if self._is_voxtral:
            return await self._transcribe_voxtral(audio_16k)
        return await self._wait_for_upstream_transcript()

    # ------------------------------------------------------------------
    # Gate + verify
    # ------------------------------------------------------------------

    async def _finish_session(self) -> None:
        sid = self._session_id
        from murdock.core.audio import to_mono_16k_pcm

        # Cancel any in-flight early probe so it doesn't race with us.
        if self._early_probe_task and not self._early_probe_task.done():
            self._early_probe_task.cancel()

        raw_audio = bytes(self._audio_buffer)
        _LOGGER.info(
            "[%s] AudioStop from HA — buffered=%d bytes, upstream_open=%s, "
            "upstream_failed=%s, early_match=%s",
            sid, len(raw_audio), self._upstream_open, self._upstream_failed,
            self._early_match,
        )

        # Tell upstream that the audio stream is complete (Wyoming path only).
        if not self._is_voxtral:
            try:
                if self._upstream_open and self._upstream_client is not None:
                    await self._upstream_client.write_event(AudioStop().event())
                    _LOGGER.debug("[%s] AudioStop sent to upstream", sid)
            except Exception as exc:
                _LOGGER.warning(
                    "[%s] Failed to signal AudioStop upstream: %s", sid, exc
                )

        if not raw_audio:
            _LOGGER.info("[%s] Empty audio buffer → empty transcript", sid)
            await self._send_transcript_to_satellite("", label="empty-buffer")
            self._responded = True
            await self._close_upstream(send_stop=False)
            self._record_event(
                outcome=OUTCOME_EMPTY, duration_sec=0.0, transcript=""
            )
            return

        try:
            audio_16k = to_mono_16k_pcm(
                raw_audio,
                self._audio_rate,
                self._audio_width,
                self._audio_channels,
            )
        except ValueError as exc:
            _LOGGER.warning("[%s] Audio normalization failed: %s", sid, exc)
            audio_16k = raw_audio

        duration = len(audio_16k) / (16000 * 2) if audio_16k else 0.0
        _LOGGER.debug("[%s] Audio stop: %.2fs (%d bytes)", sid, duration, len(audio_16k))

        settings = self.context.settings
        speaker_count = len(self.context.speakers.list_speakers())

        # Gate 1: audio too short — passthrough.
        if duration < settings.min_verify_seconds:
            _LOGGER.info(
                "[%s] SHORT (%.2fs < %.2fs) — passthrough",
                sid, duration, settings.min_verify_seconds,
            )
            await self._passthrough_response(
                reason="short",
                outcome=OUTCOME_PASSTHROUGH_SHORT,
                duration=duration,
                audio_16k=audio_16k,
            )
            return

        # Gate 2: no speakers enrolled.
        if speaker_count == 0:
            if self.context.get_passthrough_when_empty():
                _LOGGER.info("[%s] NO SPEAKERS — passthrough (opt-in)", sid)
                await self._passthrough_response(
                    reason="no-speakers",
                    outcome=OUTCOME_PASSTHROUGH_NO_SPEAKERS,
                    duration=duration,
                    audio_16k=audio_16k,
                )
                return
            _LOGGER.info(
                "[%s] NO SPEAKERS and passthrough disabled — blocking", sid
            )
            await self._block_response(
                outcome=OUTCOME_BLOCKED_NO_SPEAKERS, duration=duration
            )
            return

        # Gate 2b: liveness / TV-noise rejection.
        min_liveness = self.context.get_min_liveness_score()
        if min_liveness > 0:
            try:
                liveness = await asyncio.to_thread(analyze_liveness, audio_16k)
                _LOGGER.debug(
                    "[%s] Liveness score=%.3f (threshold=%.3f)",
                    sid, liveness.score, min_liveness,
                )
                if liveness.score < min_liveness:
                    _LOGGER.info(
                        "[%s] TV/NOISE detected (liveness=%.3f < %.3f) — blocking",
                        sid, liveness.score, min_liveness,
                    )
                    self.context.publish_recognition(
                        speaker="tv-noise",
                        confidence=0.0,
                        satellite_id=self._satellite_id,
                        is_known=False,
                    )
                    await self._block_response(
                        outcome=OUTCOME_BLOCKED_TV_NOISE,
                        duration=duration,
                    )
                    self._log_total_latency("blocked-tv-noise")
                    return
            except Exception:
                _LOGGER.debug("[%s] Liveness analysis failed", sid, exc_info=True)

        # Trim leading chime for embedding only.
        skip_seconds = self.context.get_skip_leading_seconds()
        verify_audio = audio_16k
        if skip_seconds > 0 and audio_16k:
            skip_bytes = int(skip_seconds * 16000 * 2)
            skip_bytes -= skip_bytes % 2
            if skip_bytes < len(audio_16k):
                remaining_sec = (len(audio_16k) - skip_bytes) / (16000 * 2)
                if remaining_sec >= settings.min_verify_seconds:
                    verify_audio = audio_16k[skip_bytes:]
                    _LOGGER.debug(
                        "[%s] Skipped leading %.2fs (%.2fs left for verify)",
                        sid, skip_seconds, remaining_sec,
                    )

        # --- Fast path: early match already confirmed ---
        if self._early_match:
            _LOGGER.info(
                "[%s] EARLY MATCH confirmed: %s (d=%.4f) — skipping full verify",
                sid, self._early_match, self._early_distance or 0.0,
            )
            # HA events already fired in _run_early_probe. Just get transcript.
            transcript = await self._get_transcript(verify_audio)
            await self._send_transcript_to_satellite(
                transcript, label=f"match:{self._early_match}"
            )
            self._responded = True
            await self._close_upstream(send_stop=False)
            self._log_total_latency("match(early)")
            # Emotion classification runs AFTER the transcript has been
            # handed back to the satellite, so its latency never shows up
            # to the user. The helper is a no-op when the feature is off
            # or the model file is missing.
            emotion, emotion_conf = await self._classify_emotion(
                verify_audio, duration,
            )
            self._record_event(
                outcome=OUTCOME_MATCH,
                duration_sec=duration,
                matched_speaker=self._early_match,
                distance=self._early_distance,
                threshold=self.context.get_verify_threshold(self._satellite_id),
                verify_ms=0.0,
                transcript=transcript,
                emotion=emotion,
                emotion_confidence=emotion_conf,
            )
            # Auto-enroll on early match too.
            await self._maybe_auto_enroll(
                self._early_match_id, self._early_match,
                self._early_distance, verify_audio, duration,
            )
            return

        # Adaptive extraction: if the clip has multiple speech regions
        # (target + TV / second voice), isolate the dominant enrolled
        # speaker's regions so the embedding below isn't a blend. No-op
        # fast-path for single-region utterances.
        verify_audio = await self._maybe_extract(verify_audio)

        # Gate 3: full embedder + verify (runs in parallel with STT).
        threshold = self.context.get_verify_threshold(self._satellite_id)
        require_match = self.context.get_require_match()
        if await self.context.is_tv_playing(self._satellite_id):
            threshold = max(0.0, threshold - settings.tv_threshold_boost)
            _LOGGER.debug("[%s] TV playing — tightened threshold to %.3f", sid, threshold)

        verify_start = time.monotonic()
        embedding: Optional[np.ndarray] = None
        result = None
        async with _MODEL_LOCK:
            loop = asyncio.get_running_loop()
            try:
                embedding = await loop.run_in_executor(
                    None, self.context.embedder.embed_pcm, verify_audio
                )
                result = await loop.run_in_executor(
                    None,
                    self.context.speakers.verify_embedding,
                    embedding,
                    threshold,
                )
            except ValueError as exc:
                _LOGGER.info("[%s] Embedding failed: %s", sid, exc)
        verify_ms = (time.monotonic() - verify_start) * 1000

        if result is None:
            _LOGGER.info("[%s] Embedding unavailable — blocking", sid)
            await self._block_response(
                outcome=OUTCOME_BLOCKED_EMBED_FAILED,
                duration=duration,
                threshold=threshold,
                verify_ms=verify_ms,
            )
            return

        if result.is_match and result.matched_speaker:
            _LOGGER.info(
                "[%s] MATCH: %s (d=%.4f, th=%.3f, verify=%.0fms)",
                sid, result.matched_speaker, result.distance,
                result.threshold, verify_ms,
            )
            spk = self.context.speakers.get_speaker(result.matched_speaker_id)
            transcript = await self._get_transcript(verify_audio)
            await self._send_transcript_to_satellite(
                transcript, label=f"match:{result.matched_speaker}"
            )
            self._responded = True
            await self._close_upstream(send_stop=False)
            self._log_total_latency("match")
            # Emotion classification runs AFTER the transcript has been
            # returned to the satellite, so its latency never shows up to
            # the user. The HA push is fire-and-forget, so moving it to
            # after classification only shifts *when* HA starts seeing
            # the update by a few hundred ms, still faster than any
            # automation could act on it.
            emotion, emotion_conf = await self._classify_emotion(
                verify_audio, duration,
            )
            self.context.publish_recognition(
                speaker=result.matched_speaker,
                confidence=self.context.confidence_for(result.distance),
                satellite_id=self._satellite_id,
                is_known=True,
                distance=result.distance,
                threshold=result.threshold,
                nearest_speaker=result.matched_speaker,
                role=spk.role if spk else None,
                emotion=emotion,
                emotion_confidence=emotion_conf,
            )
            self._record_event(
                outcome=OUTCOME_MATCH,
                duration_sec=duration,
                matched_speaker=result.matched_speaker,
                distance=result.distance,
                threshold=result.threshold,
                verify_ms=verify_ms,
                transcript=transcript,
                emotion=emotion,
                emotion_confidence=emotion_conf,
            )
            # Aging / auto-enroll: add fresh embedding if distance is
            # informative (not too close, not borderline).
            await self._maybe_auto_enroll(
                result.matched_speaker_id, result.matched_speaker,
                result.distance, verify_audio, duration,
            )
            return

        # No match path.
        _LOGGER.info(
            "[%s] NO MATCH (best=%.4f, th=%.3f, verify=%.0fms, scores=%s)",
            sid, result.distance, result.threshold, verify_ms,
            ", ".join(f"{n}={d:.3f}" for n, d in result.all_distances.items()),
        )

        if self.context.get_unknown_logging() and embedding is not None:
            await self._log_unknown(verify_audio, embedding, result, duration)

        best_speaker = None
        if result.all_distances:
            best_speaker = next(iter(result.all_distances))

        self.context.publish_recognition(
            speaker="unknown",
            confidence=self.context.confidence_for(result.distance),
            satellite_id=self._satellite_id,
            is_known=False,
            distance=result.distance,
            threshold=result.threshold,
            nearest_speaker=best_speaker,
        )

        if not require_match:
            _LOGGER.info("[%s] require_match=false → forwarding anyway", sid)
            transcript = await self._get_transcript(verify_audio)
            await self._send_transcript_to_satellite(
                transcript, label="unknown-forwarded"
            )
            self._responded = True
            await self._close_upstream(send_stop=False)
            self._log_total_latency("unknown-forwarded")
            self._record_event(
                outcome=OUTCOME_UNKNOWN_FORWARDED,
                duration_sec=duration,
                matched_speaker=best_speaker,
                distance=result.distance,
                threshold=result.threshold,
                verify_ms=verify_ms,
                transcript=transcript,
            )
        else:
            await self._block_response(
                outcome=OUTCOME_BLOCKED_NO_MATCH,
                duration=duration,
                distance=result.distance,
                threshold=result.threshold,
                verify_ms=verify_ms,
                matched_speaker=best_speaker,
            )
            self._log_total_latency("blocked")

    # ------------------------------------------------------------------
    # Response helpers
    # ------------------------------------------------------------------

    async def _send_transcript_to_satellite(
        self, text: str, *, label: str
    ) -> None:
        """Single choke-point for every Transcript we send back to HA.

        Centralising it means we always log (a) what we're sending and
        (b) whether the write to the satellite socket succeeded. When a
        user reports "HA didn't get anything" the logs immediately tell
        us whether the bug is upstream or satellite-side.
        """
        sid = self._session_id
        try:
            await self.write_event(Transcript(text=text).event())
            _LOGGER.info(
                "[%s] → satellite Transcript (%s): %r", sid, label, text
            )
        except Exception as exc:
            _LOGGER.error(
                "[%s] Failed to write Transcript back to satellite: %s",
                sid, exc,
            )

    async def _passthrough_response(
        self,
        *,
        reason: str,
        outcome: str,
        duration: float,
        audio_16k: Optional[bytes] = None,
    ) -> None:
        """Forward upstream transcript back to the satellite unchanged."""
        transcript = await self._get_transcript(audio_16k or bytes(self._audio_buffer))
        await self._send_transcript_to_satellite(
            transcript, label=f"passthrough:{reason}"
        )
        self._responded = True
        await self._close_upstream(send_stop=False)
        self._log_total_latency(f"passthrough:{reason}")
        self._record_event(
            outcome=outcome, duration_sec=duration, transcript=transcript
        )

    async def _block_response(
        self,
        *,
        outcome: str = OUTCOME_BLOCKED_NO_MATCH,
        duration: float = 0.0,
        matched_speaker: Optional[str] = None,
        distance: Optional[float] = None,
        threshold: Optional[float] = None,
        verify_ms: Optional[float] = None,
    ) -> None:
        """Return an empty transcript and drop the upstream stream."""
        await self._send_transcript_to_satellite("", label=f"block:{outcome}")
        self._responded = True
        await self._close_upstream(send_stop=False)
        self._record_event(
            outcome=outcome,
            duration_sec=duration,
            matched_speaker=matched_speaker,
            distance=distance,
            threshold=threshold,
            verify_ms=verify_ms,
            transcript="",
        )

    async def _maybe_auto_enroll(
        self,
        speaker_id: Optional[int],
        speaker_name: Optional[str],
        distance: Optional[float],
        verify_audio: bytes,
        duration: float,
    ) -> None:
        """Conditionally auto-enroll a fresh embedding on match.

        Only fires when auto_enroll is enabled AND the distance is
        informative (between ``auto_enroll_min_distance`` and threshold).
        Very close matches (d < min) mean the profile is already good;
        adding them just wastes DB space.
        """
        if not self.context.get_auto_enroll():
            return
        if speaker_id is None or distance is None:
            return
        min_d = self.context.settings.auto_enroll_min_distance
        threshold = self.context.get_verify_threshold(self._satellite_id)
        if distance < min_d or distance > threshold:
            return
        sid = self._session_id
        try:
            async with _MODEL_LOCK:
                loop = asyncio.get_running_loop()
                embedding = await loop.run_in_executor(
                    None, self.context.embedder.embed_pcm, verify_audio,
                )
            await asyncio.to_thread(
                self.context.speakers.auto_enroll_embedding,
                speaker_id,
                embedding,
                verify_audio,
                duration,
                self.context.settings.auto_enroll_max_samples,
                self._satellite_id,
            )
            _LOGGER.info(
                "[%s] Auto-enrolled for %s (d=%.4f, min=%.3f)",
                sid, speaker_name, distance, min_d,
            )
        except Exception:
            _LOGGER.debug(
                "[%s] Auto-enroll failed for %s", sid, speaker_name,
                exc_info=True,
            )

    async def _classify_emotion(
        self, audio_16k: bytes, duration: float,
    ) -> tuple[Optional[str], Optional[float]]:
        """Run the emotion classifier if ready. Never raises.

        Returns ``(label, confidence)`` or ``(None, None)`` when the
        feature is off, the model file is missing, the clip is too
        short, or inference fails. Any failure is logged at debug level
        and treated as "no emotion info" — we never want a classifier
        hiccup to break the recognition path.
        """
        if not self.context.emotion_ready():
            return None, None
        classifier = self.context.emotion
        if classifier is None:
            return None, None
        if duration < classifier.min_duration_sec:
            return None, None
        sid = self._session_id
        try:
            loop = asyncio.get_running_loop()
            result = await loop.run_in_executor(
                None, classifier.classify_pcm, audio_16k,
            )
        except FileNotFoundError:
            # Model file disappeared mid-session (user deleted it, or
            # the flag was flipped before the model landed). Quiet log,
            # no panic.
            _LOGGER.debug("[%s] Emotion model not available", sid)
            return None, None
        except Exception:
            _LOGGER.debug(
                "[%s] Emotion classification failed", sid, exc_info=True,
            )
            return None, None
        # Filter out the model's garbage classes before they reach HA /
        # the audit log. Surfacing "unknown" as an emotion in HA is worse
        # than surfacing nothing.
        if not result.is_meaningful:
            _LOGGER.debug(
                "[%s] Emotion classifier returned non-meaningful label %r (conf=%.3f)",
                sid, result.label, result.confidence,
            )
            return None, None
        _LOGGER.info(
            "[%s] Emotion: %s (%.2f)", sid, result.label, result.confidence,
        )
        return result.label, float(result.confidence)

    def _record_event(
        self,
        *,
        outcome: str,
        duration_sec: float,
        matched_speaker: Optional[str] = None,
        distance: Optional[float] = None,
        threshold: Optional[float] = None,
        verify_ms: Optional[float] = None,
        transcript: Optional[str] = None,
        emotion: Optional[str] = None,
        emotion_confidence: Optional[float] = None,
    ) -> None:
        """Persist a recognition audit row. Never raises."""
        try:
            self.context.recognition.record(
                session_id=self._session_id,
                satellite_id=self._satellite_id,
                duration_sec=duration_sec,
                outcome=outcome,
                matched_speaker=matched_speaker,
                distance=distance,
                threshold=threshold,
                verify_ms=verify_ms,
                transcript=transcript,
                emotion=emotion,
                emotion_confidence=emotion_confidence,
            )
        except Exception:
            _LOGGER.debug(
                "[%s] Failed to record audit row", self._session_id,
                exc_info=True,
            )

    async def _log_unknown(
        self,
        audio_16k: bytes,
        embedding: np.ndarray,
        result,
        duration: float,
    ) -> None:
        sid = self._session_id
        try:
            liveness = await asyncio.to_thread(analyze_liveness, audio_16k)
            best_speaker = None
            if result.all_distances:
                best_speaker = next(iter(result.all_distances))
            await asyncio.to_thread(
                self.context.unknown.record,
                self._session_id,
                audio_16k,
                embedding,
                duration,
                result.distance,
                best_speaker,
                self._satellite_id,
                float(liveness.score),
            )
        except Exception:
            _LOGGER.exception("[%s] Failed to log unknown sample", sid)

    def _log_total_latency(self, outcome: str) -> None:
        if self._stream_start is None:
            return
        total_ms = (time.monotonic() - self._stream_start) * 1000
        _LOGGER.info(
            "[%s] session done (%s) — wall=%.0fms", self._session_id, outcome, total_ms
        )

    async def disconnect(self) -> None:
        """Wyoming server hook — clean up upstream connection on client close."""
        try:
            await self._close_upstream(send_stop=False)
        except Exception:
            pass
        for task in (self._upstream_reader_task, self._early_probe_task):
            if task is not None and not task.done():
                task.cancel()
                try:
                    await task
                except (asyncio.CancelledError, Exception):
                    pass
