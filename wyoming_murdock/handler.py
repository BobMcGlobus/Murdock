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
import re
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
    OUTCOME_BLOCKED_EARLY_REJECT,
    OUTCOME_BLOCKED_EMBED_FAILED,
    OUTCOME_BLOCKED_NO_MATCH,
    OUTCOME_BLOCKED_NO_SPEAKERS,
    OUTCOME_BLOCKED_TV_NOISE,
    OUTCOME_BLOCKED_UNCERTAIN,
    OUTCOME_EMPTY,
    OUTCOME_MATCH,
    OUTCOME_PASSTHROUGH_NO_SPEAKERS,
    OUTCOME_PASSTHROUGH_SHORT,
    OUTCOME_UNCERTAIN_FORWARDED,
    OUTCOME_UNKNOWN_FORWARDED,
)

_LOGGER = logging.getLogger("murdock.wyoming.handler")

# Embedder is not thread-safe when the underlying ORT session is shared,
# so serialise calls across connections.
_MODEL_LOCK = asyncio.Lock()

# When media is playing in the satellite's room, the early-reject margin
# is multiplied by this — background audio is the likely culprit, so the
# reject may get bolder.
_EARLY_REJECT_MEDIA_FACTOR = 0.5

# Dual-transcript mode waits at most this long for the shadow engine
# before answering with the primary transcript alone. Short commands
# transcribe in ~1 s on the supported cloud backends, so this budget is
# generous without ever hanging the pipeline on a dead second provider.
_DUAL_SHADOW_TIMEOUT = 6.0


_TEMPLATE_VAR = re.compile(r"\{\{\s*(\w+)\s*\}\}")


def render_transcript_template(template: str, values: dict) -> str:
    """Substitute ``{{ var }}`` placeholders in a transcript template.

    Plain variable substitution — NOT Jinja2: no conditions or loops
    (the known/unknown split is handled by picking the right template).
    Unknown placeholders render empty, so a typo can't leak ``{{ foo }}``
    into the text the LLM sees.
    """
    return _TEMPLATE_VAR.sub(lambda m: str(values.get(m.group(1), "")), template)


def early_reject_decision(
    distance: float,
    effective_threshold: float,
    margin: float,
    media_playing: bool,
) -> bool:
    """Decide whether an utterance is catastrophically off-profile.

    Deliberately conservative: the reject bar sits a full ``margin``
    above the (already satellite/speaker-resolved) accept threshold, so
    an enrolled user on a bad day lands between the two and simply runs
    through the normal full verification. Media in the room halves the
    margin.
    """
    if media_playing:
        margin *= _EARLY_REJECT_MEDIA_FACTOR
    return distance >= effective_threshold + margin


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
        # Room/area of the satellite (from the active_satellite signal),
        # used to check whether media is playing in the same room.
        self._satellite_area: Optional[str] = None

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

        # Early-reject state (opt-in). Once rejected we stop buffering and
        # forwarding; the empty transcript goes out at stream end.
        self._reject_probe_task: Optional[asyncio.Task] = None
        self._reject_probed: bool = False
        self._early_rejected: bool = False
        self._reject_distance: Optional[float] = None
        self._reject_bar: Optional[float] = None
        self._reject_nearest: Optional[str] = None

        # Dual-transcript mode: cached (text, engine) of the blocking
        # shadow run, so the audit logging doesn't transcribe twice.
        self._dual_shadow: Optional[tuple[str, str]] = None

        # Sidecar hint mode: ambiguities pulled out of the transcript,
        # travelling in the recognition event instead (plan §13).
        self._transcript_hints: list = []

        # How long each STT engine took on this utterance, so the A/B
        # comparison in the log covers speed, not just wording.
        self._transcript_ms: Optional[float] = None
        self._shadow_ms: Optional[float] = None

        # Canonicalizations applied to this transcript, for the log.
        self._canonicalizations: list = []

        # Whisper detection result for this utterance (experimental).
        self._whisper: bool = False

        # Every enrolled speaker heard in this utterance (from extraction).
        self._speakers: list = []

    @property
    def upstream_uri(self) -> str:
        """Always read the live value from context so the UI can update
        it without a service restart."""
        return self.context.get_upstream_uri()

    @property
    def _is_voxtral(self) -> bool:
        """True when the active backend is cloud/HTTP (Voxtral or
        OpenAI-compatible) — i.e. buffered transcription instead of the
        streaming Wyoming upstream. Name kept for the many call sites;
        semantics are "is a buffering cloud backend".
        """
        return self.context.get_stt_backend() in ("voxtral", "openai")

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
            # After an early reject the session is decided: stop buffering
            # and forwarding, just drain the stream until AudioStop.
            if self._early_rejected or self._responded:
                return True
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

            # --- Early reject (opt-in) ---
            # Fires once there are ~1.5 s of voice AFTER the chime trim,
            # i.e. later than the accept probe — a reject needs more
            # evidence than a confirm. With no speakers enrolled there is
            # nothing to be "far away from", so the probe never arms.
            if (
                not self._reject_probed
                and not self._early_match
                and self.context.get_enable_early_reject()
            ):
                skip_bytes = int(
                    self.context.get_skip_leading_seconds() * 16000 * 2
                )
                if len(self._audio_buffer) >= skip_bytes + 48000:
                    self._reject_probed = True
                    if len(self.context.speakers.list_speakers()) > 0:
                        snapshot = bytes(self._audio_buffer)
                        self._reject_probe_task = asyncio.create_task(
                            self._run_early_reject(snapshot)
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
        self._reject_probe_task = None
        self._reject_probed = False
        self._early_rejected = False
        self._reject_distance = None
        self._reject_bar = None
        self._reject_nearest = None
        self._dual_shadow = None
        self._transcript_hints = []
        self._transcript_ms = None
        self._shadow_ms = None
        self._canonicalizations = []
        self._whisper = False
        self._speakers = []
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
        self._resolve_satellite_id()
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
            verify_kwargs = self._verify_kwargs()

            async with _MODEL_LOCK:
                loop = asyncio.get_running_loop()
                embedding = await loop.run_in_executor(
                    None, self.context.embedder.embed_pcm, audio_16k,
                )
                result = await loop.run_in_executor(
                    None,
                    lambda: self.context.speakers.verify_embedding(
                        embedding, threshold, **verify_kwargs
                    ),
                )

            # Only declare early if clearly under the effective threshold
            # (extra margin against the short-window noise).
            early_threshold = result.threshold * 0.75 if result else 0.0
            margin, uncertain = self._margin_gate_uncertain(result)
            if uncertain:
                # An ambiguous identity must never take the early
                # shortcut — fall through to the full verify on stop.
                _LOGGER.debug(
                    "[%s] Early probe: margin %.4f below gate — no early match",
                    sid, margin,
                )
            elif result is not None and result.is_match and result.distance <= early_threshold:
                self._early_match = result.matched_speaker
                self._early_match_id = result.matched_speaker_id
                self._early_distance = result.distance
                _LOGGER.info(
                    "[%s] EARLY MATCH: %s (d=%.4f, early_th=%.3f)",
                    sid, result.matched_speaker, result.distance, early_threshold,
                )
                # Push all recognition data to HA + MQTT immediately.
                spk = self.context.speakers.get_speaker(result.matched_speaker_id)
                nearest, nearest_d = self._nearest_of(result)
                self.context.publish_recognition(
                    speaker=result.matched_speaker,
                    confidence=self.context.confidence_for(result.distance),
                    satellite_id=self._satellite_id,
                    is_known=True,
                    distance=result.distance,
                    threshold=result.threshold,
                    nearest_speaker=nearest,
                    nearest_distance=nearest_d,
                    margin=margin,
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

    async def _run_early_reject(self, audio_snapshot: bytes) -> None:
        """Background task: drop the session early when the voice is
        catastrophically far from every enrolled profile.

        The reject bar is ``effective_threshold + margin`` — far above
        the accept threshold, so enrolled users on a bad day fall in the
        gap and proceed to the normal full verification. Media playing in
        the satellite's room (MQTT context) halves the margin. On reject
        we stop forwarding to the STT immediately (CPU/cloud saved, no
        unknown audio leaves the house) and answer instantly at stream
        end; the rejected audio still lands in the unknown postbox so a
        false reject is one click from becoming training data.
        """
        sid = self._session_id
        try:
            from murdock.core.audio import to_mono_16k_pcm
            audio_16k = to_mono_16k_pcm(
                audio_snapshot, self._audio_rate,
                self._audio_width, self._audio_channels,
            )
            skip_seconds = self.context.get_skip_leading_seconds()
            if skip_seconds > 0:
                skip_bytes = int(skip_seconds * 16000 * 2)
                skip_bytes -= skip_bytes % 2
                if skip_bytes < len(audio_16k):
                    audio_16k = audio_16k[skip_bytes:]
            # Need a meaningful window: ≥1 s of voice after the trim.
            if len(audio_16k) < 16000 * 2:
                return

            self._resolve_satellite_id()
            threshold = self.context.get_verify_threshold(self._satellite_id)
            margin = self.context.get_early_reject_margin()
            media_tighten = await self.context.compute_media_tightening(
                self._satellite_id, self._satellite_area
            )
            verify_kwargs = self._verify_kwargs()

            async with _MODEL_LOCK:
                loop = asyncio.get_running_loop()
                embedding = await loop.run_in_executor(
                    None, self.context.embedder.embed_pcm, audio_16k,
                )
                result = await loop.run_in_executor(
                    None,
                    lambda: self.context.speakers.verify_embedding(
                        embedding, threshold, **verify_kwargs
                    ),
                )

            if result is None or result.is_match or self._early_match:
                return
            if not early_reject_decision(
                result.distance, result.threshold, margin,
                media_playing=media_tighten > 0,
            ):
                _LOGGER.debug(
                    "[%s] Early-reject probe: not far enough (d=%.4f)",
                    sid, result.distance,
                )
                return

            # A whisper is far from every profile by construction — the
            # embedding has no voicing to match against. Rejecting it here
            # would silently kill exactly the utterances the whisper
            # feature exists to notice, so check before dropping.
            whisper_probe = self.context.detect_whisper(audio_16k)
            if whisper_probe is not None and whisper_probe.is_whisper:
                self._whisper = True
                _LOGGER.info(
                    "[%s] Early reject suppressed: utterance is whispered "
                    "(score=%.2f)", sid, whisper_probe.score,
                )
                return

            bar = result.threshold + (
                margin * _EARLY_REJECT_MEDIA_FACTOR if media_tighten > 0 else margin
            )
            nearest = next(iter(result.all_distances), None)
            self._early_rejected = True
            self._reject_distance = result.distance
            self._reject_bar = bar
            self._reject_nearest = nearest
            _LOGGER.info(
                "[%s] EARLY REJECT: d=%.4f ≥ bar=%.3f (th=%.3f, media=%s) — "
                "stopping STT forward",
                sid, result.distance, bar, result.threshold,
                media_tighten > 0,
            )
            await self._close_upstream(send_stop=False)
            # Capture for training + notify HA right away.
            if self.context.get_unknown_logging():
                try:
                    liveness = await asyncio.to_thread(analyze_liveness, audio_16k)
                    await asyncio.to_thread(
                        self.context.unknown.record,
                        self._session_id,
                        audio_16k,
                        embedding,
                        len(audio_16k) / (16000 * 2),
                        result.distance,
                        nearest,
                        self._satellite_id,
                        float(liveness.score),
                    )
                except Exception:
                    _LOGGER.debug("[%s] Early-reject capture failed", sid,
                                  exc_info=True)
            self.context.publish_recognition(
                speaker="unknown",
                confidence=self.context.confidence_for(result.distance),
                satellite_id=self._satellite_id,
                is_known=False,
                distance=result.distance,
                threshold=bar,
                nearest_speaker=nearest,
                nearest_distance=result.distance,
                margin=self._margin_of(result),
                reason="early-reject",
                whisper=self._whisper,
                speakers=self._speakers or None,
            )
        except asyncio.CancelledError:
            return
        except Exception:
            _LOGGER.debug("[%s] Early-reject probe failed", sid, exc_info=True)

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

        # Extraction already scored every speech region against the
        # enrolled speakers to pick the dominant one, so "who else spoke"
        # comes for free. The gate still follows the dominant speaker.
        self._speakers = list(getattr(result, "speakers", []) or [])
        if len(self._speakers) > 1:
            _LOGGER.info(
                "[%s] Multiple known voices: %s",
                sid,
                ", ".join(
                    f"{e['speaker']} ({e['seconds']:.1f}s)"
                    for e in self._speakers
                ),
            )

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

    async def _transcribe_cloud(self, audio_16k: bytes) -> str:
        """Transcribe buffered audio via the active cloud backend.

        On backend failure (network down, provider outage) and with the
        local fallback enabled, the same audio is transcribed one-shot
        over the Wyoming upstream instead of silently returning "".
        """
        from murdock.core.stt_backend import (
            STTBackendError,
            transcribe_via_wyoming,
        )

        sid = self._session_id
        backend = self.context.get_active_cloud_backend()
        failed = False
        if backend is None:
            _LOGGER.warning(
                "[%s] Cloud backend %r selected but not configured "
                "(missing API key / model)",
                sid, self.context.get_stt_backend(),
            )
            failed = True
        else:
            _LOGGER.info("[%s] Transcribing via %s…", sid, backend.label)
            try:
                return await backend.transcribe(
                    audio_16k,
                    rate=16000,
                    width=2,
                    channels=1,
                    language=self._language,
                )
            except STTBackendError as exc:
                _LOGGER.warning("[%s] Cloud STT failed: %s", sid, exc)
                failed = True

        if failed and self.context.get_stt_local_fallback():
            uri = self.context.get_upstream_uri()
            if uri:
                _LOGGER.info(
                    "[%s] Falling back to local Wyoming STT at %s", sid, uri
                )
                try:
                    return await transcribe_via_wyoming(
                        uri, audio_16k, language=self._language
                    )
                except STTBackendError as exc:
                    _LOGGER.warning(
                        "[%s] Local fallback also failed: %s", sid, exc
                    )
        return ""

    async def _primary_transcript(self, audio_16k: bytes) -> str:
        """Obtain the transcript from whichever primary backend is active.

        Records the elapsed time in ``_transcript_ms``. In upstream mode
        this is the *wait* for a stream that has been decoding in
        parallel since the first chunk, so it understates the engine's
        own work — in cloud mode it is the full request.
        """
        started = time.monotonic()
        try:
            if self._is_voxtral:
                return await self._transcribe_cloud(audio_16k)
            return await self._wait_for_upstream_transcript()
        finally:
            self._transcript_ms = (time.monotonic() - started) * 1000

    def _republish_hints(self, **kwargs) -> None:
        """Re-fire the recognition event once sidecar hints are known.

        Some paths publish before the transcript exists (early match, the
        no-match forward). Rather than delaying the event — automations
        want the speaker as early as possible — the hints are delivered
        by a second event with otherwise identical data. No hints, no
        second event, so the common case still fires exactly once.
        """
        if not self._transcript_hints:
            return
        self.context.publish_recognition(
            satellite_id=self._satellite_id,
            ambiguities=self._transcript_hints,
            **kwargs,
        )

    def _deliver_hints(self, result, label: str) -> str:
        """Pick the transcript rendering and stash hints for the event.

        Ambiguity markers are useful to an LLM and poison to HA's local
        intent matcher, so where they go is a setting (plan §13):
        ``inline`` keeps them in the text, ``sidecar`` moves them into the
        recognition event, ``clean`` drops them. Before any of that,
        hints that resolve against a known entity name are decided rather
        than passed on.
        """
        from murdock.core.transcript_tools import render_hints, resolve_hints

        sid = self._session_id
        mode = self.context.effective_transcript_hint_mode()
        if mode != "inline" and result.hints:
            try:
                known = self.context.get_vocabulary_store().terms()
            except Exception:
                known = []
            result = resolve_hints(result, known)

        if mode == "inline":
            if result.annotated != result.clean:
                _LOGGER.info("[%s] %s: %r", sid, label, result.annotated[:120])
            return result.annotated

        if mode == "sidecar" and result.hints:
            self._transcript_hints = render_hints(result.hints)
            _LOGGER.info(
                "[%s] %s → sidecar: %d hint(s), transcript stays clean: %r",
                sid, label, len(result.hints), result.clean[:120],
            )
        elif result.hints:
            _LOGGER.debug(
                "[%s] %s → %s: %d hint(s) dropped",
                sid, label, mode, len(result.hints),
            )
        elif result.clean != result.annotated:
            _LOGGER.info("[%s] %s: %r", sid, label, result.clean[:120])
        return result.clean

    async def _get_transcript(self, audio_16k: bytes) -> str:
        """Transcript for this utterance, fully post-processed.

        Order matters: the explicit dictionary and the dual-engine merge
        run first (see :meth:`_transcribe_and_correct`), then
        canonicalization maps whatever is left onto real entity names.
        User-written rules therefore always win over a fuzzy match.
        """
        transcript = await self._transcribe_and_correct(audio_16k)
        corrected, replacements = self.context.canonicalize_transcript(
            transcript
        )
        if replacements:
            self._canonicalizations = [r.as_dict() for r in replacements]
            _LOGGER.info(
                "[%s] Canonicalised %d span(s): %s",
                self._session_id, len(replacements),
                ", ".join(f"{r.original}→{r.replacement}" for r in replacements),
            )
        return corrected

    async def _transcribe_and_correct(self, audio_16k: bytes) -> str:
        """Primary transcript, refined by the enabled quality tiers.

        * **Dual transcript**: run the shadow engine blocking in
          parallel and merge both readings, marking disagreements as
          ``primary [oder: shadow]``. The shadow result is cached so the
          post-response audit logging doesn't transcribe twice. A slow
          or failing shadow never blocks: after ``_DUAL_SHADOW_TIMEOUT``
          the primary goes out alone.
        * **Correction dictionary**: applied to each transcript *before*
          the merge, so a term both engines got wrong (or one engine's
          known quirk) is corrected first and produces no false
          disagreement.

        Where the resulting ambiguity markers end up is decided by
        :meth:`_deliver_hints`.
        """
        sid = self._session_id
        dictionary = self.context.get_dictionary_entries()

        if self.context.dual_transcript_active() and audio_16k:
            from murdock.core.transcript_tools import merge_transcripts_ex

            shadow_task = asyncio.create_task(
                self._shadow_transcribe(audio_16k)
            )
            primary = await self._primary_transcript(audio_16k)
            shadow_text: Optional[str] = None
            try:
                shadow_text, shadow_engine = await asyncio.wait_for(
                    shadow_task, timeout=_DUAL_SHADOW_TIMEOUT
                )
                self._dual_shadow = (shadow_text, shadow_engine)
            except asyncio.TimeoutError:
                shadow_task.cancel()
                _LOGGER.warning(
                    "[%s] Dual: shadow timed out after %.0fs — primary only",
                    sid, _DUAL_SHADOW_TIMEOUT,
                )
                self._dual_shadow = ("⚠ timeout", "dual (timed out)")
            except Exception as exc:
                _LOGGER.info("[%s] Dual: shadow failed: %s — primary only", sid, exc)
                self._dual_shadow = (f"⚠ {exc}", "dual (failed)")

            dict_hints = []
            if dictionary:
                from murdock.core.transcript_tools import (
                    apply_correction_dictionary,
                    apply_correction_dictionary_ex,
                )
                primary_res = apply_correction_dictionary_ex(primary, dictionary)
                primary = primary_res.clean
                dict_hints = primary_res.hints
                if shadow_text:
                    shadow_text = apply_correction_dictionary_ex(
                        shadow_text, dictionary
                    ).clean
            if shadow_text:
                merged = merge_transcripts_ex(primary, shadow_text)
                if dict_hints:
                    # Dictionary hints ride along with the merge's own.
                    merged.hints[:0] = dict_hints
                    merged.annotated = apply_correction_dictionary(
                        merged.annotated, dictionary
                    )
                return self._deliver_hints(merged, "Dual merge")
            if dict_hints:
                return self._deliver_hints(
                    apply_correction_dictionary_ex(primary, dictionary),
                    "Dictionary applied",
                )
            return primary

        transcript = await self._primary_transcript(audio_16k)
        if dictionary and transcript:
            from murdock.core.transcript_tools import (
                apply_correction_dictionary_ex,
            )
            return self._deliver_hints(
                apply_correction_dictionary_ex(transcript, dictionary),
                "Dictionary applied",
            )
        return transcript

    # ------------------------------------------------------------------
    # Gate + verify
    # ------------------------------------------------------------------

    def _resolve_satellite_id(self) -> None:
        """Fill in the satellite id from MQTT when HA didn't send one.

        HA's Assist pipeline doesn't pass the originating device to the
        Wyoming STT stage (``Transcribe.name`` arrives as None), so the
        satellite is learned from the ``murdock/active_satellite`` topic
        that a small HA automation publishes when a satellite starts
        listening. Only fills when still unknown, so a name explicitly
        sent by a direct Wyoming satellite always wins.
        """
        # The room is useful even when HA did send a Wyoming name, so
        # refresh it independently of the id.
        try:
            area = self.context.mqtt.get_active_satellite_area()
        except Exception:
            area = None
        if area:
            self._satellite_area = area
        if self._satellite_id:
            return
        try:
            active = self.context.mqtt.get_active_satellite()
        except Exception:
            return
        if active:
            self._satellite_id = active
            _LOGGER.info(
                "[%s] Satellite resolved via MQTT: %s (area=%s)",
                self._session_id, active, self._satellite_area,
            )

    def _verify_kwargs(self, tighten: float = 0.0) -> dict:
        """Assemble the optional verify refinements for this session.

        Pulls the per-satellite sub-profile id (when the feature is on and
        the satellite is known) and the adaptive per-speaker thresholds
        (when enabled and fitted). Centralised so the early probe and the
        full verify gate behave identically.
        """
        kwargs: dict = {"tighten": tighten}
        try:
            if self._satellite_id and self.context.get_enable_satellite_profiles():
                kwargs["satellite_id"] = self._satellite_id
            if self.context.get_enable_adaptive_thresholds():
                adaptive = self.context.get_adaptive_thresholds()
                if adaptive:
                    kwargs["speaker_thresholds"] = adaptive
                    kwargs["global_threshold"] = self.context.get_verify_threshold()
        except Exception:
            _LOGGER.debug(
                "[%s] verify-kwargs assembly failed", self._session_id,
                exc_info=True,
            )
        return kwargs

    @staticmethod
    def _margin_of(result) -> Optional[float]:
        """Distance gap between the best and second-best speaker.

        None with fewer than two enrolled profiles — the margin gate
        can't say anything then and stays out of the way.
        """
        if not result or len(result.all_distances) < 2:
            return None
        dists = sorted(result.all_distances.values())
        return float(dists[1] - dists[0])

    @staticmethod
    def _nearest_of(result) -> tuple[Optional[str], Optional[float]]:
        """(name, distance) of the nearest profile for the event payload.

        For a match that's the runner-up (feeds the margin gate on the
        consumer side); for a non-match it's the best candidate.
        """
        if not result or not result.all_distances:
            return None, None
        items = sorted(result.all_distances.items(), key=lambda kv: kv[1])
        if result.is_match:
            if len(items) < 2:
                return None, None
            name, dist = items[1]
        else:
            name, dist = items[0]
        return name, float(dist)

    def _margin_gate_uncertain(self, result) -> tuple[Optional[float], bool]:
        """Apply the margin gate: (margin, uncertain).

        A match whose runner-up is closer than the configured minimum
        margin is downgraded to "uncertain" — right absolute distance,
        but two profiles are plausibly the same voice.
        """
        margin = self._margin_of(result)
        if not result or not result.is_match:
            return margin, False
        gate = self.context.get_margin_gate(self._satellite_id)
        if gate > 0 and margin is not None and margin < gate:
            return margin, True
        return margin, False

    async def _finish_session(self) -> None:
        sid = self._session_id
        self._resolve_satellite_id()
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

        # Early reject already decided this session: STT was cut off
        # mid-stream, so the empty transcript is ready instantly.
        if self._early_rejected:
            await self._send_transcript_to_satellite("", label="block:early-reject")
            self._responded = True
            await self._close_upstream(send_stop=False)
            self._record_event(
                outcome=OUTCOME_BLOCKED_EARLY_REJECT,
                duration_sec=duration,
                matched_speaker=self._reject_nearest,
                distance=self._reject_distance,
                threshold=self._reject_bar,
                transcript="",
            )
            self._log_total_latency("blocked-early-reject")
            return

        settings = self.context.settings
        speaker_count = len(self.context.speakers.list_speakers())

        # Whisper detection runs BEFORE the gates on purpose: whispered
        # speech is quiet and spectrally flat, which is exactly what the
        # liveness heuristic rejects as playback. Without checking first,
        # Murdock would discard the very utterances it should notice.
        whisper_features = self.context.detect_whisper(audio_16k)
        if whisper_features is not None and whisper_features.is_whisper:
            self._whisper = True
            _LOGGER.info(
                "[%s] WHISPER detected (score=%.2f, voiced=%.2f) — skipping "
                "the liveness gate for this utterance",
                sid, whisper_features.score, whisper_features.voiced_ratio,
            )

        # Gate 1: audio too short — passthrough.
        if duration < settings.min_verify_seconds:
            _LOGGER.info(
                "[%s] SHORT (%.2fs < %.2fs) — passthrough",
                sid, duration, settings.min_verify_seconds,
            )
            # Fire a null-speaker event so downstream state resets: this
            # utterance was never verified (plan §21).
            self.context.publish_recognition(
                speaker="unknown",
                confidence=0.0,
                satellite_id=self._satellite_id,
                is_known=False,
                reason="short",
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
            # Bootstrap path: with nothing enrolled there's nobody to match
            # against, but we still capture the utterance to the unknown
            # postbox so the user can assign it to a (new) speaker from the
            # UI and train entirely over the voice satellite.
            await self._capture_for_training(audio_16k, duration)
            self.context.publish_recognition(
                speaker="unknown",
                confidence=0.0,
                satellite_id=self._satellite_id,
                is_known=False,
                reason="no-speakers",
            )
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

        # Gate 2b: liveness / TV-noise rejection. Skipped for whispers —
        # see above; the two look alike to a spectral heuristic.
        min_liveness = self.context.get_min_liveness_score()
        if min_liveness > 0 and not self._whisper:
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
            early_spk = (
                self.context.speakers.get_speaker(self._early_match_id)
                if self._early_match_id is not None else None
            )
            self._republish_hints(
                speaker=self._early_match,
                confidence=self.context.confidence_for(
                    self._early_distance or 0.0
                ),
                is_known=True,
                distance=self._early_distance,
                threshold=self.context.get_verify_threshold(self._satellite_id),
                nearest_speaker=self._early_match,
                role=early_spk.role if early_spk else None,
            )
            await self._send_transcript_to_satellite(
                self._augment_transcript(
                    transcript, is_known=True, speaker=self._early_match,
                    role=early_spk.role if early_spk else None,
                    distance=self._early_distance, nearest=self._early_match,
                ),
                label=f"match:{self._early_match}",
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
            early_threshold = self.context.get_verify_threshold(self._satellite_id)
            event_id = self._record_event(
                outcome=OUTCOME_MATCH,
                duration_sec=duration,
                matched_speaker=self._early_match,
                distance=self._early_distance,
                threshold=early_threshold,
                verify_ms=0.0,
                transcript=transcript,
                emotion=emotion,
                emotion_confidence=emotion_conf,
                weight=self.context.compute_speaker_weight(
                    self._early_distance, early_threshold
                ),
            )
            self._spawn_shadow(event_id, verify_audio)
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
        tighten = await self.context.compute_media_tightening(
            self._satellite_id, self._satellite_area
        )
        if tighten > 0:
            _LOGGER.debug(
                "[%s] Media playing — tightening threshold by %.3f", sid, tighten,
            )
        verify_kwargs = self._verify_kwargs(tighten=tighten)

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
                    lambda: self.context.speakers.verify_embedding(
                        embedding, threshold, **verify_kwargs
                    ),
                )
            except ValueError as exc:
                _LOGGER.info("[%s] Embedding failed: %s", sid, exc)
        verify_ms = (time.monotonic() - verify_start) * 1000

        if result is None:
            _LOGGER.info("[%s] Embedding unavailable — blocking", sid)
            self.context.publish_recognition(
                speaker="unknown",
                confidence=0.0,
                satellite_id=self._satellite_id,
                is_known=False,
                reason="embed-failed",
            )
            await self._block_response(
                outcome=OUTCOME_BLOCKED_EMBED_FAILED,
                duration=duration,
                threshold=threshold,
                verify_ms=verify_ms,
            )
            return

        margin, uncertain = self._margin_gate_uncertain(result)
        weight = self.context.compute_speaker_weight(
            result.distance, result.threshold
        )

        if result.is_match and result.matched_speaker and not uncertain:
            _LOGGER.info(
                "[%s] MATCH: %s (d=%.4f, th=%.3f, verify=%.0fms%s)",
                sid, result.matched_speaker, result.distance,
                result.threshold, verify_ms,
                ", via sat-profile" if result.used_satellite_profile else "",
            )
            spk = self.context.speakers.get_speaker(result.matched_speaker_id)
            transcript = await self._get_transcript(verify_audio)
            nearest, nearest_d = self._nearest_of(result)
            # Publish BEFORE answering — and wait for it. HA starts the
            # intent stage (where the agent's prompt is built) within a
            # millisecond of receiving the transcript, so a speaker
            # published afterwards can only reach the *next* turn.
            await self.context.publish_recognition_now(
                speaker=result.matched_speaker,
                confidence=self.context.confidence_for(result.distance),
                satellite_id=self._satellite_id,
                is_known=True,
                distance=result.distance,
                threshold=result.threshold,
                nearest_speaker=nearest,
                nearest_distance=nearest_d,
                margin=margin,
                weight=weight,
                role=spk.role if spk else None,
                ambiguities=self._transcript_hints or None,
                whisper=self._whisper,
                speakers=self._speakers or None,
            )
            await self._send_transcript_to_satellite(
                self._augment_transcript(
                    transcript, is_known=True, speaker=result.matched_speaker,
                    role=spk.role if spk else None,
                    distance=result.distance, nearest=result.matched_speaker,
                ),
                label=f"match:{result.matched_speaker}",
            )
            self._responded = True
            await self._close_upstream(send_stop=False)
            self._log_total_latency("match")
            # Emotion classification runs AFTER the transcript has been
            # returned to the satellite, so its latency never shows up to
            # the user. The speaker itself already went out above; only
            # when there is an emotion to add does a second event follow.
            emotion, emotion_conf = await self._classify_emotion(
                verify_audio, duration,
            )
            if emotion:
                self.context.publish_recognition(
                    speaker=result.matched_speaker,
                    confidence=self.context.confidence_for(result.distance),
                    satellite_id=self._satellite_id,
                    is_known=True,
                    distance=result.distance,
                    threshold=result.threshold,
                    nearest_speaker=nearest,
                    nearest_distance=nearest_d,
                    margin=margin,
                    weight=weight,
                    role=spk.role if spk else None,
                    emotion=emotion,
                    emotion_confidence=emotion_conf,
                    ambiguities=self._transcript_hints or None,
                )
            event_id = self._record_event(
                outcome=OUTCOME_MATCH,
                duration_sec=duration,
                matched_speaker=result.matched_speaker,
                distance=result.distance,
                threshold=result.threshold,
                verify_ms=verify_ms,
                transcript=transcript,
                emotion=emotion,
                emotion_confidence=emotion_conf,
                weight=weight,
                margin=margin,
            )
            self._spawn_shadow(event_id, verify_audio)
            # Aging / auto-enroll: add fresh embedding if distance is
            # informative (not too close, not borderline).
            await self._maybe_auto_enroll(
                result.matched_speaker_id, result.matched_speaker,
                result.distance, verify_audio, duration,
            )
            return

        # No match — or a match downgraded by the margin gate.
        if uncertain:
            _LOGGER.info(
                "[%s] UNCERTAIN: best=%s d=%.4f th=%.3f but margin=%.4f "
                "below gate (verify=%.0fms)",
                sid, result.matched_speaker, result.distance,
                result.threshold, margin, verify_ms,
            )
            sentinel = "uncertain"
            best_speaker = result.matched_speaker
            best_distance = result.distance
            fwd_outcome = OUTCOME_UNCERTAIN_FORWARDED
            blk_outcome = OUTCOME_BLOCKED_UNCERTAIN
        else:
            _LOGGER.info(
                "[%s] NO MATCH (best=%.4f, th=%.3f, verify=%.0fms, scores=%s)",
                sid, result.distance, result.threshold, verify_ms,
                ", ".join(f"{n}={d:.3f}" for n, d in result.all_distances.items()),
            )
            # An uncertain sample is a near-match of an enrolled voice,
            # not an unknown one — keep it out of the unknown postbox.
            if self.context.get_unknown_logging() and embedding is not None:
                await self._log_unknown(verify_audio, embedding, result, duration)
            sentinel = "unknown"
            best_speaker, best_distance = self._nearest_of(result)
            fwd_outcome = OUTCOME_UNKNOWN_FORWARDED
            blk_outcome = OUTCOME_BLOCKED_NO_MATCH

        self.context.publish_recognition(
            speaker=sentinel,
            confidence=self.context.confidence_for(result.distance),
            satellite_id=self._satellite_id,
            is_known=False,
            distance=result.distance,
            threshold=result.threshold,
            nearest_speaker=best_speaker,
            nearest_distance=best_distance,
            margin=margin,
            uncertain=uncertain,
            reason=sentinel,
            ambiguities=self._transcript_hints or None,
            whisper=self._whisper,
            speakers=self._speakers or None,
        )

        if not require_match:
            _LOGGER.info("[%s] require_match=false → forwarding anyway", sid)
            transcript = await self._get_transcript(verify_audio)
            # The event above went out before the transcript existed, so
            # sidecar hints (if any) need a second, identical event.
            self._republish_hints(
                speaker=sentinel,
                confidence=self.context.confidence_for(result.distance),
                is_known=False,
                distance=result.distance,
                threshold=result.threshold,
                nearest_speaker=best_speaker,
                nearest_distance=best_distance,
                margin=margin,
                uncertain=uncertain,
                reason=sentinel,
            )
            await self._send_transcript_to_satellite(
                self._augment_transcript(
                    transcript, is_known=False,
                    distance=result.distance, nearest=best_speaker,
                ),
                label=f"{sentinel}-forwarded",
            )
            self._responded = True
            await self._close_upstream(send_stop=False)
            self._log_total_latency(f"{sentinel}-forwarded")
            event_id = self._record_event(
                outcome=fwd_outcome,
                duration_sec=duration,
                matched_speaker=best_speaker,
                distance=result.distance,
                threshold=result.threshold,
                verify_ms=verify_ms,
                transcript=transcript,
                weight=weight,
                margin=margin,
            )
            self._spawn_shadow(event_id, verify_audio)
        else:
            await self._block_response(
                outcome=blk_outcome,
                duration=duration,
                distance=result.distance,
                threshold=result.threshold,
                verify_ms=verify_ms,
                matched_speaker=best_speaker,
                weight=weight,
                margin=margin,
            )
            self._log_total_latency("blocked")

    # ------------------------------------------------------------------
    # Response helpers
    # ------------------------------------------------------------------

    def _augment_transcript(
        self,
        transcript: str,
        *,
        is_known: bool,
        speaker: Optional[str] = None,
        role: Optional[str] = None,
        distance: Optional[float] = None,
        nearest: Optional[str] = None,
    ) -> str:
        """Wrap the transcript with the recognition context, if enabled.

        Returns the transcript unchanged when the feature is off or the
        chosen template is blank. Injecting the speaker/confidence into
        the transcript (rather than the HA system prompt) guarantees the
        values reach the conversation agent fresh on every utterance —
        the system prompt is cached per conversation, so it can otherwise
        report a stale speaker after a different person talks.
        """
        if not transcript or self.context.get_speaker_context_mode() != "transcript":
            return transcript
        template = (
            self.context.get_transcript_template_known() if is_known
            else self.context.get_transcript_template_unknown()
        )
        if not template.strip():
            return transcript
        conf = (
            self.context.confidence_for(distance) if distance is not None else 0.0
        )
        values = {
            "transcript": transcript,
            "tts": transcript,
            "speaker": speaker or "",
            "role": role or "",
            "confidence": f"{conf * 100:.1f}",
            "distance": f"{distance:.4f}" if distance is not None else "",
            "nearest": nearest or "",
            "satellite": self._satellite_id or "",
        }
        try:
            return render_transcript_template(template, values)
        except Exception:
            _LOGGER.debug(
                "[%s] Transcript template render failed; sending raw",
                self._session_id, exc_info=True,
            )
            return transcript

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
        event_id = self._record_event(
            outcome=outcome, duration_sec=duration, transcript=transcript
        )
        self._spawn_shadow(event_id, audio_16k or b"")

    async def _block_response(
        self,
        *,
        outcome: str = OUTCOME_BLOCKED_NO_MATCH,
        duration: float = 0.0,
        matched_speaker: Optional[str] = None,
        distance: Optional[float] = None,
        threshold: Optional[float] = None,
        verify_ms: Optional[float] = None,
        weight: Optional[float] = None,
        margin: Optional[float] = None,
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
            weight=weight,
            margin=margin,
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
        weight: Optional[float] = None,
        margin: Optional[float] = None,
    ) -> int:
        """Persist a recognition audit row. Never raises. Returns the row
        id (0 on failure) so the A/B shadow can attach its transcript
        later."""
        try:
            return self.context.recognition.record(
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
                weight=weight,
                margin=margin,
                transcript_ms=self._transcript_ms,
                whisper=self._whisper,
                speakers=self._speakers or None,
            )
        except Exception:
            _LOGGER.debug(
                "[%s] Failed to record audit row", self._session_id,
                exc_info=True,
            )
            return 0

    def _spawn_shadow(self, event_id: int, audio_16k: bytes) -> None:
        """Attach the A/B shadow transcription to an audit event.

        In dual mode the shadow already ran (blocking, before the
        response) — its cached result is attached directly. Otherwise
        the shadow is fired after the primary transcript already went
        back to the satellite, so it never adds latency.
        """
        if not event_id:
            return
        if self._dual_shadow is not None:
            text, engine = self._dual_shadow
            self._dual_shadow = None
            self.context.recognition.set_shadow(
                event_id, text, engine, self._shadow_ms
            )
            return
        if not audio_16k:
            return
        if self.context.get_shadow_stt_backend() == "none":
            return
        asyncio.create_task(self._run_shadow(event_id, audio_16k))

    async def _shadow_transcribe(self, audio_16k: bytes) -> tuple[str, str]:
        """Transcribe with the configured shadow engine → (text, engine).

        Raises on failure/misconfiguration so callers (dual mode, the
        post-response audit run) can handle it their own way.
        """
        from murdock.core.stt_backend import (
            STTBackendError,
            transcribe_via_wyoming,
        )

        started = time.monotonic()
        try:
            kind = self.context.get_shadow_stt_backend()
            if kind == "upstream":
                uri = self.context.get_shadow_upstream_uri()
                if not uri:
                    raise STTBackendError("shadow upstream URI not configured")
                text = await transcribe_via_wyoming(
                    uri, audio_16k, language=self._language
                )
                return text, f"wyoming:{uri}"
            backend = self.context.get_shadow_backend()
            if backend is None:
                raise STTBackendError(f"shadow backend {kind!r} not configured")
            text = await backend.transcribe(audio_16k, language=self._language)
            return text, backend.label
        finally:
            self._shadow_ms = (time.monotonic() - started) * 1000

    async def _run_shadow(self, event_id: int, audio_16k: bytes) -> None:
        from murdock.core.stt_backend import STTBackendError

        sid = self._session_id
        kind = self.context.get_shadow_stt_backend()
        try:
            text, engine = await self._shadow_transcribe(audio_16k)
        except STTBackendError as exc:
            _LOGGER.info("[%s] Shadow STT failed: %s", sid, exc)
            self.context.recognition.set_shadow(
                event_id, f"⚠ {exc}", f"{kind} (failed)", self._shadow_ms
            )
            return
        except Exception:
            _LOGGER.debug("[%s] Shadow STT crashed", sid, exc_info=True)
            return
        _LOGGER.info(
            "[%s] Shadow transcript (%s): %r", sid, engine, text[:80]
        )
        self.context.recognition.set_shadow(
            event_id, text, engine, self._shadow_ms
        )

    async def _capture_for_training(self, audio_16k: bytes, duration: float) -> None:
        """Log an utterance to the unknown postbox so it can be assigned to
        a speaker from the UI.

        Used to bootstrap training over the voice satellite when no speakers
        are enrolled yet: every spoken utterance becomes assignable material,
        so the user can build a profile without the browser microphone or a
        file upload. Gated on the unknown-logging setting; never raises.
        """
        if not self.context.get_unknown_logging():
            return
        sid = self._session_id
        try:
            async with _MODEL_LOCK:
                loop = asyncio.get_running_loop()
                embedding = await loop.run_in_executor(
                    None, self.context.embedder.embed_pcm, audio_16k
                )
            try:
                liveness = await asyncio.to_thread(analyze_liveness, audio_16k)
                liveness_score = float(liveness.score)
            except Exception:
                liveness_score = None
            await asyncio.to_thread(
                self.context.unknown.record,
                self._session_id,
                audio_16k,
                embedding,
                duration,
                2.0,    # best_distance — nothing enrolled to compare against
                None,   # best_speaker
                self._satellite_id,
                liveness_score,
            )
            _LOGGER.info(
                "[%s] Captured utterance for training (no enrolled speakers)", sid
            )
        except Exception:
            _LOGGER.debug("[%s] Training capture failed", sid, exc_info=True)

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
        for task in (
            self._upstream_reader_task,
            self._early_probe_task,
            self._reject_probe_task,
        ):
            if task is not None and not task.done():
                task.cancel()
                try:
                    await task
                except (asyncio.CancelledError, Exception):
                    pass
