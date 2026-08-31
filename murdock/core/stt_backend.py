"""STT backend abstraction.

Murdock can obtain transcripts from three sources:

- **upstream** (default): proxy the audio to an external Wyoming STT
  service (faster-whisper, whisper, etc.) over a TCP socket, streaming
  live while the user is still speaking.
- **voxtral**: send the buffered audio to the Mistral Cloud API.
- **openai**: send the buffered audio to any OpenAI-compatible
  ``/v1/audio/transcriptions`` endpoint — OpenAI itself
  (gpt-4o-transcribe), Groq (whisper-large-v3-turbo), or a local
  OpenAI-compatible server such as speaches.

The handler asks the context for the active backend. Each cloud backend
exposes a single awaitable: given raw PCM audio → return text; failures
raise :class:`STTBackendError` so the caller can distinguish "the model
heard silence" from "the internet is down" — which is what makes the
local Wyoming fallback and the A/B shadow transcription possible.
"""

from __future__ import annotations

import asyncio
import io
import logging
import struct
import time
from typing import Optional

import httpx

_LOGGER = logging.getLogger("murdock.stt_backend")

#: Sampling temperature for every cloud transcription request.
#:
#: Whisper checks its own output against ``compression_ratio_threshold``
#: and ``log_prob_threshold`` after decoding, and on failure re-decodes
#: the whole clip at temperature 0.2 → 0.4 → 0.6 → 0.8 → 1.0. That is up
#: to six full passes, which is where multi-second outliers on otherwise
#: sub-second audio come from. Pinning temperature to 0 disables the
#: cascade: an utterance the model is unsure about now comes back wrong
#: quickly instead of wrong slowly, and for a voice assistant the bounded
#: latency is worth more than the retry.
_TEMPERATURE = 0


class STTBackendError(RuntimeError):
    """A cloud transcription attempt failed (network, HTTP, timeout)."""


def _normalize_language(tag: Optional[str]) -> Optional[str]:
    """Reduce a BCP-47 tag to the ISO-639-1 code these endpoints expect.

    Home Assistant hands over whatever the pipeline is configured with,
    which is commonly a full tag like ``de-DE``. OpenAI's transcription
    API documents ISO-639-1, and a local Parakeet server that cannot
    match the tag falls back to its own default — English — which turns
    a German sentence into confident English nonsense rather than an
    error.
    """
    if not tag:
        return None
    code = str(tag).strip().replace("_", "-").split("-")[0].lower()
    return code or None


def _pcm_to_wav(pcm: bytes, rate: int = 16000, width: int = 2, channels: int = 1) -> bytes:
    """Wrap raw PCM in a WAV container for upload."""
    num_samples = len(pcm) // width
    data_size = num_samples * width
    buf = io.BytesIO()
    # RIFF header
    buf.write(b"RIFF")
    buf.write(struct.pack("<I", 36 + data_size))
    buf.write(b"WAVE")
    # fmt sub-chunk
    buf.write(b"fmt ")
    buf.write(struct.pack("<I", 16))
    buf.write(struct.pack("<HHIIHH", 1, channels, rate, rate * channels * width, channels * width, width * 8))
    # data sub-chunk
    buf.write(b"data")
    buf.write(struct.pack("<I", data_size))
    buf.write(pcm)
    return buf.getvalue()


class OpenAICompatibleBackend:
    """Transcribe audio via an OpenAI-compatible transcriptions endpoint.

    One implementation covers OpenAI (gpt-4o-transcribe), Groq
    (whisper-large-v3-turbo), Mistral/Voxtral and self-hosted
    OpenAI-compatible servers (speaches, LocalAI, …) — they all speak
    multipart POST ``/v1/audio/transcriptions`` with ``file`` + ``model``.
    """

    #: Human-readable engine label used in logs and the A/B shadow column.
    name = "openai"
    #: Whether the endpoint honours the OpenAI ``prompt`` field for
    #: vocabulary biasing. Whisper-family endpoints do; Mistral's
    #: transcription API does not document it, so Voxtral opts out.
    supports_prompt = True

    def __init__(
        self,
        api_key: str,
        model: str,
        base_url: str = "https://api.openai.com",
        language: Optional[str] = None,
        timeout: float = 30.0,
        name: Optional[str] = None,
        prompt: Optional[str] = None,
    ) -> None:
        self.api_key = api_key
        self.model = model
        self.language = language
        self.timeout = timeout
        self.prompt = (prompt or "").strip() or None
        #: Breakdown of the most recent :meth:`transcribe` call. The
        #: context builds a fresh backend per transcription, so this
        #: never spans two utterances. Populated even when the request
        #: fails, so a timeout is as diagnosable as a slow success.
        self.last_timing: dict = {}
        self.base_url = (base_url or "https://api.openai.com").rstrip("/")
        # OpenRouter deviates from the OpenAI shape: JSON body with
        # base64 audio instead of a multipart file, model slugs carry a
        # provider prefix (openai/whisper-large-v3-turbo), and the API
        # root lives under /api. Detect it by host so the user can just
        # paste https://openrouter.ai and their key.
        self.is_openrouter = "openrouter.ai" in self.base_url
        if self.is_openrouter and not self.base_url.endswith("/api"):
            self.base_url += "/api"
        if name:
            self.name = name

    def _request_kwargs(self, wav_data: bytes, lang: Optional[str]) -> dict:
        """Build the httpx POST kwargs for this endpoint's request shape."""
        # Normalised here rather than at the call site so the guarantee
        # holds for every caller, not just the one that remembered.
        lang = _normalize_language(lang)
        if self.is_openrouter:
            import base64
            payload: dict = {
                "model": self.model,
                "input_audio": {
                    "data": base64.b64encode(wav_data).decode("ascii"),
                    "format": "wav",
                },
                "temperature": _TEMPERATURE,
            }
            # OpenRouter takes the language hint in the JSON body. It used
            # to be dropped here while the multipart branch honoured it,
            # so the primary engine re-detected the language every time.
            if lang:
                payload["language"] = lang
            return {"json": payload}
        files = {"file": ("audio.wav", wav_data, "audio/wav")}
        # Form fields are strings; httpx would reject a bare float.
        data: dict = {"model": self.model, "temperature": str(_TEMPERATURE)}
        if lang:
            data["language"] = lang
        # Vocabulary biasing: hand the custom terms to whisper-family
        # endpoints so names like "Fehenlichter" are recognised at the
        # source. Skipped where the field isn't documented (Voxtral) —
        # an unknown form field could fail the whole request.
        if self.prompt and self.supports_prompt:
            data["prompt"] = self.prompt
        return {"files": files, "data": data}

    @property
    def label(self) -> str:
        """Engine tag for logs / the shadow column, e.g. ``openai:whisper``."""
        return f"{self.name}:{self.model}"

    async def transcribe(
        self,
        pcm_audio: bytes,
        *,
        rate: int = 16000,
        width: int = 2,
        channels: int = 1,
        language: Optional[str] = None,
    ) -> str:
        """Send audio to the endpoint and return the transcript text.

        Raises :class:`STTBackendError` on any transport/HTTP failure so
        callers can trigger the local fallback. A successful call that
        heard nothing legitimately returns ``""``.
        """
        # `self.language` is the configured fallback: an endpoint that
        # gets no hint picks its own default, and for most of them that
        # is English.
        lang = _normalize_language(language) or _normalize_language(self.language)
        if not lang:
            _LOGGER.warning(
                "%s: no language hint — the endpoint will guess, and most "
                "default to English. Set the STT language in settings.",
                self.label,
            )
        wav_data = _pcm_to_wav(pcm_audio, rate, width, channels)
        self.last_timing = {
            "engine": self.label,
            "sent_bytes": len(wav_data),
            "audio_ms": round(len(pcm_audio) / (rate * width * channels) * 1000, 1),
        }
        t0 = time.monotonic()

        try:
            headers = {}
            if self.api_key:
                headers["Authorization"] = f"Bearer {self.api_key}"
            async with httpx.AsyncClient(
                base_url=self.base_url,
                headers=headers,
                timeout=self.timeout,
            ) as client:
                # Sent streaming so the response headers can be timed
                # separately from the body. Upload + queueing + decode all
                # land in TTFB, which is the number that actually moves;
                # a transcript body is a few hundred bytes and should be
                # noise. When it isn't, the problem is the network, not
                # the model — and that distinction is the whole point.
                request = client.build_request(
                    "POST",
                    "/v1/audio/transcriptions",
                    **self._request_kwargs(wav_data, lang),
                )
                resp = await client.send(request, stream=True)
                ttfb_ms = (time.monotonic() - t0) * 1000
                try:
                    await resp.aread()
                finally:
                    await resp.aclose()
                elapsed_ms = (time.monotonic() - t0) * 1000
                self.last_timing.update(
                    ttfb_ms=round(ttfb_ms, 1),
                    body_ms=round(elapsed_ms - ttfb_ms, 1),
                    total_ms=round(elapsed_ms, 1),
                )

                if resp.status_code != 200:
                    _LOGGER.warning(
                        "%s API error: HTTP %d — %s (%.0fms)",
                        self.label, resp.status_code, resp.text[:200], elapsed_ms,
                    )
                    raise STTBackendError(
                        f"{self.label}: HTTP {resp.status_code}"
                    )

                result = resp.json()
                text = result.get("text", "")
                _LOGGER.info(
                    "%s transcript (%.0fms: ttfb %.0f, body %.0f): %r",
                    self.label, elapsed_ms, ttfb_ms, elapsed_ms - ttfb_ms,
                    text[:100],
                )
                return text

        except STTBackendError:
            raise
        except httpx.TimeoutException as exc:
            elapsed_ms = (time.monotonic() - t0) * 1000
            self.last_timing.update(total_ms=round(elapsed_ms, 1), failed="timeout")
            _LOGGER.warning("%s timeout after %.0fms", self.label, elapsed_ms)
            raise STTBackendError(f"{self.label}: timeout") from exc
        except Exception as exc:
            elapsed_ms = (time.monotonic() - t0) * 1000
            self.last_timing.update(total_ms=round(elapsed_ms, 1), failed="error")
            _LOGGER.warning(
                "%s request failed (%.0fms): %s", self.label, elapsed_ms, exc
            )
            raise STTBackendError(f"{self.label}: {exc}") from exc


class VoxtralBackend(OpenAICompatibleBackend):
    """Transcribe audio via the Mistral /v1/audio/transcriptions API."""

    name = "voxtral"
    supports_prompt = False  # Mistral doesn't document the prompt field

    def __init__(
        self,
        api_key: str,
        model: str = "voxtral-mini-latest",
        language: Optional[str] = None,
        timeout: float = 30.0,
    ) -> None:
        super().__init__(
            api_key=api_key,
            model=model,
            base_url="https://api.mistral.ai",
            language=language,
            timeout=timeout,
        )


async def transcribe_via_wyoming(
    uri: str,
    pcm_audio: bytes,
    *,
    rate: int = 16000,
    width: int = 2,
    channels: int = 1,
    language: Optional[str] = None,
    timeout: float = 20.0,
) -> str:
    """One-shot transcription of buffered audio over a Wyoming socket.

    Used for the local fallback (cloud backend unreachable) and for the
    A/B shadow when the shadow engine is a Wyoming server: open, send
    Transcribe + AudioStart + chunks + AudioStop, wait for the
    Transcript. Raises :class:`STTBackendError` on failure.
    """
    from wyoming.asr import Transcribe, Transcript
    from wyoming.audio import AudioChunk, AudioStart, AudioStop
    from wyoming.client import AsyncClient

    t0 = time.monotonic()
    chunk_bytes = 4096

    async def _run() -> str:
        async with AsyncClient.from_uri(uri) as client:
            await client.write_event(Transcribe(language=language).event())
            await client.write_event(
                AudioStart(rate=rate, width=width, channels=channels).event()
            )
            for i in range(0, len(pcm_audio), chunk_bytes):
                await client.write_event(
                    AudioChunk(
                        audio=pcm_audio[i:i + chunk_bytes],
                        rate=rate, width=width, channels=channels,
                    ).event()
                )
            await client.write_event(AudioStop().event())
            while True:
                event = await client.read_event()
                if event is None:
                    raise STTBackendError(
                        f"wyoming {uri}: closed without transcript"
                    )
                if Transcript.is_type(event.type):
                    return Transcript.from_event(event).text or ""

    try:
        text = await asyncio.wait_for(_run(), timeout=timeout)
        _LOGGER.info(
            "wyoming one-shot transcript via %s (%.0fms): %r",
            uri, (time.monotonic() - t0) * 1000, text[:100],
        )
        return text
    except STTBackendError:
        raise
    except asyncio.TimeoutError as exc:
        raise STTBackendError(f"wyoming {uri}: timeout") from exc
    except Exception as exc:
        raise STTBackendError(f"wyoming {uri}: {exc}") from exc


class HomeAssistantSTTBackend:
    """Transcribe through a speech-to-text *entity* inside Home Assistant.

    Home Assistant Cloud's transcription is not a Wyoming service and has
    no public API of its own, so it cannot be reached the way the other
    backends are. It is, however, an ordinary ``stt.*`` entity, and Home
    Assistant exposes every such entity over ``POST /api/stt/{entity_id}``
    — the same endpoint its own web frontend uses.

    That makes any STT entity available to Murdock: Cloud, a Whisper
    add-on, anything an integration provides. The body is the raw PCM
    stream, not a WAV container; the format is declared in the header
    instead.
    """

    name = "ha"
    supports_prompt = False

    def __init__(
        self,
        base_url: str,
        token: str,
        entity_id: str,
        language: Optional[str] = None,
        timeout: float = 30.0,
    ) -> None:
        self.base_url = (base_url or "").rstrip("/")
        self.token = token or ""
        self.entity_id = (entity_id or "").strip()
        self.language = language
        self.timeout = timeout
        self.last_timing: dict = {}

    @property
    def configured(self) -> bool:
        return bool(self.base_url and self.token and self.entity_id)

    @property
    def label(self) -> str:
        return f"ha:{self.entity_id}"

    def _headers(self, lang: Optional[str], rate: int, width: int,
                 channels: int) -> dict:
        # Home Assistant wants a full locale here and is strict about the
        # field set — every one of them must be present or it answers 400.
        language = lang or "en-US"
        if "-" not in language and "_" not in language:
            language = f"{language}-{language.upper()}"
        return {
            "Authorization": f"Bearer {self.token}",
            "X-Speech-Content": (
                f"format=wav; codec=pcm; sample_rate={rate}; "
                f"bit_rate={width * 8}; channel={channels}; "
                f"language={language}"
            ),
        }

    async def transcribe(
        self,
        pcm_audio: bytes,
        *,
        rate: int = 16000,
        width: int = 2,
        channels: int = 1,
        language: Optional[str] = None,
    ) -> str:
        lang = language or self.language
        self.last_timing = {
            "engine": self.label,
            "sent_bytes": len(pcm_audio),
            "audio_ms": round(len(pcm_audio) / (rate * width * channels) * 1000, 1),
        }
        t0 = time.monotonic()
        try:
            async with httpx.AsyncClient(
                base_url=self.base_url, timeout=self.timeout
            ) as client:
                request = client.build_request(
                    "POST",
                    f"/api/stt/{self.entity_id}",
                    headers=self._headers(lang, rate, width, channels),
                    content=pcm_audio,
                )
                resp = await client.send(request, stream=True)
                ttfb_ms = (time.monotonic() - t0) * 1000
                try:
                    await resp.aread()
                finally:
                    await resp.aclose()
                elapsed_ms = (time.monotonic() - t0) * 1000
                self.last_timing.update(
                    ttfb_ms=round(ttfb_ms, 1),
                    body_ms=round(elapsed_ms - ttfb_ms, 1),
                    total_ms=round(elapsed_ms, 1),
                )
                if resp.status_code != 200:
                    _LOGGER.warning(
                        "%s API error: HTTP %d — %s (%.0fms)",
                        self.label, resp.status_code, resp.text[:200], elapsed_ms,
                    )
                    raise STTBackendError(
                        f"{self.label}: HTTP {resp.status_code}"
                    )
                data = resp.json()
                # A refusal comes back as 200 with result="error", which
                # is not the same thing as hearing silence.
                if str(data.get("result", "success")).lower() != "success":
                    raise STTBackendError(f"{self.label}: {data.get('result')}")
                text = data.get("text") or ""
                _LOGGER.info(
                    "%s transcript (%.0fms): %r", self.label, elapsed_ms, text[:100]
                )
                return text
        except STTBackendError:
            raise
        except httpx.TimeoutException as exc:
            self.last_timing.update(
                total_ms=round((time.monotonic() - t0) * 1000, 1), failed="timeout"
            )
            raise STTBackendError(f"{self.label}: timeout") from exc
        except Exception as exc:
            self.last_timing.update(
                total_ms=round((time.monotonic() - t0) * 1000, 1), failed="error"
            )
            raise STTBackendError(f"{self.label}: {exc}") from exc
