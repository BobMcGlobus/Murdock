"""STT backend abstraction.

Murdock can obtain transcripts from two sources:

- **upstream** (default): proxy the audio to an external Wyoming STT
  service (faster-whisper, whisper, etc.) over a TCP socket.
- **voxtral**: send the buffered audio to the Mistral Cloud API
  (Voxtral endpoint) and receive a transcript directly.

The handler asks ``get_stt_backend(ctx)`` for the active backend. Each
backend exposes a single awaitable: given raw PCM audio → return text.
The Wyoming proxy path is handled implicitly by the handler (existing
streaming logic), so this module only provides the *cloud* backends.
"""

from __future__ import annotations

import io
import logging
import struct
import time
from typing import Optional

import httpx

_LOGGER = logging.getLogger("murdock.stt_backend")


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


class VoxtralBackend:
    """Transcribe audio via the Mistral /v1/audio/transcriptions API."""

    def __init__(
        self,
        api_key: str,
        model: str = "voxtral-mini-latest",
        language: Optional[str] = None,
        timeout: float = 30.0,
    ) -> None:
        self.api_key = api_key
        self.model = model
        self.language = language
        self.timeout = timeout
        self._base_url = "https://api.mistral.ai"

    async def transcribe(
        self,
        pcm_audio: bytes,
        *,
        rate: int = 16000,
        width: int = 2,
        channels: int = 1,
        language: Optional[str] = None,
    ) -> str:
        """Send audio to Voxtral and return the transcript text.

        Args:
            pcm_audio: Raw PCM bytes (mono 16-bit 16 kHz expected).
            rate: Sample rate.
            width: Sample width in bytes.
            channels: Channel count.
            language: ISO language hint (e.g. "de"). Falls back to
                instance default if not set.

        Returns:
            Transcript text, or empty string on failure.
        """
        lang = language or self.language
        wav_data = _pcm_to_wav(pcm_audio, rate, width, channels)
        t0 = time.monotonic()

        try:
            async with httpx.AsyncClient(
                base_url=self._base_url,
                headers={"Authorization": f"Bearer {self.api_key}"},
                timeout=self.timeout,
            ) as client:
                # Build multipart form data
                files = {"file": ("audio.wav", wav_data, "audio/wav")}
                data: dict = {"model": self.model}
                if lang:
                    data["language"] = lang

                resp = await client.post(
                    "/v1/audio/transcriptions",
                    files=files,
                    data=data,
                )

                elapsed_ms = (time.monotonic() - t0) * 1000

                if resp.status_code != 200:
                    _LOGGER.warning(
                        "Voxtral API error: HTTP %d — %s (%.0fms)",
                        resp.status_code, resp.text[:200], elapsed_ms,
                    )
                    return ""

                result = resp.json()
                text = result.get("text", "")
                usage = result.get("usage", {})
                audio_sec = usage.get("prompt_audio_seconds", 0)
                _LOGGER.info(
                    "Voxtral transcript (%.0fms, %.1fs audio): %r",
                    elapsed_ms, audio_sec, text[:100],
                )
                return text

        except httpx.TimeoutException:
            elapsed_ms = (time.monotonic() - t0) * 1000
            _LOGGER.warning("Voxtral API timeout after %.0fms", elapsed_ms)
            return ""
        except Exception as exc:
            elapsed_ms = (time.monotonic() - t0) * 1000
            _LOGGER.warning(
                "Voxtral API request failed (%.0fms): %s", elapsed_ms, exc
            )
            return ""
