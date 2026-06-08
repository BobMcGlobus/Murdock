"""Audio helpers: PCM <-> float, WAV encoding, resampling."""

from __future__ import annotations

import io
import logging
import os
import shutil
import subprocess
import tempfile
import wave
from typing import Tuple

import numpy as np

_LOGGER = logging.getLogger("murdock.core.audio")

# Common magic bytes so we don't have to shell out to ffmpeg for the
# fast path of "user uploaded a plain PCM WAV".
_WAV_MAGIC = b"RIFF"
_WAV_WAVE = b"WAVE"


def pcm16_bytes_to_float32(audio_bytes: bytes) -> np.ndarray:
    """Convert 16-bit signed little-endian PCM bytes to float32 in [-1, 1]."""
    if not audio_bytes:
        return np.zeros(0, dtype=np.float32)
    samples = np.frombuffer(audio_bytes, dtype=np.int16)
    return samples.astype(np.float32) / 32768.0


def float32_to_pcm16_bytes(samples: np.ndarray) -> bytes:
    """Convert float32 samples in [-1, 1] to 16-bit signed PCM bytes."""
    clipped = np.clip(samples, -1.0, 1.0)
    return (clipped * 32767.0).astype(np.int16).tobytes()


def encode_wav(
    audio_bytes: bytes,
    sample_rate: int = 16000,
    sample_width: int = 2,
    channels: int = 1,
) -> bytes:
    """Wrap raw PCM bytes in a WAV container."""
    buffer = io.BytesIO()
    with wave.open(buffer, "wb") as wf:
        wf.setnchannels(channels)
        wf.setsampwidth(sample_width)
        wf.setframerate(sample_rate)
        wf.writeframes(audio_bytes)
    return buffer.getvalue()


def decode_wav(wav_bytes: bytes) -> Tuple[bytes, int, int, int]:
    """Decode a WAV file and return (pcm_bytes, rate, width, channels).

    Caller is responsible for resampling if needed.
    """
    with wave.open(io.BytesIO(wav_bytes), "rb") as wf:
        rate = wf.getframerate()
        width = wf.getsampwidth()
        channels = wf.getnchannels()
        frames = wf.readframes(wf.getnframes())
    return frames, rate, width, channels


def _looks_like_wav(data: bytes) -> bool:
    return len(data) >= 12 and data[:4] == _WAV_MAGIC and data[8:12] == _WAV_WAVE


def _ffmpeg_decode_to_pcm16(data: bytes) -> bytes:
    """Decode arbitrary audio via ffmpeg → raw 16 kHz mono 16-bit PCM bytes.

    The input is written to a temporary file instead of piped on stdin.
    Two reasons:

    1. MP4/M4A containers (as produced by iTunes, QuickTime, iPhone voice
       memos, …) often place the ``moov`` atom at the *end* of the file.
       Demuxing requires seeking to it, which is impossible on a stdin
       pipe — ffmpeg then aborts with "partial file / Invalid data found
       when processing input".
    2. Some formats (WebM, OGG) work on pipes but MP4 is common enough
       that a temp file is the simpler, uniformly-reliable path.

    Output is raw ``s16le`` (not ``wav``) because ffmpeg can not seek
    backwards to fix up the WAV header size fields when writing to
    ``pipe:1``, which would otherwise cause Python's stdlib ``wave``
    module to return a bogus frame count — silently producing near-empty
    PCM and speaker embeddings that match nothing.
    """
    ffmpeg = shutil.which("ffmpeg")
    if not ffmpeg:
        raise RuntimeError(
            "ffmpeg not found. Install ffmpeg (the Dockerfile does this "
            "automatically) to enable non-WAV uploads."
        )

    # Write the upload to a temp file so ffmpeg can seek inside it.
    tmp = tempfile.NamedTemporaryFile(
        prefix="murdock-upload-", suffix=".bin", delete=False
    )
    try:
        tmp.write(data)
        tmp.flush()
        tmp.close()
        cmd = [
            ffmpeg,
            "-hide_banner",
            "-loglevel", "error",
            "-nostdin",
            "-i", tmp.name,
            "-vn",                 # drop any video streams (mp4/m4a cover art)
            "-map", "0:a:0",       # take the first audio stream
            "-ac", "1",
            "-ar", "16000",
            "-f", "s16le",
            "-acodec", "pcm_s16le",
            "pipe:1",
        ]
        try:
            proc = subprocess.run(
                cmd,
                capture_output=True,
                timeout=60,
                check=False,
            )
        except subprocess.TimeoutExpired as exc:
            raise ValueError("ffmpeg timed out decoding the upload") from exc
        if proc.returncode != 0 or not proc.stdout:
            err = (proc.stderr or b"").decode("utf-8", "replace").strip()
            _LOGGER.warning("ffmpeg decode failed: %s", err)
            raise ValueError(
                "Unsupported or corrupt audio file"
                + (f": {err}" if err else "")
            )
        return proc.stdout
    finally:
        try:
            os.unlink(tmp.name)
        except OSError:
            pass


def decode_audio_any(data: bytes) -> Tuple[bytes, int, int, int]:
    """Decode arbitrary audio (WAV, MP3, MP4/M4A, OGG, FLAC, WebM…) to PCM.

    Returns ``(pcm_bytes, rate, width, channels)`` in the same shape as
    :func:`decode_wav`. WAV inputs are decoded in-process; everything else
    is transcoded through ffmpeg directly to 16 kHz mono 16-bit raw PCM.
    """
    if not data:
        raise ValueError("Empty upload")
    if _looks_like_wav(data):
        try:
            return decode_wav(data)
        except Exception:
            # Fall through — some "WAV" files use unusual codecs that the
            # stdlib ``wave`` module can't handle (e.g. float32 PCM, ADPCM,
            # or GSM). ffmpeg handles all of those.
            _LOGGER.info("stdlib wave failed on RIFF input; falling back to ffmpeg")
    pcm = _ffmpeg_decode_to_pcm16(data)
    if not pcm:
        raise ValueError("ffmpeg produced no audio — file may be empty or corrupt")
    # Already 16 kHz mono 16-bit, so downstream to_mono_16k_pcm() is a no-op.
    return pcm, 16000, 2, 1


def to_mono_16k_pcm(
    audio_bytes: bytes, rate: int, width: int, channels: int
) -> bytes:
    """Convert arbitrary PCM audio to 16 kHz mono 16-bit PCM.

    Uses linear interpolation for resampling. Good enough for
    speaker-recognition input; we don't need studio fidelity.
    """
    if width != 2:
        # Only 16-bit PCM supported; reject other widths.
        raise ValueError(f"Unsupported sample width: {width} bytes")

    samples = np.frombuffer(audio_bytes, dtype=np.int16)
    if channels > 1:
        samples = samples.reshape(-1, channels).mean(axis=1).astype(np.int16)

    if rate != 16000 and len(samples) > 0:
        duration = len(samples) / rate
        target_len = int(round(duration * 16000))
        if target_len <= 0:
            return b""
        x_old = np.linspace(0.0, 1.0, num=len(samples), endpoint=False)
        x_new = np.linspace(0.0, 1.0, num=target_len, endpoint=False)
        resampled = np.interp(x_new, x_old, samples.astype(np.float32))
        samples = resampled.astype(np.int16)

    return samples.tobytes()


def rms_dbfs(audio_bytes: bytes) -> float:
    """Return the RMS level of PCM audio in dBFS (0 = full scale)."""
    if not audio_bytes:
        return -120.0
    samples = np.frombuffer(audio_bytes, dtype=np.int16).astype(np.float32)
    if len(samples) == 0:
        return -120.0
    rms = float(np.sqrt(np.mean(samples**2) + 1e-12))
    if rms <= 1e-9:
        return -120.0
    return 20.0 * float(np.log10(rms / 32768.0))
