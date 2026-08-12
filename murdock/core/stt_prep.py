"""Conditioning the audio that goes to a cloud STT endpoint.

Two cheap fixes for the same failure: transcription models hallucinate
in silence, and hallucinated text is what trips Whisper's own quality
thresholds into re-decoding the clip.

- **Head/tail trim.** A satellite hands over the whole capture window,
  which for a two-word command is mostly room tone. Only the leading and
  trailing silence is removed — pauses *inside* the utterance stay. That
  restraint is deliberate: splicing out internal gaps saves a few
  hundred milliseconds of upload and risks clipping word onsets and
  leaving discontinuities at every join, which makes the decoder guess
  more, not less. The win here is almost entirely at the two ends.
- **RMS normalisation.** A whispered or distant utterance arrives far
  below the level the model saw in training. The gain is capped so a
  clip that is *only* room tone doesn't get amplified into something the
  decoder feels obliged to interpret.

This path is for the upload only. Speaker embeddings are computed on a
separate copy and must not be touched — ``fbank.py`` does its own
per-utterance mean normalisation, and changing the level underneath it
would shift every distance.
"""

from __future__ import annotations

import logging
from typing import Optional

import numpy as np

from .audio import float32_to_pcm16_bytes, pcm16_bytes_to_float32

_LOGGER = logging.getLogger("murdock.stt_prep")

_SAMPLE_RATE = 16000

#: Kept either side of the speech region. Enough that a plosive onset or
#: a trailing fricative never lands on the cut.
_PAD_SEC = 0.20

#: Never hand the endpoint less than this. Below roughly a quarter
#: second there is nothing to transcribe anyway, and a too-eager trim
#: turning a real command into a stub is a worse failure than a slightly
#: long upload.
_MIN_KEEP_SEC = 0.30

#: If the detected speech spans less than this fraction of the clip, the
#: VAD is distrusted and nothing is trimmed. Silero under-detects quiet
#: and whispered speech, which is exactly the audio a user is most
#: likely to have to repeat — losing most of it to a confident-looking
#: trim would be the worst outcome this module could produce.
_MIN_SPEECH_FRACTION = 0.15

#: Target level for normalisation, in dBFS.
_TARGET_DBFS = -20.0

#: Upper bound on the gain applied. 20 dB rescues a genuine whisper;
#: beyond that the clip is mostly noise and amplifying it just invites
#: the decoder to invent words.
_MAX_GAIN_DB = 20.0

#: Below this the clip is treated as silence and left alone entirely.
_SILENCE_FLOOR_DBFS = -55.0


def _dbfs(samples: np.ndarray) -> float:
    """RMS level of float32 [-1, 1] audio in dBFS."""
    if samples.size == 0:
        return -120.0
    rms = float(np.sqrt(np.mean(np.square(samples)) + 1e-12))
    if rms <= 1e-9:
        return -120.0
    return 20.0 * float(np.log10(rms))


def trim_head_and_tail(pcm_bytes: bytes, vad) -> bytes:
    """Drop silence before the first and after the last speech segment.

    Returns the input unchanged when the VAD found no speech, when the
    clip is already tight, or when trimming would leave too little —
    this must never be the reason a command goes missing.
    """
    if not pcm_bytes or vad is None:
        return pcm_bytes
    try:
        audio = pcm16_bytes_to_float32(pcm_bytes)
        result = vad.analyze_waveform(audio)
    except Exception:
        _LOGGER.debug("VAD pass failed; uploading untrimmed", exc_info=True)
        return pcm_bytes

    if not result.segments:
        return pcm_bytes

    total_sec = len(audio) / _SAMPLE_RATE
    speech_sec = sum(max(0.0, e - s) for s, e in result.segments)
    if total_sec > 0 and (speech_sec / total_sec) < _MIN_SPEECH_FRACTION:
        _LOGGER.debug(
            "VAD found only %.2fs of speech in %.2fs — uploading untrimmed",
            speech_sec, total_sec,
        )
        return pcm_bytes

    first_start = min(s for s, _ in result.segments)
    last_end = max(e for _, e in result.segments)
    start = max(0, int((first_start - _PAD_SEC) * _SAMPLE_RATE))
    end = min(len(audio), int((last_end + _PAD_SEC) * _SAMPLE_RATE))
    if end <= start:
        return pcm_bytes
    if (end - start) < int(_MIN_KEEP_SEC * _SAMPLE_RATE):
        return pcm_bytes
    if start == 0 and end == len(audio):
        return pcm_bytes
    return float32_to_pcm16_bytes(audio[start:end])


def normalize_level(pcm_bytes: bytes) -> bytes:
    """Bring the clip toward ``_TARGET_DBFS``, within a capped gain.

    Only ever applied on the upload copy. Attenuation is applied as
    readily as gain — a clipped, too-hot capture is as hard to decode as
    a faint one.
    """
    if not pcm_bytes:
        return pcm_bytes
    audio = pcm16_bytes_to_float32(pcm_bytes)
    level = _dbfs(audio)
    if level <= _SILENCE_FLOOR_DBFS:
        return pcm_bytes
    gain_db = min(_TARGET_DBFS - level, _MAX_GAIN_DB)
    if abs(gain_db) < 1.0:
        return pcm_bytes
    scaled = audio * float(10.0 ** (gain_db / 20.0))
    # Guard the peak so normalisation can never introduce clipping.
    peak = float(np.max(np.abs(scaled))) if scaled.size else 0.0
    if peak > 0.99:
        scaled = scaled * (0.99 / peak)
    return float32_to_pcm16_bytes(scaled)


def prepare_for_upload(
    pcm_bytes: bytes,
    vad,
    *,
    trim: bool = True,
    normalize: bool = True,
) -> tuple[bytes, Optional[dict]]:
    """Condition audio for a cloud STT request.

    Returns ``(pcm, info)``; ``info`` is ``None`` when nothing changed,
    otherwise a small dict for the recognition log so the effect is
    visible rather than assumed.
    """
    if not pcm_bytes:
        return pcm_bytes, None
    original_len = len(pcm_bytes)
    out = pcm_bytes
    if trim:
        out = trim_head_and_tail(out, vad)
    trimmed_len = len(out)
    level_before = _dbfs(pcm16_bytes_to_float32(out))
    if normalize:
        out = normalize_level(out)
    if len(out) == original_len and out == pcm_bytes:
        return pcm_bytes, None
    info = {
        "trimmed_ms": round(
            (original_len - trimmed_len) / (_SAMPLE_RATE * 2) * 1000, 1
        ),
        "level_before_dbfs": round(level_before, 1),
        "level_after_dbfs": round(_dbfs(pcm16_bytes_to_float32(out)), 1),
    }
    return out, info
