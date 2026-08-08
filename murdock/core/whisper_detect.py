"""Detect whispered speech.

Whispering is acoustically unambiguous in a way most voice properties are
not: the vocal folds don't vibrate, so there is **no fundamental
frequency and no harmonic structure** — just filtered noise. Everything
else follows from that:

* **Harmonicity** — the strongest signal by far. Voiced speech has a tall
  autocorrelation peak at the pitch period; whispered speech has none.
* **Spectral tilt** — voiced speech concentrates energy low (the glottal
  source rolls off steeply); whispered speech is comparatively flat and
  bright.
* **Zero-crossing rate** — noise-like excitation crosses zero far more
  often than a periodic wave.
* **Level** — whispering is quiet, but level alone says nothing (a
  distant shout is quiet too), so it only refines the decision.

Why this lives *before* the liveness gate: whispered speech is quiet and
spectrally flat, which is exactly what the TV/playback heuristic rejects.
Without an explicit check, Murdock would throw away precisely the
utterances it is meant to notice.

Deliberately **not** used to relax speaker verification. Whispering does
wreck speaker embeddings — but "whisper detected, so let this through"
would turn the feature into a way past the gate. Whisper is reported, not
trusted.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

import numpy as np

from .audio import pcm16_bytes_to_float32

_LOGGER = logging.getLogger("murdock.whisper")

_SAMPLE_RATE = 16000
_FRAME = 512
_HOP = 256

# Pitch search range for the harmonicity probe: 70–350 Hz covers adult
# speech comfortably. Expressed as autocorrelation lags.
_MIN_LAG = int(_SAMPLE_RATE / 350)
_MAX_LAG = int(_SAMPLE_RATE / 70)

# Frames quieter than this are silence, not whispering — scoring them
# would drown the decision in room tone.
_SILENCE_RMS = 0.004

# Decision bar for :attr:`WhisperFeatures.is_whisper`. Tuned so ordinary
# speech sits far below and a genuine whisper clears it; adjustable per
# install because microphones and rooms differ wildly.
DEFAULT_THRESHOLD = 0.62


@dataclass
class WhisperFeatures:
    """Per-utterance whisper evidence."""

    score: float          # 0 = clearly voiced, 1 = clearly whispered
    harmonicity: float    # mean autocorrelation peak (voicing strength)
    zero_crossing_rate: float
    spectral_tilt: float  # high-band / low-band energy
    rms: float
    voiced_ratio: float   # share of frames with real periodicity
    threshold: float = DEFAULT_THRESHOLD

    @property
    def is_whisper(self) -> bool:
        return self.score >= self.threshold


def _frames(audio: np.ndarray) -> np.ndarray:
    if audio.size < _FRAME:
        return np.zeros((0, _FRAME), dtype=np.float32)
    n = 1 + (audio.size - _FRAME) // _HOP
    idx = np.arange(_FRAME)[None, :] + _HOP * np.arange(n)[:, None]
    return audio[idx]


def _harmonicity(frame: np.ndarray) -> float:
    """Normalised autocorrelation peak in the pitch range (0…1).

    High for a periodic (voiced) frame, near zero for noise. This is a
    cheap stand-in for a full harmonics-to-noise ratio, which is all the
    decision needs.
    """
    frame = frame - frame.mean()
    energy = float(np.dot(frame, frame))
    if energy <= 1e-9:
        return 0.0
    corr = np.correlate(frame, frame, mode="full")[frame.size - 1:]
    window = corr[_MIN_LAG:_MAX_LAG]
    if window.size == 0:
        return 0.0
    return float(np.clip(window.max() / energy, 0.0, 1.0))


def _zcr(frame: np.ndarray) -> float:
    return float(np.mean(np.abs(np.diff(np.sign(frame))) > 0))


def _tilt(frame: np.ndarray) -> float:
    """Energy above 2 kHz relative to energy below it."""
    spec = np.abs(np.fft.rfft(frame * np.hanning(frame.size)))
    freqs = np.fft.rfftfreq(frame.size, 1.0 / _SAMPLE_RATE)
    low = float(np.sum(spec[freqs < 2000] ** 2))
    high = float(np.sum(spec[freqs >= 2000] ** 2))
    if low + high <= 1e-12:
        return 0.0
    return high / (low + high)


def _combine(
    harmonicity: float, zcr: float, tilt: float, voiced_ratio: float
) -> float:
    """Blend the signals into one score.

    Voicing dominates: whispering is *defined* by its absence, while a
    bright or noisy voiced sound is still not a whisper. The other two
    signals mainly guard against a breathy-but-voiced speaker scoring
    high on harmonicity alone.
    """
    # Absence of periodicity, the primary evidence.
    voicing_absence = 1.0 - float(np.clip(voiced_ratio, 0.0, 1.0))
    # Secondary: noise-like excitation.
    zcr_score = float(np.clip((zcr - 0.10) / 0.25, 0.0, 1.0))
    tilt_score = float(np.clip((tilt - 0.25) / 0.35, 0.0, 1.0))
    return float(
        np.clip(0.6 * voicing_absence + 0.2 * zcr_score + 0.2 * tilt_score,
                0.0, 1.0)
    )


def analyze(
    audio: np.ndarray, threshold: float = DEFAULT_THRESHOLD
) -> WhisperFeatures:
    """Score a mono 16 kHz float32 waveform."""
    if audio.size == 0:
        return WhisperFeatures(0.0, 0.0, 0.0, 0.0, 0.0, 0.0, threshold)

    frames = _frames(audio.astype(np.float32))
    if frames.shape[0] == 0:
        return WhisperFeatures(0.0, 0.0, 0.0, 0.0, 0.0, 0.0, threshold)

    rms_per_frame = np.sqrt(np.mean(frames ** 2, axis=1))
    loud = rms_per_frame >= _SILENCE_RMS
    if not np.any(loud):
        # Nothing but room tone — no opinion rather than a false positive.
        return WhisperFeatures(
            0.0, 0.0, 0.0, 0.0, float(rms_per_frame.mean()), 0.0, threshold
        )

    active = frames[loud]
    harmonics = np.array([_harmonicity(f) for f in active])
    zcrs = np.array([_zcr(f) for f in active])
    tilts = np.array([_tilt(f) for f in active])

    # A frame counts as voiced when its periodicity is unmistakable.
    voiced_ratio = float(np.mean(harmonics > 0.5))
    score = _combine(
        float(harmonics.mean()), float(zcrs.mean()), float(tilts.mean()),
        voiced_ratio,
    )
    return WhisperFeatures(
        score=score,
        harmonicity=float(harmonics.mean()),
        zero_crossing_rate=float(zcrs.mean()),
        spectral_tilt=float(tilts.mean()),
        rms=float(rms_per_frame[loud].mean()),
        voiced_ratio=voiced_ratio,
        threshold=threshold,
    )


def analyze_pcm(
    pcm_bytes: bytes, threshold: float = DEFAULT_THRESHOLD
) -> WhisperFeatures:
    """Score 16-bit mono 16 kHz PCM."""
    if not pcm_bytes:
        return WhisperFeatures(0.0, 0.0, 0.0, 0.0, 0.0, 0.0, threshold)
    return analyze(pcm16_bytes_to_float32(pcm_bytes), threshold)
