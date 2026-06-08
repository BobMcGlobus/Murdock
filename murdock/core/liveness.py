"""Lightweight spectral liveness heuristics.

The MVP liveness score is a cheap sanity check for "is this probably TV
or radio audio rather than a live speaker?". It is intentionally not a
hard gate — it just annotates unknown-sample reviews and biases the
verification threshold when a media player is known to be playing.

Signals combined into a single [0, 1] score (1 = very likely live voice):

* **Spectral rolloff** — TV/music tends to have energy pushed higher in
  the spectrum than conversational speech.
* **Crest factor** — heavily compressed broadcast audio has a lower peak
  to RMS ratio than natural, unprocessed voice.
* **High-frequency energy ratio** — an extra check against bright,
  consumer-processed sources.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .audio import pcm16_bytes_to_float32

_SAMPLE_RATE = 16000
_FFT_SIZE = 1024


@dataclass
class LivenessFeatures:
    rolloff_hz: float
    crest_factor: float
    hf_ratio: float
    score: float

    @property
    def is_probably_tv(self) -> bool:
        return self.score < 0.35


def _spectrogram(audio: np.ndarray, fft_size: int = _FFT_SIZE) -> np.ndarray:
    """Simple magnitude spectrogram for heuristic feature extraction."""
    if audio.size < fft_size:
        return np.zeros((0, fft_size // 2 + 1), dtype=np.float32)
    hop = fft_size // 2
    num_frames = 1 + (audio.size - fft_size) // hop
    window = np.hanning(fft_size).astype(np.float32)
    indices = np.arange(fft_size)[None, :] + hop * np.arange(num_frames)[:, None]
    frames = audio[indices] * window
    spec = np.abs(np.fft.rfft(frames, axis=1)).astype(np.float32)
    return spec


def _spectral_rolloff(spec: np.ndarray, sample_rate: int, percentile: float = 0.85) -> float:
    if spec.size == 0:
        return 0.0
    total = spec.sum(axis=1, keepdims=True) + 1e-9
    cumsum = np.cumsum(spec, axis=1)
    target = percentile * total
    idx = np.argmax(cumsum >= target, axis=1)
    freqs = np.linspace(0.0, sample_rate / 2.0, spec.shape[1])
    return float(np.mean(freqs[idx]))


def _crest_factor(audio: np.ndarray) -> float:
    if audio.size == 0:
        return 0.0
    peak = float(np.max(np.abs(audio)))
    rms = float(np.sqrt(np.mean(audio**2) + 1e-12))
    if rms < 1e-6:
        return 0.0
    return peak / rms


def _hf_ratio(spec: np.ndarray) -> float:
    if spec.size == 0:
        return 0.0
    total = float(spec.sum()) + 1e-9
    hf_start = int(spec.shape[1] * 0.5)  # everything above ~4 kHz
    hf = float(spec[:, hf_start:].sum())
    return hf / total


def _combine(rolloff: float, crest: float, hf: float) -> float:
    """Combine the individual signals into a single 0–1 liveness score."""
    # Rolloff: live speech typically peaks at 3-5 kHz, TV/music 5-8 kHz.
    rolloff_score = float(np.clip(1.0 - max(rolloff - 3500.0, 0.0) / 4500.0, 0.0, 1.0))
    # Crest factor: natural speech ~4-8, compressed broadcast ~2-3.
    crest_score = float(np.clip((crest - 2.0) / 5.0, 0.0, 1.0))
    # HF ratio: less than 25% high-freq energy for live speech.
    hf_score = float(np.clip(1.0 - max(hf - 0.25, 0.0) / 0.5, 0.0, 1.0))
    return float(0.45 * rolloff_score + 0.35 * crest_score + 0.20 * hf_score)


def analyze_pcm(pcm_bytes: bytes) -> LivenessFeatures:
    audio = pcm16_bytes_to_float32(pcm_bytes)
    if audio.size == 0:
        return LivenessFeatures(0.0, 0.0, 0.0, 0.0)
    spec = _spectrogram(audio)
    rolloff = _spectral_rolloff(spec, _SAMPLE_RATE)
    crest = _crest_factor(audio)
    hf = _hf_ratio(spec)
    score = _combine(rolloff, crest, hf)
    return LivenessFeatures(
        rolloff_hz=rolloff, crest_factor=crest, hf_ratio=hf, score=score
    )
