"""Sample quality scoring for speaker enrollment audio.

Computes a composite quality score (0–1) from multiple signals:

* **Speech ratio** — VAD-detected speech vs total duration.
* **SNR estimate** — RMS ratio of speech segments vs silence segments.
* **Liveness** — spectral heuristics (rolloff, crest factor, HF ratio).
* **Embedding consistency** — split audio into windows, embed each, check
  variance.  High variance ⇒ multiple speakers or heavy noise.
* **Centroid distance** — how well this sample fits the speaker's existing
  profile.  Only available when a centroid already exists.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Dict, List, Optional

import numpy as np

from .audio import pcm16_bytes_to_float32
from .embeddings import CAMPPlusEmbedder
from .liveness import LivenessFeatures, analyze_pcm as liveness_analyze
from .vad import SileroVAD, VADResult

_LOGGER = logging.getLogger("voiceid.sample_quality")

# Minimum audio length in samples for the multi-window consistency check.
# ~1.5 s at 16 kHz ⇒ 24000 samples — below that we skip the check.
_MIN_CONSISTENCY_SAMPLES = 24000
_SAMPLE_RATE = 16000

# Default weights for the composite score (sum to 1.0).
DEFAULT_WEIGHTS: Dict[str, float] = {
    "speech_ratio": 0.25,
    "snr": 0.20,
    "liveness": 0.15,
    "consistency": 0.25,
    "centroid_distance": 0.15,
}


@dataclass
class QualityBreakdown:
    """Detailed per-component scores (each 0–1, higher = better)."""

    speech_ratio_score: float = 0.0
    snr_score: float = 0.0
    liveness_score: float = 0.0
    consistency_score: float = 0.0
    centroid_distance_score: float = 0.0

    # Raw values for UI display
    speech_ratio: float = 0.0
    snr_db: float = 0.0
    liveness_raw: float = 0.0
    consistency_std: float = 0.0
    centroid_distance: Optional[float] = None

    composite: float = 0.0
    weights_used: Dict[str, float] = field(default_factory=dict)

    def to_dict(self) -> Dict:
        return {
            "speech_ratio_score": round(self.speech_ratio_score, 3),
            "snr_score": round(self.snr_score, 3),
            "liveness_score": round(self.liveness_score, 3),
            "consistency_score": round(self.consistency_score, 3),
            "centroid_distance_score": round(self.centroid_distance_score, 3),
            "speech_ratio": round(self.speech_ratio, 3),
            "snr_db": round(self.snr_db, 1),
            "liveness_raw": round(self.liveness_raw, 3),
            "consistency_std": round(self.consistency_std, 4),
            "centroid_distance": round(self.centroid_distance, 4) if self.centroid_distance is not None else None,
            "composite": round(self.composite, 3),
            "weights_used": {k: round(v, 3) for k, v in self.weights_used.items()},
        }


def _speech_ratio_score(ratio: float) -> float:
    """Map speech ratio to 0–1.  Best at 70-90%, falls off below 40%."""
    if ratio >= 0.7:
        return 1.0
    if ratio <= 0.2:
        return 0.0
    return float(np.clip((ratio - 0.2) / 0.5, 0.0, 1.0))


def _snr_score(snr_db: float) -> float:
    """Map estimated SNR (dB) to 0–1.  20+ dB is excellent, <5 dB is bad."""
    if snr_db >= 20.0:
        return 1.0
    if snr_db <= 3.0:
        return 0.0
    return float(np.clip((snr_db - 3.0) / 17.0, 0.0, 1.0))


def _liveness_to_score(liveness: float) -> float:
    """Liveness already 0–1, just apply a slight curve to penalize low values."""
    return float(np.clip(liveness, 0.0, 1.0))


def _consistency_score(std: float) -> float:
    """Map embedding std across windows to 0–1.  Lower std = better.

    For L2-normalized 192-dim embeddings, same-speaker windows typically
    have std < 0.05, multi-speaker or noisy audio > 0.12.
    """
    if std <= 0.03:
        return 1.0
    if std >= 0.15:
        return 0.0
    return float(np.clip(1.0 - (std - 0.03) / 0.12, 0.0, 1.0))


def _centroid_distance_to_score(distance: float) -> float:
    """Map cosine distance to existing centroid to 0–1.

    Very close (< 0.15) = excellent fit, > 0.45 = poor fit.
    """
    if distance <= 0.10:
        return 1.0
    if distance >= 0.50:
        return 0.0
    return float(np.clip(1.0 - (distance - 0.10) / 0.40, 0.0, 1.0))


def _estimate_snr(audio: np.ndarray, vad_result: VADResult) -> float:
    """Estimate SNR from speech vs non-speech RMS levels."""
    if not vad_result.segments or audio.size == 0:
        return 0.0

    speech_samples: List[np.ndarray] = []
    noise_samples: List[np.ndarray] = []
    prev_end = 0

    for start_s, end_s in vad_result.segments:
        start = int(start_s * _SAMPLE_RATE)
        end = int(end_s * _SAMPLE_RATE)
        # Noise before this segment
        if prev_end < start:
            noise_samples.append(audio[prev_end:start])
        speech_samples.append(audio[start:end])
        prev_end = end

    # Trailing noise
    if prev_end < len(audio):
        noise_samples.append(audio[prev_end:])

    if not speech_samples:
        return 0.0

    speech_all = np.concatenate(speech_samples)
    speech_rms = float(np.sqrt(np.mean(speech_all ** 2) + 1e-12))

    if not noise_samples:
        # All speech — assume high SNR
        return 30.0

    noise_all = np.concatenate(noise_samples)
    if noise_all.size < 100:
        return 25.0  # Very little noise to measure

    noise_rms = float(np.sqrt(np.mean(noise_all ** 2) + 1e-12))
    if noise_rms < 1e-9:
        return 40.0

    snr = 20.0 * float(np.log10(speech_rms / noise_rms))
    return max(snr, 0.0)


def _embedding_consistency(
    audio: np.ndarray,
    embedder: CAMPPlusEmbedder,
) -> float:
    """Split audio into overlapping windows, embed each, return std of distances.

    A high standard deviation across window embeddings suggests multiple
    speakers or heavy interference.
    """
    if audio.size < _MIN_CONSISTENCY_SAMPLES:
        return 0.0  # Too short to assess — return best-case

    # ~1.5 s windows with 50% overlap
    window_size = int(1.5 * _SAMPLE_RATE)
    hop = window_size // 2
    embeddings: List[np.ndarray] = []

    for start in range(0, audio.size - window_size + 1, hop):
        chunk = audio[start:start + window_size]
        pcm_bytes = (chunk * 32767.0).astype(np.int16).tobytes()
        try:
            emb = embedder.embed_pcm(pcm_bytes)
            embeddings.append(emb)
        except ValueError:
            continue  # Window too quiet or short

    if len(embeddings) < 2:
        return 0.0  # Can't assess with fewer than 2 windows

    # Compute pairwise cosine distances and return std
    stacked = np.stack(embeddings)
    centroid = stacked.mean(axis=0)
    norm = float(np.linalg.norm(centroid))
    if norm > 1e-9:
        centroid = centroid / norm

    distances = []
    for emb in embeddings:
        d = 1.0 - float(np.dot(emb, centroid))
        distances.append(d)

    return float(np.std(distances))


def score_sample(
    pcm_bytes: bytes,
    embedder: CAMPPlusEmbedder,
    vad: Optional[SileroVAD] = None,
    centroid: Optional[np.ndarray] = None,
    weights: Optional[Dict[str, float]] = None,
) -> QualityBreakdown:
    """Compute composite quality score for an audio sample.

    Parameters
    ----------
    pcm_bytes : bytes
        Raw 16 kHz mono 16-bit PCM.
    embedder : CAMPPlusEmbedder
        Speaker-embedding model for consistency and centroid checks.
    vad : SileroVAD, optional
        Voice activity detector.  If None, speech_ratio and SNR default
        to neutral (0.5) so they don't dominate the score.
    centroid : np.ndarray, optional
        Existing speaker centroid for centroid_distance scoring.
        If None, centroid_distance component is redistributed.
    weights : dict, optional
        Override scoring weights.  Keys: speech_ratio, snr, liveness,
        consistency, centroid_distance.  Values should sum to 1.0.
    """
    w = dict(weights or DEFAULT_WEIGHTS)

    audio = pcm16_bytes_to_float32(pcm_bytes)
    breakdown = QualityBreakdown(weights_used=w)

    # --- Speech ratio + SNR ---
    vad_result: Optional[VADResult] = None
    if vad is not None:
        try:
            vad_result = vad.analyze_pcm(pcm_bytes)
        except Exception as exc:
            _LOGGER.warning("VAD failed during quality scoring: %s", exc)

    if vad_result is not None and vad_result.peak_probability > 0.1:
        breakdown.speech_ratio = vad_result.speech_ratio
        breakdown.speech_ratio_score = _speech_ratio_score(vad_result.speech_ratio)
        snr = _estimate_snr(audio, vad_result)
        breakdown.snr_db = snr
        breakdown.snr_score = _snr_score(snr)
    else:
        # No VAD — use neutral scores
        breakdown.speech_ratio = 0.5
        breakdown.speech_ratio_score = 0.5
        breakdown.snr_db = 10.0
        breakdown.snr_score = 0.5

    # --- Liveness ---
    try:
        liveness = liveness_analyze(pcm_bytes)
        breakdown.liveness_raw = liveness.score
        breakdown.liveness_score = _liveness_to_score(liveness.score)
    except Exception as exc:
        _LOGGER.warning("Liveness analysis failed: %s", exc)
        breakdown.liveness_raw = 0.5
        breakdown.liveness_score = 0.5

    # --- Embedding consistency ---
    try:
        consistency_std = _embedding_consistency(audio, embedder)
        breakdown.consistency_std = consistency_std
        breakdown.consistency_score = _consistency_score(consistency_std)
    except Exception as exc:
        _LOGGER.warning("Consistency check failed: %s", exc)
        breakdown.consistency_std = 0.0
        breakdown.consistency_score = 0.5

    # --- Centroid distance ---
    if centroid is not None:
        try:
            emb = embedder.embed_pcm(pcm_bytes)
            dist = CAMPPlusEmbedder.cosine_distance(emb, centroid)
            breakdown.centroid_distance = dist
            breakdown.centroid_distance_score = _centroid_distance_to_score(dist)
        except Exception as exc:
            _LOGGER.warning("Centroid distance check failed: %s", exc)
            breakdown.centroid_distance = None
            breakdown.centroid_distance_score = 0.5
    else:
        # No centroid — redistribute weight to other components
        breakdown.centroid_distance = None
        breakdown.centroid_distance_score = 0.0

    # --- Composite ---
    if centroid is None:
        # Redistribute centroid weight proportionally
        cd_weight = w.get("centroid_distance", 0.0)
        remaining = 1.0 - cd_weight
        if remaining > 0:
            effective_w = {
                k: v / remaining
                for k, v in w.items()
                if k != "centroid_distance"
            }
        else:
            effective_w = {k: 0.25 for k in w if k != "centroid_distance"}
    else:
        effective_w = w

    composite = 0.0
    composite += effective_w.get("speech_ratio", 0.0) * breakdown.speech_ratio_score
    composite += effective_w.get("snr", 0.0) * breakdown.snr_score
    composite += effective_w.get("liveness", 0.0) * breakdown.liveness_score
    composite += effective_w.get("consistency", 0.0) * breakdown.consistency_score
    composite += effective_w.get("centroid_distance", 0.0) * breakdown.centroid_distance_score

    breakdown.composite = float(np.clip(composite, 0.0, 1.0))
    return breakdown


def speaker_training_quality(
    sample_scores: List[float],
) -> float:
    """Aggregate per-sample scores into a speaker training quality (0–1).

    Takes into account both the average quality and consistency of samples.
    A speaker with 5 high-quality samples is better trained than one with
    20 mediocre ones.
    """
    if not sample_scores:
        return 0.0
    arr = np.array(sample_scores, dtype=np.float64)
    avg = float(arr.mean())
    # Bonus for having enough samples (diminishing returns after ~8)
    count_factor = float(np.clip(len(sample_scores) / 8.0, 0.0, 1.0))
    # Penalty for inconsistent quality
    std_penalty = float(np.clip(1.0 - arr.std(), 0.5, 1.0)) if len(arr) > 1 else 1.0
    quality = avg * 0.6 + count_factor * 0.25 + std_penalty * 0.15
    return float(np.clip(quality, 0.0, 1.0))
