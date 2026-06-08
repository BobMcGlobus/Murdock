"""Emotion classification via onnxruntime.

Loads an emotion-recognition ONNX model (emotion2vec+ base by default) and
turns mono 16 kHz audio into one of nine emotion labels plus a confidence.

Design goals:

- **No PyTorch.** Everything runs on onnxruntime so the model fits into the
  existing HA add-on image without doubling its size.
- **Lazy loading.** The ONNX session is only materialised on first call;
  users who never enable emotion detection pay zero RAM cost.
- **Signature-agnostic.** emotion2vec-family exports come in two flavours:
  a single-input ``waveform[1, N]`` and a two-input ``(waveform, lengths)``
  variant. The classifier inspects the loaded session and adapts.
- **Post-hoc label override.** The 9-class label order of the standard
  emotion2vec+ export is hard-coded, but callers can pass a custom list
  for experimental models.

The standard emotion2vec+ class order (matching the FunASR/ModelScope
checkpoints) is::

    0 angry   1 disgusted   2 fearful   3 happy      4 neutral
    5 other   6 sad         7 surprised 8 unknown

"other" and "unknown" are garbage classes that the training set uses for
non-speech / ambiguous clips. The raw softmax still returns them; higher
layers can filter them out before pushing to HA.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from threading import Lock
from typing import Dict, List, Optional, Tuple

import numpy as np

from .audio import pcm16_bytes_to_float32

_LOGGER = logging.getLogger("murdock.emotion")

# Standard emotion2vec+ label order. Keep in sync with whatever checkpoint
# users download. If we ever add an alternative model with different
# classes, pass a custom label list to the constructor.
DEFAULT_LABELS: Tuple[str, ...] = (
    "angry",
    "disgusted",
    "fearful",
    "happy",
    "neutral",
    "other",
    "sad",
    "surprised",
    "unknown",
)

# Classes we don't want to surface to users — they indicate the classifier
# itself was uncertain or the clip wasn't really speech. Callers can use
# ``EmotionResult.is_meaningful`` to filter these out cleanly.
NON_EMOTION_LABELS: frozenset = frozenset({"other", "unknown"})

# Minimum audio length for a stable emotion prediction. Shorter than this
# and the SSL backbone produces essentially random logits.
_MIN_DURATION_SEC = 1.0
_SAMPLE_RATE = 16000


@dataclass(frozen=True)
class EmotionResult:
    """Output of :meth:`EmotionClassifier.classify_pcm`."""

    label: str
    confidence: float
    scores: Dict[str, float]

    @property
    def is_meaningful(self) -> bool:
        """False for ``other`` / ``unknown`` — useful to gate HA pushes."""
        return self.label not in NON_EMOTION_LABELS

    def top_k(self, k: int = 3) -> List[Tuple[str, float]]:
        """Return the top-k ``(label, score)`` pairs by score, descending."""
        return sorted(self.scores.items(), key=lambda kv: kv[1], reverse=True)[:k]


class EmotionClassifier:
    """ONNX-backed speech emotion recognizer.

    The model is loaded lazily on first call so that enabling the feature
    doesn't bloat the murdock process RAM for users who never turn it on.
    Thread-safe: a lock serialises access to the single ``InferenceSession``.
    """

    def __init__(
        self,
        model_path: Path,
        labels: Optional[List[str]] = None,
        sample_rate: int = _SAMPLE_RATE,
        min_duration_sec: float = _MIN_DURATION_SEC,
    ) -> None:
        self.model_path = Path(model_path)
        self.sample_rate = sample_rate
        self.min_duration_sec = float(min_duration_sec)
        self.labels: List[str] = list(labels) if labels else list(DEFAULT_LABELS)
        self._session = None  # type: ignore[assignment]
        self._input_names: List[str] = []
        self._output_name: Optional[str] = None
        self._wants_lengths: bool = False
        self._lock = Lock()

    # ------------------------------------------------------------------
    # Session lifecycle
    # ------------------------------------------------------------------

    def is_available(self) -> bool:
        """Whether the model file is on disk. Does not actually load it."""
        return self.model_path.exists()

    def _ensure_session(self) -> None:
        if self._session is not None:
            return
        if not self.model_path.exists():
            raise FileNotFoundError(
                f"Emotion ONNX model not found at {self.model_path}. "
                "Run scripts/download_models.sh (with emotion enabled) or "
                "mount the model file manually."
            )
        # Imported lazily so importing the module (for type hints, tests,
        # the web UI startup path) doesn't force an onnxruntime import.
        import onnxruntime as ort

        sess_opts = ort.SessionOptions()
        sess_opts.intra_op_num_threads = 1
        sess_opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        self._session = ort.InferenceSession(
            str(self.model_path),
            sess_options=sess_opts,
            providers=["CPUExecutionProvider"],
        )
        self._input_names = [inp.name for inp in self._session.get_inputs()]
        self._output_name = self._session.get_outputs()[0].name
        # emotion2vec+ is published in two common shapes: a one-input variant
        # (waveform only) and a two-input variant (waveform + lengths). The
        # lengths input is named ``lengths``, ``wav_lengths`` or similar
        # across exports — detect by count, which is unambiguous enough.
        self._wants_lengths = len(self._input_names) >= 2
        _LOGGER.info(
            "Loaded emotion classifier from %s (inputs=%s, output=%s)",
            self.model_path, self._input_names, self._output_name,
        )

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def classify_pcm(self, pcm_bytes: bytes) -> EmotionResult:
        """Classify 16-bit signed PCM bytes at ``self.sample_rate``."""
        audio = pcm16_bytes_to_float32(pcm_bytes)
        return self.classify_waveform(audio)

    def classify_waveform(self, audio: np.ndarray) -> EmotionResult:
        """Classify a float32 mono waveform in ``[-1, 1]``."""
        if audio.size == 0:
            raise ValueError("Cannot classify empty audio")
        duration = audio.size / float(self.sample_rate)
        if duration < self.min_duration_sec:
            raise ValueError(
                f"Audio too short for emotion classification: "
                f"{duration:.2f}s < {self.min_duration_sec:.2f}s"
            )

        # Ensure shape [1, N] float32. Some ONNX exports are strict about
        # contiguous memory — ``ascontiguousarray`` is cheap insurance.
        waveform = np.ascontiguousarray(audio.astype(np.float32))[np.newaxis, :]

        with self._lock:
            self._ensure_session()
            feeds: Dict[str, np.ndarray] = {self._input_names[0]: waveform}
            if self._wants_lengths and len(self._input_names) >= 2:
                # Common names across exports: lengths / wav_lengths / input_lengths.
                # Feeding the second input by position is more robust than
                # guessing the name.
                feeds[self._input_names[1]] = np.asarray(
                    [waveform.shape[1]], dtype=np.int64
                )
            outputs = self._session.run([self._output_name], feeds)

        logits = np.asarray(outputs[0]).reshape(-1).astype(np.float32)
        probs = _softmax(logits)

        if probs.size != len(self.labels):
            # Don't crash — pad or truncate the label list so we still
            # return something useful. Log loudly so users notice the
            # mismatch in their logs.
            _LOGGER.warning(
                "Emotion model emitted %d scores but %d labels are configured; "
                "output will be truncated/padded.",
                probs.size, len(self.labels),
            )
            labels = list(self.labels)
            if probs.size > len(labels):
                labels.extend(f"class_{i}" for i in range(len(labels), probs.size))
            else:
                probs = np.pad(probs, (0, len(labels) - probs.size))
        else:
            labels = self.labels

        scores = {label: float(score) for label, score in zip(labels, probs)}
        best_idx = int(np.argmax(probs))
        return EmotionResult(
            label=labels[best_idx],
            confidence=float(probs[best_idx]),
            scores=scores,
        )


# ----------------------------------------------------------------------
# Helpers
# ----------------------------------------------------------------------


def _softmax(x: np.ndarray) -> np.ndarray:
    """Numerically stable softmax over the last axis.

    Some ONNX exports already apply softmax on-graph (outputs look like
    probabilities summing to ~1). Re-applying softmax to probabilities is
    harmless *as long as* the values are bounded — which they are — so we
    don't bother trying to detect the case. The result is still a valid
    distribution and the argmax doesn't move.
    """
    x = x.astype(np.float32)
    if x.size == 0:
        return x
    shifted = x - np.max(x)
    exp = np.exp(shifted)
    total = float(np.sum(exp))
    if total <= 1e-9:
        # Degenerate output — return uniform rather than NaN.
        return np.full_like(x, 1.0 / x.size)
    return exp / total
