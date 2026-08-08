"""Emotion classification via onnxruntime.

Loads emotion2vec+ base and turns mono 16 kHz audio into one of nine
emotion labels plus a confidence.

The published ONNX export is the **feature extractor only**: it emits
frame-level features of shape ``[1, frames, 768]``. The classification
head is a single linear layer shipped alongside as a small binary
(``emotion_head.bin``: two int32 — classes, dim — then the weight matrix
and bias as float32). Utterance-level inference is therefore mean-pooling
over frames followed by that linear layer, which is exactly what the
upstream implementation does. Both files are needed; with only the ONNX
there is no classifier at all.

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
- **No silent guessing.** If the model's output doesn't match the labels
  and no head can turn it into class scores, classification fails rather
  than inventing label names — a confidently wrong emotion is worse than
  an absent one.

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
        head_path: Optional[Path] = None,
        sample_rate: int = _SAMPLE_RATE,
        min_duration_sec: float = _MIN_DURATION_SEC,
    ) -> None:
        self.model_path = Path(model_path)
        # The classification head lives next to the ONNX; without it the
        # export is a feature extractor and cannot name an emotion.
        self.head_path = (
            Path(head_path) if head_path
            else self.model_path.with_name("emotion_head.bin")
        )
        self.sample_rate = sample_rate
        self.min_duration_sec = float(min_duration_sec)
        self.labels: List[str] = list(labels) if labels else list(DEFAULT_LABELS)
        self._session = None  # type: ignore[assignment]
        self._input_names: List[str] = []
        self._output_name: Optional[str] = None
        self._wants_lengths: bool = False
        self._head: Optional[Tuple[np.ndarray, np.ndarray]] = None
        self._lock = Lock()

    # ------------------------------------------------------------------
    # Session lifecycle
    # ------------------------------------------------------------------

    def is_available(self) -> bool:
        """Whether a *usable* model is on disk. Does not load it.

        Both parts are required: the ONNX alone only yields frame
        features, which cannot be turned into an emotion.
        """
        return self.model_path.exists() and self.head_path.exists()

    def _load_head(self) -> Optional[Tuple[np.ndarray, np.ndarray]]:
        """Read the linear head: int32 classes, int32 dim, W, bias."""
        if self._head is not None:
            return self._head
        if not self.head_path.exists():
            return None
        raw = self.head_path.read_bytes()
        if len(raw) < 8:
            _LOGGER.warning("Emotion head %s is truncated", self.head_path)
            return None
        n_classes, dim = np.frombuffer(raw[:8], dtype="<i4")
        expected = 8 + int(n_classes) * int(dim) * 4 + int(n_classes) * 4
        if len(raw) != expected:
            _LOGGER.warning(
                "Emotion head %s has %d bytes, expected %d for %dx%d — ignoring",
                self.head_path, len(raw), expected, n_classes, dim,
            )
            return None
        end_w = 8 + int(n_classes) * int(dim) * 4
        weight = np.frombuffer(raw[8:end_w], dtype="<f4").reshape(
            int(n_classes), int(dim)
        )
        bias = np.frombuffer(raw[end_w:], dtype="<f4")
        self._head = (weight, bias)
        _LOGGER.info(
            "Loaded emotion head: %d classes x %d dims", n_classes, dim
        )
        return self._head

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

        raw = np.asarray(outputs[0])

        # Frame-level features [1, frames, dim] → mean-pool → linear head.
        # This is how emotion2vec+ does utterance-level classification.
        if raw.ndim == 3 or (raw.ndim == 2 and raw.shape[-1] not in (len(self.labels),)):
            head = self._load_head()
            if head is None:
                raise RuntimeError(
                    f"Emotion model returned features of shape {raw.shape}, "
                    f"not {len(self.labels)} class scores, and no usable head "
                    f"was found at {self.head_path}. Both files are required."
                )
            weight, bias = head
            pooled = raw.reshape(-1, weight.shape[1]).mean(axis=0)
            logits = (weight @ pooled + bias).astype(np.float32)
        else:
            logits = raw.reshape(-1).astype(np.float32)

        probs = _softmax(logits)
        if probs.size != len(self.labels):
            # Never invent label names: a confidently wrong emotion is
            # worse than none at all.
            raise RuntimeError(
                f"Emotion model produced {probs.size} scores but "
                f"{len(self.labels)} labels are configured — refusing to "
                f"guess. Check that the model and label list match."
            )
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
