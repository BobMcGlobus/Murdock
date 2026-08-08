"""Tests for the emotion classifier's two-part model handling.

emotion2vec+ ships as a feature extractor (ONNX, frame-level 768-dim
output) plus a separate linear head. The classifier used to paper over
that by inventing label names like ``class_44737`` — these tests pin the
correct behaviour: pool + head when features arrive, and a loud failure
rather than a guess when nothing can produce class scores.
"""

from __future__ import annotations

import struct

import numpy as np
import pytest

from murdock.core.emotion import DEFAULT_LABELS, EmotionClassifier

_DIM = 768
_CLASSES = 9


def _write_head(path, n_classes=_CLASSES, dim=_DIM, seed=3):
    rng = np.random.default_rng(seed)
    weight = rng.normal(0, 0.05, (n_classes, dim)).astype("<f4")
    bias = rng.normal(0, 0.01, n_classes).astype("<f4")
    path.write_bytes(
        struct.pack("<ii", n_classes, dim) + weight.tobytes() + bias.tobytes()
    )
    return weight, bias


class _FakeSession:
    """Stands in for onnxruntime, returning a chosen output shape."""

    def __init__(self, output):
        self._output = output

    def get_inputs(self):
        return [type("I", (), {"name": "input"})()]

    def get_outputs(self):
        return [type("O", (), {"name": "output"})()]

    def run(self, _names, _feeds):
        return [self._output]


def _classifier(tmp_path, output, *, with_head=True):
    model = tmp_path / "emotion.onnx"
    model.write_bytes(b"not really onnx")
    if with_head:
        _write_head(tmp_path / "emotion_head.bin")
    c = EmotionClassifier(model_path=model)
    c._session = _FakeSession(output)
    c._input_names = ["input"]
    c._output_name = "output"
    return c


def _audio(seconds=3.0):
    rng = np.random.default_rng(11)
    return rng.normal(0, 0.1, int(seconds * 16000)).astype(np.float32)


def test_head_file_layout_round_trip(tmp_path):
    weight, bias = _write_head(tmp_path / "emotion_head.bin")
    c = EmotionClassifier(model_path=tmp_path / "emotion.onnx")
    loaded = c._load_head()
    assert loaded is not None
    np.testing.assert_allclose(loaded[0], weight)
    np.testing.assert_allclose(loaded[1], bias)


def test_frame_features_are_pooled_through_the_head(tmp_path):
    frames = np.ones((1, 149, _DIM), dtype=np.float32) * 0.01
    c = _classifier(tmp_path, frames)
    result = c.classify_waveform(_audio())
    assert result.label in DEFAULT_LABELS
    assert set(result.scores) == set(DEFAULT_LABELS)
    assert sum(result.scores.values()) == pytest.approx(1.0, abs=1e-5)


def test_direct_class_scores_still_work(tmp_path):
    """A model that already outputs 9 logits needs no head."""
    logits = np.array([[0.1, 0.2, 0.3, 5.0, 0.1, 0.0, 0.1, 0.2, 0.0]], dtype=np.float32)
    c = _classifier(tmp_path, logits, with_head=False)
    result = c.classify_waveform(_audio())
    assert result.label == "happy"
    assert result.confidence > 0.9


def test_features_without_a_head_fail_loudly(tmp_path):
    """The old code invented labels here. It must refuse instead."""
    frames = np.ones((1, 149, _DIM), dtype=np.float32)
    c = _classifier(tmp_path, frames, with_head=False)
    with pytest.raises(RuntimeError, match="no usable head"):
        c.classify_waveform(_audio())


def test_head_with_wrong_class_count_is_refused(tmp_path):
    """A head that disagrees with the label list must not be guessed at."""
    _write_head(tmp_path / "emotion_head.bin", n_classes=5)
    model = tmp_path / "emotion.onnx"
    model.write_bytes(b"x")
    c = EmotionClassifier(model_path=model)
    c._session = _FakeSession(np.ones((1, 20, _DIM), dtype=np.float32))
    c._input_names = ["input"]
    c._output_name = "output"
    with pytest.raises(RuntimeError, match="refusing to"):
        c.classify_waveform(_audio())


def test_unexpected_output_shape_never_invents_labels(tmp_path):
    """The old code answered with made-up names like ``class_44737``."""
    c = _classifier(tmp_path, np.zeros((1, 5), dtype=np.float32), with_head=False)
    with pytest.raises(RuntimeError) as excinfo:
        c.classify_waveform(_audio())
    assert "class_" not in str(excinfo.value)


def test_availability_needs_both_files(tmp_path):
    model = tmp_path / "emotion.onnx"
    model.write_bytes(b"x")
    c = EmotionClassifier(model_path=model)
    # ONNX alone is only a feature extractor — not usable.
    assert c.is_available() is False
    _write_head(tmp_path / "emotion_head.bin")
    assert EmotionClassifier(model_path=model).is_available() is True


def test_corrupt_head_is_ignored(tmp_path):
    (tmp_path / "emotion_head.bin").write_bytes(struct.pack("<ii", 9, 768) + b"short")
    c = EmotionClassifier(model_path=tmp_path / "emotion.onnx")
    assert c._load_head() is None


def test_custom_head_path(tmp_path):
    head = tmp_path / "elsewhere.bin"
    _write_head(head)
    c = EmotionClassifier(
        model_path=tmp_path / "emotion.onnx", head_path=head
    )
    assert c.head_path == head
    assert c._load_head() is not None
