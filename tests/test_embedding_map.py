"""Tests for the embedding map (PCA projection) and speaker health.

Uses a deterministic fake embedder (hash of the audio bytes → unit
vector), so distances are stable per sample without ONNX models.
"""

from __future__ import annotations

import hashlib

import numpy as np
import pytest

from murdock.core.db import open_db
from murdock.core.embedding_map import compute_embedding_map, pca_2d
from murdock.core.speaker_store import SpeakerStore
from murdock.core.unknown_store import UnknownStore

_RATE = 16000


class FakeEmbedder:
    """Deterministic per-audio embedder: sha256(audio) → 192-dim unit vec."""

    EMBEDDING_DIM = 192

    def embed_pcm(self, pcm: bytes) -> np.ndarray:
        digest = hashlib.sha256(pcm).digest() * 6  # 192 bytes
        v = np.frombuffer(digest[:192], dtype=np.uint8).astype(np.float32)
        v -= v.mean()
        return v / (np.linalg.norm(v) or 1.0)


def _pcm(seed: int, secs: float = 2.0) -> bytes:
    """Distinct, deterministic 16-bit PCM per seed."""
    rng = np.random.default_rng(seed)
    return (rng.integers(-2000, 2000, int(secs * _RATE))
            .astype(np.int16).tobytes())


def _store(tmp_path):
    conn = open_db(tmp_path / "m.db")
    return SpeakerStore(conn=conn, embedder=FakeEmbedder(), vad=None), conn


# --- pca_2d ----------------------------------------------------------------


def test_pca_separates_clusters():
    rng = np.random.default_rng(0)
    a = rng.normal(0.0, 0.05, (20, 192)) + np.eye(192)[0] * 2
    b = rng.normal(0.0, 0.05, (20, 192)) - np.eye(192)[0] * 2
    coords, explained = pca_2d(np.vstack([a, b]))
    assert coords.shape == (40, 2)
    # Cluster means should be far apart in 2-D relative to spread.
    ca, cb = coords[:20].mean(axis=0), coords[20:].mean(axis=0)
    gap = float(np.linalg.norm(ca - cb))
    # Within-cluster scatter: mean distance of points to their own mean.
    spread = float(
        np.linalg.norm(coords[:20] - ca, axis=1).mean()
        + np.linalg.norm(coords[20:] - cb, axis=1).mean()
    )
    assert gap > spread * 3
    # PC1 dominates this synthetic data.
    assert explained[0] > 0.5
    assert 0.0 <= explained[1] <= explained[0]


def test_pca_rejects_too_few_points():
    with pytest.raises(ValueError):
        pca_2d(np.zeros((1, 192)))


# --- compute_embedding_map ---------------------------------------------------


def test_map_points_and_kinds(tmp_path):
    store, conn = _store(tmp_path)
    for i in range(3):
        store.enroll("anna", _pcm(i), 2.0, skip_vad=True)
    for i in range(3, 6):
        store.enroll("ben", _pcm(i), 2.0, skip_vad=True)
    unknown = UnknownStore(conn)
    unknown.record("sess-x", _pcm(99), FakeEmbedder().embed_pcm(_pcm(99)),
                   2.0, 0.9, None, None, None)

    data = compute_embedding_map(store, unknown, include_unknown=True)
    kinds = [p["kind"] for p in data["points"]]
    assert kinds.count("sample") == 6
    assert kinds.count("centroid") == 2
    assert kinds.count("unknown") == 1
    # Every point has plot coordinates; samples carry centroid distance.
    for p in data["points"]:
        assert "x" in p and "y" in p
        if p["kind"] == "sample":
            assert p["distance"] >= 0.0
    assert data["computed_ms"] >= 0


def test_map_excludes_unknown_when_asked(tmp_path):
    store, conn = _store(tmp_path)
    for i in range(4):
        store.enroll("anna", _pcm(i), 2.0, skip_vad=True)
    unknown = UnknownStore(conn)
    unknown.record("sess-x", _pcm(99), FakeEmbedder().embed_pcm(_pcm(99)),
                   2.0, 0.9, None, None, None)
    data = compute_embedding_map(store, unknown, include_unknown=False)
    assert all(p["kind"] != "unknown" for p in data["points"])


def test_map_not_enough_data(tmp_path):
    store, _conn = _store(tmp_path)
    store.enroll("anna", _pcm(1), 2.0, skip_vad=True)
    data = compute_embedding_map(store, None, include_unknown=False)
    assert data["points"] == []
    assert "note" in data


# --- speaker_health ----------------------------------------------------------


def test_health_stats(tmp_path):
    store, _conn = _store(tmp_path)
    res = None
    for i in range(4):
        res = store.enroll("anna", _pcm(i), 2.0, skip_vad=True)
    h = store.speaker_health(res.speaker_id)
    assert h["embedded_count"] == 4
    assert len(h["samples"]) == 4
    # Random fake embeddings are near-orthogonal → clear positive spread.
    assert h["spread_avg"] > 0
    assert h["spread_max"] >= h["spread_avg"]
    for s in h["samples"]:
        assert s["age_days"] >= 0
        assert s["centroid_distance"] >= 0


def test_health_empty_speaker(tmp_path):
    store, _conn = _store(tmp_path)
    spk = store.create_speaker("leer")
    h = store.speaker_health(spk.id)
    assert h["embedded_count"] == 0
    assert h["samples"] == []
    assert h["spread_avg"] is None
