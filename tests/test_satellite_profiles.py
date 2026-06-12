"""Tests for per-satellite voice sub-profiles and adaptive thresholds."""

from __future__ import annotations

import hashlib

import numpy as np
import pytest

from murdock.core.calibration import compute_adaptive_thresholds
from murdock.core.db import open_db
from murdock.core.speaker_store import SpeakerStore

_RATE = 16000


class FakeEmbedder:
    """Deterministic per-audio embedder: sha256(audio) → 192-dim unit vec."""

    EMBEDDING_DIM = 192

    def embed_pcm(self, pcm: bytes) -> np.ndarray:
        digest = hashlib.sha256(pcm).digest() * 6
        v = np.frombuffer(digest[:192], dtype=np.uint8).astype(np.float32)
        v -= v.mean()
        return v / (np.linalg.norm(v) or 1.0)


def _pcm(seed: int, secs: float = 2.0) -> bytes:
    rng = np.random.default_rng(seed)
    return (rng.integers(-2000, 2000, int(secs * _RATE))
            .astype(np.int16).tobytes())


def _store(tmp_path):
    conn = open_db(tmp_path / "m.db")
    return SpeakerStore(conn=conn, embedder=FakeEmbedder(), vad=None)


# --- sub-centroid lifecycle ---------------------------------------------------


def test_sub_centroid_needs_min_samples(tmp_path):
    store = _store(tmp_path)
    res = None
    # 3 kitchen samples → sub-profile; 2 bath samples → below the minimum.
    for i in range(3):
        res = store.enroll("anna", _pcm(i), 2.0, skip_vad=True,
                           satellite_id="kitchen")
    for i in range(3, 5):
        res = store.enroll("anna", _pcm(i), 2.0, skip_vad=True,
                           satellite_id="bath")
    profiles = store.list_satellite_profiles(res.speaker_id)
    assert profiles == {"kitchen": 3}


def test_untagged_samples_build_no_profiles(tmp_path):
    store = _store(tmp_path)
    res = None
    for i in range(4):
        res = store.enroll("anna", _pcm(i), 2.0, skip_vad=True)
    assert store.list_satellite_profiles(res.speaker_id) == {}


def test_sub_centroids_rebuilt_on_sample_delete(tmp_path):
    store = _store(tmp_path)
    res = None
    for i in range(3):
        res = store.enroll("anna", _pcm(i), 2.0, skip_vad=True,
                           satellite_id="kitchen")
    sid = res.speaker_id
    assert store.list_satellite_profiles(sid) == {"kitchen": 3}
    # Dropping below the minimum removes the sub-profile.
    sample_id = store.list_samples(sid)[0]["id"]
    store.delete_sample(sample_id)
    assert store.list_satellite_profiles(sid) == {}


# --- verification with sub-profiles --------------------------------------------


class ClusteredEmbedder:
    """Embedder that models microphone coloring.

    The first two int16 samples of the PCM encode (group, idx); the
    embedding is the group's base direction plus per-sample noise — so
    same-mic recordings cluster tightly while a different mic points
    elsewhere, which is exactly the bias sub-profiles exist to fix.
    """

    EMBEDDING_DIM = 192

    def embed_pcm(self, pcm: bytes) -> np.ndarray:
        arr = np.frombuffer(pcm[:8], dtype=np.int16)
        group, idx = int(arr[0]), int(arr[1])
        base = np.random.default_rng(group).normal(size=192)
        noise = np.random.default_rng(1000 + idx).normal(size=192)
        v = base + 0.3 * noise
        return (v / np.linalg.norm(v)).astype(np.float32)


def _cpcm(group: int, idx: int, secs: float = 2.0) -> bytes:
    rng = np.random.default_rng(group * 100 + idx)
    a = rng.integers(-2000, 2000, int(secs * _RATE)).astype(np.int16)
    a[0], a[1] = group, idx
    return a.tobytes()


def test_satellite_match_uses_better_centroid(tmp_path):
    conn = open_db(tmp_path / "m.db")
    store = SpeakerStore(conn=conn, embedder=ClusteredEmbedder(), vad=None)
    # 3 kitchen-mic samples (group 1) + 2 from another mic (group 2): the
    # global centroid blends both colorings, the kitchen sub-centroid is
    # pure same-mic.
    for i in range(3):
        store.enroll("anna", _cpcm(1, i), 2.0, skip_vad=True,
                     satellite_id="kitchen")
    for i in range(3, 5):
        store.enroll("anna", _cpcm(2, i), 2.0, skip_vad=True)

    query = ClusteredEmbedder().embed_pcm(_cpcm(1, 9))  # fresh kitchen take
    base = store.verify_embedding(query, threshold=2.0)
    with_sat = store.verify_embedding(query, threshold=2.0, satellite_id="kitchen")
    assert with_sat.distance < base.distance
    assert with_sat.used_satellite_profile is True
    assert base.used_satellite_profile is False
    # Unknown satellite → falls back to the global centroid.
    other = store.verify_embedding(query, threshold=2.0, satellite_id="garage")
    assert other.distance == pytest.approx(base.distance, abs=1e-4)


# --- adaptive per-speaker thresholds in verify ----------------------------------


def test_adaptive_threshold_gates_winner(tmp_path):
    store = _store(tmp_path)
    for i in range(3):
        store.enroll("anna", _pcm(i), 2.0, skip_vad=True)
    stranger = FakeEmbedder().embed_pcm(_pcm(99))

    strict = store.verify_embedding(stranger, threshold=0.3)
    assert strict.is_match is False
    lenient = store.verify_embedding(
        stranger, threshold=0.3,
        speaker_thresholds={"anna": 1.8}, global_threshold=0.3,
    )
    assert lenient.is_match is True
    assert lenient.threshold == pytest.approx(1.8)


def test_satellite_delta_rides_on_adaptive(tmp_path):
    store = _store(tmp_path)
    for i in range(3):
        store.enroll("anna", _pcm(i), 2.0, skip_vad=True)
    emb = FakeEmbedder().embed_pcm(_pcm(0))
    # base 0.4 vs global 0.3 → +0.1 room delta on top of the adaptive 1.0.
    r = store.verify_embedding(
        emb, threshold=0.4,
        speaker_thresholds={"anna": 1.0}, global_threshold=0.3,
    )
    assert r.threshold == pytest.approx(1.1)


def test_tighten_applies_to_both_paths(tmp_path):
    store = _store(tmp_path)
    for i in range(3):
        store.enroll("anna", _pcm(i), 2.0, skip_vad=True)
    emb = FakeEmbedder().embed_pcm(_pcm(0))
    plain = store.verify_embedding(
        emb, threshold=0.4, global_threshold=0.3, tighten=0.1
    )
    assert plain.threshold == pytest.approx(0.3)  # base − tighten
    adaptive = store.verify_embedding(
        emb, threshold=0.4, global_threshold=0.3, tighten=0.1,
        speaker_thresholds={"anna": 1.0},
    )
    assert adaptive.threshold == pytest.approx(1.0)  # 1.0 + 0.1 − 0.1


# --- compute_adaptive_thresholds -------------------------------------------------


def test_adaptive_midpoint_inside_band():
    per = {"anna": {
        "genuine": [0.18, 0.20, 0.22, 0.21, 0.19],
        "impostor": [0.32, 0.35, 0.31, 0.36, 0.34],
    }}
    out = compute_adaptive_thresholds(per, global_threshold=0.30)
    # midpoint of g95 (~0.218) and i05 (~0.312) ≈ 0.265, inside ±0.08.
    assert 0.22 <= out["anna"] <= 0.38


def test_adaptive_clamped_to_band():
    per = {"anna": {
        "genuine": [0.10] * 6,
        "impostor": [0.90] * 6,
    }}
    out = compute_adaptive_thresholds(per, global_threshold=0.30)
    assert out["anna"] == pytest.approx(0.38)  # capped at global + 0.08


def test_adaptive_skips_overlap_and_small_samples():
    per = {
        "overlap": {"genuine": [0.30] * 6, "impostor": [0.25] * 6},
        "tiny": {"genuine": [0.1, 0.1], "impostor": [0.6] * 6},
    }
    out = compute_adaptive_thresholds(per, global_threshold=0.30)
    assert out == {}


# --- collect_calibration_data per-speaker stats ----------------------------------


def test_collect_returns_per_speaker_stats(tmp_path):
    store = _store(tmp_path)
    for i in range(3):
        store.enroll("anna", _pcm(i), 2.0, skip_vad=True)
    for i in range(10, 13):
        store.enroll("ben", _pcm(i), 2.0, skip_vad=True)
    distances, labels, per_speaker = store.collect_calibration_data()
    assert len(distances) == len(labels)
    assert set(per_speaker) == {"anna", "ben"}
    for stats in per_speaker.values():
        assert len(stats["genuine"]) == 3   # LOO per own sample
        assert len(stats["impostor"]) == 3  # the other speaker's samples
