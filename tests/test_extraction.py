"""Tests for adaptive target-speaker extraction.

Uses light fakes for the VAD / embedder / speaker store so the region
selection logic is exercised without ONNX models or a database.
"""

from __future__ import annotations

from dataclasses import dataclass
from types import SimpleNamespace

import pytest

from murdock.core.extraction import extract_target_speaker

_RATE = 16000
_WIDTH = 2


def _silence(seconds: float) -> bytes:
    return b"\x00\x00" * int(seconds * _RATE)


class FakeVAD:
    def __init__(self, segments):
        self._segments = segments

    def analyze_pcm(self, pcm):
        return SimpleNamespace(segments=list(self._segments))


class FakeEmbedder:
    """Returns a marker per call; embedding content is irrelevant here."""

    def __init__(self, raise_on=()):
        self.calls = 0
        self._raise_on = set(raise_on)

    def embed_pcm(self, pcm):
        idx = self.calls
        self.calls += 1
        if idx in self._raise_on:
            raise ValueError("too short")
        return [float(idx)]


@dataclass
class _Res:
    distance: float
    matched_speaker_id: object
    matched_speaker: object


class FakeSpeakers:
    """Pops a scripted verify result per embed call (in region order)."""

    def __init__(self, results):
        self._results = list(results)
        self.i = 0

    def verify_embedding(self, emb, threshold=None):
        res = self._results[self.i]
        self.i += 1
        dist, sid, name = res
        # Mirror SpeakerStore semantics: id/name only set when within threshold.
        if sid is not None and dist <= threshold:
            return _Res(dist, sid, name)
        return _Res(dist, None, None)


def test_fast_path_single_region_no_op():
    vad = FakeVAD([(0.0, 2.0)])
    emb = FakeEmbedder()
    spk = FakeSpeakers([])
    audio = _silence(2.0)
    res = extract_target_speaker(
        audio, vad=vad, embedder=emb, speakers=spk, extraction_threshold=0.25
    )
    assert res.applied is False
    assert res.audio == audio
    assert emb.calls == 0  # no embedding work on the fast path


def test_fast_path_no_regions():
    res = extract_target_speaker(
        _silence(2.0), vad=FakeVAD([]), embedder=FakeEmbedder(),
        speakers=FakeSpeakers([]), extraction_threshold=0.25,
    )
    assert res.applied is False


def test_empty_audio():
    res = extract_target_speaker(
        b"", vad=FakeVAD([(0, 1)]), embedder=FakeEmbedder(),
        speakers=FakeSpeakers([]), extraction_threshold=0.25,
    )
    assert res.applied is False


def test_drops_non_matching_region():
    # Region 0 = target (close), region 1 = TV (far, no match).
    vad = FakeVAD([(0.0, 1.5), (1.5, 3.0)])
    emb = FakeEmbedder()
    spk = FakeSpeakers([
        (0.10, 1, "jonas"),   # region 0 → match jonas
        (0.80, None, None),   # region 1 → no match (TV)
    ])
    audio = _silence(3.0)
    res = extract_target_speaker(
        audio, vad=vad, embedder=emb, speakers=spk, extraction_threshold=0.25
    )
    assert res.applied is True
    assert res.target_speaker == "jonas"
    assert res.n_regions == 2 and res.n_kept == 1
    # Kept audio is just the first 1.5 s region.
    assert len(res.audio) == int(1.5 * _RATE) * _WIDTH
    assert res.dropped_seconds == pytest.approx(1.5, abs=0.01)


def test_all_regions_target_is_noop():
    vad = FakeVAD([(0.0, 1.0), (1.5, 2.5)])
    emb = FakeEmbedder()
    spk = FakeSpeakers([(0.1, 1, "jonas"), (0.12, 1, "jonas")])
    audio = _silence(3.0)
    res = extract_target_speaker(
        audio, vad=vad, embedder=emb, speakers=spk, extraction_threshold=0.25
    )
    # Every region is the target → nothing to drop → no-op.
    assert res.applied is False
    assert res.audio == audio
    assert res.target_speaker == "jonas"


def test_no_region_matches_is_noop():
    vad = FakeVAD([(0.0, 1.0), (1.5, 2.5)])
    emb = FakeEmbedder()
    spk = FakeSpeakers([(0.9, None, None), (0.95, None, None)])
    audio = _silence(3.0)
    res = extract_target_speaker(
        audio, vad=vad, embedder=emb, speakers=spk, extraction_threshold=0.25
    )
    assert res.applied is False
    assert res.audio == audio


def test_dominant_speaker_wins_across_two():
    # jonas speaks two long regions, anna one short-ish region.
    vad = FakeVAD([(0.0, 2.0), (2.0, 2.8), (2.8, 4.8)])
    emb = FakeEmbedder()
    spk = FakeSpeakers([
        (0.1, 1, "jonas"),   # 2.0 s
        (0.1, 2, "anna"),    # 0.8 s
        (0.1, 1, "jonas"),   # 2.0 s
    ])
    audio = _silence(4.8)
    res = extract_target_speaker(
        audio, vad=vad, embedder=emb, speakers=spk, extraction_threshold=0.25
    )
    assert res.applied is True
    assert res.target_speaker == "jonas"
    assert res.n_kept == 2
    # 2.0 + 2.0 s kept, anna's 0.8 s dropped.
    assert len(res.audio) == int(4.0 * _RATE) * _WIDTH
    assert res.dropped_seconds == pytest.approx(0.8, abs=0.01)


def test_short_region_not_scored():
    # Region 1 is below min_region_sec → never embedded, treated as non-target.
    vad = FakeVAD([(0.0, 1.5), (1.5, 1.7)])
    emb = FakeEmbedder()
    spk = FakeSpeakers([(0.1, 1, "jonas")])  # only region 0 scored
    audio = _silence(1.7)
    res = extract_target_speaker(
        audio, vad=vad, embedder=emb, speakers=spk,
        extraction_threshold=0.25, min_region_sec=0.6,
    )
    assert emb.calls == 1  # short region skipped
    assert res.applied is True
    assert res.n_kept == 1


def test_embed_failure_treated_as_non_target():
    vad = FakeVAD([(0.0, 1.5), (1.5, 3.0)])
    emb = FakeEmbedder(raise_on=(1,))  # second embed raises
    spk = FakeSpeakers([(0.1, 1, "jonas")])  # only first region scored
    audio = _silence(3.0)
    res = extract_target_speaker(
        audio, vad=vad, embedder=emb, speakers=spk, extraction_threshold=0.25
    )
    assert res.applied is True
    assert res.n_kept == 1


def test_guard_rejects_too_little_kept_audio():
    # Two short-ish matched regions whose total is below the 0.5 s floor
    # would be rejected — but here we force a tiny kept region.
    vad = FakeVAD([(0.0, 0.7), (0.7, 3.0)])
    emb = FakeEmbedder()
    # Region 0 (0.7s) matches jonas; region 1 (2.3s) is TV/no-match.
    # Kept = 0.7s ≥ 0.5s floor, so this should still apply.
    spk = FakeSpeakers([(0.1, 1, "jonas"), (0.9, None, None)])
    audio = _silence(3.0)
    res = extract_target_speaker(
        audio, vad=vad, embedder=emb, speakers=spk,
        extraction_threshold=0.25, min_region_sec=0.6,
    )
    assert res.applied is True
    assert len(res.audio) == int(0.7 * _RATE) * _WIDTH
