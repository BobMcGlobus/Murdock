"""Tests for the canonical recognition-event payload, the speaker
weight (plan §11) and the margin gate (plan §21)."""

from __future__ import annotations

import time
from types import SimpleNamespace

import pytest

from murdock.core.context import AppContext
from murdock.core.db import open_db
from murdock.core.event_payload import build_recognition_payload
from murdock.core.speaker_store import VerificationResult
from wyoming_murdock.handler import MurdockHandler


def _ctx(tmp_path):
    db = open_db(tmp_path / "m.db")
    return AppContext(
        settings=SimpleNamespace(), db=db, embedder=None, vad=None,
        speakers=None, unknown=None, ha=None, mqtt=None, recognition=None,
    )


# ----------------------------------------------------------------------
# build_recognition_payload
# ----------------------------------------------------------------------


def test_payload_known_speaker():
    p = build_recognition_payload(
        speaker="Jonas", is_known=True, confidence=0.94321,
        satellite_id="wohnzimmer", distance=0.31, threshold=0.38,
        nearest_speaker="Alex", nearest_distance=0.61, margin=0.30,
        role="admin",
    )
    assert p["speaker"] == "Jonas"
    assert p["is_known"] is True
    assert "reason" not in p
    assert p["confidence"] == pytest.approx(0.9432)
    assert p["nearest_speaker"] == "Alex"
    assert p["nearest_distance"] == pytest.approx(0.61)
    assert p["margin"] == pytest.approx(0.30)
    assert p["timestamp"] == pytest.approx(time.time(), abs=5.0)


def test_payload_unknown_nulls_speaker_and_moves_sentinel_to_reason():
    p = build_recognition_payload(
        speaker="unknown", is_known=False, confidence=0.2,
        satellite_id="kueche",
    )
    assert p["speaker"] is None
    assert p["reason"] == "unknown"
    p = build_recognition_payload(
        speaker="tv-noise", is_known=False, confidence=0.0,
        satellite_id="kueche",
    )
    assert p["speaker"] is None
    assert p["reason"] == "tv-noise"


def test_payload_explicit_reason_wins():
    p = build_recognition_payload(
        speaker="unknown", is_known=False, confidence=0.0,
        satellite_id=None, reason="short",
    )
    assert p["reason"] == "short"


def test_payload_uncertain_flag():
    p = build_recognition_payload(
        speaker="uncertain", is_known=False, confidence=0.8,
        satellite_id="bad", uncertain=True, reason="uncertain",
        nearest_speaker="Jonas", nearest_distance=0.35, margin=0.02,
    )
    assert p["speaker"] is None
    assert p["uncertain"] is True
    assert p["reason"] == "uncertain"
    assert p["nearest_speaker"] == "Jonas"


# ----------------------------------------------------------------------
# Speaker weight (plan §11)
# ----------------------------------------------------------------------


def test_weight_verified_is_one(tmp_path):
    ctx = _ctx(tmp_path)
    assert ctx.compute_speaker_weight(0.30, 0.38) == pytest.approx(1.0)
    assert ctx.compute_speaker_weight(0.38, 0.38) == pytest.approx(1.0)


def test_weight_decays_linearly_to_ceiling(tmp_path):
    ctx = _ctx(tmp_path)
    # Worked example from the plan: d=0.4981, th=0.380, ceiling=0.53 → ≈0.21
    assert ctx.compute_speaker_weight(0.4981, 0.380) == pytest.approx(0.213, abs=0.01)
    assert ctx.compute_speaker_weight(0.53, 0.380) == pytest.approx(0.0)
    assert ctx.compute_speaker_weight(0.90, 0.380) == pytest.approx(0.0)


def test_weight_none_without_inputs(tmp_path):
    ctx = _ctx(tmp_path)
    assert ctx.compute_speaker_weight(None, 0.38) is None
    assert ctx.compute_speaker_weight(0.40, None) is None


# ----------------------------------------------------------------------
# Margin gate settings + helpers
# ----------------------------------------------------------------------


def _result(distances, is_match=True, threshold=0.38):
    ordered = dict(sorted(distances.items(), key=lambda kv: kv[1]))
    best_name, best_d = next(iter(ordered.items()))
    return VerificationResult(
        is_match=is_match,
        matched_speaker=best_name if is_match else None,
        matched_speaker_id=1 if is_match else None,
        distance=best_d,
        threshold=threshold,
        all_distances=ordered,
    )


def test_margin_gate_settings_chain(tmp_path):
    ctx = _ctx(tmp_path)
    assert ctx.get_margin_gate() == 0.0
    ctx.set_margin_gate(0.05)
    assert ctx.get_margin_gate() == pytest.approx(0.05)
    assert ctx.get_margin_gate("sat1") == pytest.approx(0.05)
    ctx.set_satellite_margin_gate("sat1", 0.12)
    assert ctx.get_margin_gate("sat1") == pytest.approx(0.12)
    assert ctx.get_satellite_margin_gates() == {"sat1": pytest.approx(0.12)}
    ctx.set_satellite_margin_gate("sat1", None)
    assert ctx.get_margin_gate("sat1") == pytest.approx(0.05)


def test_margin_of_needs_two_speakers():
    assert MurdockHandler._margin_of(
        _result({"Jonas": 0.30})
    ) is None
    assert MurdockHandler._margin_of(
        _result({"Jonas": 0.30, "Alex": 0.50})
    ) == pytest.approx(0.20)


def test_nearest_of_match_returns_runner_up():
    r = _result({"Jonas": 0.30, "Alex": 0.50})
    name, dist = MurdockHandler._nearest_of(r)
    assert (name, dist) == ("Alex", pytest.approx(0.50))


def test_nearest_of_no_match_returns_best():
    r = _result({"Jonas": 0.45, "Alex": 0.60}, is_match=False)
    name, dist = MurdockHandler._nearest_of(r)
    assert (name, dist) == ("Jonas", pytest.approx(0.45))


def test_margin_gate_downgrades_close_call(tmp_path):
    ctx = _ctx(tmp_path)
    ctx.set_margin_gate(0.10)
    handler = SimpleNamespace(
        context=ctx, _satellite_id="sat1",
        _margin_of=MurdockHandler._margin_of,
    )
    gate = MurdockHandler._margin_gate_uncertain
    # Comfortable margin → stays a match.
    margin, uncertain = gate(handler, _result({"Jonas": 0.30, "Alex": 0.55}))
    assert uncertain is False
    # Runner-up too close → uncertain.
    margin, uncertain = gate(handler, _result({"Jonas": 0.30, "Alex": 0.36}))
    assert uncertain is True
    assert margin == pytest.approx(0.06)
    # Non-matches are never "uncertain".
    margin, uncertain = gate(
        handler, _result({"Jonas": 0.50, "Alex": 0.52}, is_match=False)
    )
    assert uncertain is False
    # Single enrolled speaker → gate cannot apply.
    margin, uncertain = gate(handler, _result({"Jonas": 0.30}))
    assert uncertain is False


def test_recognition_log_persists_weight_and_margin(tmp_path):
    from murdock.core.recognition_log import RecognitionLog

    db = open_db(tmp_path / "log.db")
    log = RecognitionLog(db)
    row_id = log.record(
        session_id="s1", satellite_id="sat1", duration_sec=1.5,
        outcome="match", matched_speaker="Jonas", distance=0.31,
        threshold=0.38, weight=1.0, margin=0.2,
    )
    assert row_id > 0
    ev = log.list_events(limit=1)[0]
    assert ev.weight == pytest.approx(1.0)
    assert ev.margin == pytest.approx(0.2)
