"""Tests for the settings-backup helpers and the threshold recommendation."""

from __future__ import annotations

import numpy as np
import pytest

from murdock.api.routes_backup import apply_settings, dump_settings
from murdock.api.routes_settings import recommend_threshold
from murdock.core.db import open_db, set_setting


# --- settings dump / apply ----------------------------------------------------


def test_settings_dump_apply_round_trip(tmp_path):
    src = open_db(tmp_path / "src.db")
    set_setting(src, "verify_threshold", "0.42")
    set_setting(src, "mqtt_host", "core-mosquitto")
    set_setting(src, "media_restrictions", '{"sat1": {"media_player.tv": 0.2}}')

    dumped = dump_settings(src)
    assert dumped["verify_threshold"] == "0.42"
    assert dumped["mqtt_host"] == "core-mosquitto"

    dst = open_db(tmp_path / "dst.db")
    n = apply_settings(dst, dumped)
    assert n == len(dumped)
    assert dump_settings(dst) == dumped


def test_apply_settings_skips_non_strings(tmp_path):
    dst = open_db(tmp_path / "dst.db")
    n = apply_settings(dst, {"good": "1", "bad": 5, 7: "x"})
    assert n == 1
    assert dump_settings(dst) == {"good": "1"}


# --- threshold recommendation --------------------------------------------------


def _dists(mean, std, n, seed):
    rng = np.random.default_rng(seed)
    return list(np.clip(rng.normal(mean, std, n), 0.0, 2.0))


def test_reco_clean_separation():
    genuine = _dists(0.15, 0.03, 50, 1)
    impostor = _dists(0.60, 0.05, 30, 2)
    r = recommend_threshold(genuine, impostor, current=0.30)
    assert r["status"] == "ok"
    assert r["separation"] > 0
    # Recommendation lies in the gap between the two distributions.
    assert r["genuine_p95"] < r["recommended"] < r["impostor_p05"]


def test_reco_overlap_flagged():
    genuine = _dists(0.30, 0.08, 50, 3)
    impostor = _dists(0.35, 0.08, 50, 4)
    r = recommend_threshold(genuine, impostor, current=0.30)
    assert r["status"] == "overlap"
    assert r["separation"] < 0
    assert r["recommended"] is not None


def test_reco_insufficient_data():
    r = recommend_threshold([0.1] * 5, [0.6] * 2, current=0.30)
    assert r["status"] == "insufficient_data"
    assert "recommended" not in r or r.get("recommended") is None
    assert r["genuine_count"] == 5
    assert r["impostor_count"] == 2


def test_reco_clamped():
    # Pathologically low distances should still yield a sane floor.
    r = recommend_threshold([0.001] * 20, [0.02] * 10, current=0.30)
    assert r["recommended"] >= 0.05


# --- STT engine timings -------------------------------------------------------


def test_recognition_log_persists_stt_timings(tmp_path):
    """The A/B comparison needs speed, not just wording."""
    from murdock.core.db import open_db as _open
    from murdock.core.recognition_log import RecognitionLog

    log = RecognitionLog(_open(tmp_path / "t.db"))
    row_id = log.record(
        session_id="s1", satellite_id="sat1", duration_sec=2.0,
        outcome="match", matched_speaker="Jonas", transcript="hallo",
        transcript_ms=412.5,
    )
    assert row_id > 0
    ev = log.list_events(limit=1)[0]
    assert ev.transcript_ms == pytest.approx(412.5)
    # The shadow arrives later, via set_shadow.
    assert ev.shadow_ms is None

    assert log.set_shadow(row_id, "hallo!", "openai:whisper", 1183.0) is True
    ev = log.list_events(limit=1)[0]
    assert ev.shadow_transcript == "hallo!"
    assert ev.shadow_ms == pytest.approx(1183.0)


def test_set_shadow_without_timing_keeps_none(tmp_path):
    from murdock.core.db import open_db as _open
    from murdock.core.recognition_log import RecognitionLog

    log = RecognitionLog(_open(tmp_path / "t2.db"))
    row_id = log.record(
        session_id="s", satellite_id=None, duration_sec=1.0, outcome="match"
    )
    log.set_shadow(row_id, "text", "engine")
    assert log.list_events(limit=1)[0].shadow_ms is None


def test_recognition_api_exposes_timings(tmp_path):
    import asyncio
    from types import SimpleNamespace

    from murdock.api.routes_recognition import list_events
    from murdock.core.db import open_db as _open
    from murdock.core.recognition_log import RecognitionLog

    db = _open(tmp_path / "t3.db")
    log = RecognitionLog(db)
    row_id = log.record(
        session_id="s1", satellite_id="sat1", duration_sec=1.0,
        outcome="match", matched_speaker="Jonas", transcript="hi",
        transcript_ms=300.0, weight=1.0, margin=0.2,
    )
    log.set_shadow(row_id, "hi", "shadow-engine", 900.0)

    ctx = SimpleNamespace(
        recognition=log,
        unknown=SimpleNamespace(map_sessions_to_samples=lambda ids: {}),
        # The log renders a room name where one was announced.
        satellite_label=lambda sid: sid or "",
    )
    out = asyncio.run(list_events(limit=10, outcome=None, speaker=None, ctx=ctx))
    ev = out.events[0]
    assert ev.transcript_ms == pytest.approx(300.0)
    assert ev.shadow_ms == pytest.approx(900.0)
    assert ev.weight == pytest.approx(1.0)
    assert ev.margin == pytest.approx(0.2)
