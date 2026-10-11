"""Latency per engine, from the utterances it actually handled."""

from __future__ import annotations

from murdock.core.db import open_db
from murdock.core.recognition_log import RecognitionLog


def _event(log, timing):
    return log.record(
        session_id="s", satellite_id="wz", duration_sec=1.0, outcome="match",
        transcript="x", transcript_timing=timing,
    )


def test_mains_and_shadows_are_counted_per_engine(tmp_path):
    log = RecognitionLog(open_db(tmp_path / "m.db"))
    for ms in (800, 900, 1000, 1100, 8000):
        e = _event(log, {"main_engine": "MAI", "main_ms": ms, "main_error": False})
        log.add_shadow_result(e, service_id=2, engine="Kroko", transcript="x", ms=100.0)
    _event(log, {"main_engine": "MAI", "main_ms": 8400, "main_error": True})

    stats = log.engine_stats()
    mai, kroko = stats["MAI"], stats["Kroko"]
    assert mai["n"] == 6 and mai["errors"] == 1
    assert mai["p50_ms"] == 1000.0
    assert mai["p95_ms"] > 5000, "the slow tail is what the p95 is for"
    assert kroko["n"] == 5 and kroko["p50_ms"] == 100.0


def test_an_overtaken_main_is_counted_from_its_late_answer_only(tmp_path):
    """Not twice, and not as a fast answer it never gave."""
    log = RecognitionLog(open_db(tmp_path / "m.db"))
    e = _event(log, {"main_engine": "MAI", "main_ms": None, "overtaken": True})
    log.add_shadow_result(e, service_id=1, engine="MAI", transcript="Licht an", ms=6200.0)
    stats = log.engine_stats()
    assert stats["MAI"]["n"] == 1
    assert stats["MAI"]["p50_ms"] == 6200.0
