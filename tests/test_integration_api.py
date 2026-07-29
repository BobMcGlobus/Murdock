"""Tests for the HA-integration sync endpoints (/api/satellites, /api/state)."""

from __future__ import annotations

import asyncio
from types import SimpleNamespace

import pytest

from murdock import __version__
from murdock.api.routes_integration import get_state, get_version, list_satellites
from murdock.core.db import open_db
from murdock.core.recognition_log import RecognitionLog


def _ctx(tmp_path):
    db = open_db(tmp_path / "m.db")
    return SimpleNamespace(db=db), RecognitionLog(db)


def test_version_endpoint():
    out = asyncio.run(get_version())
    assert out.version == __version__


def test_satellites_grouped_and_sorted(tmp_path):
    ctx, log = _ctx(tmp_path)
    for _ in range(3):
        log.record(session_id="s", satellite_id="wohnzimmer",
                   duration_sec=1.0, outcome="match", matched_speaker="Jonas")
    log.record(session_id="s", satellite_id="kueche",
               duration_sec=1.0, outcome="blocked-no-match")
    # Rows without a satellite are ignored.
    log.record(session_id="s", satellite_id=None,
               duration_sec=1.0, outcome="match")

    out = asyncio.run(list_satellites(ctx))
    ids = [s.satellite_id for s in out.satellites]
    assert ids == ["wohnzimmer", "kueche"]
    assert out.satellites[0].seen_events == 3
    assert out.satellites[0].last_seen is not None


def test_state_returns_latest_row_per_satellite(tmp_path):
    ctx, log = _ctx(tmp_path)
    log.record(session_id="s1", satellite_id="wohnzimmer", duration_sec=1.0,
               outcome="blocked-no-match", matched_speaker="Jonas",
               distance=0.55, threshold=0.38)
    log.record(session_id="s2", satellite_id="wohnzimmer", duration_sec=1.0,
               outcome="match", matched_speaker="Jonas",
               distance=0.31, threshold=0.38, weight=1.0, margin=0.2)
    log.record(session_id="s3", satellite_id="kueche", duration_sec=1.0,
               outcome="unknown-forwarded", matched_speaker="Alex",
               distance=0.51, threshold=0.38)

    out = asyncio.run(get_state(ctx))
    by_id = {s.satellite_id: s for s in out.satellites}
    assert set(by_id) == {"wohnzimmer", "kueche"}

    wz = by_id["wohnzimmer"]
    assert wz.speaker == "Jonas"
    assert wz.outcome == "match"
    assert wz.weight == pytest.approx(1.0)
    assert wz.margin == pytest.approx(0.2)

    # Non-match outcomes carry no speaker even when a nearest candidate
    # was logged in matched_speaker.
    kü = by_id["kueche"]
    assert kü.speaker is None
    assert kü.outcome == "unknown-forwarded"


def test_state_empty_log(tmp_path):
    ctx, _ = _ctx(tmp_path)
    out = asyncio.run(get_state(ctx))
    assert out.satellites == []
