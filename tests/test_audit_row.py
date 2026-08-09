"""The handler's audit-row call must actually reach the database.

A regression guard for a real outage: an unbounded search-and-replace put
a publish-only keyword (`speakers=`) into the call that writes the
recognition log. Every insert raised TypeError, the handler swallowed it
at debug level, and the log silently stayed empty for a whole release.

Nothing in the old suite exercised the handler's call site against a real
RecognitionLog, so nothing caught it. These tests do exactly that.
"""

from __future__ import annotations

import inspect
from types import SimpleNamespace

import pytest

from murdock.core.db import open_db
from murdock.core.recognition_log import RecognitionLog
from wyoming_murdock.handler import MurdockHandler


def _handler(tmp_path):
    """A stand-in carrying only what _record_event touches."""
    log = RecognitionLog(open_db(tmp_path / "m.db"))
    fake = SimpleNamespace(
        context=SimpleNamespace(recognition=log),
        _session_id="s1",
        _satellite_id="wohnzimmer",
        _transcript_ms=412.0,
        _whisper=True,
        _whisper_score=0.83,
        _speakers=[{"speaker": "Jonas", "seconds": 2.4},
                   {"speaker": "Anna", "seconds": 0.9}],
    )
    return fake, log


def test_record_event_writes_a_row(tmp_path):
    fake, log = _handler(tmp_path)
    row_id = MurdockHandler._record_event(
        fake,
        outcome="match",
        duration_sec=3.3,
        matched_speaker="Jonas",
        distance=0.21,
        threshold=0.38,
        verify_ms=93.0,
        transcript="hallo",
        weight=1.0,
        margin=0.3,
    )
    assert row_id > 0, "the audit row was not written"
    events = log.list_events(limit=5)
    assert len(events) == 1
    ev = events[0]
    assert ev.matched_speaker == "Jonas"
    assert ev.whisper is True
    assert ev.whisper_score == pytest.approx(0.83)
    assert [s["speaker"] for s in ev.speakers] == ["Jonas", "Anna"]


def test_every_handler_kwarg_is_accepted_by_the_log(tmp_path):
    """Guards the exact failure mode: an argument the store doesn't take."""
    accepted = set(inspect.signature(RecognitionLog.record).parameters)
    # What the handler forwards, per _record_event.
    forwarded = {
        "session_id", "satellite_id", "duration_sec", "outcome",
        "matched_speaker", "distance", "threshold", "verify_ms",
        "transcript", "emotion", "emotion_confidence", "weight", "margin",
        "transcript_ms", "whisper", "whisper_score", "speakers",
    }
    missing = forwarded - accepted
    assert not missing, f"RecognitionLog.record() cannot accept: {missing}"


def test_record_event_reports_failure_instead_of_hiding_it(tmp_path, caplog):
    """A broken write must be loud — silence is what cost us a release."""
    fake, _ = _handler(tmp_path)

    class Exploding:
        def record(self, **kwargs):
            raise TypeError("unexpected keyword argument 'nonsense'")

    fake.context = SimpleNamespace(recognition=Exploding())
    with caplog.at_level("WARNING"):
        row_id = MurdockHandler._record_event(
            fake, outcome="match", duration_sec=1.0
        )
    assert row_id == 0
    assert any(
        "Failed to record audit row" in r.message for r in caplog.records
    ), "the failure was not logged at warning level"


def test_whisper_score_is_kept_even_below_the_bar(tmp_path):
    """Tuning needs "0.55, just under", not just a boolean."""
    fake, log = _handler(tmp_path)
    fake._whisper = False
    fake._whisper_score = 0.55
    MurdockHandler._record_event(fake, outcome="match", duration_sec=1.0)
    ev = log.list_events(limit=1)[0]
    assert ev.whisper is False
    assert ev.whisper_score == pytest.approx(0.55)


def test_speakers_roster_survives_a_round_trip(tmp_path):
    fake, log = _handler(tmp_path)
    MurdockHandler._record_event(fake, outcome="match", duration_sec=1.0)
    ev = log.list_events(limit=1)[0]
    assert ev.speakers[0]["seconds"] == pytest.approx(2.4)


def test_no_roster_stores_nothing(tmp_path):
    fake, log = _handler(tmp_path)
    fake._speakers = []
    MurdockHandler._record_event(fake, outcome="match", duration_sec=1.0)
    assert log.list_events(limit=1)[0].speakers == []
