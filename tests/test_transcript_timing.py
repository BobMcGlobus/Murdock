"""The timing breakdown that lands next to a transcript in the log.

Which service answered and how long the wait after AudioStop was are
the two numbers that explain a slow answer; the fallback chain itself
is covered in test_stt_roles_runtime.py.
"""

from __future__ import annotations

from types import SimpleNamespace

from wyoming_murdock.handler import MurdockHandler


def test_timing_is_untouched_when_no_fallback_answered():
    fake = SimpleNamespace(
        _transcript_timing={"ttfb_ms": 400.0}, _rescued_by=None,
        _gate_ms=None, _answer_ms=None,
    )
    assert MurdockHandler._timing_with_rescue(fake) == {"ttfb_ms": 400.0}


def test_the_gap_after_audiostop_is_measured():
    """The stretch a user actually waits for was never in the log.

    An engine that reports 0 ms while Home Assistant waits seven seconds
    is not a contradiction — it just means the time went somewhere the
    request breakdown never covered.
    """
    fake = SimpleNamespace(
        _transcript_timing={"ttfb_ms": 0.0},
        _rescued_by="Voxtral",
        _gate_ms=310.4,
        _answer_ms=6951.2,
    )
    timing = MurdockHandler._timing_with_rescue(fake)
    assert timing["answer_ms"] == 6951.2
    assert timing["gate_ms"] == 310.4
    assert timing["rescued_by"] == "Voxtral"


def test_timings_are_omitted_when_never_taken():
    fake = SimpleNamespace(
        _transcript_timing=None, _rescued_by=None,
        _gate_ms=None, _answer_ms=None,
    )
    assert MurdockHandler._timing_with_rescue(fake) is None
