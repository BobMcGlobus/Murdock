"""Tests for the enrollment coach.

The point of the coach is that a healthy profile produces *silence* — a
panel that always has something to say gets ignored. So every rule is
tested from both sides: it fires when it should, and stays quiet when the
data is fine.
"""

from __future__ import annotations

import pytest

from murdock.core.enrollment_coach import (
    DRIFT_MIN_EVENTS,
    MIN_EVENTS_FOR_SATELLITE_ADVICE,
    MIN_HEALTHY_SAMPLES,
    SATELLITE_PROFILE_MIN_SAMPLES,
    analyse,
    analyse_speaker,
)

THRESHOLD = 0.38


def _samples(n, quality=0.8, satellite=None):
    return [
        {"quality_score": quality, "satellite_id": satellite} for _ in range(n)
    ]


def _events(n, distance=0.25, satellite="wohnzimmer"):
    return [
        {"distance": distance, "satellite_id": satellite, "created_at": 1000 - i}
        for i in range(n)
    ]


def _codes(findings):
    return {f.code for f in findings}


def test_healthy_profile_says_nothing():
    findings = analyse_speaker(
        name="Jonas",
        samples=_samples(6, quality=0.85, satellite="wohnzimmer"),
        events=_events(20, distance=0.20),
        threshold=THRESHOLD,
    )
    assert findings == []


def test_too_few_samples():
    findings = analyse_speaker(
        name="Jonas", samples=_samples(1), events=[], threshold=THRESHOLD
    )
    assert "few_samples" in _codes(findings)
    detail = next(f for f in findings if f.code == "few_samples").detail
    assert detail["samples"] == 1
    assert detail["recommended"] == MIN_HEALTHY_SAMPLES


def test_enough_samples_is_quiet():
    findings = analyse_speaker(
        name="Jonas",
        samples=_samples(MIN_HEALTHY_SAMPLES, satellite="wohnzimmer"),
        events=[],
        threshold=THRESHOLD,
    )
    assert "few_samples" not in _codes(findings)


def test_low_quality_is_flagged():
    findings = analyse_speaker(
        name="Jonas",
        samples=_samples(5, quality=0.3, satellite="wohnzimmer"),
        events=[],
        threshold=THRESHOLD,
    )
    assert "low_quality" in _codes(findings)


def test_missing_quality_scores_are_not_a_finding():
    """Older samples have no score; absence is not evidence of badness."""
    samples = [{"quality_score": None, "satellite_id": "wohnzimmer"}] * 5
    findings = analyse_speaker(
        name="Jonas", samples=samples, events=[], threshold=THRESHOLD
    )
    assert "low_quality" not in _codes(findings)


def test_satellite_without_samples_is_the_headline_advice():
    findings = analyse_speaker(
        name="Anna",
        samples=_samples(5, satellite="wohnzimmer"),
        events=_events(12, satellite="kueche"),
        threshold=THRESHOLD,
    )
    sat = next(f for f in findings if f.code == "satellite_coverage")
    assert sat.detail["satellite_id"] == "kueche"
    assert sat.detail["samples"] == 0
    assert sat.detail["missing"] == SATELLITE_PROFILE_MIN_SAMPLES
    assert "kueche" in sat.message


def test_satellite_with_enough_samples_is_quiet():
    findings = analyse_speaker(
        name="Anna",
        samples=_samples(SATELLITE_PROFILE_MIN_SAMPLES, satellite="kueche"),
        events=_events(20, satellite="kueche"),
        threshold=THRESHOLD,
    )
    assert "satellite_coverage" not in _codes(findings)


def test_rarely_used_satellite_is_not_worth_advice():
    findings = analyse_speaker(
        name="Anna",
        samples=_samples(5, satellite="wohnzimmer"),
        events=_events(MIN_EVENTS_FOR_SATELLITE_ADVICE - 1, satellite="bad"),
        threshold=THRESHOLD,
    )
    assert "satellite_coverage" not in _codes(findings)


def test_tight_headroom():
    findings = analyse_speaker(
        name="Jonas",
        samples=_samples(5, satellite="wohnzimmer"),
        events=_events(10, distance=0.355),  # 0.025 below the threshold
        threshold=THRESHOLD,
    )
    tight = next(f for f in findings if f.code == "tight_headroom")
    assert tight.detail["headroom"] == pytest.approx(0.025, abs=0.001)


def test_comfortable_distance_is_quiet():
    findings = analyse_speaker(
        name="Jonas",
        samples=_samples(5, satellite="wohnzimmer"),
        events=_events(10, distance=0.15),
        threshold=THRESHOLD,
    )
    assert "tight_headroom" not in _codes(findings)


def test_drift_is_detected_from_newest_first_events():
    recent = _events(DRIFT_MIN_EVENTS // 2, distance=0.30)
    older = _events(DRIFT_MIN_EVENTS // 2, distance=0.18)
    findings = analyse_speaker(
        name="Jonas",
        samples=_samples(5, satellite="wohnzimmer"),
        events=recent + older,
        threshold=THRESHOLD,
    )
    drift = next(f for f in findings if f.code == "drift")
    assert drift.detail["recent_mean"] > drift.detail["older_mean"]


def test_improving_profile_is_not_drift():
    recent = _events(DRIFT_MIN_EVENTS // 2, distance=0.18)
    older = _events(DRIFT_MIN_EVENTS // 2, distance=0.30)
    findings = analyse_speaker(
        name="Jonas",
        samples=_samples(5, satellite="wohnzimmer"),
        events=recent + older,
        threshold=THRESHOLD,
    )
    assert "drift" not in _codes(findings)


def test_too_few_events_for_a_drift_verdict():
    findings = analyse_speaker(
        name="Jonas",
        samples=_samples(5, satellite="wohnzimmer"),
        events=_events(4, distance=0.30),
        threshold=THRESHOLD,
    )
    assert "drift" not in _codes(findings)


def test_analyse_sorts_warnings_first():
    findings = analyse(
        speakers=[{"id": 1, "name": "Jonas"}, {"id": 2, "name": "Anna"}],
        samples_by_speaker={
            1: _samples(5, quality=0.3, satellite="wohnzimmer"),  # info
            2: _samples(1),                                        # warn
        },
        events_by_speaker={},
        threshold=THRESHOLD,
    )
    assert findings[0].severity == "warn"
    assert findings[0].speaker == "Anna"


def test_findings_serialise():
    findings = analyse_speaker(
        name="Jonas", samples=_samples(1), events=[], threshold=THRESHOLD
    )
    d = findings[0].as_dict()
    assert set(d) == {"speaker", "code", "severity", "message", "detail"}


def test_no_speakers_no_findings():
    assert analyse(
        speakers=[], samples_by_speaker={}, events_by_speaker={},
        threshold=THRESHOLD,
    ) == []
