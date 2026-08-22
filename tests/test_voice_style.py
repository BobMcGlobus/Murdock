"""Coarse "how was this said", judged against the room's own baseline."""

from __future__ import annotations

from murdock.core.voice_style import (
    STYLE_ANIMATED,
    STYLE_NORMAL,
    STYLE_QUIET,
    STYLE_WHISPERED,
    VoiceStyleTracker,
)


def _warm(t, sat="wohnzimmer", level=0.05, n=8):
    for _ in range(n):
        t.classify(sat, level)


def test_it_says_normal_until_it_has_seen_enough():
    """A confident label from two data points is a guess, not a measurement."""
    t = VoiceStyleTracker()
    assert t.baseline_for("wohnzimmer") is None
    for _ in range(3):
        assert t.classify("wohnzimmer", 0.9) == STYLE_NORMAL
    assert t.baseline_for("wohnzimmer") is None


def test_quiet_and_animated_are_relative_to_the_room():
    t = VoiceStyleTracker()
    _warm(t)
    assert t.classify("wohnzimmer", 0.05) == STYLE_NORMAL
    assert t.classify("wohnzimmer", 0.02) == STYLE_QUIET
    assert t.classify("wohnzimmer", 0.15) == STYLE_ANIMATED


def test_two_rooms_calibrate_separately():
    """A far mic and a near one must not share a threshold."""
    t = VoiceStyleTracker()
    _warm(t, "wohnzimmer", 0.05)
    _warm(t, "kueche", 0.4)
    # The same absolute level is loud in one room and quiet in the other.
    assert t.classify("wohnzimmer", 0.2) == STYLE_ANIMATED
    assert t.classify("kueche", 0.2) == STYLE_QUIET


def test_whispering_wins_over_the_level_comparison():
    """It is a measured acoustic property, not a loudness judgement."""
    t = VoiceStyleTracker()
    _warm(t)
    assert t.classify("wohnzimmer", 0.2, whispered=True) == STYLE_WHISPERED
    assert t.classify("wohnzimmer", 0.01, whispered=True) == STYLE_WHISPERED


def test_silence_never_produces_a_confident_label():
    t = VoiceStyleTracker()
    _warm(t)
    assert t.classify("wohnzimmer", 0.0) == STYLE_NORMAL
    assert t.classify("wohnzimmer", None) == STYLE_NORMAL


def test_the_baseline_follows_the_room_over_time():
    """Somebody who starts speaking up should stop reading as animated."""
    t = VoiceStyleTracker()
    _warm(t, "wohnzimmer", 0.05)
    assert t.classify("wohnzimmer", 0.15) == STYLE_ANIMATED
    _warm(t, "wohnzimmer", 0.15, n=30)
    assert t.classify("wohnzimmer", 0.15) == STYLE_NORMAL
