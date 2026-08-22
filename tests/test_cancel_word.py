"""Cancelling a turn the wake word should never have started."""

from __future__ import annotations

import pytest

from murdock.core.cancel_word import is_cancel, parse_cancel_words

WORDS = ["Abbruch", "stop", "vergiss es"]


def test_parsing_tolerates_the_ways_people_write_lists():
    assert parse_cancel_words("Abbruch, stop") == ["Abbruch", "stop"]
    assert parse_cancel_words("Abbruch; stop") == ["Abbruch", "stop"]
    assert parse_cancel_words("  Abbruch ,  stop  ") == ["Abbruch", "stop"]
    assert parse_cancel_words("") == []
    assert parse_cancel_words("   ") == []


@pytest.mark.parametrize("text", [
    "Abbruch",
    "abbruch",
    "Abbruch!",
    "Abbruch bitte",
    "Stop",
    "vergiss es",
    "Vergiss es, danke",
])
def test_a_cancellation_is_recognised(text):
    assert is_cancel(text, WORDS)


@pytest.mark.parametrize("text", [
    "kein Abbruch nötig",
    "Abbruch der Verhandlungen war absehbar gewesen",
    "Schalte das Licht im Wohnzimmer an",
    "Wann ist der nächste Stopp der Straßenbahn",
    "",
    "   ",
])
def test_a_normal_request_survives(text):
    """Killing a real request is far worse than missing a cancellation."""
    assert not is_cancel(text, WORDS)


def test_a_mangled_transcript_still_cancels():
    """Short words come back damaged often enough to matter."""
    assert is_cancel("Abruch", WORDS)
    assert is_cancel("ab Bruch", ["Abbruch"]) or is_cancel("abbruch", ["Abbruch"])


def test_nothing_configured_never_cancels():
    assert not is_cancel("Abbruch", [])
    assert not is_cancel("stop", [])


def _ctx(tmp_path):
    from murdock.config import Settings
    from murdock.core.context import AppContext
    from murdock.core.db import open_db

    return AppContext(
        settings=Settings(), db=open_db(tmp_path / "m.db"), embedder=None,
        vad=None, speakers=None, unknown=None, ha=None, mqtt=None,
        recognition=None,
    )


def test_the_feature_is_off_until_configured(tmp_path):
    """An upgrade must not start swallowing requests on its own."""
    ctx = _ctx(tmp_path)
    assert ctx.get_cancel_words() == []
    assert not ctx.is_cancel_phrase("Abbruch")

    ctx.set_cancel_words("Abbruch, stop")
    assert ctx.get_cancel_words() == ["Abbruch", "stop"]
    assert ctx.is_cancel_phrase("Abbruch bitte")
    assert not ctx.is_cancel_phrase("kein Abbruch nötig")

    ctx.set_cancel_words("")
    assert not ctx.is_cancel_phrase("Abbruch")


def test_the_liveness_bar_rises_with_media(tmp_path):
    """Media tightening used to move only the verify threshold."""
    ctx = _ctx(tmp_path)
    assert ctx.get_liveness_media_boost() == 0.15
    ctx.set_liveness_media_boost(0.3)
    assert ctx.get_liveness_media_boost() == 0.3
    # Clamped, so a typo cannot make every utterance impossible.
    ctx.set_liveness_media_boost(5.0)
    assert ctx.get_liveness_media_boost() == 0.9
    ctx.set_liveness_media_boost(-1.0)
    assert ctx.get_liveness_media_boost() == 0.0


def test_satellite_names_survive_and_fall_back(tmp_path):
    """The UI should show a room, not an entity id — but never nothing."""
    ctx = _ctx(tmp_path)
    assert ctx.satellite_label("assist_satellite.arbeitszimmer") == \
        "assist_satellite.arbeitszimmer"

    ctx.remember_satellite_name("assist_satellite.arbeitszimmer", "Arbeitszimmer")
    assert ctx.satellite_label("assist_satellite.arbeitszimmer") == "Arbeitszimmer"
    # Persisted, so a satellite stays named before it next speaks.
    assert ctx.get_satellite_names() == {
        "assist_satellite.arbeitszimmer": "Arbeitszimmer"
    }
    # Junk is ignored rather than stored as a blank label.
    ctx.remember_satellite_name("assist_satellite.arbeitszimmer", "   ")
    ctx.remember_satellite_name("", "Küche")
    assert ctx.satellite_label("assist_satellite.arbeitszimmer") == "Arbeitszimmer"
    assert ctx.satellite_label(None) == ""
