"""Tests for transcript-template rendering (speaker context injection)."""

from __future__ import annotations

from murdock.config import Settings
from wyoming_murdock.handler import render_transcript_template


def test_basic_substitution():
    tpl = '{{ transcript }} [speaker: {{ speaker }}, {{ confidence }}%]'
    out = render_transcript_template(tpl, {
        "transcript": "turn on the light",
        "speaker": "jonas",
        "confidence": "94.5",
    })
    assert out == "turn on the light [speaker: jonas, 94.5%]"


def test_whitespace_variants_in_placeholder():
    tpl = "{{transcript}}|{{ transcript }}|{{  transcript  }}"
    out = render_transcript_template(tpl, {"transcript": "x"})
    assert out == "x|x|x"


def test_tts_alias_and_transcript_are_independent_keys():
    # The handler maps both to the same value; the renderer treats them
    # as plain keys, so each resolves from the dict it's given.
    tpl = "{{ tts }} / {{ transcript }}"
    out = render_transcript_template(tpl, {"tts": "a", "transcript": "a"})
    assert out == "a / a"


def test_unknown_placeholder_renders_empty():
    out = render_transcript_template("{{ transcript }}{{ bogus }}",
                                     {"transcript": "hi"})
    assert out == "hi"


def test_multiline_and_literals_preserved():
    tpl = "{{ transcript }}\n---\nrole: {{ role }}"
    out = render_transcript_template(tpl, {"transcript": "hello", "role": "Admin"})
    assert out == "hello\n---\nrole: Admin"


def test_no_placeholders_is_identity():
    assert render_transcript_template("just text", {"x": "y"}) == "just text"


def test_default_templates_render_with_full_context():
    s = Settings()
    known = render_transcript_template(s.transcript_template_known, {
        "transcript": "dim the lights",
        "speaker": "anna", "role": "Familie", "confidence": "88.0",
    })
    assert "dim the lights" in known
    assert "anna" in known and "Familie" in known and "88.0" in known
    assert "{{" not in known  # every placeholder resolved

    unknown = render_transcript_template(s.transcript_template_unknown, {
        "transcript": "who am i", "nearest": "ben", "confidence": "20.0",
    })
    assert "who am i" in unknown and "ben" in unknown
    assert "{{" not in unknown


def test_speaker_context_mode_defaults_to_none():
    assert Settings().speaker_context_mode == "none"


def _ctx(tmp_path):
    from types import SimpleNamespace

    from murdock.core.context import AppContext
    from murdock.core.db import open_db

    db = open_db(tmp_path / "m.db")
    settings = SimpleNamespace(speaker_context_mode="none")
    return AppContext(
        settings=settings, db=db, embedder=None, vad=None, speakers=None,
        unknown=None, ha=None, mqtt=None, recognition=None,
    )


def test_context_mode_get_set(tmp_path):
    ctx = _ctx(tmp_path)
    assert ctx.get_speaker_context_mode() == "none"
    ctx.set_speaker_context_mode("transcript")
    assert ctx.get_speaker_context_mode() == "transcript"


def test_context_mode_rejects_invalid(tmp_path):
    import pytest

    ctx = _ctx(tmp_path)
    with pytest.raises(ValueError):
        ctx.set_speaker_context_mode("bogus")


def test_context_mode_legacy_flag_maps_to_transcript(tmp_path):
    from murdock.core.db import set_setting

    ctx = _ctx(tmp_path)
    # A pre-release install that had the old boolean flag enabled.
    set_setting(ctx.db, "enable_transcript_template", "true")
    assert ctx.get_speaker_context_mode() == "transcript"
