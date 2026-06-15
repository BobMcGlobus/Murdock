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


def test_template_defaults_off():
    assert Settings().enable_transcript_template is False
