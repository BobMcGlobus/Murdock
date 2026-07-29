"""Tests for structured transcript hints and the sidecar mode (plan §13)."""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from murdock.core.context import AppContext
from murdock.core.db import open_db
from murdock.core.transcript_tools import (
    HINT_ADDITIONAL,
    HINT_ALTERNATIVE,
    HINT_READING,
    apply_correction_dictionary,
    apply_correction_dictionary_ex,
    merge_transcripts,
    merge_transcripts_ex,
    parse_correction_dictionary,
    render_hints,
    resolve_hints,
)


def _ctx(tmp_path, *, mqtt_connected=False, ha_configured=False):
    db = open_db(tmp_path / "m.db")
    return AppContext(
        settings=SimpleNamespace(
            enable_stt_vocabulary=True, stt_vocabulary="",
        ),
        db=db, embedder=None, vad=None, speakers=None, unknown=None,
        ha=SimpleNamespace(configured=ha_configured),
        mqtt=SimpleNamespace(connected=mqtt_connected),
        recognition=None,
    )


# ----------------------------------------------------------------------
# Dictionary: clean vs annotated vs hints
# ----------------------------------------------------------------------


def test_dictionary_replace_applies_to_both_renderings():
    entries = parse_correction_dictionary("fehlende Lichter -> Fehenlichter")
    r = apply_correction_dictionary_ex("mach die fehlende Lichter an", entries)
    assert r.clean == "mach die Fehenlichter an"
    assert r.annotated == "mach die Fehenlichter an"
    assert r.hints == []


def test_dictionary_annotate_splits_clean_and_hint():
    entries = parse_correction_dictionary("Bad-Lightstrip ~> Bed-Lightstrip")
    r = apply_correction_dictionary_ex("schalte Bad-Lightstrip ein", entries)
    # Clean keeps what was heard — no marker in the text.
    assert r.clean == "schalte Bad-Lightstrip ein"
    assert r.annotated == "schalte Bad-Lightstrip [oder: Bed-Lightstrip] ein"
    assert len(r.hints) == 1
    assert r.hints[0].original == "Bad-Lightstrip"
    assert r.hints[0].alternative == "Bed-Lightstrip"
    assert r.hints[0].kind == HINT_ALTERNATIVE


def test_legacy_dictionary_wrapper_unchanged():
    entries = parse_correction_dictionary("Bad-Lightstrip ~> Bed-Lightstrip")
    assert apply_correction_dictionary("schalte Bad-Lightstrip ein", entries) == (
        "schalte Bad-Lightstrip [oder: Bed-Lightstrip] ein"
    )


# ----------------------------------------------------------------------
# Merge: hints carry the exact spans
# ----------------------------------------------------------------------


def test_merge_replace_yields_hint():
    r = merge_transcripts_ex("mach das Licht an", "mach das Nicht an")
    assert r.clean == "mach das Licht an"
    assert "[oder: Nicht]" in r.annotated
    assert [(h.original, h.alternative) for h in r.hints] == [("Licht", "Nicht")]


def test_merge_insert_yields_additional_hint():
    r = merge_transcripts_ex("mach das Licht an", "mach das Licht nicht an")
    assert r.clean == "mach das Licht an"
    hint = r.hints[0]
    assert hint.kind == HINT_ADDITIONAL
    assert hint.original == ""
    assert hint.alternative == "nicht"


def test_merge_divergent_yields_reading_hint():
    r = merge_transcripts_ex("mach das Licht an", "wie ist das Wetter morgen")
    assert r.clean == "mach das Licht an"
    assert r.hints[0].kind == HINT_READING
    assert "alternative Lesart" in r.annotated


def test_merge_identical_has_no_hints():
    r = merge_transcripts_ex("mach das Licht an", "Mach das licht an!")
    assert r.hints == []
    assert r.clean == r.annotated == "mach das Licht an"


def test_legacy_merge_wrapper_unchanged():
    assert merge_transcripts("mach das Licht an", "mach das Nicht an") == (
        "mach das Licht [oder: Nicht] an"
    )


def test_render_hints_shape():
    r = merge_transcripts_ex("mach das Licht an", "mach das Nicht an")
    assert render_hints(r.hints) == [
        {"original": "Licht", "alternative": "Nicht", "kind": "alternative"}
    ]


# ----------------------------------------------------------------------
# Decide before marking
# ----------------------------------------------------------------------


def test_resolve_hints_rewrites_to_known_entity():
    r = merge_transcripts_ex("schalte Bad-Lightstrip ein",
                             "schalte Bed-Lightstrip ein")
    resolved = resolve_hints(r, ["Bed-Lightstrip", "Deckenlampe"])
    # Exactly one reading is a real entity → decided, not marked.
    assert resolved.clean == "schalte Bed-Lightstrip ein"
    assert resolved.hints == []


def test_resolve_hints_drops_when_heard_reading_is_the_entity():
    r = merge_transcripts_ex("schalte Bed-Lightstrip ein",
                             "schalte Bad-Lightstrip ein")
    resolved = resolve_hints(r, ["Bed-Lightstrip"])
    assert resolved.clean == "schalte Bed-Lightstrip ein"
    assert resolved.hints == []


def test_resolve_hints_keeps_genuine_ambiguity():
    r = merge_transcripts_ex("schalte Bett-Licht ein", "schalte Bad-Licht ein")
    # Both are entities → genuinely ambiguous, keep the hint.
    resolved = resolve_hints(r, ["Bett-Licht", "Bad-Licht"])
    assert len(resolved.hints) == 1
    # Neither is an entity → nothing to decide from, keep the hint.
    resolved = resolve_hints(r, ["Deckenlampe"])
    assert len(resolved.hints) == 1


def test_resolve_hints_without_vocabulary_is_noop():
    r = merge_transcripts_ex("mach das Licht an", "mach das Nicht an")
    assert resolve_hints(r, None) is r
    assert resolve_hints(r, []) is r


def test_resolve_hints_ignores_non_alternative_kinds():
    r = merge_transcripts_ex("mach das Licht an", "mach das Licht nicht an")
    resolved = resolve_hints(r, ["nicht"])
    assert len(resolved.hints) == 1


# ----------------------------------------------------------------------
# Mode setting + auto resolution
# ----------------------------------------------------------------------


def test_hint_mode_default_is_inline(tmp_path):
    ctx = _ctx(tmp_path)
    assert ctx.get_transcript_hint_mode() == "inline"
    assert ctx.effective_transcript_hint_mode() == "inline"


def test_hint_mode_round_trip(tmp_path):
    ctx = _ctx(tmp_path)
    for mode in ("sidecar", "clean", "auto", "inline"):
        ctx.set_transcript_hint_mode(mode)
        assert ctx.get_transcript_hint_mode() == mode
    with pytest.raises(ValueError):
        ctx.set_transcript_hint_mode("nonsense")


def test_auto_needs_a_sink(tmp_path):
    ctx = _ctx(tmp_path)
    ctx.set_transcript_hint_mode("auto")
    # No sink → inline markers, otherwise the hints would vanish.
    assert ctx.effective_transcript_hint_mode() == "inline"

    ctx_mqtt = _ctx(tmp_path / "a", mqtt_connected=True)
    ctx_mqtt.set_transcript_hint_mode("auto")
    assert ctx_mqtt.effective_transcript_hint_mode() == "sidecar"

    ctx_ha = _ctx(tmp_path / "b", ha_configured=True)
    ctx_ha.set_transcript_hint_mode("auto")
    assert ctx_ha.effective_transcript_hint_mode() == "sidecar"


# ----------------------------------------------------------------------
# Handler delivery
# ----------------------------------------------------------------------


def _deliver(ctx, result, label="test"):
    """Call the real _deliver_hints against a minimal fake handler."""
    from wyoming_murdock.handler import MurdockHandler

    fake = SimpleNamespace(
        context=ctx, _session_id="s1", _transcript_hints=[]
    )
    text = MurdockHandler._deliver_hints(fake, result, label)
    return text, fake._transcript_hints


def test_deliver_inline_keeps_markers(tmp_path):
    ctx = _ctx(tmp_path)
    r = merge_transcripts_ex("mach das Licht an", "mach das Nicht an")
    text, hints = _deliver(ctx, r)
    assert text == r.annotated
    assert hints == []


def test_deliver_sidecar_moves_hints_out(tmp_path):
    ctx = _ctx(tmp_path)
    ctx.set_transcript_hint_mode("sidecar")
    r = merge_transcripts_ex("mach das Licht an", "mach das Nicht an")
    text, hints = _deliver(ctx, r)
    assert text == "mach das Licht an"
    assert "[oder:" not in text
    assert hints == [
        {"original": "Licht", "alternative": "Nicht", "kind": "alternative"}
    ]


def test_deliver_clean_drops_hints(tmp_path):
    ctx = _ctx(tmp_path)
    ctx.set_transcript_hint_mode("clean")
    r = merge_transcripts_ex("mach das Licht an", "mach das Nicht an")
    text, hints = _deliver(ctx, r)
    assert text == "mach das Licht an"
    assert hints == []


def test_deliver_sidecar_resolves_against_vocabulary(tmp_path):
    ctx = _ctx(tmp_path)
    ctx.set_transcript_hint_mode("sidecar")
    ctx.get_vocabulary_store().save_snapshot(
        {"version": 1, "entities": [
            {"entity_id": "light.bed", "name": "Bed-Lightstrip", "aliases": []}
        ]}
    )
    r = merge_transcripts_ex("schalte Bad-Lightstrip ein",
                             "schalte Bed-Lightstrip ein")
    text, hints = _deliver(ctx, r)
    # Decided, not marked: clean text carries the real entity name.
    assert text == "schalte Bed-Lightstrip ein"
    assert hints == []
