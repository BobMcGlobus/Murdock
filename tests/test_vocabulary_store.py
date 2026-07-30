"""Tests for the HA registry vocabulary snapshots (plan §9/§10)."""

from __future__ import annotations

import asyncio

import pytest
from types import SimpleNamespace

from murdock.core.context import AppContext
from murdock.core.db import open_db
from murdock.core.vocabulary_store import (
    VocabularyStore,
    _MAX_SNAPSHOTS,
    normalize_term,
)


def _store(tmp_path):
    return VocabularyStore(open_db(tmp_path / "v.db"))


_SNAPSHOT = {
    "version": 7,
    "generated_at": "2026-07-29T12:00:00+00:00",
    "entities": [
        {"entity_id": "light.bett", "name": "Bett-Lightstrip",
         "aliases": ["Bettlicht"], "area": "Schlafzimmer", "domain": "light"},
        {"entity_id": "light.decke", "name": "Deckenlampe",
         "aliases": [], "area": "Wohnzimmer", "domain": "light"},
        # Duplicate name with different case → deduped.
        {"entity_id": "light.decke2", "name": "deckenlampe",
         "aliases": [], "area": "Küche", "domain": "light"},
    ],
    "areas": [{"name": "Wohnzimmer"}, {"name": "Schlafzimmer"}],
    "floors": ["Obergeschoss"],
}


def test_normalize_term():
    assert normalize_term("Bett-Lightstrip") == "bett-lightstrip"
    assert normalize_term("  Küche  ") == "kuche"
    assert normalize_term("Straße") == "strasse"


def test_save_and_terms(tmp_path):
    store = _store(tmp_path)
    meta = store.save_snapshot(_SNAPSHOT)
    assert meta["snapshot_id"] > 0
    assert meta["version"] == 7

    terms = store.terms()
    # Entity names first, then aliases, then areas/floors; dupes gone.
    assert terms[0] == "Bett-Lightstrip"
    assert "Bettlicht" in terms
    assert "Obergeschoss" in terms
    assert terms.count("Deckenlampe") == 1
    assert "deckenlampe" not in terms  # case-dupe removed

    # Cap applies in priority order.
    assert store.terms(limit=2) == ["Bett-Lightstrip", "Deckenlampe"]


def test_latest_returns_newest_and_history_trims(tmp_path):
    store = _store(tmp_path)
    for v in range(1, _MAX_SNAPSHOTS + 3):
        store.save_snapshot({"version": v, "entities": []})
    snap = store.latest()
    assert snap["version"] == str(_MAX_SNAPSHOTS + 2)
    count = store.conn.execute(
        "SELECT COUNT(*) AS n FROM vocabulary_snapshots"
    ).fetchone()["n"]
    assert count == _MAX_SNAPSHOTS


def test_empty_store(tmp_path):
    store = _store(tmp_path)
    assert store.latest() is None
    assert store.terms() == []
    assert store.normalized_terms() == {}


# ----------------------------------------------------------------------
# Effective vocabulary merge (context)
# ----------------------------------------------------------------------


def _ctx(tmp_path):
    db = open_db(tmp_path / "m.db")
    return AppContext(
        settings=SimpleNamespace(
            enable_stt_vocabulary=True, stt_vocabulary="",
            stt_backend="upstream", enable_stt_dictionary=False,
            stt_dictionary="",
        ),
        db=db, embedder=None, vad=None, speakers=None, unknown=None,
        ha=None, mqtt=None, recognition=None,
    )


def test_effective_vocabulary_merges_ha_terms(tmp_path):
    ctx = _ctx(tmp_path)
    ctx.set_stt_vocabulary("Fehenlichter")
    ctx.get_vocabulary_store().save_snapshot(_SNAPSHOT)
    vocab = ctx.get_effective_vocabulary()
    assert vocab.startswith("Fehenlichter")
    assert "Bett-Lightstrip" in vocab
    # Terms already present in the manual list aren't repeated.
    ctx.set_stt_vocabulary("Bett-Lightstrip")
    vocab = ctx.get_effective_vocabulary()
    assert vocab.count("Bett-Lightstrip") == 1


def test_effective_vocabulary_disabled_stays_empty(tmp_path):
    ctx = _ctx(tmp_path)
    ctx.get_vocabulary_store().save_snapshot(_SNAPSHOT)
    ctx.set_enable_stt_vocabulary(False)
    assert ctx.get_effective_vocabulary() == ""


def test_vocabulary_endpoints_round_trip(tmp_path):
    from murdock.api.routes_integration import (
        VocabularyIn,
        get_vocabulary,
        push_vocabulary,
    )

    ctx = _ctx(tmp_path)
    meta = asyncio.run(get_vocabulary(ctx))
    assert meta.available is False

    out = asyncio.run(
        push_vocabulary(VocabularyIn(**_SNAPSHOT), ctx)
    )
    assert out.term_count > 0
    assert out.version == "7"

    meta = asyncio.run(get_vocabulary(ctx))
    assert meta.available is True
    assert meta.entity_count == 3
    assert "Bett-Lightstrip" in meta.terms_preview


def test_vocabulary_meta_exposes_full_list_and_effective_prompt(tmp_path):
    """The UI panel needs the whole term list plus what actually goes out."""
    import asyncio

    from murdock.api.routes_integration import VocabularyIn, get_vocabulary, push_vocabulary
    from murdock.core.context import _HA_VOCAB_TERM_CAP

    ctx = _ctx(tmp_path)
    ctx.set_stt_vocabulary("Fehenlichter")

    entities = [
        {"entity_id": f"light.l{i}", "name": f"Lampe {i:02d}", "aliases": []}
        for i in range(_HA_VOCAB_TERM_CAP + 5)
    ]
    asyncio.run(push_vocabulary(
        VocabularyIn(version=3, entities=entities, areas=[], floors=[]), ctx
    ))

    meta = asyncio.run(get_vocabulary(ctx))
    assert meta.term_count == _HA_VOCAB_TERM_CAP + 5
    # Full list for the panel, preview stays capped for cheap clients.
    assert len(meta.terms) == _HA_VOCAB_TERM_CAP + 5
    assert len(meta.terms_preview) == 25
    assert meta.term_cap == _HA_VOCAB_TERM_CAP
    # The effective prompt is manual terms + capped HA terms.
    sent = meta.effective_prompt.split(", ")
    assert sent[0] == "Fehenlichter"
    assert len(sent) == _HA_VOCAB_TERM_CAP + 1
    assert meta.terms[_HA_VOCAB_TERM_CAP] not in meta.effective_prompt
    assert meta.effective_enabled is True


def test_vocabulary_meta_reports_disabled_tier(tmp_path):
    import asyncio

    from murdock.api.routes_integration import get_vocabulary

    ctx = _ctx(tmp_path)
    ctx.get_vocabulary_store().save_snapshot(_SNAPSHOT)
    ctx.set_enable_stt_vocabulary(False)
    meta = asyncio.run(get_vocabulary(ctx))
    assert meta.available is True
    assert meta.effective_enabled is False
    assert meta.effective_prompt == ""


# ----------------------------------------------------------------------
# Curated bias-prompt selection
# ----------------------------------------------------------------------


def test_selection_defaults_to_automatic(tmp_path):
    ctx = _ctx(tmp_path)
    ctx.get_vocabulary_store().save_snapshot(_SNAPSHOT)
    assert ctx.get_vocab_selection() is None
    # Automatic = the first N by priority.
    assert ctx.get_selected_ha_terms() == ctx.get_vocabulary_store().terms()


def test_explicit_selection_wins(tmp_path):
    ctx = _ctx(tmp_path)
    ctx.get_vocabulary_store().save_snapshot(_SNAPSHOT)
    ctx.set_vocab_selection(["Bettlicht", "Obergeschoss"])
    assert ctx.get_vocab_selection() == ["Bettlicht", "Obergeschoss"]
    # Snapshot order is preserved, not the order they were picked in.
    assert ctx.get_selected_ha_terms() == ["Bettlicht", "Obergeschoss"]
    vocab = ctx.get_effective_vocabulary()
    assert "Bettlicht" in vocab
    assert "Bett-Lightstrip" not in vocab


def test_empty_selection_sends_no_mirrored_terms(tmp_path):
    ctx = _ctx(tmp_path)
    ctx.get_vocabulary_store().save_snapshot(_SNAPSHOT)
    ctx.set_stt_vocabulary("Fehenlichter")
    ctx.set_vocab_selection([])
    assert ctx.get_selected_ha_terms() == []
    assert ctx.get_effective_vocabulary() == "Fehenlichter"


def test_selection_can_return_to_automatic(tmp_path):
    ctx = _ctx(tmp_path)
    ctx.get_vocabulary_store().save_snapshot(_SNAPSHOT)
    ctx.set_vocab_selection(["Bettlicht"])
    ctx.set_vocab_selection(None)
    assert ctx.get_vocab_selection() is None
    assert len(ctx.get_selected_ha_terms()) > 1


def test_selection_ignores_terms_that_vanished(tmp_path):
    ctx = _ctx(tmp_path)
    ctx.get_vocabulary_store().save_snapshot(_SNAPSHOT)
    ctx.set_vocab_selection(["Bettlicht", "Entity die es nicht mehr gibt"])
    assert ctx.get_selected_ha_terms() == ["Bettlicht"]


def test_backend_prompt_support_is_reported_honestly(tmp_path):
    ctx = _ctx(tmp_path)
    # The local upstream has no prompt field at all.
    ctx.set_stt_backend("upstream")
    assert ctx.active_backend_supports_prompt() is False
    # Voxtral doesn't document one.
    ctx.set_stt_backend("voxtral")
    assert ctx.active_backend_supports_prompt() is False
    # OpenAI-compatible does...
    ctx.set_stt_backend("openai")
    ctx.set_openai_base_url("https://api.openai.com")
    assert ctx.active_backend_supports_prompt() is True
    # ...except OpenRouter, whose request shape skips it.
    ctx.set_openai_base_url("https://openrouter.ai")
    assert ctx.active_backend_supports_prompt() is False


# ----------------------------------------------------------------------
# Canonicalizer wiring
# ----------------------------------------------------------------------


def test_canonicalizer_is_off_by_default(tmp_path):
    ctx = _ctx(tmp_path)
    ctx.get_vocabulary_store().save_snapshot(_SNAPSHOT)
    text, reps = ctx.canonicalize_transcript("mach die Dekenlampe an")
    assert text == "mach die Dekenlampe an"
    assert reps == []


def test_canonicalizer_uses_the_full_uncapped_vocabulary(tmp_path):
    ctx = _ctx(tmp_path)
    ctx.set_enable_canonicalizer(True)
    ctx.set_stt_vocabulary("Fehenlichter")
    ctx.get_vocabulary_store().save_snapshot(_SNAPSHOT)
    # Manual terms are candidates too.
    terms = ctx.get_canonicalizer_terms()
    assert "Fehenlichter" in terms
    assert "Deckenlampe" in terms
    text, reps = ctx.canonicalize_transcript("mach die Dekenlampe an")
    assert text == "mach die Deckenlampe an"
    assert reps[0].replacement == "Deckenlampe"


def test_canonicalizer_hits_are_counted_and_promotable(tmp_path):
    import asyncio

    from murdock.api.routes_integration import (
        PromoteHitIn,
        get_canonicalizer_hits,
        promote_canonicalizer_hit,
    )

    ctx = _ctx(tmp_path)
    ctx.set_enable_canonicalizer(True)
    ctx.get_vocabulary_store().save_snapshot(_SNAPSHOT)
    for _ in range(3):
        ctx.canonicalize_transcript("mach die Dekenlampe an")

    out = asyncio.run(get_canonicalizer_hits(limit=10, ctx=ctx))
    assert len(out.hits) == 1
    hit = out.hits[0]
    assert (hit.original, hit.replacement) == ("Dekenlampe", "Deckenlampe")
    assert hit.count == 3

    # Promoting writes an explicit rule, enables the tier and clears the tally.
    after = asyncio.run(promote_canonicalizer_hit(
        PromoteHitIn(original="Dekenlampe", replacement="Deckenlampe"), ctx
    ))
    assert after.hits == []
    assert "Dekenlampe -> Deckenlampe" in ctx.get_stt_dictionary()
    assert ctx.get_enable_stt_dictionary() is True


def test_promote_rejects_empty_input(tmp_path):
    import asyncio

    from fastapi import HTTPException

    from murdock.api.routes_integration import PromoteHitIn, promote_canonicalizer_hit

    ctx = _ctx(tmp_path)
    with pytest.raises(HTTPException):
        asyncio.run(promote_canonicalizer_hit(
            PromoteHitIn(original="  ", replacement="x"), ctx
        ))
