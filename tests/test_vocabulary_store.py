"""Tests for the HA registry vocabulary snapshots (plan §9/§10)."""

from __future__ import annotations

import asyncio
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
