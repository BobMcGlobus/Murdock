"""Endpoints for the Home Assistant custom integration (plan §21).

Serves the integration's config flow and initial sync:

- ``GET /api/version``     — proxy version (connection test)
- ``GET /api/satellites``  — satellites the proxy has seen
- ``GET /api/state``       — last recognition per satellite

The integration reconstructs its in-memory speaker state from
``/api/state`` after an HA restart; live updates then arrive via the
``speaker_recognition_detected`` event.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional, Union

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel, Field

from murdock import __version__
from murdock.core.context import AppContext
from murdock.core.recognition_log import OUTCOME_MATCH

from .deps import get_context

_LOGGER = logging.getLogger("murdock.api.integration")

router = APIRouter(prefix="/api", tags=["integration"])


class VersionOut(BaseModel):
    version: str


class SatelliteOut(BaseModel):
    satellite_id: str
    seen_events: int = 0
    last_seen: Optional[float] = None  # epoch seconds


class SatelliteList(BaseModel):
    satellites: List[SatelliteOut]


class SatelliteStateOut(BaseModel):
    satellite_id: str
    # Name for a verified match, null otherwise — same semantics as the
    # ``speaker_recognition_detected`` event.
    speaker: Optional[str] = None
    outcome: str
    distance: Optional[float] = None
    threshold: Optional[float] = None
    weight: Optional[float] = None
    margin: Optional[float] = None
    recognized_at: float  # epoch seconds


class StateOut(BaseModel):
    satellites: List[SatelliteStateOut]


@router.get("/version", response_model=VersionOut)
async def get_version() -> VersionOut:
    return VersionOut(version=__version__)


@router.get("/satellites", response_model=SatelliteList)
async def list_satellites(ctx: AppContext = Depends(get_context)) -> SatelliteList:
    """All satellites seen in the recognition log, busiest first."""
    cur = ctx.db.execute(
        "SELECT satellite_id, COUNT(*) AS n, MAX(created_at) AS last "
        "FROM recognition_events "
        "WHERE satellite_id IS NOT NULL AND satellite_id != '' "
        "GROUP BY satellite_id "
        "ORDER BY n DESC"
    )
    return SatelliteList(
        satellites=[
            SatelliteOut(
                satellite_id=row["satellite_id"],
                seen_events=row["n"],
                last_seen=row["last"],
            )
            for row in cur.fetchall()
        ]
    )


@router.get("/state", response_model=StateOut)
async def get_state(ctx: AppContext = Depends(get_context)) -> StateOut:
    """Latest recognition per satellite (initial sync after HA restart).

    The integration applies its own freshness window on
    ``recognized_at``, so stale rows are harmless here.
    """
    # Latest row per satellite via MAX(id): the autoincrement id is
    # strictly monotonic, unlike created_at, which can collide for
    # back-to-back events.
    cur = ctx.db.execute(
        "SELECT e.satellite_id, e.outcome, e.matched_speaker, e.distance, "
        "e.threshold, e.weight, e.margin, e.created_at "
        "FROM recognition_events e "
        "JOIN ("
        "  SELECT satellite_id, MAX(id) AS mid "
        "  FROM recognition_events "
        "  WHERE satellite_id IS NOT NULL AND satellite_id != '' "
        "  GROUP BY satellite_id"
        ") latest ON e.id = latest.mid"
    )
    states: List[SatelliteStateOut] = []
    for row in cur.fetchall():
        is_match = row["outcome"] == OUTCOME_MATCH
        states.append(
            SatelliteStateOut(
                satellite_id=row["satellite_id"],
                speaker=row["matched_speaker"] if is_match else None,
                outcome=row["outcome"],
                distance=row["distance"],
                threshold=row["threshold"],
                weight=row["weight"],
                margin=row["margin"],
                recognized_at=row["created_at"],
            )
        )
    return StateOut(satellites=states)


# ----------------------------------------------------------------------
# Vocabulary mirroring (plan §9)
# ----------------------------------------------------------------------


class VocabularyIn(BaseModel):
    """Registry snapshot pushed by the integration.

    Entities/areas/floors are kept schemaless dicts on purpose — the
    integration owns the shape, Murdock persists it verbatim and only
    extracts names/aliases for its indexes.
    """

    version: Optional[Union[int, str]] = None
    generated_at: Optional[str] = None
    entities: List[Dict[str, Any]] = Field(default_factory=list)
    areas: List[Any] = Field(default_factory=list)
    floors: List[Any] = Field(default_factory=list)


class VocabularyStoredOut(BaseModel):
    snapshot_id: int
    version: Optional[str] = None
    term_count: int


class VocabularyMetaOut(BaseModel):
    available: bool
    snapshot_id: Optional[int] = None
    version: Optional[str] = None
    generated_at: Optional[str] = None
    created_at: Optional[float] = None
    entity_count: int = 0
    term_count: int = 0
    terms_preview: List[str] = Field(default_factory=list)
    # Full mirrored term list, in the priority order the cap applies to
    # (entity names → aliases → areas → floors).
    terms: List[str] = Field(default_factory=list)
    # How many of those actually reach the STT backend.
    term_cap: int = 0
    # The prompt that really goes out: manual terms + capped HA terms.
    # Empty when the vocabulary tier is switched off.
    effective_prompt: str = ""
    effective_enabled: bool = False
    # Manually maintained terms, always sent, never capped away.
    manual_terms: List[str] = Field(default_factory=list)
    # Curated mirrored terms; null means "automatic, first N by priority".
    selection: Optional[List[str]] = None
    selected: List[str] = Field(default_factory=list)
    # False when the active backend ignores the prompt entirely.
    backend_supports_prompt: bool = False


@router.post("/vocabulary", response_model=VocabularyStoredOut)
async def push_vocabulary(
    body: VocabularyIn, ctx: AppContext = Depends(get_context)
) -> VocabularyStoredOut:
    """Persist a registry snapshot (versioned; survives HA outages)."""
    store = ctx.get_vocabulary_store()
    try:
        meta = store.save_snapshot(body.model_dump())
    except Exception as exc:
        _LOGGER.exception("Failed to store vocabulary snapshot")
        raise HTTPException(status_code=500, detail=str(exc))
    return VocabularyStoredOut(
        snapshot_id=meta["snapshot_id"],
        version=str(meta["version"]) if meta["version"] is not None else None,
        term_count=meta["term_count"],
    )


@router.get("/vocabulary", response_model=VocabularyMetaOut)
async def get_vocabulary(
    ctx: AppContext = Depends(get_context)
) -> VocabularyMetaOut:
    """Latest snapshot + what the STT backend actually receives.

    Serves both the integration's verify step and the Web UI panel, so
    "which terms are in play" is answerable without reading the DB.
    """
    from murdock.core.context import _HA_VOCAB_TERM_CAP

    store = ctx.get_vocabulary_store()
    enabled = ctx.get_enable_stt_vocabulary()
    effective = ctx.get_effective_vocabulary()
    snap = store.latest()
    if snap is None:
        return VocabularyMetaOut(
            available=False,
            term_cap=_HA_VOCAB_TERM_CAP,
            effective_prompt=effective,
            effective_enabled=enabled,
            manual_terms=ctx.get_manual_vocabulary_terms(),
            backend_supports_prompt=ctx.active_backend_supports_prompt(),
        )
    terms = store.terms()
    return VocabularyMetaOut(
        available=True,
        snapshot_id=snap["snapshot_id"],
        version=snap["version"],
        generated_at=snap["generated_at"],
        created_at=snap["created_at"],
        entity_count=len(snap["payload"].get("entities") or []),
        term_count=len(terms),
        terms_preview=terms[:25],
        terms=terms,
        term_cap=_HA_VOCAB_TERM_CAP,
        effective_prompt=effective,
        effective_enabled=enabled,
        manual_terms=ctx.get_manual_vocabulary_terms(),
        selection=ctx.get_vocab_selection(),
        selected=ctx.get_selected_ha_terms(),
        backend_supports_prompt=ctx.active_backend_supports_prompt(),
    )


class VocabularySelectionIn(BaseModel):
    """Which mirrored terms go into the bias prompt.

    ``auto`` returns to "first N by priority"; otherwise ``terms`` is the
    explicit list (an empty list is a valid choice: send none of them).
    """

    auto: bool = False
    terms: Optional[List[str]] = None


@router.put("/vocabulary/selection", response_model=VocabularyMetaOut)
async def put_vocabulary_selection(
    body: VocabularySelectionIn, ctx: AppContext = Depends(get_context)
) -> VocabularyMetaOut:
    if body.auto:
        ctx.set_vocab_selection(None)
    else:
        ctx.set_vocab_selection(body.terms or [])
    return await get_vocabulary(ctx)


class CanonicalizerHitOut(BaseModel):
    original: str
    replacement: str
    count: int
    first_seen: float
    last_seen: float


class CanonicalizerHitsOut(BaseModel):
    hits: List[CanonicalizerHitOut] = Field(default_factory=list)


@router.get("/canonicalizer/hits", response_model=CanonicalizerHitsOut)
async def get_canonicalizer_hits(
    limit: int = 20, ctx: AppContext = Depends(get_context)
) -> CanonicalizerHitsOut:
    """Corrections the canonicalizer keeps making.

    A pair that recurs is no longer a fuzzy guess — it is a fact about
    this household, and belongs in the explicit dictionary.
    """
    hits = ctx.get_canonicalizer_hits(limit=limit)
    return CanonicalizerHitsOut(
        hits=[
            CanonicalizerHitOut(
                original=h.original,
                replacement=h.replacement,
                count=h.count,
                first_seen=h.first_seen,
                last_seen=h.last_seen,
            )
            for h in hits
        ]
    )


class PromoteHitIn(BaseModel):
    original: str
    replacement: str


@router.post("/canonicalizer/promote", response_model=CanonicalizerHitsOut)
async def promote_canonicalizer_hit(
    body: PromoteHitIn, ctx: AppContext = Depends(get_context)
) -> CanonicalizerHitsOut:
    """Turn a recurring correction into an explicit dictionary rule.

    Deliberately a user action, not an automatism (plan §11: "Vorschlag,
    nicht Automatik"). Once promoted it runs as a deterministic
    replacement, so the fuzzy path never has to decide it again.
    """
    original = (body.original or "").strip()
    replacement = (body.replacement or "").strip()
    if not original or not replacement:
        raise HTTPException(status_code=400, detail="original and replacement required")

    line = f"{original} -> {replacement}"
    existing = ctx.get_stt_dictionary()
    lines = [ln for ln in existing.splitlines()]
    if not any(ln.strip() == line for ln in lines):
        if lines and lines[-1].strip():
            lines.append(line)
        elif lines:
            lines[-1] = line
        else:
            lines = [line]
        ctx.set_stt_dictionary("\n".join(lines))
        # Enable the tier — a promoted rule that never runs is a trap.
        if not ctx.get_enable_stt_dictionary():
            ctx.set_enable_stt_dictionary(True)
            _LOGGER.info("Correction dictionary enabled by rule promotion")
    try:
        from murdock.core.canonicalizer_hits import CanonicalizerHits

        CanonicalizerHits(ctx.db).forget(original, replacement)
    except Exception:
        _LOGGER.debug("Could not clear promoted hit", exc_info=True)
    return await get_canonicalizer_hits(limit=20, ctx=ctx)
