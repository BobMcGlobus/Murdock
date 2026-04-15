"""Runtime settings endpoints."""

from __future__ import annotations

import asyncio
import logging
import os
from typing import List, Optional

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel, Field

from voiceid.core.context import AppContext

from .deps import get_context

_LOGGER = logging.getLogger("voiceid.api.settings")

router = APIRouter(prefix="/api/settings", tags=["settings"])


class SettingsOut(BaseModel):
    verify_threshold: float
    unknown_logging: bool
    unknown_ttl_hours: int
    require_speaker_match: bool
    passthrough_when_no_speakers: bool
    min_verify_seconds: float
    skip_leading_seconds: float
    min_liveness_score: float
    auto_enroll: bool
    upstream_uri: str
    upstream_uri_default: str
    upstream_uri_source: str  # "env" | "override"
    listen_uri: str
    ha_configured: bool
    ha_url: str = ""
    ha_token_set: bool = False
    ha_input_text_entity: str = "input_text.current_speaker"
    ha_tv_entity: str = ""
    ha_confidence_entity: str = ""
    ha_distance_entity: str = ""
    ha_nearest_entity: str = ""
    ha_role_entity: str = ""
    ha_emotion_entity: str = ""
    advertised_languages: List[str] = Field(default_factory=list)
    advertised_languages_source: str = "auto"  # "auto" | "override"
    quality_weights: dict = Field(default_factory=dict)
    quality_weights_source: str = "default"  # "default" | "override"
    # Emotion detection status — plumbing-only in this release. `enabled`
    # reflects the user's opt-in; `model_available` reflects whether a
    # usable ONNX file is on disk. The UI shows "no model yet" when the
    # flag is on but the model is missing, preventing confusion about
    # why recognition events lack emotion data.
    enable_emotion: bool = False
    emotion_model_available: bool = False


class SettingsPatch(BaseModel):
    verify_threshold: Optional[float] = Field(default=None, ge=0.0, le=2.0)
    unknown_logging: Optional[bool] = None
    require_speaker_match: Optional[bool] = None
    passthrough_when_no_speakers: Optional[bool] = None
    skip_leading_seconds: Optional[float] = Field(default=None, ge=0.0, le=5.0)
    min_liveness_score: Optional[float] = Field(default=None, ge=0.0, le=1.0)
    auto_enroll: Optional[bool] = None
    # None → field not touched
    # ""   → clear override, fall back to env default
    # "…"  → set override (host:port accepted, tcp:// auto-prefixed)
    upstream_uri: Optional[str] = None
    # None  → field not touched
    # []    → clear override, fall back to upstream auto-detect
    # [...] → hard override
    advertised_languages: Optional[List[str]] = None
    # Home Assistant fields (None = not touched, "" = clear)
    ha_url: Optional[str] = None
    ha_token: Optional[str] = None
    ha_input_text_entity: Optional[str] = None
    ha_tv_entity: Optional[str] = None
    ha_confidence_entity: Optional[str] = None
    ha_distance_entity: Optional[str] = None
    ha_nearest_entity: Optional[str] = None
    ha_role_entity: Optional[str] = None
    ha_emotion_entity: Optional[str] = None
    # Emotion detection (experimental). Setting enable_emotion to True while
    # no model is on disk is harmless — the handler gates on both flag AND
    # model availability before attempting inference.
    enable_emotion: Optional[bool] = None
    quality_weights: Optional[dict] = None  # {"speech_ratio":0.25,...} — pass {} to reset


class RestartResponse(BaseModel):
    ok: bool
    message: str


def _build_settings_out(ctx: AppContext) -> SettingsOut:
    override = ctx.get_advertised_languages()
    if override:
        languages = override
        source = "override"
    else:
        # Whatever the cache currently holds; may be [] if upstream is
        # unreachable, in which case the UI will say "auto (none yet)".
        cache = getattr(ctx, "info_cache", None)
        languages = list(getattr(cache, "_languages", []) or [])
        source = "auto"
    return SettingsOut(
        verify_threshold=ctx.get_verify_threshold(),
        unknown_logging=ctx.get_unknown_logging(),
        unknown_ttl_hours=ctx.settings.unknown_ttl_hours,
        require_speaker_match=ctx.get_require_match(),
        passthrough_when_no_speakers=ctx.get_passthrough_when_empty(),
        min_verify_seconds=ctx.settings.min_verify_seconds,
        skip_leading_seconds=ctx.get_skip_leading_seconds(),
        min_liveness_score=ctx.get_min_liveness_score(),
        auto_enroll=ctx.get_auto_enroll(),
        upstream_uri=ctx.get_upstream_uri(),
        upstream_uri_default=ctx.settings.upstream_uri,
        upstream_uri_source=ctx.get_upstream_uri_source(),
        listen_uri=ctx.settings.listen_uri,
        ha_configured=ctx.ha.configured,
        ha_url=ctx.get_ha_url(),
        ha_token_set=ctx.has_ha_token(),
        ha_input_text_entity=ctx.get_ha_input_text_entity(),
        ha_tv_entity=ctx.get_ha_tv_entity(),
        ha_confidence_entity=ctx.get_ha_confidence_entity(),
        ha_distance_entity=ctx.get_ha_distance_entity(),
        ha_nearest_entity=ctx.get_ha_nearest_entity(),
        ha_role_entity=ctx.get_ha_role_entity(),
        ha_emotion_entity=ctx.get_ha_emotion_entity(),
        advertised_languages=languages,
        advertised_languages_source=source,
        quality_weights=ctx.get_quality_weights(),
        quality_weights_source=ctx.get_quality_weights_source(),
        enable_emotion=ctx.get_enable_emotion(),
        emotion_model_available=ctx.emotion_model_available(),
    )


@router.get("", response_model=SettingsOut)
async def get_settings_route(ctx: AppContext = Depends(get_context)):
    return _build_settings_out(ctx)


@router.patch("", response_model=SettingsOut)
async def patch_settings(
    body: SettingsPatch, ctx: AppContext = Depends(get_context)
):
    has_field = any(
        getattr(body, f) is not None
        for f in body.model_fields
    )
    if not has_field:
        raise HTTPException(status_code=400, detail="No fields to update")
    if body.verify_threshold is not None:
        ctx.set_verify_threshold(body.verify_threshold)
    if body.unknown_logging is not None:
        ctx.set_unknown_logging(body.unknown_logging)
    if body.require_speaker_match is not None:
        ctx.set_require_match(body.require_speaker_match)
    if body.passthrough_when_no_speakers is not None:
        ctx.set_passthrough_when_empty(body.passthrough_when_no_speakers)
    if body.skip_leading_seconds is not None:
        ctx.set_skip_leading_seconds(body.skip_leading_seconds)
    if body.min_liveness_score is not None:
        ctx.set_min_liveness_score(body.min_liveness_score)
    if body.auto_enroll is not None:
        ctx.set_auto_enroll(body.auto_enroll)
    if body.upstream_uri is not None:
        ctx.set_upstream_uri(body.upstream_uri)
    if body.advertised_languages is not None:
        # An empty list means "clear override" — fall through to
        # upstream auto-detect. A non-empty list is a hard override.
        ctx.set_advertised_languages(body.advertised_languages or None)
    # Home Assistant settings — reconfigure the live client when changed.
    ha_changed = False
    if body.ha_url is not None:
        ctx.set_ha_url(body.ha_url)
        ha_changed = True
    if body.ha_token is not None:
        ctx.set_ha_token(body.ha_token)
        ha_changed = True
    if body.ha_input_text_entity is not None:
        ctx.set_ha_input_text_entity(body.ha_input_text_entity)
        ha_changed = True
    if body.ha_tv_entity is not None:
        ctx.set_ha_tv_entity(body.ha_tv_entity)
        ha_changed = True
    if body.ha_confidence_entity is not None:
        ctx.set_ha_confidence_entity(body.ha_confidence_entity)
        ha_changed = True
    if body.ha_distance_entity is not None:
        ctx.set_ha_distance_entity(body.ha_distance_entity)
        ha_changed = True
    if body.ha_nearest_entity is not None:
        ctx.set_ha_nearest_entity(body.ha_nearest_entity)
        ha_changed = True
    if body.ha_role_entity is not None:
        ctx.set_ha_role_entity(body.ha_role_entity)
        ha_changed = True
    if body.ha_emotion_entity is not None:
        ctx.set_ha_emotion_entity(body.ha_emotion_entity)
        ha_changed = True
    if ha_changed:
        await ctx.apply_ha_settings()
    if body.enable_emotion is not None:
        ctx.set_enable_emotion(body.enable_emotion)
    if body.quality_weights is not None:
        # Empty dict resets to defaults.
        if body.quality_weights == {}:
            ctx.set_quality_weights(None)
        else:
            ctx.set_quality_weights(body.quality_weights)
    return _build_settings_out(ctx)


class UpstreamPingOut(BaseModel):
    ok: bool
    upstream_uri: str
    latency_ms: Optional[float] = None
    languages: List[str] = Field(default_factory=list)
    error: Optional[str] = None


@router.post("/ping-upstream", response_model=UpstreamPingOut)
async def ping_upstream(ctx: AppContext = Depends(get_context)):
    """Open a Wyoming connection to the upstream STT and describe it.

    Pure diagnostic endpoint: lets the user verify from the Settings tab
    that VoiceID can actually reach its configured upstream, instead of
    having to interpret the container logs.
    """
    import time as _time

    from wyoming.client import AsyncClient
    from wyoming.info import Describe, Info

    # Use the LIVE URI (UI override > compose default), not the static
    # env value, so the button actually pings what the handler would.
    upstream_uri = ctx.get_upstream_uri()
    t0 = _time.monotonic()
    try:
        async with AsyncClient.from_uri(upstream_uri) as client:
            await client.write_event(Describe().event())
            # Bounded wait: the upstream should respond within a second.
            deadline = _time.monotonic() + 5.0
            langs: List[str] = []
            while _time.monotonic() < deadline:
                event = await asyncio.wait_for(
                    client.read_event(),
                    timeout=max(0.1, deadline - _time.monotonic()),
                )
                if event is None:
                    break
                if Info.is_type(event.type):
                    info = Info.from_event(event)
                    for asr in info.asr:
                        for model in asr.models:
                            for lang in model.languages:
                                if lang and lang not in langs:
                                    langs.append(lang)
                    break
            latency_ms = (_time.monotonic() - t0) * 1000
            return UpstreamPingOut(
                ok=True,
                upstream_uri=upstream_uri,
                latency_ms=latency_ms,
                languages=langs,
            )
    except Exception as exc:
        _LOGGER.warning("ping-upstream failed: %s", exc)
        return UpstreamPingOut(
            ok=False, upstream_uri=upstream_uri, error=str(exc)
        )


@router.post("/refresh-languages", response_model=SettingsOut)
async def refresh_languages(ctx: AppContext = Depends(get_context)):
    """Force a fresh Describe round-trip against the upstream STT.

    Lets the admin nudge VoiceID to re-query the upstream without
    restarting the container — useful after fixing DNS, starting the
    STT container, etc.
    """
    cache = getattr(ctx, "info_cache", None)
    if cache is None:
        raise HTTPException(status_code=503, detail="info cache not ready")
    try:
        await cache.get_languages(force=True)
    except Exception as exc:
        _LOGGER.exception("Manual language refresh failed")
        raise HTTPException(
            status_code=502, detail=f"upstream refresh failed: {exc}"
        ) from exc
    return _build_settings_out(ctx)


class HATestOut(BaseModel):
    ok: bool
    error: Optional[str] = None


@router.post("/test-ha", response_model=HATestOut)
async def test_ha(ctx: AppContext = Depends(get_context)):
    """Quick connectivity check against the configured HA instance."""
    if not ctx.ha.configured:
        return HATestOut(ok=False, error="not configured (URL or token missing)")
    try:
        import httpx
        async with httpx.AsyncClient(
            base_url=ctx.ha.base_url or "",
            headers={
                "Authorization": f"Bearer {ctx.ha.token or ''}",
                "Content-Type": "application/json",
            },
            timeout=5.0,
        ) as client:
            resp = await client.get("/api/")
            if resp.status_code == 200:
                return HATestOut(ok=True)
            return HATestOut(ok=False, error=f"HTTP {resp.status_code}")
    except Exception as exc:
        return HATestOut(ok=False, error=str(exc))


# ----------------------------------------------------------------------
# Per-satellite verify-threshold overrides
# ----------------------------------------------------------------------


class SatelliteThresholdEntry(BaseModel):
    satellite_id: str
    threshold: Optional[float] = None  # None = no override, falls back to global
    seen_events: int = 0               # recognition_events count (for sorting)
    last_seen: Optional[float] = None  # epoch seconds of last event


class SatelliteThresholdList(BaseModel):
    default_threshold: float
    entries: List[SatelliteThresholdEntry]


class SatelliteThresholdPatch(BaseModel):
    satellite_id: str
    # null → clear the override; float → set it
    threshold: Optional[float] = Field(default=None, ge=0.0, le=2.0)


def _list_known_satellites(ctx: AppContext) -> list[tuple[str, int, float]]:
    """Return ``(satellite_id, count, last_seen)`` triples from recognition_events."""
    cur = ctx.db.execute(
        "SELECT satellite_id, COUNT(*) AS n, MAX(created_at) AS last "
        "FROM recognition_events "
        "WHERE satellite_id IS NOT NULL AND satellite_id != '' "
        "GROUP BY satellite_id "
        "ORDER BY n DESC"
    )
    return [(row["satellite_id"], row["n"], row["last"]) for row in cur.fetchall()]


@router.get("/satellite-thresholds", response_model=SatelliteThresholdList)
async def list_satellite_thresholds(ctx: AppContext = Depends(get_context)):
    """Return known satellites + their per-satellite threshold overrides.

    The UI uses this to render one row per satellite the proxy has seen,
    letting the admin tune a per-room threshold without having to know
    the exact IDs in advance.
    """
    overrides = ctx.get_satellite_thresholds()
    known = _list_known_satellites(ctx)
    entries: List[SatelliteThresholdEntry] = []
    covered: set[str] = set()
    for sid, count, last in known:
        entries.append(
            SatelliteThresholdEntry(
                satellite_id=sid,
                threshold=overrides.get(sid),
                seen_events=count,
                last_seen=last,
            )
        )
        covered.add(sid)
    # Also surface overrides for satellites we have no events for yet
    # (e.g. pre-configured manually before the first recognition).
    for sid, th in overrides.items():
        if sid not in covered:
            entries.append(
                SatelliteThresholdEntry(satellite_id=sid, threshold=th)
            )
    return SatelliteThresholdList(
        default_threshold=ctx.get_verify_threshold(),
        entries=entries,
    )


@router.patch("/satellite-thresholds", response_model=SatelliteThresholdList)
async def patch_satellite_threshold(
    body: SatelliteThresholdPatch, ctx: AppContext = Depends(get_context)
):
    """Set or clear a per-satellite threshold override.

    ``threshold = null`` clears the override for that satellite.
    """
    try:
        ctx.set_satellite_threshold(body.satellite_id, body.threshold)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc))
    return await list_satellite_thresholds(ctx)


@router.post("/restart", response_model=RestartResponse)
async def restart_service(ctx: AppContext = Depends(get_context)):
    """Hard-restart the VoiceID process so container supervisors (Docker,
    systemd) bring it back up. Returns immediately, then exits after a
    short delay so the HTTP response actually reaches the UI."""

    async def _exit_soon() -> None:
        await asyncio.sleep(0.5)
        _LOGGER.warning("Restart requested via API — exiting process")
        os._exit(0)

    asyncio.create_task(_exit_soon())
    return RestartResponse(
        ok=True, message="restart scheduled — service will exit in 500ms"
    )
