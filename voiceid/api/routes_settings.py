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
    advertised_languages: List[str] = Field(default_factory=list)
    advertised_languages_source: str = "auto"  # "auto" | "override"


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
        advertised_languages=languages,
        advertised_languages_source=source,
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
