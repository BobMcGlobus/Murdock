"""Speech-to-text services: the connections, their roles, a test button."""

from __future__ import annotations

import asyncio
import logging
import time
from typing import List, Optional

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel, Field

from murdock.core.context import AppContext
from murdock.core.stt_services import (
    KIND_FIELDS,
    KIND_HA,
    KIND_WYOMING,
    KINDS,
    SECRET_KEYS,
    SttRoles,
    SttService,
    _clean_config,
)

from .deps import get_context

_LOGGER = logging.getLogger("murdock.api.stt_services")

router = APIRouter(prefix="/api/stt-services", tags=["stt-services"])


class ServiceOut(BaseModel):
    id: int
    name: str
    kind: str
    config: dict
    secrets_set: List[str] = Field(default_factory=list)
    created_at: float = 0.0
    remote: bool = False
    complete: bool = True


class RolesModel(BaseModel):
    main: Optional[int] = None
    fallbacks: List[int] = Field(default_factory=list)
    shadows: List[int] = Field(default_factory=list)
    fallback_on_empty: bool = True


class ServicesOut(BaseModel):
    kinds: dict
    services: List[ServiceOut]
    roles: RolesModel


class ServiceIn(BaseModel):
    name: str = ""
    kind: str
    config: dict = Field(default_factory=dict)


class ServicePatch(BaseModel):
    name: Optional[str] = None
    config: Optional[dict] = None


class TestIn(BaseModel):
    # Test what the form holds, not only what was saved; with ``id`` an
    # empty secret falls back to the stored one, as a save would.
    id: Optional[int] = None
    kind: Optional[str] = None
    config: dict = Field(default_factory=dict)


class TestOut(BaseModel):
    ok: bool
    latency_ms: Optional[float] = None
    languages: List[str] = Field(default_factory=list)
    language_ok: Optional[bool] = None
    configured_language: str = ""
    transcript: Optional[str] = None
    error: Optional[str] = None


def _out(ctx: AppContext, service: SttService) -> ServiceOut:
    return ServiceOut(
        **service.public(),
        remote=ctx.is_service_remote(service),
        complete=ctx.build_stt_backend(service) is not None,
    )


def _services_out(ctx: AppContext) -> ServicesOut:
    store = ctx.stt_services
    roles = store.roles()
    return ServicesOut(
        kinds={k: list(v) for k, v in KIND_FIELDS.items()},
        services=[_out(ctx, s) for s in store.list()],
        roles=RolesModel(
            main=roles.main, fallbacks=roles.fallbacks,
            shadows=roles.shadows, fallback_on_empty=roles.fallback_on_empty,
        ),
    )


@router.get("", response_model=ServicesOut)
async def list_services(ctx: AppContext = Depends(get_context)):
    return _services_out(ctx)


@router.post("", response_model=ServicesOut)
async def create_service(body: ServiceIn, ctx: AppContext = Depends(get_context)):
    if body.kind not in KINDS:
        raise HTTPException(status_code=400, detail=f"unknown kind {body.kind!r}")
    try:
        ctx.stt_services.create(body.name, body.kind, body.config)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc))
    ctx.sync_upstream_from_services()
    return _services_out(ctx)


@router.put("/roles", response_model=ServicesOut)
async def set_roles(body: RolesModel, ctx: AppContext = Depends(get_context)):
    ctx.stt_services.set_roles(SttRoles(
        main=body.main, fallbacks=list(body.fallbacks),
        shadows=list(body.shadows), fallback_on_empty=body.fallback_on_empty,
    ))
    ctx.sync_upstream_from_services()
    return _services_out(ctx)


@router.patch("/{service_id}", response_model=ServicesOut)
async def update_service(
    service_id: int, body: ServicePatch, ctx: AppContext = Depends(get_context)
):
    try:
        ctx.stt_services.update(service_id, name=body.name, config=body.config)
    except KeyError:
        raise HTTPException(status_code=404, detail="no such service")
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc))
    ctx.sync_upstream_from_services()
    return _services_out(ctx)


@router.delete("/{service_id}", response_model=ServicesOut)
async def delete_service(service_id: int, ctx: AppContext = Depends(get_context)):
    if ctx.stt_services.get(service_id) is None:
        raise HTTPException(status_code=404, detail="no such service")
    try:
        ctx.stt_services.delete(service_id)
    except ValueError as exc:
        raise HTTPException(status_code=409, detail=str(exc))
    ctx.sync_upstream_from_services()
    return _services_out(ctx)


async def _describe_wyoming(uri: str) -> TestOut:
    from wyoming.client import AsyncClient
    from wyoming.info import Describe, Info

    t0 = time.monotonic()
    async with AsyncClient.from_uri(uri) as client:
        await client.write_event(Describe().event())
        deadline = t0 + 5.0
        while time.monotonic() < deadline:
            event = await asyncio.wait_for(
                client.read_event(), timeout=max(0.1, deadline - time.monotonic())
            )
            if event is None:
                break
            if not Info.is_type(event.type):
                continue
            langs: List[str] = []
            for asr in Info.from_event(event).asr:
                for model in asr.models:
                    for lang in model.languages:
                        if lang and lang not in langs:
                            langs.append(lang)
            return TestOut(
                ok=True, latency_ms=(time.monotonic() - t0) * 1000, languages=langs
            )
    return TestOut(ok=False, error="connected, but no answer to Describe")


def _language_ok(want: str, languages: List[str]) -> Optional[bool]:
    from murdock.core.stt_backend import _normalize_language

    want = _normalize_language(want)
    if not want or not languages:
        return None
    return any(_normalize_language(x) == want for x in languages)


@router.post("/test", response_model=TestOut)
async def test_service(body: TestIn, ctx: AppContext = Depends(get_context)):
    """Check a service the cheapest way its protocol allows.

    Wyoming answers Describe and Home Assistant lists an entity's
    capabilities without transcribing anything. The HTTP APIs have no
    such call, so they get one second of silence — the smallest request
    that proves the key, the model name and the route all work.
    """
    stored = ctx.stt_services.get(body.id) if body.id is not None else None
    kind = body.kind or (stored.kind if stored else None)
    if kind not in KINDS:
        raise HTTPException(status_code=400, detail="unknown or missing kind")
    config = dict(stored.config) if stored is not None and stored.kind == kind else {}
    for key, value in (body.config or {}).items():
        if key in SECRET_KEYS and not value:
            continue
        config[key] = value
    probe = SttService(
        id=body.id or 0,
        name=stored.name if stored is not None else kind,
        kind=kind,
        config=_clean_config(kind, config),
    )
    lang = ctx.get_stt_language() or ""

    backend = ctx.build_stt_backend(probe)
    if backend is None:
        return TestOut(
            ok=False, configured_language=lang,
            error="incomplete — a required field is empty",
        )

    try:
        if kind == KIND_WYOMING:
            out = await _describe_wyoming(probe.config["uri"])
            out.configured_language = lang
            if out.ok:
                out.language_ok = _language_ok(lang, out.languages)
            return out
        t0 = time.monotonic()
        if kind == KIND_HA:
            caps = await backend.capabilities()
            languages = [str(x) for x in (caps.get("languages") or [])]
            return TestOut(
                ok=True,
                latency_ms=(time.monotonic() - t0) * 1000,
                languages=languages,
                language_ok=_language_ok(lang, languages),
                configured_language=lang,
            )
        text = await asyncio.wait_for(
            backend.transcribe(b"\x00\x00" * 16000),
            timeout=ctx.service_timeout(probe) + 2,
        )
        return TestOut(
            ok=True, latency_ms=(time.monotonic() - t0) * 1000,
            transcript=text or "", configured_language=lang,
        )
    except asyncio.TimeoutError:
        return TestOut(ok=False, configured_language=lang, error="timed out")
    except Exception as exc:
        _LOGGER.info("STT service test failed (%s): %s", kind, exc)
        return TestOut(
            ok=False, configured_language=lang,
            error=str(exc) or type(exc).__name__,
        )
