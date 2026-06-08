"""FastAPI application for Murdock Web UI."""

from __future__ import annotations

import logging
from pathlib import Path

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, HTMLResponse
from fastapi.staticfiles import StaticFiles

from murdock import __version__
from murdock.core.context import AppContext

from .routes_backup import router as backup_router
from .routes_recognition import router as recognition_router
from .routes_settings import router as settings_router
from .routes_speakers import router as speakers_router
from .routes_unknown import router as unknown_router

_LOGGER = logging.getLogger("murdock.api")

_STATIC_DIR = Path(__file__).parent.parent / "ui" / "static"


def create_app(context: AppContext) -> FastAPI:
    """Build the FastAPI application bound to an :class:`AppContext`."""
    app = FastAPI(
        title="Murdock",
        description="Speaker recognition proxy for Home Assistant",
        version=__version__,
    )
    app.state.context = context

    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_methods=["*"],
        allow_headers=["*"],
    )

    app.include_router(speakers_router)
    app.include_router(unknown_router)
    app.include_router(settings_router)
    app.include_router(recognition_router)
    app.include_router(backup_router)

    @app.get("/api/health")
    async def health() -> dict:
        return {
            "status": "ok",
            "version": __version__,
            "speakers": len(context.speakers.list_speakers()),
        }

    if _STATIC_DIR.exists():
        app.mount(
            "/static",
            StaticFiles(directory=str(_STATIC_DIR)),
            name="static",
        )

        # Cache the raw HTML once — we template-replace per request, but
        # the file itself is tiny and never changes at runtime.
        _INDEX_HTML = (_STATIC_DIR / "index.html").read_text(encoding="utf-8")

        @app.get("/", include_in_schema=False)
        async def index(request: Request) -> HTMLResponse:
            # Home Assistant's ingress proxy forwards requests with an
            # X-Ingress-Path header pointing at the rewritten base (e.g.
            # ``/api/hassio_ingress/<token>``). We inject that into
            # window.API_BASE so the SPA's absolute /api/... fetches get
            # re-routed through ingress instead of hitting HA's own API.
            ingress_path = request.headers.get("X-Ingress-Path", "")
            ingress_path = ingress_path.rstrip("/")
            html = _INDEX_HTML.replace("__API_BASE__", ingress_path)
            return HTMLResponse(html)
    else:
        _LOGGER.warning("UI static dir not found at %s", _STATIC_DIR)

    return app
