"""FastAPI application for VoiceID Web UI."""

from __future__ import annotations

import logging
from pathlib import Path

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from voiceid import __version__
from voiceid.core.context import AppContext

from .routes_recognition import router as recognition_router
from .routes_settings import router as settings_router
from .routes_speakers import router as speakers_router
from .routes_unknown import router as unknown_router

_LOGGER = logging.getLogger("voiceid.api")

_STATIC_DIR = Path(__file__).parent.parent / "ui" / "static"


def create_app(context: AppContext) -> FastAPI:
    """Build the FastAPI application bound to an :class:`AppContext`."""
    app = FastAPI(
        title="VoiceID",
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

        @app.get("/", include_in_schema=False)
        async def index() -> FileResponse:
            return FileResponse(str(_STATIC_DIR / "index.html"))
    else:
        _LOGGER.warning("UI static dir not found at %s", _STATIC_DIR)

    return app
