"""Shared FastAPI dependencies."""

from __future__ import annotations

from fastapi import Request

from voiceid.core.context import AppContext


def get_context(request: Request) -> AppContext:
    """Fetch the shared :class:`AppContext` from the application state."""
    return request.app.state.context
