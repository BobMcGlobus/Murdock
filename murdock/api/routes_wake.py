"""Wake word voiceprints and the rescue mode of the wake word proxy.

Experimental, and only there when the proxy runs (``WAKE_PROXY_UPSTREAM``).
Behind the web API's authentication like every other route.
"""

from __future__ import annotations

from typing import Optional, Union

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel, Field

from murdock.core.context import AppContext

from .deps import get_context

router = APIRouter(prefix="/api/wake", tags=["wake"])


def _proxy(ctx: AppContext):
    proxy = getattr(ctx, "wake_proxy", None)
    if proxy is None:
        raise HTTPException(status_code=404, detail="wake word proxy is not running")
    return proxy


class SettingsIn(BaseModel):
    mode: Optional[str] = None
    threshold: Optional[float] = Field(default=None, ge=0.0, le=1.0)
    min_samples: Optional[int] = Field(default=None, ge=1, le=40)
    min_margin: Optional[float] = Field(default=None, ge=0.0, le=1.0)


class EnrollIn(BaseModel):
    speaker: Union[int, str]
    seconds: float = Field(default=180, ge=10, le=1800)


@router.get("/status")
async def status(recent: int = 20, ctx: AppContext = Depends(get_context)):
    return _proxy(ctx).status(recent=max(0, min(recent, 300)))


@router.patch("/settings")
async def update_settings(body: SettingsIn, ctx: AppContext = Depends(get_context)):
    try:
        return _proxy(ctx).prints.update_settings(**body.model_dump())
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@router.post("/enroll")
async def open_enrollment(body: EnrollIn, ctx: AppContext = Depends(get_context)):
    """The next wake words, on any satellite, are this person's."""
    proxy = _proxy(ctx)
    speakers = proxy.tools.speakers() if proxy.tools else {}
    wanted = body.speaker
    match = None
    for sid, name in speakers.items():
        if (isinstance(wanted, int) and sid == wanted) or (
            isinstance(wanted, str) and name.casefold() == wanted.casefold()
        ):
            match = (sid, name)
            break
    if match is None:
        raise HTTPException(status_code=404, detail=f"no enrolled speaker {wanted!r}")
    proxy.prints.open_window(match[0], match[1], body.seconds)
    return proxy.status(recent=0)["enrolling"]


@router.delete("/enroll")
async def close_enrollment(ctx: AppContext = Depends(get_context)):
    _proxy(ctx).prints.close_window()
    return {"enrolling": None}


@router.delete("/voiceprints/{speaker_id}")
async def forget(speaker_id: int, ctx: AppContext = Depends(get_context)):
    if not _proxy(ctx).prints.remove(speaker_id):
        raise HTTPException(status_code=404, detail="no wake voiceprint for that speaker")
    return {"removed": speaker_id}
