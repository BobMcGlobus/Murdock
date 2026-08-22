"""One call that answers "is this working, and what should I do next?".

The web UI used to open on a wall of forms with no indication of which
ones mattered yet. Everything here already existed somewhere — spread
across the speaker list, the settings page and the recognition log — but
nobody could see it at once, least of all somebody who had just
installed Murdock and did not yet know the vocabulary.

The checklist is deliberately ordered by dependency rather than by
importance: a transcription backend is useless without a speaker to
recognise, and a speaker is useless if nothing reaches Home Assistant.
"""

from __future__ import annotations

import time
from typing import List, Optional

from fastapi import APIRouter, Depends
from pydantic import BaseModel

from murdock import __version__
from murdock.core.context import AppContext

from .deps import get_context

router = APIRouter(prefix="/api/status", tags=["status"])

#: Below this a speaker's voiceprint is thin enough to be worth saying so.
_HEALTHY_SAMPLES = 3


class SetupStep(BaseModel):
    key: str
    done: bool
    # What is still missing, when it is. Free of jargon on purpose: this
    # is the one screen a new user reads before they know any.
    detail: str = ""


class StatusOut(BaseModel):
    version: str
    speakers: int
    samples: int
    thin_profiles: List[str] = []
    stt_backend: str
    delivery: str
    mqtt_connected: bool
    ha_configured: bool
    events_24h: int = 0
    matches_24h: int = 0
    unknown_24h: int = 0
    last_event_at: Optional[float] = None
    setup: List[SetupStep] = []
    setup_complete: bool = False


@router.get("", response_model=StatusOut)
async def status(ctx: AppContext = Depends(get_context)):
    speakers = ctx.speakers.list_speakers()
    counts = {s.name: s.enrollment_count for s in speakers}
    samples = sum(counts.values())
    thin = sorted(n for n, c in counts.items() if c < _HEALTHY_SAMPLES)

    stats = ctx.recognition.stats(since_seconds=24 * 3600)
    per_outcome = stats.get("per_outcome", {})
    events = sum(per_outcome.values())
    matches = per_outcome.get("match", 0)
    unknown = per_outcome.get("unknown-forwarded", 0)

    recent = ctx.recognition.list_events(limit=1)
    last_at = recent[0].created_at if recent else None

    backend = ctx.get_stt_backend()
    mqtt_ok = bool(ctx.mqtt and ctx.mqtt.connected)
    ha_ok = bool(ctx.ha and ctx.ha.configured)

    setup = [
        SetupStep(
            key="stt",
            done=True,
            detail=backend,
        ),
        SetupStep(
            key="speakers",
            done=bool(speakers),
            detail="" if speakers else "none",
        ),
        SetupStep(
            key="samples",
            done=bool(speakers) and not thin,
            detail=", ".join(thin),
        ),
        SetupStep(
            key="delivery",
            done=mqtt_ok or ha_ok,
            detail="mqtt" if mqtt_ok else ("rest" if ha_ok else ""),
        ),
        SetupStep(
            key="first_recognition",
            done=last_at is not None,
            detail="",
        ),
    ]

    return StatusOut(
        version=__version__,
        speakers=len(speakers),
        samples=samples,
        thin_profiles=thin,
        stt_backend=backend,
        delivery="mqtt" if mqtt_ok else ("rest" if ha_ok else "none"),
        mqtt_connected=mqtt_ok,
        ha_configured=ha_ok,
        events_24h=events,
        matches_24h=matches,
        unknown_24h=unknown,
        last_event_at=last_at,
        setup=setup,
        setup_complete=all(s.done for s in setup),
    )
