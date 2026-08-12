"""REST endpoints for the recognition audit log."""

from __future__ import annotations

from typing import List, Optional

from fastapi import APIRouter, Depends, Query
from pydantic import BaseModel

from murdock.core.context import AppContext

from .deps import get_context

router = APIRouter(prefix="/api/recognition", tags=["recognition"])


class RecognitionEventOut(BaseModel):
    id: int
    created_at: float
    session_id: str
    satellite_id: Optional[str]
    duration_sec: float
    outcome: str
    matched_speaker: Optional[str]
    distance: Optional[float]
    threshold: Optional[float]
    verify_ms: Optional[float]
    transcript: Optional[str]
    emotion: Optional[str] = None
    emotion_confidence: Optional[float] = None
    # Set when a captured (untagged) unknown sample exists for this
    # session, so the UI can offer "assign to speaker" on blocked entries.
    unknown_sample_id: Optional[int] = None
    # A/B shadow engine result, filled in asynchronously.
    shadow_transcript: Optional[str] = None
    shadow_engine: Optional[str] = None
    # Wall-clock time per STT engine, so the A/B view compares speed too.
    transcript_ms: Optional[float] = None
    shadow_ms: Optional[float] = None
    # {"ttfb_ms", "body_ms", "total_ms", "sent_bytes", "audio_ms", …}
    transcript_timing: Optional[dict] = None
    weight: Optional[float] = None
    margin: Optional[float] = None
    whisper: bool = False
    whisper_score: Optional[float] = None
    speakers: List[dict] = []


class RecognitionListOut(BaseModel):
    events: List[RecognitionEventOut]


class RecognitionStatsOut(BaseModel):
    window_seconds: float
    per_outcome: dict
    per_speaker: dict


class ClearedOut(BaseModel):
    deleted: int


@router.get("", response_model=RecognitionListOut)
async def list_events(
    limit: int = Query(default=100, ge=1, le=1000),
    outcome: Optional[str] = Query(default=None),
    speaker: Optional[str] = Query(default=None),
    ctx: AppContext = Depends(get_context),
) -> RecognitionListOut:
    events = ctx.recognition.list_events(
        limit=limit, outcome=outcome, speaker=speaker
    )
    # Link each event to its captured audio (if any) so the UI can offer
    # "assign to speaker" directly on blocked/unknown entries.
    session_map = ctx.unknown.map_sessions_to_samples(
        [e.session_id for e in events]
    )
    return RecognitionListOut(
        events=[
            RecognitionEventOut(
                id=e.id,
                created_at=e.created_at,
                session_id=e.session_id,
                satellite_id=e.satellite_id,
                duration_sec=e.duration_sec,
                outcome=e.outcome,
                matched_speaker=e.matched_speaker,
                distance=e.distance,
                threshold=e.threshold,
                verify_ms=e.verify_ms,
                transcript=e.transcript,
                emotion=e.emotion,
                emotion_confidence=e.emotion_confidence,
                unknown_sample_id=session_map.get(e.session_id),
                shadow_transcript=e.shadow_transcript,
                shadow_engine=e.shadow_engine,
                transcript_ms=e.transcript_ms,
                shadow_ms=e.shadow_ms,
                transcript_timing=e.transcript_timing,
                weight=e.weight,
                margin=e.margin,
                whisper=e.whisper,
                whisper_score=e.whisper_score,
                speakers=e.speakers,
            )
            for e in events
        ]
    )


@router.get("/stats", response_model=RecognitionStatsOut)
async def stats(
    hours: float = Query(default=24.0, gt=0.0, le=24.0 * 30),
    ctx: AppContext = Depends(get_context),
) -> RecognitionStatsOut:
    data = ctx.recognition.stats(since_seconds=hours * 3600.0)
    return RecognitionStatsOut(**data)


@router.delete("", response_model=ClearedOut)
async def clear(ctx: AppContext = Depends(get_context)) -> ClearedOut:
    return ClearedOut(deleted=ctx.recognition.clear())


class TestEventOut(BaseModel):
    ok: bool
    id: int
    message: str


@router.post("/test", response_model=TestEventOut)
async def insert_test_event(
    ctx: AppContext = Depends(get_context),
) -> TestEventOut:
    """Write a synthetic event to the audit log.

    Diagnostic helper so the user can verify the Recognition tab works
    end-to-end without going through the Wyoming pipeline. If this
    button populates the tab but a real voice command does not, we know
    the bug is in the Wyoming handler, not in the logging plumbing.
    """
    import uuid
    row_id = ctx.recognition.record(
        session_id="test-" + uuid.uuid4().hex[:8],
        satellite_id="web-ui-test",
        duration_sec=2.5,
        outcome="match",
        matched_speaker="(test entry)",
        distance=0.0,
        threshold=ctx.get_verify_threshold(),
        verify_ms=0.0,
        transcript="This is a test entry created from the Web UI.",
    )
    if row_id:
        return TestEventOut(
            ok=True,
            id=row_id,
            message=f"Inserted test event #{row_id}",
        )
    return TestEventOut(
        ok=False, id=0, message="Insert failed — check container logs"
    )
