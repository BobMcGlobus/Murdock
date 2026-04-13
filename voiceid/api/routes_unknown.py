"""Unknown-sample review and tagging endpoints."""

from __future__ import annotations

import logging
from typing import List, Optional

from fastapi import APIRouter, Depends, HTTPException, Response
from pydantic import BaseModel

from voiceid.core.audio import decode_wav, to_mono_16k_pcm
from voiceid.core.context import AppContext

from .deps import get_context

_LOGGER = logging.getLogger("voiceid.api.unknown")

router = APIRouter(prefix="/api/unknown", tags=["unknown"])


class UnknownOut(BaseModel):
    id: int
    session_id: str
    satellite_id: Optional[str]
    duration_sec: float
    best_distance: float
    best_speaker: Optional[str]
    liveness_score: Optional[float]
    tag: Optional[str]
    created_at: float


class AssignBody(BaseModel):
    speaker_name: str
    create_if_missing: bool = True


@router.get("", response_model=List[UnknownOut])
async def list_unknown(
    include_tagged: bool = False,
    limit: int = 200,
    ctx: AppContext = Depends(get_context),
):
    return [
        UnknownOut(
            id=s.id,
            session_id=s.session_id,
            satellite_id=s.satellite_id,
            duration_sec=s.duration_sec,
            best_distance=s.best_distance,
            best_speaker=s.best_speaker,
            liveness_score=s.liveness_score,
            tag=s.tag,
            created_at=s.created_at,
        )
        for s in ctx.unknown.list_samples(include_tagged=include_tagged, limit=limit)
    ]


@router.get("/{sample_id}/audio")
async def unknown_audio(sample_id: int, ctx: AppContext = Depends(get_context)):
    audio = ctx.unknown.get_audio(sample_id)
    if audio is None:
        raise HTTPException(status_code=404, detail="Unknown sample not found")
    return Response(content=audio, media_type="audio/wav")


@router.post("/{sample_id}/tag")
async def tag_unknown(
    sample_id: int,
    body: dict,
    ctx: AppContext = Depends(get_context),
):
    tag = body.get("tag")
    if not tag:
        raise HTTPException(status_code=400, detail="Missing 'tag'")
    if not ctx.unknown.tag(sample_id, tag):
        raise HTTPException(status_code=404, detail="Unknown sample not found")
    return {"tagged": True, "tag": tag}


@router.post("/{sample_id}/assign")
async def assign_unknown(
    sample_id: int,
    body: AssignBody,
    ctx: AppContext = Depends(get_context),
):
    """Promote an unknown sample to an enrollment for a speaker."""
    wav = ctx.unknown.get_audio(sample_id)
    if wav is None:
        raise HTTPException(status_code=404, detail="Unknown sample not found")

    existing = ctx.speakers.get_speaker_by_name(body.speaker_name)
    if existing is None and not body.create_if_missing:
        raise HTTPException(status_code=404, detail="Speaker does not exist")

    pcm, rate, width, channels = decode_wav(wav)
    pcm = to_mono_16k_pcm(pcm, rate, width, channels)
    duration = len(pcm) / (16000 * 2)
    try:
        result = ctx.speakers.enroll(
            speaker_name=body.speaker_name,
            pcm_bytes=pcm,
            duration_sec=duration,
            source="unknown",
            filename=f"unknown-{sample_id}.wav",
            skip_vad=True,  # user explicitly approved this sample
        )
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc))

    ctx.unknown.tag(sample_id, f"assigned:{body.speaker_name}")
    return {
        "speaker_id": result.speaker_id,
        "speaker_name": result.speaker_name,
        "total_samples": result.total_samples,
    }


@router.delete("/{sample_id}")
async def delete_unknown(sample_id: int, ctx: AppContext = Depends(get_context)):
    if not ctx.unknown.delete(sample_id):
        raise HTTPException(status_code=404, detail="Unknown sample not found")
    return {"deleted": True}


@router.post("/cleanup")
async def cleanup_now(ctx: AppContext = Depends(get_context)):
    deleted = ctx.unknown.cleanup_expired()
    return {"deleted": deleted}
