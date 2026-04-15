"""Unknown-sample review and tagging endpoints."""

from __future__ import annotations

import logging
from typing import List, Optional

from fastapi import APIRouter, Depends, HTTPException, Response
from pydantic import BaseModel

from voiceid.core.audio import decode_wav, to_mono_16k_pcm
from voiceid.core.context import AppContext
from voiceid.core.unknown_cluster import (
    DEFAULT_CLUSTER_THRESHOLD,
    cluster_unknown_samples,
)

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


class ClusterMemberOut(BaseModel):
    sample_id: int
    distance_to_centroid: float
    duration_sec: float
    best_speaker: Optional[str]
    best_distance: float
    liveness_score: Optional[float]
    satellite_id: Optional[str]
    created_at: float
    tag: Optional[str]


class ClusterOut(BaseModel):
    cluster_id: int
    size: int
    avg_distance: float
    satellites: List[str]
    members: List[ClusterMemberOut]


class ClusterListOut(BaseModel):
    threshold: float
    clusters: List[ClusterOut]


class BulkAssignBody(BaseModel):
    speaker_name: str
    create_if_missing: bool = True
    sample_ids: List[int]


class BulkAssignResult(BaseModel):
    speaker_id: Optional[int]
    speaker_name: str
    assigned: int
    skipped: int
    total_samples: Optional[int]
    errors: List[str] = []


@router.get("/clusters", response_model=ClusterListOut)
async def unknown_clusters(
    threshold: float = DEFAULT_CLUSTER_THRESHOLD,
    include_tagged: bool = False,
    limit: int = 500,
    ctx: AppContext = Depends(get_context),
):
    """Greedy-cluster untagged unknown samples by voice similarity.

    The UI uses this to let the admin label five recordings of the same
    unknown voice in one click instead of five.
    """
    if threshold < 0.0 or threshold > 1.5:
        raise HTTPException(
            status_code=400, detail="threshold must be between 0.0 and 1.5"
        )
    clusters = cluster_unknown_samples(
        ctx.unknown,
        threshold=threshold,
        include_tagged=include_tagged,
        limit=limit,
    )
    return ClusterListOut(
        threshold=threshold,
        clusters=[
            ClusterOut(
                cluster_id=c.cluster_id,
                size=c.size,
                avg_distance=c.avg_distance,
                satellites=c.satellites,
                members=[
                    ClusterMemberOut(
                        sample_id=m.sample_id,
                        distance_to_centroid=m.distance_to_centroid,
                        duration_sec=m.duration_sec,
                        best_speaker=m.best_speaker,
                        best_distance=m.best_distance,
                        liveness_score=m.liveness_score,
                        satellite_id=m.satellite_id,
                        created_at=m.created_at,
                        tag=m.tag,
                    )
                    for m in c.members
                ],
            )
            for c in clusters
        ],
    )


@router.post("/bulk-assign", response_model=BulkAssignResult)
async def bulk_assign_unknown(
    body: BulkAssignBody,
    ctx: AppContext = Depends(get_context),
):
    """Promote multiple unknown samples to the same speaker in one call.

    Used by the cluster UI: click *Assign cluster → Anna* and every
    sample in the cluster gets enrolled. Errors on individual samples
    are collected but don't fail the whole batch — we'd rather import 4
    of 5 than roll everything back because one WAV is corrupt.
    """
    name = body.speaker_name.strip()
    if not name:
        raise HTTPException(status_code=400, detail="speaker_name is required")
    if not body.sample_ids:
        raise HTTPException(status_code=400, detail="sample_ids must not be empty")

    existing = ctx.speakers.get_speaker_by_name(name)
    if existing is None and not body.create_if_missing:
        raise HTTPException(status_code=404, detail="Speaker does not exist")

    assigned = 0
    skipped = 0
    errors: List[str] = []
    speaker_id: Optional[int] = existing.id if existing else None
    total_samples: Optional[int] = None

    for sid in body.sample_ids:
        wav = ctx.unknown.get_audio(sid)
        if wav is None:
            skipped += 1
            errors.append(f"#{sid}: not found")
            continue
        try:
            pcm, rate, width, channels = decode_wav(wav)
            pcm = to_mono_16k_pcm(pcm, rate, width, channels)
            duration = len(pcm) / (16000 * 2)
            result = ctx.speakers.enroll(
                speaker_name=name,
                pcm_bytes=pcm,
                duration_sec=duration,
                source="unknown",
                filename=f"unknown-{sid}.wav",
                skip_vad=True,  # user explicitly clustered these together
            )
            ctx.unknown.tag(sid, f"assigned:{name}")
            assigned += 1
            speaker_id = result.speaker_id
            total_samples = result.total_samples
        except Exception as exc:  # pragma: no cover — keep batch resilient
            skipped += 1
            errors.append(f"#{sid}: {exc}")
            _LOGGER.warning("Bulk-assign failed for sample %s: %s", sid, exc)

    return BulkAssignResult(
        speaker_id=speaker_id,
        speaker_name=name,
        assigned=assigned,
        skipped=skipped,
        total_samples=total_samples,
        errors=errors,
    )


@router.delete("/{sample_id}")
async def delete_unknown(sample_id: int, ctx: AppContext = Depends(get_context)):
    if not ctx.unknown.delete(sample_id):
        raise HTTPException(status_code=404, detail="Unknown sample not found")
    return {"deleted": True}


@router.post("/cleanup")
async def cleanup_now(ctx: AppContext = Depends(get_context)):
    deleted = ctx.unknown.cleanup_expired()
    return {"deleted": deleted}
