"""Speaker management REST endpoints."""

from __future__ import annotations

import logging
from typing import List, Optional

import numpy as np
from fastapi import APIRouter, Depends, File, Form, HTTPException, Response, UploadFile
from pydantic import BaseModel

from murdock.core.audio import decode_audio_any, to_mono_16k_pcm
from murdock.core.context import AppContext
from murdock.core.speaker_store import VALID_ROLES, VALID_SAMPLE_SOURCES

from .deps import get_context

_LOGGER = logging.getLogger("murdock.api.speakers")

router = APIRouter(prefix="/api/speakers", tags=["speakers"])


class SpeakerOut(BaseModel):
    id: int
    name: str
    ha_user_id: Optional[str]
    role: Optional[str]
    enrollment_count: int
    created_at: float
    updated_at: float


class SampleOut(BaseModel):
    id: int
    duration_sec: float
    source: Optional[str] = None
    filename: Optional[str] = None
    created_at: float
    quality_score: Optional[float] = None
    satellite_id: Optional[str] = None


class QualityBreakdownOut(BaseModel):
    composite: float
    speech_ratio_score: float
    snr_score: float
    liveness_score: float
    consistency_score: float
    centroid_distance_score: float
    speech_ratio: float
    snr_db: float
    liveness_raw: float
    consistency_std: float
    centroid_distance: Optional[float] = None


class EnrollResponse(BaseModel):
    speaker_id: int
    speaker_name: str
    sample_id: int
    total_samples: int
    vad_speech_seconds: Optional[float] = None
    vad_speech_ratio: Optional[float] = None
    warnings: List[str] = []
    quality: Optional[QualityBreakdownOut] = None


class TrainingQualityOut(BaseModel):
    speaker_id: int
    training_quality: float
    sample_count: int
    scored_count: int
    avg_sample_score: Optional[float] = None


class RescoreResponse(BaseModel):
    speaker_id: int
    rescored: int


class SpeakerPatch(BaseModel):
    name: Optional[str] = None
    ha_user_id: Optional[str] = None
    role: Optional[str] = None
    clear_ha_user_id: bool = False
    clear_role: bool = False


def _speaker_to_out(s) -> "SpeakerOut":
    return SpeakerOut(
        id=s.id,
        name=s.name,
        ha_user_id=s.ha_user_id,
        role=s.role,
        enrollment_count=s.enrollment_count,
        created_at=s.created_at,
        updated_at=s.updated_at,
    )


@router.get("/roles")
async def list_roles():
    return {"roles": list(VALID_ROLES), "sources": list(VALID_SAMPLE_SOURCES)}


@router.get("", response_model=List[SpeakerOut])
async def list_speakers(ctx: AppContext = Depends(get_context)):
    return [_speaker_to_out(s) for s in ctx.speakers.list_speakers()]


@router.patch("/{speaker_id}", response_model=SpeakerOut)
async def edit_speaker(
    speaker_id: int,
    body: SpeakerPatch,
    ctx: AppContext = Depends(get_context),
):
    if (
        body.name is None
        and body.ha_user_id is None
        and body.role is None
        and not body.clear_ha_user_id
        and not body.clear_role
    ):
        raise HTTPException(status_code=400, detail="No fields to update")
    try:
        speaker = ctx.speakers.update_speaker(
            speaker_id=speaker_id,
            name=body.name,
            ha_user_id=body.ha_user_id,
            role=body.role,
            clear_ha_user_id=body.clear_ha_user_id,
            clear_role=body.clear_role,
        )
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc))
    return _speaker_to_out(speaker)


@router.delete("/{speaker_id}")
async def delete_speaker(speaker_id: int, ctx: AppContext = Depends(get_context)):
    ok = ctx.speakers.delete_speaker(speaker_id)
    if not ok:
        raise HTTPException(status_code=404, detail="Speaker not found")
    return {"deleted": True}


@router.get("/{speaker_id}/samples", response_model=List[SampleOut])
async def list_samples(speaker_id: int, ctx: AppContext = Depends(get_context)):
    speaker = ctx.speakers.get_speaker(speaker_id)
    if not speaker:
        raise HTTPException(status_code=404, detail="Speaker not found")
    return [SampleOut(**row) for row in ctx.speakers.list_samples(speaker_id)]


@router.get("/samples/{sample_id}/audio")
async def get_sample_audio(sample_id: int, ctx: AppContext = Depends(get_context)):
    audio = ctx.speakers.get_sample_audio(sample_id)
    if audio is None:
        raise HTTPException(status_code=404, detail="Sample not found")
    return Response(content=audio, media_type="audio/wav")


@router.delete("/samples/{sample_id}")
async def delete_sample(sample_id: int, ctx: AppContext = Depends(get_context)):
    if not ctx.speakers.delete_sample(sample_id):
        raise HTTPException(status_code=404, detail="Sample not found")
    return {"deleted": True}


@router.post("/enroll", response_model=EnrollResponse)
async def enroll(
    name: str = Form(...),
    ha_user_id: Optional[str] = Form(None),
    role: Optional[str] = Form(None),
    source: Optional[str] = Form(None),
    filename: Optional[str] = Form(None),
    satellite_id: Optional[str] = Form(None),
    audio: UploadFile = File(...),
    ctx: AppContext = Depends(get_context),
):
    """Enroll a new sample for ``name``.

    Accepts WAV files. Browser MediaRecorder typically produces WebM/Opus,
    which should be converted client-side before upload — we keep the
    server simple and dependency-light.
    """
    raw = await audio.read()
    if not raw:
        raise HTTPException(status_code=400, detail="Empty upload")

    try:
        pcm, rate, width, channels = decode_audio_any(raw)
    except ValueError as exc:
        raise HTTPException(status_code=415, detail=str(exc))
    except Exception as exc:
        _LOGGER.exception("Audio decode failed")
        raise HTTPException(
            status_code=415,
            detail=f"Could not decode audio upload: {exc}",
        )

    pcm = to_mono_16k_pcm(pcm, rate, width, channels)
    duration = len(pcm) / (16000 * 2)

    # Default the stored filename to the upload's own name when present, so
    # the samples list shows something useful even if the UI forgot to send
    # an explicit filename field.
    effective_filename = filename or (audio.filename if audio.filename else None)

    try:
        result = ctx.speakers.enroll(
            speaker_name=name.strip(),
            pcm_bytes=pcm,
            duration_sec=duration,
            ha_user_id=ha_user_id,
            role=role,
            source=source,
            filename=effective_filename,
            satellite_id=satellite_id,
        )
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc))
    except Exception as exc:
        _LOGGER.exception("Enrollment failed")
        raise HTTPException(status_code=500, detail=str(exc))

    quality_out: Optional[QualityBreakdownOut] = None
    if result.quality is not None:
        q = result.quality
        quality_out = QualityBreakdownOut(
            composite=q.composite,
            speech_ratio_score=q.speech_ratio_score,
            snr_score=q.snr_score,
            liveness_score=q.liveness_score,
            consistency_score=q.consistency_score,
            centroid_distance_score=q.centroid_distance_score,
            speech_ratio=q.speech_ratio,
            snr_db=q.snr_db,
            liveness_raw=q.liveness_raw,
            consistency_std=q.consistency_std,
            centroid_distance=q.centroid_distance,
        )

    return EnrollResponse(
        speaker_id=result.speaker_id,
        speaker_name=result.speaker_name,
        sample_id=result.sample_id,
        total_samples=result.total_samples,
        vad_speech_seconds=result.vad.speech_seconds if result.vad else None,
        vad_speech_ratio=result.vad.speech_ratio if result.vad else None,
        warnings=result.warnings,
        quality=quality_out,
    )


@router.get("/{speaker_id}/quality", response_model=TrainingQualityOut)
async def speaker_quality(speaker_id: int, ctx: AppContext = Depends(get_context)):
    """Return aggregate training quality for a speaker."""
    speaker = ctx.speakers.get_speaker(speaker_id)
    if not speaker:
        raise HTTPException(status_code=404, detail="Speaker not found")
    samples = ctx.speakers.list_samples(speaker_id)
    scored = [s["quality_score"] for s in samples if s.get("quality_score") is not None]
    training = ctx.speakers.get_speaker_training_quality(speaker_id)
    avg = float(sum(scored) / len(scored)) if scored else None
    return TrainingQualityOut(
        speaker_id=speaker_id,
        training_quality=training,
        sample_count=len(samples),
        scored_count=len(scored),
        avg_sample_score=avg,
    )


@router.post("/{speaker_id}/rescore", response_model=RescoreResponse)
async def rescore_speaker(speaker_id: int, ctx: AppContext = Depends(get_context)):
    """Recompute quality scores for all samples of this speaker."""
    speaker = ctx.speakers.get_speaker(speaker_id)
    if not speaker:
        raise HTTPException(status_code=404, detail="Speaker not found")
    import asyncio
    count = await asyncio.to_thread(ctx.speakers.rescore_all_samples, speaker_id)
    return RescoreResponse(speaker_id=speaker_id, rescored=count)


@router.post("/rescore-all", response_model=List[RescoreResponse])
async def rescore_all(ctx: AppContext = Depends(get_context)):
    """Recompute quality scores for all samples of all speakers."""
    import asyncio
    results: List[RescoreResponse] = []
    for speaker in ctx.speakers.list_speakers():
        count = await asyncio.to_thread(
            ctx.speakers.rescore_all_samples, speaker.id
        )
        results.append(RescoreResponse(speaker_id=speaker.id, rescored=count))
    return results


@router.post("/verify")
async def verify_audio(
    audio: UploadFile = File(...),
    ctx: AppContext = Depends(get_context),
):
    """Ad-hoc verification endpoint — useful for threshold tuning from the UI."""
    raw = await audio.read()
    try:
        pcm, rate, width, channels = decode_audio_any(raw)
    except ValueError as exc:
        raise HTTPException(status_code=415, detail=str(exc))
    except Exception as exc:
        _LOGGER.exception("Audio decode failed")
        raise HTTPException(status_code=415, detail=f"Could not decode audio: {exc}")
    pcm = to_mono_16k_pcm(pcm, rate, width, channels)
    result = ctx.speakers.verify_pcm(pcm)
    return {
        "is_match": result.is_match,
        "matched_speaker": result.matched_speaker,
        "distance": result.distance,
        "threshold": result.threshold,
        "all_distances": result.all_distances,
    }


@router.post("/debug/vad")
async def debug_vad(
    audio: UploadFile = File(...),
    ctx: AppContext = Depends(get_context),
):
    """Run Silero VAD on a sample and return raw diagnostics.

    Useful when enrollment keeps reporting "0% speech" — this surfaces
    the actual model spec, the audio statistics, and the per-window
    probabilities so a misconfiguration is easy to spot.
    """
    if ctx.vad is None:
        raise HTTPException(status_code=503, detail="VAD not loaded")

    raw = await audio.read()
    try:
        pcm, rate, width, channels = decode_audio_any(raw)
    except ValueError as exc:
        raise HTTPException(status_code=415, detail=str(exc))
    pcm = to_mono_16k_pcm(pcm, rate, width, channels)

    from murdock.core.audio import pcm16_bytes_to_float32, rms_dbfs

    audio_f32 = pcm16_bytes_to_float32(pcm)
    duration = len(audio_f32) / 16000.0
    peak = float(np.max(np.abs(audio_f32))) if audio_f32.size else 0.0
    dbfs = rms_dbfs(pcm)

    result = ctx.vad.analyze_pcm(pcm)

    # Reach into the VAD wrapper for the model spec.
    try:
        ctx.vad._ensure_session()  # type: ignore[attr-defined]
        spec = {
            "input_names": ctx.vad._input_names,  # type: ignore[attr-defined]
            "output_names": ctx.vad._output_names,  # type: ignore[attr-defined]
            "is_v5": ctx.vad._is_v5,  # type: ignore[attr-defined]
        }
    except Exception as exc:
        spec = {"error": str(exc)}

    return {
        "audio": {
            "decoded_bytes": len(pcm),
            "duration_sec": duration,
            "peak_amplitude": peak,
            "rms_dbfs": dbfs,
            "input_rate": rate,
            "input_channels": channels,
        },
        "model": spec,
        "result": {
            "speech_seconds": result.speech_seconds,
            "speech_ratio": result.speech_ratio,
            "peak_probability": result.peak_probability,
            "segments": result.segments,
        },
    }
