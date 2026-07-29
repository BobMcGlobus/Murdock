"""Runtime settings endpoints."""

from __future__ import annotations

import asyncio
import logging
import os
from typing import List, Optional

import numpy as np

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel, Field

from murdock.core.context import AppContext

from .deps import get_context

_LOGGER = logging.getLogger("murdock.api.settings")

router = APIRouter(prefix="/api/settings", tags=["settings"])


class SettingsOut(BaseModel):
    verify_threshold: float
    # Margin gate (plan §21): min distance between best and second-best
    # speaker before a match counts; 0 = off.
    margin_gate: float = 0.0
    unknown_logging: bool
    unknown_ttl_hours: int
    require_speaker_match: bool
    passthrough_when_no_speakers: bool
    min_verify_seconds: float
    skip_leading_seconds: float
    min_liveness_score: float
    auto_enroll: bool
    enable_extraction: bool = True
    extraction_threshold: float = 0.25
    extraction_min_region_sec: float = 0.6
    enable_calibration: bool = True
    calibration_fitted: bool = False
    calibration_n_genuine: int = 0
    calibration_n_impostor: int = 0
    calibration_fitted_at: float = 0.0
    enable_satellite_profiles: bool = True
    enable_adaptive_thresholds: bool = True
    adaptive_thresholds: dict = {}
    enable_early_reject: bool = False
    early_reject_margin: float = 0.25
    speaker_context_mode: str = "none"  # "none" | "transcript"
    # Where ambiguity markers go: inline | sidecar | clean | auto
    transcript_hint_mode: str = "inline"
    effective_transcript_hint_mode: str = "inline"
    transcript_template_known: str = ""
    transcript_template_unknown: str = ""
    upstream_uri: str
    upstream_uri_default: str
    upstream_uri_source: str  # "env" | "override"
    listen_uri: str
    ha_configured: bool
    ha_url: str = ""
    ha_token_set: bool = False
    ha_input_text_entity: str = "input_text.current_speaker"
    ha_tv_entity: str = ""
    ha_confidence_entity: str = ""
    ha_distance_entity: str = ""
    ha_nearest_entity: str = ""
    ha_role_entity: str = ""
    ha_emotion_entity: str = ""
    stt_backend: str = "upstream"  # "upstream" | "voxtral" | "openai"
    mistral_api_key_set: bool = False
    mistral_model: str = "voxtral-mini-latest"
    # OpenAI-compatible cloud backend (OpenAI, Groq, local speaches, …)
    openai_base_url: str = "https://api.openai.com"
    openai_api_key_set: bool = False
    openai_model: str = "gpt-4o-transcribe"
    # Local Wyoming fallback for cloud backend failures
    stt_local_fallback: bool = False
    # A/B shadow engine (transcript logged, never returned via Wyoming)
    shadow_stt_backend: str = "none"  # none | upstream | voxtral | openai
    shadow_upstream_uri: str = ""
    shadow_mistral_model: str = "voxtral-small-latest"
    shadow_mistral_api_key_set: bool = False
    shadow_openai_base_url: str = ""
    shadow_openai_api_key_set: bool = False
    shadow_openai_model: str = ""
    # Transcript quality tiers
    enable_stt_vocabulary: bool = False
    stt_vocabulary: str = ""
    enable_stt_dictionary: bool = False
    stt_dictionary: str = ""
    enable_dual_transcript: bool = False
    # MQTT integration (recommended over the REST/token path)
    mqtt_enabled: bool = False
    mqtt_host: str = ""
    mqtt_port: int = 1883
    mqtt_username: str = ""
    mqtt_password_set: bool = False
    mqtt_topic_prefix: str = "murdock"
    mqtt_discovery_prefix: str = "homeassistant"
    mqtt_connected: bool = False
    advertised_languages: List[str] = Field(default_factory=list)
    advertised_languages_source: str = "auto"  # "auto" | "override"
    quality_weights: dict = Field(default_factory=dict)
    quality_weights_source: str = "default"  # "default" | "override"
    # Emotion detection status — plumbing-only in this release. `enabled`
    # reflects the user's opt-in; `model_available` reflects whether a
    # usable ONNX file is on disk. The UI shows "no model yet" when the
    # flag is on but the model is missing, preventing confusion about
    # why recognition events lack emotion data.
    enable_emotion: bool = False
    emotion_model_available: bool = False


class SettingsPatch(BaseModel):
    verify_threshold: Optional[float] = Field(default=None, ge=0.0, le=2.0)
    margin_gate: Optional[float] = Field(default=None, ge=0.0, le=1.0)
    unknown_logging: Optional[bool] = None
    require_speaker_match: Optional[bool] = None
    passthrough_when_no_speakers: Optional[bool] = None
    skip_leading_seconds: Optional[float] = Field(default=None, ge=0.0, le=5.0)
    min_liveness_score: Optional[float] = Field(default=None, ge=0.0, le=1.0)
    auto_enroll: Optional[bool] = None
    enable_extraction: Optional[bool] = None
    extraction_threshold: Optional[float] = Field(default=None, ge=0.0, le=2.0)
    extraction_min_region_sec: Optional[float] = Field(default=None, ge=0.0, le=5.0)
    enable_calibration: Optional[bool] = None
    enable_satellite_profiles: Optional[bool] = None
    enable_adaptive_thresholds: Optional[bool] = None
    enable_early_reject: Optional[bool] = None
    early_reject_margin: Optional[float] = Field(default=None, ge=0.05, le=1.0)
    speaker_context_mode: Optional[str] = None  # "none" | "transcript"
    transcript_hint_mode: Optional[str] = None
    transcript_template_known: Optional[str] = None
    transcript_template_unknown: Optional[str] = None
    # None → field not touched
    # ""   → clear override, fall back to env default
    # "…"  → set override (host:port accepted, tcp:// auto-prefixed)
    upstream_uri: Optional[str] = None
    # None  → field not touched
    # []    → clear override, fall back to upstream auto-detect
    # [...] → hard override
    advertised_languages: Optional[List[str]] = None
    # Home Assistant fields (None = not touched, "" = clear)
    ha_url: Optional[str] = None
    ha_token: Optional[str] = None
    ha_input_text_entity: Optional[str] = None
    ha_tv_entity: Optional[str] = None
    ha_confidence_entity: Optional[str] = None
    ha_distance_entity: Optional[str] = None
    ha_nearest_entity: Optional[str] = None
    ha_role_entity: Optional[str] = None
    ha_emotion_entity: Optional[str] = None
    # Emotion detection (experimental). Setting enable_emotion to True while
    # no model is on disk is harmless — the handler gates on both flag AND
    # model availability before attempting inference.
    enable_emotion: Optional[bool] = None
    quality_weights: Optional[dict] = None  # {"speech_ratio":0.25,...} — pass {} to reset
    # STT backend selection
    stt_backend: Optional[str] = None  # "upstream" | "voxtral" | "openai"
    mistral_api_key: Optional[str] = None
    mistral_model: Optional[str] = None
    openai_base_url: Optional[str] = None
    openai_api_key: Optional[str] = None
    openai_model: Optional[str] = None
    stt_local_fallback: Optional[bool] = None
    shadow_stt_backend: Optional[str] = None
    shadow_upstream_uri: Optional[str] = None
    shadow_mistral_model: Optional[str] = None
    shadow_mistral_api_key: Optional[str] = None
    shadow_openai_base_url: Optional[str] = None
    shadow_openai_api_key: Optional[str] = None
    shadow_openai_model: Optional[str] = None
    enable_stt_vocabulary: Optional[bool] = None
    stt_vocabulary: Optional[str] = None
    enable_stt_dictionary: Optional[bool] = None
    stt_dictionary: Optional[str] = None
    enable_dual_transcript: Optional[bool] = None
    # MQTT integration
    mqtt_enabled: Optional[bool] = None
    mqtt_host: Optional[str] = None
    mqtt_port: Optional[int] = Field(default=None, ge=1, le=65535)
    mqtt_username: Optional[str] = None
    mqtt_password: Optional[str] = None
    mqtt_topic_prefix: Optional[str] = None
    mqtt_discovery_prefix: Optional[str] = None


class RestartResponse(BaseModel):
    ok: bool
    message: str


def _build_settings_out(ctx: AppContext) -> SettingsOut:
    override = ctx.get_advertised_languages()
    if override:
        languages = override
        source = "override"
    else:
        # Whatever the cache currently holds; may be [] if upstream is
        # unreachable, in which case the UI will say "auto (none yet)".
        cache = getattr(ctx, "info_cache", None)
        languages = list(getattr(cache, "_languages", []) or [])
        source = "auto"
    return SettingsOut(
        verify_threshold=ctx.get_verify_threshold(),
        margin_gate=ctx.get_margin_gate(),
        unknown_logging=ctx.get_unknown_logging(),
        unknown_ttl_hours=ctx.settings.unknown_ttl_hours,
        require_speaker_match=ctx.get_require_match(),
        passthrough_when_no_speakers=ctx.get_passthrough_when_empty(),
        min_verify_seconds=ctx.settings.min_verify_seconds,
        skip_leading_seconds=ctx.get_skip_leading_seconds(),
        min_liveness_score=ctx.get_min_liveness_score(),
        auto_enroll=ctx.get_auto_enroll(),
        enable_extraction=ctx.get_enable_extraction(),
        extraction_threshold=ctx.get_extraction_threshold(),
        extraction_min_region_sec=ctx.get_extraction_min_region_sec(),
        enable_calibration=ctx.get_enable_calibration(),
        calibration_fitted=ctx.get_calibrator().fitted,
        calibration_n_genuine=ctx.get_calibrator().n_genuine,
        calibration_n_impostor=ctx.get_calibrator().n_impostor,
        calibration_fitted_at=ctx.get_calibrator().fitted_at,
        enable_satellite_profiles=ctx.get_enable_satellite_profiles(),
        enable_adaptive_thresholds=ctx.get_enable_adaptive_thresholds(),
        adaptive_thresholds=ctx.get_adaptive_thresholds(),
        enable_early_reject=ctx.get_enable_early_reject(),
        early_reject_margin=ctx.get_early_reject_margin(),
        speaker_context_mode=ctx.get_speaker_context_mode(),
        transcript_hint_mode=ctx.get_transcript_hint_mode(),
        effective_transcript_hint_mode=ctx.effective_transcript_hint_mode(),
        transcript_template_known=ctx.get_transcript_template_known(),
        transcript_template_unknown=ctx.get_transcript_template_unknown(),
        upstream_uri=ctx.get_upstream_uri(),
        upstream_uri_default=ctx.settings.upstream_uri,
        upstream_uri_source=ctx.get_upstream_uri_source(),
        listen_uri=ctx.settings.listen_uri,
        stt_backend=ctx.get_stt_backend(),
        mistral_api_key_set=ctx.has_mistral_api_key(),
        mistral_model=ctx.get_mistral_model(),
        openai_base_url=ctx.get_openai_base_url(),
        openai_api_key_set=ctx.has_openai_api_key(),
        openai_model=ctx.get_openai_model(),
        stt_local_fallback=ctx.get_stt_local_fallback(),
        shadow_stt_backend=ctx.get_shadow_stt_backend(),
        shadow_upstream_uri=ctx.get_shadow_upstream_uri(),
        shadow_mistral_model=ctx.get_shadow_mistral_model(),
        shadow_mistral_api_key_set=ctx.has_shadow_mistral_api_key(),
        shadow_openai_base_url=ctx.get_shadow_openai_base_url(),
        shadow_openai_api_key_set=ctx.has_shadow_openai_api_key(),
        shadow_openai_model=ctx.get_shadow_openai_model(),
        enable_stt_vocabulary=ctx.get_enable_stt_vocabulary(),
        stt_vocabulary=ctx.get_stt_vocabulary(),
        enable_stt_dictionary=ctx.get_enable_stt_dictionary(),
        stt_dictionary=ctx.get_stt_dictionary(),
        enable_dual_transcript=ctx.get_enable_dual_transcript(),
        mqtt_enabled=ctx.get_mqtt_enabled(),
        mqtt_host=ctx.get_mqtt_host(),
        mqtt_port=ctx.get_mqtt_port(),
        mqtt_username=ctx.get_mqtt_username(),
        mqtt_password_set=ctx.has_mqtt_password(),
        mqtt_topic_prefix=ctx.get_mqtt_topic_prefix(),
        mqtt_discovery_prefix=ctx.get_mqtt_discovery_prefix(),
        mqtt_connected=ctx.mqtt.connected,
        ha_configured=ctx.ha.configured,
        ha_url=ctx.get_ha_url(),
        ha_token_set=ctx.has_ha_token(),
        ha_input_text_entity=ctx.get_ha_input_text_entity(),
        ha_tv_entity=ctx.get_ha_tv_entity(),
        ha_confidence_entity=ctx.get_ha_confidence_entity(),
        ha_distance_entity=ctx.get_ha_distance_entity(),
        ha_nearest_entity=ctx.get_ha_nearest_entity(),
        ha_role_entity=ctx.get_ha_role_entity(),
        ha_emotion_entity=ctx.get_ha_emotion_entity(),
        advertised_languages=languages,
        advertised_languages_source=source,
        quality_weights=ctx.get_quality_weights(),
        quality_weights_source=ctx.get_quality_weights_source(),
        enable_emotion=ctx.get_enable_emotion(),
        emotion_model_available=ctx.emotion_model_available(),
    )


@router.get("", response_model=SettingsOut)
async def get_settings_route(ctx: AppContext = Depends(get_context)):
    return _build_settings_out(ctx)


@router.patch("", response_model=SettingsOut)
async def patch_settings(
    body: SettingsPatch, ctx: AppContext = Depends(get_context)
):
    has_field = any(
        getattr(body, f) is not None
        for f in body.model_fields
    )
    if not has_field:
        raise HTTPException(status_code=400, detail="No fields to update")
    if body.verify_threshold is not None:
        ctx.set_verify_threshold(body.verify_threshold)
    if body.margin_gate is not None:
        ctx.set_margin_gate(body.margin_gate)
    if body.unknown_logging is not None:
        ctx.set_unknown_logging(body.unknown_logging)
    if body.require_speaker_match is not None:
        ctx.set_require_match(body.require_speaker_match)
    if body.passthrough_when_no_speakers is not None:
        ctx.set_passthrough_when_empty(body.passthrough_when_no_speakers)
    if body.skip_leading_seconds is not None:
        ctx.set_skip_leading_seconds(body.skip_leading_seconds)
    if body.min_liveness_score is not None:
        ctx.set_min_liveness_score(body.min_liveness_score)
    if body.auto_enroll is not None:
        ctx.set_auto_enroll(body.auto_enroll)
    if body.enable_extraction is not None:
        ctx.set_enable_extraction(body.enable_extraction)
    if body.extraction_threshold is not None:
        ctx.set_extraction_threshold(body.extraction_threshold)
    if body.extraction_min_region_sec is not None:
        ctx.set_extraction_min_region_sec(body.extraction_min_region_sec)
    if body.enable_calibration is not None:
        ctx.set_enable_calibration(body.enable_calibration)
    if body.enable_satellite_profiles is not None:
        ctx.set_enable_satellite_profiles(body.enable_satellite_profiles)
    if body.enable_adaptive_thresholds is not None:
        ctx.set_enable_adaptive_thresholds(body.enable_adaptive_thresholds)
    if body.enable_early_reject is not None:
        ctx.set_enable_early_reject(body.enable_early_reject)
    if body.early_reject_margin is not None:
        ctx.set_early_reject_margin(body.early_reject_margin)
    if body.speaker_context_mode is not None:
        try:
            ctx.set_speaker_context_mode(body.speaker_context_mode)
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc))
    if body.transcript_hint_mode is not None:
        try:
            ctx.set_transcript_hint_mode(body.transcript_hint_mode)
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc))
    if body.transcript_template_known is not None:
        ctx.set_transcript_template_known(body.transcript_template_known)
    if body.transcript_template_unknown is not None:
        ctx.set_transcript_template_unknown(body.transcript_template_unknown)
    if body.upstream_uri is not None:
        ctx.set_upstream_uri(body.upstream_uri)
    if body.advertised_languages is not None:
        # An empty list means "clear override" — fall through to
        # upstream auto-detect. A non-empty list is a hard override.
        ctx.set_advertised_languages(body.advertised_languages or None)
    # Home Assistant settings — reconfigure the live client when changed.
    ha_changed = False
    if body.ha_url is not None:
        ctx.set_ha_url(body.ha_url)
        ha_changed = True
    if body.ha_token is not None:
        ctx.set_ha_token(body.ha_token)
        ha_changed = True
    if body.ha_input_text_entity is not None:
        ctx.set_ha_input_text_entity(body.ha_input_text_entity)
        ha_changed = True
    if body.ha_tv_entity is not None:
        ctx.set_ha_tv_entity(body.ha_tv_entity)
        ha_changed = True
    if body.ha_confidence_entity is not None:
        ctx.set_ha_confidence_entity(body.ha_confidence_entity)
        ha_changed = True
    if body.ha_distance_entity is not None:
        ctx.set_ha_distance_entity(body.ha_distance_entity)
        ha_changed = True
    if body.ha_nearest_entity is not None:
        ctx.set_ha_nearest_entity(body.ha_nearest_entity)
        ha_changed = True
    if body.ha_role_entity is not None:
        ctx.set_ha_role_entity(body.ha_role_entity)
        ha_changed = True
    if body.ha_emotion_entity is not None:
        ctx.set_ha_emotion_entity(body.ha_emotion_entity)
        ha_changed = True
    if ha_changed:
        await ctx.apply_ha_settings()
    if body.enable_emotion is not None:
        ctx.set_enable_emotion(body.enable_emotion)
    if body.quality_weights is not None:
        # Empty dict resets to defaults.
        if body.quality_weights == {}:
            ctx.set_quality_weights(None)
        else:
            ctx.set_quality_weights(body.quality_weights)
    # STT backend selection
    if body.stt_backend is not None:
        try:
            ctx.set_stt_backend(body.stt_backend)
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc))
    if body.openai_base_url is not None:
        ctx.set_openai_base_url(body.openai_base_url)
    if body.openai_api_key is not None:
        ctx.set_openai_api_key(body.openai_api_key)
    if body.openai_model is not None:
        ctx.set_openai_model(body.openai_model)
    if body.stt_local_fallback is not None:
        ctx.set_stt_local_fallback(body.stt_local_fallback)
    if body.shadow_stt_backend is not None:
        try:
            ctx.set_shadow_stt_backend(body.shadow_stt_backend)
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc))
    if body.shadow_upstream_uri is not None:
        ctx.set_shadow_upstream_uri(body.shadow_upstream_uri)
    if body.shadow_mistral_model is not None:
        ctx.set_shadow_mistral_model(body.shadow_mistral_model)
    if body.shadow_mistral_api_key is not None:
        ctx.set_shadow_mistral_api_key(body.shadow_mistral_api_key)
    if body.shadow_openai_base_url is not None:
        ctx.set_shadow_openai_base_url(body.shadow_openai_base_url)
    if body.shadow_openai_api_key is not None:
        ctx.set_shadow_openai_api_key(body.shadow_openai_api_key)
    if body.shadow_openai_model is not None:
        ctx.set_shadow_openai_model(body.shadow_openai_model)
    if body.enable_stt_vocabulary is not None:
        ctx.set_enable_stt_vocabulary(body.enable_stt_vocabulary)
    if body.stt_vocabulary is not None:
        ctx.set_stt_vocabulary(body.stt_vocabulary)
    if body.enable_stt_dictionary is not None:
        ctx.set_enable_stt_dictionary(body.enable_stt_dictionary)
    if body.stt_dictionary is not None:
        ctx.set_stt_dictionary(body.stt_dictionary)
    if body.enable_dual_transcript is not None:
        ctx.set_enable_dual_transcript(body.enable_dual_transcript)
    if body.mistral_api_key is not None:
        ctx.set_mistral_api_key(body.mistral_api_key)
    if body.mistral_model is not None:
        ctx.set_mistral_model(body.mistral_model)
    # MQTT settings — restart the client when any of them change.
    mqtt_changed = False
    if body.mqtt_enabled is not None:
        ctx.set_mqtt_enabled(body.mqtt_enabled)
        mqtt_changed = True
    if body.mqtt_host is not None:
        ctx.set_mqtt_host(body.mqtt_host)
        mqtt_changed = True
    if body.mqtt_port is not None:
        ctx.set_mqtt_port(body.mqtt_port)
        mqtt_changed = True
    if body.mqtt_username is not None:
        ctx.set_mqtt_username(body.mqtt_username)
        mqtt_changed = True
    if body.mqtt_password is not None:
        ctx.set_mqtt_password(body.mqtt_password)
        mqtt_changed = True
    if body.mqtt_topic_prefix is not None:
        ctx.set_mqtt_topic_prefix(body.mqtt_topic_prefix)
        mqtt_changed = True
    if body.mqtt_discovery_prefix is not None:
        ctx.set_mqtt_discovery_prefix(body.mqtt_discovery_prefix)
        mqtt_changed = True
    if mqtt_changed:
        await ctx.apply_mqtt_settings()
    return _build_settings_out(ctx)


class UpstreamPingOut(BaseModel):
    ok: bool
    upstream_uri: str
    latency_ms: Optional[float] = None
    languages: List[str] = Field(default_factory=list)
    error: Optional[str] = None


@router.post("/ping-upstream", response_model=UpstreamPingOut)
async def ping_upstream(ctx: AppContext = Depends(get_context)):
    """Open a Wyoming connection to the upstream STT and describe it.

    Pure diagnostic endpoint: lets the user verify from the Settings tab
    that Murdock can actually reach its configured upstream, instead of
    having to interpret the container logs.
    """
    import time as _time

    from wyoming.client import AsyncClient
    from wyoming.info import Describe, Info

    # Use the LIVE URI (UI override > compose default), not the static
    # env value, so the button actually pings what the handler would.
    upstream_uri = ctx.get_upstream_uri()
    t0 = _time.monotonic()
    try:
        async with AsyncClient.from_uri(upstream_uri) as client:
            await client.write_event(Describe().event())
            # Bounded wait: the upstream should respond within a second.
            deadline = _time.monotonic() + 5.0
            langs: List[str] = []
            while _time.monotonic() < deadline:
                event = await asyncio.wait_for(
                    client.read_event(),
                    timeout=max(0.1, deadline - _time.monotonic()),
                )
                if event is None:
                    break
                if Info.is_type(event.type):
                    info = Info.from_event(event)
                    for asr in info.asr:
                        for model in asr.models:
                            for lang in model.languages:
                                if lang and lang not in langs:
                                    langs.append(lang)
                    break
            latency_ms = (_time.monotonic() - t0) * 1000
            return UpstreamPingOut(
                ok=True,
                upstream_uri=upstream_uri,
                latency_ms=latency_ms,
                languages=langs,
            )
    except Exception as exc:
        _LOGGER.warning("ping-upstream failed: %s", exc)
        return UpstreamPingOut(
            ok=False, upstream_uri=upstream_uri, error=str(exc)
        )


@router.post("/refresh-languages", response_model=SettingsOut)
async def refresh_languages(ctx: AppContext = Depends(get_context)):
    """Force a fresh Describe round-trip against the upstream STT.

    Lets the admin nudge Murdock to re-query the upstream without
    restarting the container — useful after fixing DNS, starting the
    STT container, etc.
    """
    cache = getattr(ctx, "info_cache", None)
    if cache is None:
        raise HTTPException(status_code=503, detail="info cache not ready")
    try:
        await cache.get_languages(force=True)
    except Exception as exc:
        _LOGGER.exception("Manual language refresh failed")
        raise HTTPException(
            status_code=502, detail=f"upstream refresh failed: {exc}"
        ) from exc
    return _build_settings_out(ctx)


class ThresholdRecommendationOut(BaseModel):
    status: str  # "ok" | "overlap" | "insufficient_data"
    current: float
    recommended: Optional[float] = None
    genuine_count: int = 0
    impostor_count: int = 0
    genuine_p95: Optional[float] = None
    impostor_p05: Optional[float] = None
    separation: Optional[float] = None
    window_hours: float = 0.0


def recommend_threshold(
    genuine: List[float],
    impostor: List[float],
    current: float,
    min_genuine: int = 10,
    min_impostor: int = 5,
) -> dict:
    """Derive a verify-threshold suggestion from observed distances.

    genuine = distances of accepted matches, impostor = best distances of
    blocked/unknown utterances. The suggestion is the midpoint between
    the genuine 95th percentile and the impostor 5th percentile — i.e.
    the middle of the gap between "your voice on a bad day" and "the
    closest stranger". A negative separation means the distributions
    overlap; the midpoint then balances false rejects against false
    accepts, flagged so the UI can warn.
    """
    if len(genuine) < min_genuine or len(impostor) < min_impostor:
        return {
            "status": "insufficient_data",
            "current": current,
            "genuine_count": len(genuine),
            "impostor_count": len(impostor),
        }
    g95 = float(np.percentile(np.asarray(genuine, dtype=np.float64), 95))
    i05 = float(np.percentile(np.asarray(impostor, dtype=np.float64), 5))
    separation = i05 - g95
    recommended = max(0.05, min(1.5, (g95 + i05) / 2.0))
    return {
        "status": "ok" if separation > 0 else "overlap",
        "current": current,
        "recommended": round(recommended, 3),
        "genuine_count": len(genuine),
        "impostor_count": len(impostor),
        "genuine_p95": round(g95, 4),
        "impostor_p05": round(i05, 4),
        "separation": round(separation, 4),
    }


@router.get("/threshold-recommendation", response_model=ThresholdRecommendationOut)
async def threshold_recommendation(
    hours: float = 24.0 * 30,
    ctx: AppContext = Depends(get_context),
):
    """Suggest a verify threshold from the recognition log's distances.

    Empirical replacement for tuning by feel: matches give the genuine
    distance distribution, blocked/unknown events the impostor one.
    """
    import time as _time

    from murdock.core.recognition_log import (
        OUTCOME_BLOCKED_NO_MATCH,
        OUTCOME_MATCH,
        OUTCOME_UNKNOWN_FORWARDED,
    )

    cutoff = _time.time() - hours * 3600.0
    rows = ctx.db.execute(
        "SELECT outcome, distance FROM recognition_events "
        "WHERE distance IS NOT NULL AND created_at > ?",
        (cutoff,),
    ).fetchall()
    genuine = [float(r["distance"]) for r in rows if r["outcome"] == OUTCOME_MATCH]
    impostor = [
        float(r["distance"]) for r in rows
        if r["outcome"] in (OUTCOME_BLOCKED_NO_MATCH, OUTCOME_UNKNOWN_FORWARDED)
    ]
    data = recommend_threshold(genuine, impostor, ctx.get_verify_threshold())
    data["window_hours"] = hours
    return ThresholdRecommendationOut(**data)


class CalibrationOut(BaseModel):
    fitted: bool
    n_genuine: int
    n_impostor: int
    fitted_at: float
    a: float
    b: float
    adaptive_thresholds: dict = {}


@router.post("/recalibrate", response_model=CalibrationOut)
async def recalibrate(ctx: AppContext = Depends(get_context)):
    """Re-fit confidence calibration from the current enrollments now.

    Synchronous (the caller asked for it) but offloaded to a thread so the
    event loop keeps serving. Re-embeds every stored sample, so it can
    take a few seconds with many speakers. Also recomputes the adaptive
    per-speaker thresholds from the same pass.
    """
    calibrator = await asyncio.to_thread(ctx.recalibrate)
    return CalibrationOut(
        fitted=calibrator.fitted,
        n_genuine=calibrator.n_genuine,
        n_impostor=calibrator.n_impostor,
        fitted_at=calibrator.fitted_at,
        a=calibrator.a,
        b=calibrator.b,
        adaptive_thresholds=ctx.get_adaptive_thresholds(),
    )


class HATestOut(BaseModel):
    ok: bool
    error: Optional[str] = None


@router.post("/test-ha", response_model=HATestOut)
async def test_ha(ctx: AppContext = Depends(get_context)):
    """Quick connectivity check against the configured HA instance."""
    if not ctx.ha.configured:
        return HATestOut(ok=False, error="not configured (URL or token missing)")
    try:
        import httpx
        async with httpx.AsyncClient(
            base_url=ctx.ha.base_url or "",
            headers={
                "Authorization": f"Bearer {ctx.ha.token or ''}",
                "Content-Type": "application/json",
            },
            timeout=5.0,
        ) as client:
            resp = await client.get("/api/")
            if resp.status_code == 200:
                return HATestOut(ok=True)
            return HATestOut(ok=False, error=f"HTTP {resp.status_code}")
    except Exception as exc:
        return HATestOut(ok=False, error=str(exc))


class MQTTTestOut(BaseModel):
    ok: bool
    connected: bool = False
    error: Optional[str] = None


@router.post("/test-mqtt", response_model=MQTTTestOut)
async def test_mqtt(ctx: AppContext = Depends(get_context)):
    """Open a short-lived MQTT connection to verify broker reachability.

    Independent of the live client so the user can validate creds before
    flipping the enable switch. Uses aiomqtt directly with a timeout.
    """
    host = ctx.get_mqtt_host()
    if not host:
        return MQTTTestOut(ok=False, error="no broker host configured")
    try:
        import aiomqtt
    except ImportError:
        return MQTTTestOut(ok=False, error="aiomqtt not installed")
    try:
        kwargs: dict = {
            "hostname": host,
            "port": ctx.get_mqtt_port(),
            "keepalive": 10,
        }
        if ctx.get_mqtt_username():
            kwargs["username"] = ctx.get_mqtt_username()
        if ctx.get_mqtt_password():
            kwargs["password"] = ctx.get_mqtt_password()

        async def _probe() -> None:
            async with aiomqtt.Client(**kwargs):
                return

        await asyncio.wait_for(_probe(), timeout=8.0)
        return MQTTTestOut(ok=True, connected=ctx.mqtt.connected)
    except asyncio.TimeoutError:
        return MQTTTestOut(ok=False, error="connection timed out")
    except Exception as exc:
        return MQTTTestOut(ok=False, error=str(exc))


class MQTTContextOut(BaseModel):
    connected: bool
    context: dict = Field(default_factory=dict)
    active_satellite: Optional[str] = None
    active_satellite_area: Optional[str] = None


@router.get("/mqtt-context", response_model=MQTTContextOut)
async def mqtt_context(ctx: AppContext = Depends(get_context)):
    """Return the context Murdock has received from HA over MQTT.

    Shows what HA has pushed onto the retained ``context/<room>/<key>``
    topics (TV state, presence) and the last announced active satellite.
    Useful for verifying the context-push / satellite automations without
    digging through broker tooling.
    """
    return MQTTContextOut(
        connected=ctx.mqtt.connected,
        context=ctx.mqtt.context_snapshot(),
        active_satellite=ctx.mqtt.get_active_satellite(),
        active_satellite_area=ctx.mqtt.get_active_satellite_area(),
    )


# ----------------------------------------------------------------------
# Per-satellite verify-threshold overrides
# ----------------------------------------------------------------------


class SatelliteThresholdEntry(BaseModel):
    satellite_id: str
    threshold: Optional[float] = None  # None = no override, falls back to global
    seen_events: int = 0               # recognition_events count (for sorting)
    last_seen: Optional[float] = None  # epoch seconds of last event


class SatelliteThresholdList(BaseModel):
    default_threshold: float
    entries: List[SatelliteThresholdEntry]


class SatelliteThresholdPatch(BaseModel):
    satellite_id: str
    # null → clear the override; float → set it
    threshold: Optional[float] = Field(default=None, ge=0.0, le=2.0)


def _list_known_satellites(ctx: AppContext) -> list[tuple[str, int, float]]:
    """Return ``(satellite_id, count, last_seen)`` triples from recognition_events."""
    cur = ctx.db.execute(
        "SELECT satellite_id, COUNT(*) AS n, MAX(created_at) AS last "
        "FROM recognition_events "
        "WHERE satellite_id IS NOT NULL AND satellite_id != '' "
        "GROUP BY satellite_id "
        "ORDER BY n DESC"
    )
    return [(row["satellite_id"], row["n"], row["last"]) for row in cur.fetchall()]


@router.get("/satellite-thresholds", response_model=SatelliteThresholdList)
async def list_satellite_thresholds(ctx: AppContext = Depends(get_context)):
    """Return known satellites + their per-satellite threshold overrides.

    The UI uses this to render one row per satellite the proxy has seen,
    letting the admin tune a per-room threshold without having to know
    the exact IDs in advance.
    """
    overrides = ctx.get_satellite_thresholds()
    known = _list_known_satellites(ctx)
    entries: List[SatelliteThresholdEntry] = []
    covered: set[str] = set()
    for sid, count, last in known:
        entries.append(
            SatelliteThresholdEntry(
                satellite_id=sid,
                threshold=overrides.get(sid),
                seen_events=count,
                last_seen=last,
            )
        )
        covered.add(sid)
    # Also surface overrides for satellites we have no events for yet
    # (e.g. pre-configured manually before the first recognition).
    for sid, th in overrides.items():
        if sid not in covered:
            entries.append(
                SatelliteThresholdEntry(satellite_id=sid, threshold=th)
            )
    return SatelliteThresholdList(
        default_threshold=ctx.get_verify_threshold(),
        entries=entries,
    )


@router.patch("/satellite-thresholds", response_model=SatelliteThresholdList)
async def patch_satellite_threshold(
    body: SatelliteThresholdPatch, ctx: AppContext = Depends(get_context)
):
    """Set or clear a per-satellite threshold override.

    ``threshold = null`` clears the override for that satellite.
    """
    try:
        ctx.set_satellite_threshold(body.satellite_id, body.threshold)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc))
    return await list_satellite_thresholds(ctx)


# ----------------------------------------------------------------------
# Per-satellite margin-gate overrides (plan §21)
# ----------------------------------------------------------------------


class SatelliteMarginEntry(BaseModel):
    satellite_id: str
    margin: Optional[float] = None     # None = no override, falls back to global
    seen_events: int = 0
    last_seen: Optional[float] = None


class SatelliteMarginList(BaseModel):
    default_margin: float
    entries: List[SatelliteMarginEntry]


class SatelliteMarginPatch(BaseModel):
    satellite_id: str
    # null → clear the override; float → set it
    margin: Optional[float] = Field(default=None, ge=0.0, le=1.0)


@router.get("/satellite-margin-gates", response_model=SatelliteMarginList)
async def list_satellite_margin_gates(ctx: AppContext = Depends(get_context)):
    """Known satellites + their per-satellite margin-gate overrides."""
    overrides = ctx.get_satellite_margin_gates()
    known = _list_known_satellites(ctx)
    entries: List[SatelliteMarginEntry] = []
    covered: set[str] = set()
    for sid, count, last in known:
        entries.append(
            SatelliteMarginEntry(
                satellite_id=sid,
                margin=overrides.get(sid),
                seen_events=count,
                last_seen=last,
            )
        )
        covered.add(sid)
    for sid, mg in overrides.items():
        if sid not in covered:
            entries.append(SatelliteMarginEntry(satellite_id=sid, margin=mg))
    return SatelliteMarginList(
        default_margin=ctx.get_margin_gate(),
        entries=entries,
    )


@router.patch("/satellite-margin-gates", response_model=SatelliteMarginList)
async def patch_satellite_margin_gate(
    body: SatelliteMarginPatch, ctx: AppContext = Depends(get_context)
):
    """Set or clear a per-satellite margin-gate override (null clears)."""
    try:
        ctx.set_satellite_margin_gate(body.satellite_id, body.margin)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc))
    return await list_satellite_margin_gates(ctx)


# ----------------------------------------------------------------------
# Media-restriction matrix (per-satellite × per-source threshold deltas)
# ----------------------------------------------------------------------


class MediaSourceEntry(BaseModel):
    entity_id: str
    area: Optional[str] = None
    playing: bool = False


class MediaRestrictionEntry(BaseModel):
    satellite_id: str
    media_entity: str
    delta: float


class MediaRestrictionsOut(BaseModel):
    default_boost: float
    satellites: List[str]
    media: List[MediaSourceEntry]
    restrictions: List[MediaRestrictionEntry]


class MediaRestrictionPatch(BaseModel):
    satellite_id: str
    media_entity: str
    # null clears the cell; a value tightens by that much while playing
    delta: Optional[float] = Field(default=None, ge=0.0, le=2.0)


def _build_media_restrictions(ctx: AppContext) -> MediaRestrictionsOut:
    matrix = ctx.get_media_restrictions()
    entries = [
        MediaRestrictionEntry(satellite_id=sat, media_entity=ent, delta=float(d))
        for sat, cell in matrix.items()
        for ent, d in cell.items()
    ]
    # Satellites: those seen in recognition events plus any in the matrix.
    sats = [sid for sid, _n, _last in _list_known_satellites(ctx)]
    for sat in matrix:
        if sat not in sats:
            sats.append(sat)
    # Media sources: currently-known players plus any referenced in the
    # matrix that aren't currently publishing.
    media = [MediaSourceEntry(**m) for m in ctx.mqtt.known_media()]
    known_ids = {m.entity_id for m in media}
    for cell in matrix.values():
        for ent in cell:
            if ent not in known_ids:
                media.append(MediaSourceEntry(entity_id=ent))
                known_ids.add(ent)
    return MediaRestrictionsOut(
        default_boost=float(ctx.settings.tv_threshold_boost),
        satellites=sats,
        media=media,
        restrictions=entries,
    )


@router.get("/media-restrictions", response_model=MediaRestrictionsOut)
async def list_media_restrictions(ctx: AppContext = Depends(get_context)):
    """Return the per-satellite × per-source restriction matrix.

    Plus the satellites Murdock has seen and the media players currently
    announced over MQTT, so the UI can offer them as rows/columns.
    """
    return _build_media_restrictions(ctx)


@router.patch("/media-restrictions", response_model=MediaRestrictionsOut)
async def patch_media_restriction(
    body: MediaRestrictionPatch, ctx: AppContext = Depends(get_context)
):
    """Set or clear one matrix cell (``delta = null`` clears it)."""
    try:
        ctx.set_media_restriction(body.satellite_id, body.media_entity, body.delta)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc))
    return _build_media_restrictions(ctx)


@router.post("/restart", response_model=RestartResponse)
async def restart_service(ctx: AppContext = Depends(get_context)):
    """Hard-restart the Murdock process so container supervisors (Docker,
    systemd) bring it back up. Returns immediately, then exits after a
    short delay so the HTTP response actually reaches the UI."""

    async def _exit_soon() -> None:
        await asyncio.sleep(0.5)
        _LOGGER.warning("Restart requested via API — exiting process")
        os._exit(0)

    asyncio.create_task(_exit_soon())
    return RestartResponse(
        ok=True, message="restart scheduled — service will exit in 500ms"
    )
