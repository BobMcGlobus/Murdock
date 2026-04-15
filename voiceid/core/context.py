"""Shared runtime context: stores, embedder, VAD, and HA client.

Both the Wyoming proxy and the FastAPI web UI import from this module so
that they share a single database connection and set of models.
"""

from __future__ import annotations

import logging
import sqlite3
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, List, Optional

from ..config import Settings, get_settings
from .db import get_setting, open_db, set_setting
from .embeddings import CAMPPlusEmbedder
from .emotion import EmotionClassifier
from .ha_integration import HomeAssistantClient
from .recognition_log import RecognitionLog
from .speaker_store import SpeakerStore
from .unknown_store import UnknownStore
from .vad import SileroVAD

if TYPE_CHECKING:
    from .info_cache import UpstreamInfoCache


def _normalize_wyoming_uri(value: str) -> str:
    """Accept sloppy inputs and turn them into valid Wyoming URIs.

    Users routinely forget the ``tcp://`` prefix and just paste a bare
    ``host:port`` from their router or a docker-compose file. Silently
    adding the scheme is much more forgiving than failing with a cryptic
    "unknown URI scheme" error deep inside the Wyoming client.
    """
    value = (value or "").strip()
    if not value:
        return ""
    if "://" in value:
        return value
    return f"tcp://{value}"

_LOGGER = logging.getLogger("voiceid.context")


@dataclass
class AppContext:
    settings: Settings
    db: sqlite3.Connection
    embedder: CAMPPlusEmbedder
    vad: Optional[SileroVAD]
    speakers: SpeakerStore
    unknown: UnknownStore
    ha: HomeAssistantClient
    recognition: RecognitionLog
    emotion: Optional[EmotionClassifier] = None
    info_cache: "Optional[UpstreamInfoCache]" = None
    _overrides: dict = field(default_factory=dict)

    # ------------------------------------------------------------------
    # Runtime-configurable settings (persisted in the settings table)
    # ------------------------------------------------------------------

    def get_verify_threshold(self, satellite_id: Optional[str] = None) -> float:
        """Return the effective verify threshold.

        If ``satellite_id`` is given and a per-satellite override is set,
        that wins over the global override, which in turn wins over the
        env/config default. Keeping the fallback chain explicit (satellite
        → global → env) avoids surprising behaviour when someone sets a
        global override and then a per-satellite one from a noisy room.
        """
        if satellite_id:
            sat_override = get_setting(self.db, f"threshold_sat_{satellite_id}")
            if sat_override:
                try:
                    return float(sat_override)
                except ValueError:
                    pass
        override = get_setting(self.db, "verify_threshold")
        if override is not None:
            try:
                return float(override)
            except ValueError:
                pass
        return self.settings.verify_threshold

    def set_verify_threshold(self, value: float) -> None:
        set_setting(self.db, "verify_threshold", str(value))
        self.speakers.threshold = value

    def get_satellite_thresholds(self) -> dict:
        """Return ``{satellite_id: threshold}`` for all explicitly set overrides."""
        cur = self.db.execute(
            "SELECT key, value FROM settings WHERE key LIKE 'threshold_sat_%' AND value != ''"
        )
        out: dict = {}
        for key, value in cur.fetchall():
            sid = key[len("threshold_sat_"):]
            if not sid or not value:
                continue
            try:
                out[sid] = float(value)
            except (TypeError, ValueError):
                continue
        return out

    def set_satellite_threshold(self, satellite_id: str, value: Optional[float]) -> None:
        """Persist (or clear) a per-satellite threshold override."""
        sid = (satellite_id or "").strip()
        if not sid:
            raise ValueError("satellite_id is required")
        if value is None:
            set_setting(self.db, f"threshold_sat_{sid}", "")
        else:
            set_setting(self.db, f"threshold_sat_{sid}", f"{float(value):.4f}")

    def get_unknown_logging(self) -> bool:
        override = get_setting(self.db, "unknown_logging")
        if override is not None:
            return override.lower() in ("1", "true", "yes", "on")
        return self.settings.unknown_logging

    def set_unknown_logging(self, enabled: bool) -> None:
        set_setting(self.db, "unknown_logging", "true" if enabled else "false")

    def get_require_match(self) -> bool:
        override = get_setting(self.db, "require_speaker_match")
        if override is not None:
            return override.lower() in ("1", "true", "yes", "on")
        return self.settings.require_speaker_match

    def set_require_match(self, enabled: bool) -> None:
        set_setting(
            self.db, "require_speaker_match", "true" if enabled else "false"
        )

    def get_passthrough_when_empty(self) -> bool:
        override = get_setting(self.db, "passthrough_when_no_speakers")
        if override is not None:
            return override.lower() in ("1", "true", "yes", "on")
        return self.settings.passthrough_when_no_speakers

    def set_passthrough_when_empty(self, enabled: bool) -> None:
        set_setting(
            self.db,
            "passthrough_when_no_speakers",
            "true" if enabled else "false",
        )

    def get_skip_leading_seconds(self) -> float:
        override = get_setting(self.db, "skip_leading_seconds")
        if override is not None:
            try:
                return max(0.0, float(override))
            except ValueError:
                pass
        return max(0.0, float(self.settings.skip_leading_seconds))

    def set_skip_leading_seconds(self, value: float) -> None:
        set_setting(self.db, "skip_leading_seconds", f"{max(0.0, float(value)):.3f}")

    # ------------------------------------------------------------------
    # Liveness gate (background-voice rejection)
    # ------------------------------------------------------------------

    def get_min_liveness_score(self) -> float:
        override = get_setting(self.db, "min_liveness_score")
        if override is not None:
            try:
                return max(0.0, float(override))
            except ValueError:
                pass
        return self.settings.min_liveness_score

    def set_min_liveness_score(self, value: float) -> None:
        set_setting(self.db, "min_liveness_score", f"{max(0.0, float(value)):.3f}")

    # ------------------------------------------------------------------
    # Sample-quality scoring weights (advanced tuning)
    # ------------------------------------------------------------------

    _QUALITY_WEIGHT_KEYS: tuple = (
        "speech_ratio", "snr", "liveness", "consistency", "centroid_distance",
    )

    def get_quality_weights(self) -> dict:
        """Return the effective weights dict (defaults if not overridden)."""
        from .sample_quality import DEFAULT_WEIGHTS
        result = dict(DEFAULT_WEIGHTS)
        for key in self._QUALITY_WEIGHT_KEYS:
            raw = get_setting(self.db, f"quality_weight_{key}")
            if raw is not None:
                try:
                    result[key] = max(0.0, float(raw))
                except ValueError:
                    pass
        # Normalize to sum to 1.0 so composition is stable.
        total = sum(result.values())
        if total > 1e-6:
            result = {k: v / total for k, v in result.items()}
        return result

    def get_quality_weights_source(self) -> str:
        """'override' if any weight is persisted, else 'default'."""
        for key in self._QUALITY_WEIGHT_KEYS:
            if get_setting(self.db, f"quality_weight_{key}") is not None:
                return "override"
        return "default"

    def set_quality_weights(self, weights: Optional[dict]) -> None:
        """Persist weights override (or clear when None)."""
        if weights is None:
            for key in self._QUALITY_WEIGHT_KEYS:
                set_setting(self.db, f"quality_weight_{key}", "")
            self.speakers.set_quality_weights(None)
            return
        for key in self._QUALITY_WEIGHT_KEYS:
            if key in weights:
                set_setting(
                    self.db, f"quality_weight_{key}", f"{max(0.0, float(weights[key])):.4f}"
                )
        self.speakers.set_quality_weights(self.get_quality_weights())

    # ------------------------------------------------------------------
    # Auto-enroll / aging
    # ------------------------------------------------------------------

    def get_auto_enroll(self) -> bool:
        override = get_setting(self.db, "auto_enroll")
        if override is not None:
            return override.lower() in ("1", "true", "yes", "on")
        return self.settings.auto_enroll

    def set_auto_enroll(self, enabled: bool) -> None:
        set_setting(self.db, "auto_enroll", "true" if enabled else "false")

    # ------------------------------------------------------------------
    # Emotion detection (experimental, opt-in)
    # ------------------------------------------------------------------
    #
    # The flag is honoured live (no restart needed) and guards every CPU
    # path in the classifier. Even with the flag on, the handler skips
    # classification whenever the model file is missing so users can
    # safely pre-flip the switch while waiting for a model release.

    def get_enable_emotion(self) -> bool:
        override = get_setting(self.db, "enable_emotion")
        if override is not None:
            return override.lower() in ("1", "true", "yes", "on")
        return self.settings.enable_emotion

    def set_enable_emotion(self, enabled: bool) -> None:
        set_setting(self.db, "enable_emotion", "true" if enabled else "false")

    def emotion_model_available(self) -> bool:
        """Whether the on-disk model file exists and the classifier can run."""
        return self.emotion is not None and self.emotion.is_available()

    def emotion_ready(self) -> bool:
        """True when the feature is both enabled AND a model is on disk."""
        return self.get_enable_emotion() and self.emotion_model_available()

    # ------------------------------------------------------------------
    # Home Assistant integration (live-editable, persisted in DB)
    # ------------------------------------------------------------------

    def get_ha_url(self) -> str:
        override = get_setting(self.db, "ha_url")
        if override is not None:
            return override
        return self.settings.ha_url or ""

    def set_ha_url(self, value: str) -> None:
        set_setting(self.db, "ha_url", (value or "").strip())

    def get_ha_token(self) -> str:
        override = get_setting(self.db, "ha_token")
        if override is not None:
            return override
        return self.settings.ha_token or ""

    def has_ha_token(self) -> bool:
        return bool(self.get_ha_token())

    def set_ha_token(self, value: str) -> None:
        set_setting(self.db, "ha_token", value or "")

    def get_ha_input_text_entity(self) -> str:
        override = get_setting(self.db, "ha_input_text_entity")
        if override is not None and override:
            return override
        return self.settings.ha_input_text_entity

    def set_ha_input_text_entity(self, value: str) -> None:
        set_setting(self.db, "ha_input_text_entity", (value or "").strip())

    def get_ha_tv_entity(self) -> str:
        override = get_setting(self.db, "ha_tv_entity")
        if override is not None:
            return override
        return self.settings.ha_tv_entity or ""

    def set_ha_tv_entity(self, value: str) -> None:
        set_setting(self.db, "ha_tv_entity", (value or "").strip())

    def get_ha_confidence_entity(self) -> str:
        return get_setting(self.db, "ha_confidence_entity") or ""

    def set_ha_confidence_entity(self, value: str) -> None:
        set_setting(self.db, "ha_confidence_entity", (value or "").strip())

    def get_ha_distance_entity(self) -> str:
        return get_setting(self.db, "ha_distance_entity") or ""

    def set_ha_distance_entity(self, value: str) -> None:
        set_setting(self.db, "ha_distance_entity", (value or "").strip())

    def get_ha_nearest_entity(self) -> str:
        return get_setting(self.db, "ha_nearest_entity") or ""

    def set_ha_nearest_entity(self, value: str) -> None:
        set_setting(self.db, "ha_nearest_entity", (value or "").strip())

    def get_ha_role_entity(self) -> str:
        return get_setting(self.db, "ha_role_entity") or ""

    def set_ha_role_entity(self, value: str) -> None:
        set_setting(self.db, "ha_role_entity", (value or "").strip())

    def get_ha_emotion_entity(self) -> str:
        return get_setting(self.db, "ha_emotion_entity") or ""

    def set_ha_emotion_entity(self, value: str) -> None:
        set_setting(self.db, "ha_emotion_entity", (value or "").strip())

    async def apply_ha_settings(self) -> None:
        """Push current HA settings into the live client."""
        await self.ha.reconfigure(
            base_url=self.get_ha_url() or None,
            token=self.get_ha_token() or None,
            input_text_entity=self.get_ha_input_text_entity(),
            tv_entity=self.get_ha_tv_entity() or None,
            confidence_entity=self.get_ha_confidence_entity() or None,
            distance_entity=self.get_ha_distance_entity() or None,
            nearest_entity=self.get_ha_nearest_entity() or None,
            role_entity=self.get_ha_role_entity() or None,
        )
        _LOGGER.info(
            "HA client reconfigured: url=%s configured=%s",
            self.get_ha_url() or "(none)",
            self.ha.configured,
        )

    # ------------------------------------------------------------------
    # Advertised languages (Wyoming Info)
    # ------------------------------------------------------------------
    #
    # Stored as a comma-separated string in the settings table so the
    # Wyoming info_cache can read it through a callback on every
    # Describe, meaning UI edits take effect without a service restart.

    def get_advertised_languages(self) -> Optional[List[str]]:
        raw = get_setting(self.db, "advertised_languages")
        if raw is None:
            return None
        parts = [p.strip() for p in raw.split(",") if p.strip()]
        return parts or None

    # ------------------------------------------------------------------
    # Upstream URI (live-editable so users don't have to redeploy the
    # container just to swap STT backends)
    # ------------------------------------------------------------------

    def get_upstream_uri(self) -> str:
        override = get_setting(self.db, "upstream_uri")
        if override:
            return _normalize_wyoming_uri(override)
        return self.settings.upstream_uri

    def get_upstream_uri_source(self) -> str:
        """Return ``"override"`` if the live setting overrides the env,
        or ``"env"`` if we're using the compose-time default."""
        override = get_setting(self.db, "upstream_uri")
        return "override" if override else "env"

    def set_upstream_uri(self, value: Optional[str]) -> str:
        """Persist a new upstream URI override and propagate it live.

        Returns the normalised value actually stored. Passing an empty
        string clears the override and falls back to the env default.
        """
        if value is None or not value.strip():
            set_setting(self.db, "upstream_uri", "")
            effective = self.settings.upstream_uri
        else:
            normalized = _normalize_wyoming_uri(value)
            set_setting(self.db, "upstream_uri", normalized)
            effective = normalized
        # Propagate into the shared info_cache so Describe responses,
        # ping-upstream and Upstream-refresh all see the new value
        # without waiting for a restart. The handler reads the URI
        # directly from context on each session open, so no further
        # plumbing is required for the STT forwarding path.
        if self.info_cache is not None:
            try:
                self.info_cache.upstream_uri = effective
                self.info_cache.invalidate()
                _LOGGER.info("Upstream URI updated live: %s", effective)
            except Exception:
                _LOGGER.exception("Failed to propagate upstream URI to info_cache")
        return effective

    def set_advertised_languages(self, langs: Optional[List[str]]) -> None:
        if langs:
            cleaned = [l.strip() for l in langs if l and l.strip()]
            value = ",".join(cleaned)
        else:
            value = ""
        set_setting(self.db, "advertised_languages", value)
        # Drop the upstream cache so the next Describe reflects the new
        # override (or the upstream fallback if the override was cleared).
        if self.info_cache is not None:
            try:
                self.info_cache.invalidate()
            except Exception:
                _LOGGER.exception("Failed to invalidate info_cache")


_CONTEXT: Optional[AppContext] = None


def build_context(settings: Optional[Settings] = None) -> AppContext:
    """Construct the shared application context.

    Idempotent — the first call builds and caches, subsequent calls return
    the cached instance.
    """
    global _CONTEXT
    if _CONTEXT is not None:
        return _CONTEXT

    settings = settings or get_settings()
    db = open_db(settings.db_path)
    embedder = CAMPPlusEmbedder(settings.cam_model_path)

    vad: Optional[SileroVAD] = None
    try:
        vad = SileroVAD(
            settings.vad_model_path,
            speech_threshold=settings.vad_speech_threshold,
        )
    except Exception as exc:
        _LOGGER.warning(
            "VAD not available (%s). Enrollment quality checks disabled.", exc
        )

    speakers = SpeakerStore(
        conn=db,
        embedder=embedder,
        vad=vad,
        threshold=settings.verify_threshold,
        min_enrollment_speech_ratio=settings.vad_min_speech_ratio,
    )

    unknown = UnknownStore(
        conn=db,
        ttl_hours=settings.unknown_ttl_hours,
        cleanup_interval_minutes=settings.unknown_cleanup_interval_minutes,
    )

    recognition = RecognitionLog(conn=db)

    # Emotion classifier: always constructed, never auto-loaded. The ONNX
    # session is materialised on the first classify() call — if the model
    # file is missing at that point we raise FileNotFoundError, which the
    # handler catches and silently skips. Constructing it unconditionally
    # keeps the wiring identical regardless of the feature flag.
    emotion = EmotionClassifier(
        settings.emotion_model_path,
        min_duration_sec=settings.emotion_min_seconds,
    )

    # HA client: check DB overrides first, fall back to env.
    _ha_url = get_setting(db, "ha_url")
    _ha_token = get_setting(db, "ha_token")
    _ha_entity = get_setting(db, "ha_input_text_entity")
    _ha_tv = get_setting(db, "ha_tv_entity")
    ha = HomeAssistantClient(
        base_url=_ha_url if _ha_url is not None else settings.ha_url,
        token=_ha_token if _ha_token is not None else settings.ha_token,
        input_text_entity=(_ha_entity or None) if _ha_entity is not None else settings.ha_input_text_entity,
        tv_entity=_ha_tv if _ha_tv is not None else settings.ha_tv_entity,
        confidence_entity=get_setting(db, "ha_confidence_entity") or None,
        distance_entity=get_setting(db, "ha_distance_entity") or None,
        nearest_entity=get_setting(db, "ha_nearest_entity") or None,
        role_entity=get_setting(db, "ha_role_entity") or None,
    )

    _CONTEXT = AppContext(
        settings=settings,
        db=db,
        embedder=embedder,
        vad=vad,
        speakers=speakers,
        unknown=unknown,
        ha=ha,
        recognition=recognition,
        emotion=emotion,
    )
    # Apply the persisted threshold override, if any.
    _CONTEXT.speakers.threshold = _CONTEXT.get_verify_threshold()
    # Apply persisted quality-weight override, if any.
    if _CONTEXT.get_quality_weights_source() == "override":
        _CONTEXT.speakers.set_quality_weights(_CONTEXT.get_quality_weights())
    return _CONTEXT


def get_context() -> AppContext:
    if _CONTEXT is None:
        return build_context()
    return _CONTEXT


def reset_context() -> None:
    global _CONTEXT
    if _CONTEXT is not None:
        try:
            _CONTEXT.db.close()
        except Exception:
            pass
    _CONTEXT = None
