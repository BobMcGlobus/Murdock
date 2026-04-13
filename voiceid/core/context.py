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
    info_cache: "Optional[UpstreamInfoCache]" = None
    _overrides: dict = field(default_factory=dict)

    # ------------------------------------------------------------------
    # Runtime-configurable settings (persisted in the settings table)
    # ------------------------------------------------------------------

    def get_verify_threshold(self) -> float:
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

    ha = HomeAssistantClient(
        base_url=settings.ha_url,
        token=settings.ha_token,
        input_text_entity=settings.ha_input_text_entity,
        tv_entity=settings.ha_tv_entity,
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
    )
    # Apply the persisted threshold override, if any.
    _CONTEXT.speakers.threshold = _CONTEXT.get_verify_threshold()
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
