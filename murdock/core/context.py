"""Shared runtime context: stores, embedder, VAD, and HA client.

Both the Wyoming proxy and the FastAPI web UI import from this module so
that they share a single database connection and set of models.
"""

from __future__ import annotations

import asyncio
import json
import logging
import sqlite3
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, List, Optional

from ..config import Settings, get_settings
from .calibration import Calibrator, calibrator_from_pairs
from .db import get_setting, open_db, set_setting
from .embeddings import CAMPPlusEmbedder
from .emotion import EmotionClassifier
from .ha_integration import HomeAssistantClient
from .mqtt_integration import MQTTClient
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

_LOGGER = logging.getLogger("murdock.context")

# Speaker weight (plan §11): distance above the verify threshold decays
# the weight linearly, reaching 0.0 at threshold + this delta. Matches
# the plan's worked example (th=0.380, ceiling≈0.53).
_WEIGHT_CEILING_DELTA = 0.15

# Cap on HA registry terms merged into the STT bias prompt (plan §10:
# ~20–30; more terms raise the false-replacement rate on noisy input).
_HA_VOCAB_TERM_CAP = 25

# Upper bound for the ordering-critical recognition publish that runs
# before the transcript goes back. A local HA/broker answers in single
# digit milliseconds; this only caps pathological cases.
_PUBLISH_NOW_TIMEOUT = 1.0


@dataclass
class AppContext:
    settings: Settings
    db: sqlite3.Connection
    embedder: CAMPPlusEmbedder
    vad: Optional[SileroVAD]
    speakers: SpeakerStore
    unknown: UnknownStore
    ha: HomeAssistantClient
    mqtt: MQTTClient
    recognition: RecognitionLog
    emotion: Optional[EmotionClassifier] = None
    info_cache: "Optional[UpstreamInfoCache]" = None
    _overrides: dict = field(default_factory=dict)
    _calibrator: Calibrator = field(default_factory=Calibrator)
    _recalibration_task: "Optional[asyncio.Task]" = None
    _recalibration_pending: bool = False

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

    # ------------------------------------------------------------------
    # Margin gate + speaker weight (integration plan §11/§21)
    # ------------------------------------------------------------------

    def get_margin_gate(self, satellite_id: Optional[str] = None) -> float:
        """Minimum required distance between best and second-best speaker.

        0.0 disables the gate. Same fallback chain as the verify
        threshold: per-satellite override → global setting → off.
        """
        if satellite_id:
            sat_override = get_setting(self.db, f"margin_gate_sat_{satellite_id}")
            if sat_override:
                try:
                    return float(sat_override)
                except ValueError:
                    pass
        override = get_setting(self.db, "margin_gate")
        if override:
            try:
                return float(override)
            except ValueError:
                pass
        return 0.0

    def set_margin_gate(self, value: float) -> None:
        set_setting(self.db, "margin_gate", f"{float(value):.4f}")

    def get_satellite_margin_gates(self) -> dict:
        """Return ``{satellite_id: margin}`` for all explicit overrides."""
        cur = self.db.execute(
            "SELECT key, value FROM settings WHERE key LIKE 'margin_gate_sat_%' AND value != ''"
        )
        out: dict = {}
        for key, value in cur.fetchall():
            sid = key[len("margin_gate_sat_"):]
            if not sid or not value:
                continue
            try:
                out[sid] = float(value)
            except (TypeError, ValueError):
                continue
        return out

    def set_satellite_margin_gate(
        self, satellite_id: str, value: Optional[float]
    ) -> None:
        sid = (satellite_id or "").strip()
        if not sid:
            raise ValueError("satellite_id is required")
        if value is None:
            set_setting(self.db, f"margin_gate_sat_{sid}", "")
        else:
            set_setting(self.db, f"margin_gate_sat_{sid}", f"{float(value):.4f}")

    def compute_speaker_weight(
        self, distance: Optional[float], threshold: Optional[float]
    ) -> Optional[float]:
        """Speaker weight per plan §11: 1.0 when verified, decaying
        linearly to 0.0 at ``threshold + _WEIGHT_CEILING_DELTA``.

        A near-miss (short command, blurry embedding) keeps a partial
        weight instead of dropping to zero — speaker-bound dictionary
        entries then require a stricter lexical match rather than being
        switched off entirely.
        """
        if distance is None or threshold is None:
            return None
        ceiling = threshold + _WEIGHT_CEILING_DELTA
        if distance <= threshold:
            return 1.0
        if distance >= ceiling:
            return 0.0
        return (ceiling - distance) / (ceiling - threshold)

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

    # ------------------------------------------------------------------
    # Adaptive target-speaker extraction
    # ------------------------------------------------------------------

    def get_enable_extraction(self) -> bool:
        override = get_setting(self.db, "enable_extraction")
        if override is not None:
            return override.lower() in ("1", "true", "yes", "on")
        return self.settings.enable_extraction

    def set_enable_extraction(self, enabled: bool) -> None:
        set_setting(self.db, "enable_extraction", "true" if enabled else "false")

    def get_enable_stt_prep(self) -> bool:
        override = get_setting(self.db, "enable_stt_prep")
        if override is not None:
            return override.lower() in ("1", "true", "yes", "on")
        return self.settings.enable_stt_prep

    def set_enable_stt_prep(self, enabled: bool) -> None:
        set_setting(self.db, "enable_stt_prep", "true" if enabled else "false")

    def get_extraction_threshold(self, satellite_id: Optional[str] = None) -> float:
        """Effective extraction distance ceiling.

        Mirrors the verify-threshold fallback chain: per-satellite override
        → global override → env/config default. A noisy room can therefore
        run a more lenient extraction threshold than a quiet one.
        """
        if satellite_id:
            sat = get_setting(self.db, f"extraction_threshold_sat_{satellite_id}")
            if sat:
                try:
                    return float(sat)
                except ValueError:
                    pass
        override = get_setting(self.db, "extraction_threshold")
        if override is not None:
            try:
                return float(override)
            except ValueError:
                pass
        return self.settings.extraction_threshold

    def set_extraction_threshold(self, value: float) -> None:
        set_setting(self.db, "extraction_threshold", f"{max(0.0, float(value)):.4f}")

    def get_extraction_min_region_sec(self) -> float:
        override = get_setting(self.db, "extraction_min_region_sec")
        if override is not None:
            try:
                return max(0.0, float(override))
            except ValueError:
                pass
        return max(0.0, float(self.settings.extraction_min_region_sec))

    def set_extraction_min_region_sec(self, value: float) -> None:
        set_setting(self.db, "extraction_min_region_sec", f"{max(0.0, float(value)):.3f}")

    # ------------------------------------------------------------------
    # Confidence calibration (Platt scaling)
    # ------------------------------------------------------------------

    def get_enable_calibration(self) -> bool:
        override = get_setting(self.db, "enable_calibration")
        if override is not None:
            return override.lower() in ("1", "true", "yes", "on")
        return self.settings.enable_calibration

    def set_enable_calibration(self, enabled: bool) -> None:
        set_setting(self.db, "enable_calibration", "true" if enabled else "false")

    def load_calibration(self) -> None:
        """Load persisted Platt parameters from the settings table."""
        raw = get_setting(self.db, "calibration")
        if raw:
            try:
                self._calibrator = Calibrator.from_dict(json.loads(raw))
                return
            except Exception:
                _LOGGER.warning("Stored calibration is corrupt; ignoring")
        self._calibrator = Calibrator()

    def get_calibrator(self) -> Calibrator:
        return self._calibrator

    def confidence_for(self, distance: float) -> float:
        """Return calibrated P(same speaker) if available, else 1 - distance.

        This is the single source of truth for the ``confidence`` value we
        report to HA / MQTT, so the whole pipeline reads the same number.
        """
        if self.get_enable_calibration() and self._calibrator.fitted:
            prob = self._calibrator.probability(distance)
            if prob is not None:
                return prob
        return max(0.0, 1.0 - float(distance))

    def recalibrate(self) -> Calibrator:
        """Re-fit the calibrator from the current enrollments (blocking).

        Heavy (re-embeds every stored sample), so call it off the event
        loop (``asyncio.to_thread``) or via :meth:`schedule_recalibration`.
        Persists the result to the settings table — including the
        adaptive per-speaker thresholds derived from the same pass.
        """
        from .calibration import compute_adaptive_thresholds

        distances, labels, per_speaker = self.speakers.collect_calibration_data()
        calibrator = calibrator_from_pairs(distances, labels)
        self._calibrator = calibrator
        set_setting(self.db, "calibration", json.dumps(calibrator.to_dict()))
        adaptive = compute_adaptive_thresholds(
            per_speaker, self.get_verify_threshold()
        )
        set_setting(self.db, "adaptive_thresholds", json.dumps(adaptive))
        if adaptive:
            _LOGGER.info(
                "Adaptive thresholds updated: %s",
                ", ".join(f"{n}={t}" for n, t in adaptive.items()),
            )
        return calibrator

    # ------------------------------------------------------------------
    # Adaptive per-speaker thresholds + per-satellite voice profiles
    # ------------------------------------------------------------------

    def get_enable_adaptive_thresholds(self) -> bool:
        override = get_setting(self.db, "enable_adaptive_thresholds")
        if override is not None:
            return override.lower() in ("1", "true", "yes", "on")
        return self.settings.enable_adaptive_thresholds

    def set_enable_adaptive_thresholds(self, enabled: bool) -> None:
        set_setting(
            self.db, "enable_adaptive_thresholds", "true" if enabled else "false"
        )

    def get_adaptive_thresholds(self) -> dict:
        """Return ``{speaker_name: threshold}`` (empty when none computed)."""
        raw = get_setting(self.db, "adaptive_thresholds")
        if not raw:
            return {}
        try:
            data = json.loads(raw)
            return {
                str(k): float(v) for k, v in data.items()
            } if isinstance(data, dict) else {}
        except (ValueError, TypeError):
            return {}

    def get_enable_satellite_profiles(self) -> bool:
        override = get_setting(self.db, "enable_satellite_profiles")
        if override is not None:
            return override.lower() in ("1", "true", "yes", "on")
        return self.settings.enable_satellite_profiles

    def set_enable_satellite_profiles(self, enabled: bool) -> None:
        set_setting(
            self.db, "enable_satellite_profiles", "true" if enabled else "false"
        )

    # ------------------------------------------------------------------
    # Early reject (opt-in TV/radio killer)
    # ------------------------------------------------------------------

    def get_enable_early_reject(self) -> bool:
        override = get_setting(self.db, "enable_early_reject")
        if override is not None:
            return override.lower() in ("1", "true", "yes", "on")
        return self.settings.enable_early_reject

    def set_enable_early_reject(self, enabled: bool) -> None:
        set_setting(
            self.db, "enable_early_reject", "true" if enabled else "false"
        )

    def get_early_reject_margin(self) -> float:
        override = get_setting(self.db, "early_reject_margin")
        if override is not None:
            try:
                return max(0.05, float(override))
            except ValueError:
                pass
        return max(0.05, float(self.settings.early_reject_margin))

    def set_early_reject_margin(self, value: float) -> None:
        set_setting(
            self.db, "early_reject_margin", f"{max(0.05, float(value)):.3f}"
        )

    # ------------------------------------------------------------------
    # Transcript augmentation (inject recognition context into the text)
    # ------------------------------------------------------------------

    SPEAKER_CONTEXT_MODES = ("none", "transcript")

    def get_speaker_context_mode(self) -> str:
        """How the speaker reaches the conversation agent: none | transcript."""
        override = get_setting(self.db, "speaker_context_mode")
        if override in self.SPEAKER_CONTEXT_MODES:
            return override
        # Back-compat with the pre-release boolean flag.
        legacy = get_setting(self.db, "enable_transcript_template")
        if legacy is not None and legacy.lower() in ("1", "true", "yes", "on"):
            return "transcript"
        if self.settings.speaker_context_mode in self.SPEAKER_CONTEXT_MODES:
            return self.settings.speaker_context_mode
        return "none"

    def set_speaker_context_mode(self, mode: str) -> None:
        if mode not in self.SPEAKER_CONTEXT_MODES:
            raise ValueError(
                f"Invalid speaker_context_mode: {mode!r} "
                f"(allowed: {', '.join(self.SPEAKER_CONTEXT_MODES)})"
            )
        set_setting(self.db, "speaker_context_mode", mode)

    # ------------------------------------------------------------------
    # Transcript hint delivery (plan §13)
    # ------------------------------------------------------------------

    TRANSCRIPT_HINT_MODES = ("inline", "sidecar", "clean", "auto")

    def get_transcript_hint_mode(self) -> str:
        """Where ambiguity markers go: inline | sidecar | clean | auto.

        * ``inline`` — ``[oder: …]`` stays in the transcript (default,
          the pre-0.8 behaviour).
        * ``sidecar`` — transcript stays clean, ambiguities travel in the
          recognition event; the HA integration turns them into a
          "Transkript-Hinweis" prompt line.
        * ``clean`` — markers are dropped entirely.
        * ``auto`` — sidecar when an event sink is available, else inline.
        """
        override = get_setting(self.db, "transcript_hint_mode")
        if override in self.TRANSCRIPT_HINT_MODES:
            return override
        return "inline"

    def set_transcript_hint_mode(self, mode: str) -> None:
        if mode not in self.TRANSCRIPT_HINT_MODES:
            raise ValueError(
                f"Invalid transcript_hint_mode: {mode!r} "
                f"(allowed: {', '.join(self.TRANSCRIPT_HINT_MODES)})"
            )
        set_setting(self.db, "transcript_hint_mode", mode)

    def effective_transcript_hint_mode(self) -> str:
        """Resolve ``auto`` against the sinks that are actually wired.

        Sidecar only helps when something downstream reads the event, so
        ``auto`` falls back to inline markers when neither MQTT nor the
        HA REST push is configured.
        """
        mode = self.get_transcript_hint_mode()
        if mode != "auto":
            return mode
        has_sink = False
        try:
            has_sink = bool(self.mqtt and self.mqtt.connected)
        except Exception:
            pass
        if not has_sink:
            try:
                has_sink = bool(self.ha and self.ha.configured)
            except Exception:
                pass
        return "sidecar" if has_sink else "inline"

    def get_transcript_template_known(self) -> str:
        override = get_setting(self.db, "transcript_template_known")
        if override is not None:
            return override
        return self.settings.transcript_template_known

    def set_transcript_template_known(self, value: str) -> None:
        set_setting(self.db, "transcript_template_known", value or "")

    def get_transcript_template_unknown(self) -> str:
        override = get_setting(self.db, "transcript_template_unknown")
        if override is not None:
            return override
        return self.settings.transcript_template_unknown

    def set_transcript_template_unknown(self, value: str) -> None:
        set_setting(self.db, "transcript_template_unknown", value or "")

    def schedule_recalibration(self) -> None:
        """Fire-and-forget a recalibration on the running loop, debounced.

        If a recalibration is already running, mark one as pending so a
        single follow-up pass picks up the latest enrollments instead of
        queueing one task per enrolment.
        """
        if not self.get_enable_calibration():
            return
        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:
            return
        if self._recalibration_task and not self._recalibration_task.done():
            self._recalibration_pending = True
            return

        async def _run() -> None:
            try:
                while True:
                    self._recalibration_pending = False
                    try:
                        await asyncio.to_thread(self.recalibrate)
                    except Exception:
                        _LOGGER.exception("Recalibration failed")
                    if not self._recalibration_pending:
                        break
            finally:
                self._recalibration_task = None

        self._recalibration_task = loop.create_task(_run())

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
            emotion_entity=self.get_ha_emotion_entity() or None,
        )
        _LOGGER.info(
            "HA client reconfigured: url=%s configured=%s",
            self.get_ha_url() or "(none)",
            self.ha.configured,
        )

    # ------------------------------------------------------------------
    # MQTT integration (auto-discovery, no token required)
    # ------------------------------------------------------------------

    def get_mqtt_enabled(self) -> bool:
        override = get_setting(self.db, "mqtt_enabled")
        if override is not None:
            return override.lower() in ("1", "true", "yes", "on")
        return self.settings.mqtt_enabled

    def set_mqtt_enabled(self, enabled: bool) -> None:
        set_setting(self.db, "mqtt_enabled", "true" if enabled else "false")

    def get_mqtt_host(self) -> str:
        override = get_setting(self.db, "mqtt_host")
        if override is not None:
            return override
        return self.settings.mqtt_host or ""

    def set_mqtt_host(self, value: str) -> None:
        set_setting(self.db, "mqtt_host", (value or "").strip())

    def get_mqtt_port(self) -> int:
        raw = get_setting(self.db, "mqtt_port")
        if raw:
            try:
                return int(raw)
            except ValueError:
                pass
        return self.settings.mqtt_port

    def set_mqtt_port(self, value: int) -> None:
        set_setting(self.db, "mqtt_port", str(max(1, min(65535, value))))

    def get_mqtt_username(self) -> str:
        override = get_setting(self.db, "mqtt_username")
        if override is not None:
            return override
        return self.settings.mqtt_username or ""

    def set_mqtt_username(self, value: str) -> None:
        set_setting(self.db, "mqtt_username", (value or "").strip())

    def get_mqtt_password(self) -> str:
        override = get_setting(self.db, "mqtt_password")
        if override is not None:
            return override
        return self.settings.mqtt_password or ""

    def has_mqtt_password(self) -> bool:
        return bool(self.get_mqtt_password())

    def set_mqtt_password(self, value: str) -> None:
        set_setting(self.db, "mqtt_password", value or "")

    def get_mqtt_topic_prefix(self) -> str:
        return get_setting(self.db, "mqtt_topic_prefix") or self.settings.mqtt_topic_prefix

    def set_mqtt_topic_prefix(self, value: str) -> None:
        set_setting(self.db, "mqtt_topic_prefix", (value or "murdock").strip())

    def get_mqtt_discovery_prefix(self) -> str:
        return get_setting(self.db, "mqtt_discovery_prefix") or self.settings.mqtt_discovery_prefix

    def set_mqtt_discovery_prefix(self, value: str) -> None:
        set_setting(self.db, "mqtt_discovery_prefix", (value or "homeassistant").strip())

    async def apply_mqtt_settings(self) -> None:
        """Reconfigure and (re)start the MQTT client with current settings."""
        if self.get_mqtt_enabled():
            await self.mqtt.reconfigure(
                host=self.get_mqtt_host(),
                port=self.get_mqtt_port(),
                username=self.get_mqtt_username(),
                password=self.get_mqtt_password(),
                topic_prefix=self.get_mqtt_topic_prefix(),
                discovery_prefix=self.get_mqtt_discovery_prefix(),
            )
        else:
            await self.mqtt.stop()
        _LOGGER.info(
            "MQTT client reconfigured: host=%s enabled=%s connected=%s",
            self.get_mqtt_host() or "(none)",
            self.get_mqtt_enabled(),
            self.mqtt.connected,
        )

    # ------------------------------------------------------------------
    # Unified recognition output + context lookups (HA REST + MQTT)
    # ------------------------------------------------------------------
    #
    # The handler talks to these instead of poking ha/mqtt directly, so
    # the two transports can run side by side (or either alone). MQTT is
    # the recommended path; the REST client stays as a legacy fallback
    # for users who haven't set up a broker.

    def publish_recognition(
        self,
        *,
        speaker: str,
        confidence: float,
        satellite_id: Optional[str],
        is_known: bool,
        distance: Optional[float] = None,
        threshold: Optional[float] = None,
        nearest_speaker: Optional[str] = None,
        nearest_distance: Optional[float] = None,
        weight: Optional[float] = None,
        margin: Optional[float] = None,
        uncertain: bool = False,
        reason: Optional[str] = None,
        role: Optional[str] = None,
        emotion: Optional[str] = None,
        emotion_confidence: Optional[float] = None,
        ambiguities: Optional[list] = None,
        whisper: bool = False,
        whisper_score: Optional[float] = None,
        speakers: Optional[list] = None,
    ) -> None:
        """Fire-and-forget a recognition result to every configured sink.

        Pushes to MQTT (if connected) and HA REST (if configured). Both
        are scheduled as background tasks so neither blocks the Wyoming
        pipeline.
        """
        if weight is None:
            weight = self.compute_speaker_weight(distance, threshold)
        kwargs = dict(
            speaker=speaker,
            confidence=confidence,
            satellite_id=satellite_id,
            is_known=is_known,
            distance=distance,
            threshold=threshold,
            nearest_speaker=nearest_speaker,
            nearest_distance=nearest_distance,
            weight=weight,
            margin=margin,
            uncertain=uncertain,
            reason=reason,
            role=role,
            emotion=emotion,
            emotion_confidence=emotion_confidence,
            ambiguities=ambiguities,
            whisper=whisper,
            whisper_score=whisper_score,
            speakers=speakers,
        )
        if self.mqtt.connected:
            self.mqtt.publish_async(self.mqtt.publish_recognition(**kwargs))
        if self.ha.configured:
            self.ha.publish_async(self.ha.push_recognition(**kwargs))

    async def publish_recognition_now(self, **kwargs) -> None:
        """Publish and *wait* for the sinks — ordering-critical variant.

        Home Assistant starts the intent stage within a millisecond of
        receiving the transcript, and the conversation agent's prompt is
        built there. A fire-and-forget push therefore loses the race and
        the speaker only lands in time for the *next* turn. Awaiting the
        publish before answering the satellite is what makes the speaker
        available for the current turn.

        Bounded by ``_PUBLISH_NOW_TIMEOUT`` and never raises: an
        unreachable HA or a stalled broker must not hold up a transcript.
        """
        if kwargs.get("weight") is None:
            kwargs["weight"] = self.compute_speaker_weight(
                kwargs.get("distance"), kwargs.get("threshold")
            )
        coros = []
        if self.mqtt.connected:
            coros.append(self.mqtt.publish_recognition(**kwargs))
        if self.ha.configured:
            coros.append(self.ha.push_recognition(**kwargs))
        if not coros:
            return
        try:
            await asyncio.wait_for(
                asyncio.gather(*coros, return_exceptions=True),
                timeout=_PUBLISH_NOW_TIMEOUT,
            )
        except asyncio.TimeoutError:
            _LOGGER.warning(
                "Recognition publish exceeded %.1fs — continuing",
                _PUBLISH_NOW_TIMEOUT,
            )
        except Exception:
            _LOGGER.debug("Recognition publish failed", exc_info=True)

    async def is_tv_playing(self, satellite_id: Optional[str] = None) -> bool:
        """Return True if a TV is playing in the relevant room.

        Resolution order:
          1. MQTT context cache (HA pushes ``murdock/context/<room>/tv``)
          2. HA REST poll on the configured media_player entity (legacy)

        The satellite_id doubles as the room key for the per-room MQTT
        topic — satellites are named after their room in practice.
        """
        if self.mqtt.connected:
            mqtt_state = self.mqtt.is_tv_playing(room=satellite_id)
            if mqtt_state is not None:
                return mqtt_state
        # Legacy fallback: poll HA REST if a token + entity are set.
        return await self.ha.is_tv_playing()

    # ------------------------------------------------------------------
    # Media-aware gating: per-satellite × per-source restriction matrix
    # ------------------------------------------------------------------
    #
    # Stored as JSON in the settings table:
    #   {"<satellite_id>": {"<media_entity_id>": <threshold_delta>}}
    # A delta is how much that source tightens (lowers) the verify
    # threshold for that satellite while it's playing. An explicit 0
    # means "this source does not restrict this satellite" (e.g. a radio
    # in another room). Sources with no explicit entry fall back to the
    # global tv_threshold_boost when they play in the satellite's room.

    def get_media_restrictions(self) -> dict:
        raw = get_setting(self.db, "media_restrictions")
        if not raw:
            return {}
        try:
            data = json.loads(raw)
            return data if isinstance(data, dict) else {}
        except (ValueError, TypeError):
            return {}

    def set_media_restriction(
        self,
        satellite_id: str,
        media_entity: str,
        delta: Optional[float],
    ) -> None:
        """Set (or clear, when delta is None) one matrix cell."""
        sat = (satellite_id or "").strip()
        ent = (media_entity or "").strip()
        if not sat or not ent:
            raise ValueError("satellite_id and media_entity are required")
        matrix = self.get_media_restrictions()
        cell = matrix.setdefault(sat, {})
        if delta is None:
            cell.pop(ent, None)
            if not cell:
                matrix.pop(sat, None)
        else:
            cell[ent] = round(max(0.0, float(delta)), 4)
        set_setting(self.db, "media_restrictions", json.dumps(matrix))

    async def compute_media_tightening(
        self,
        satellite_id: Optional[str],
        satellite_area: Optional[str],
    ) -> float:
        """How much to tighten (lower) the verify threshold right now.

        Looks at every currently-playing media player and applies, per
        source: the explicit matrix delta for this satellite if set, else
        the global boost when the source shares the satellite's room, else
        nothing. The strongest single source wins (max, not sum, so two
        playing devices don't over-restrict). Falls back to the legacy
        room-TV / REST signal when no per-source media context exists.
        """
        default_boost = float(self.settings.tv_threshold_boost)
        matrix = self.get_media_restrictions()
        sat_rules = matrix.get(satellite_id or "", {}) if satellite_id else {}
        deltas: List[float] = [0.0]

        if self.mqtt.connected:
            for media in self.mqtt.playing_media():
                ent = media.get("entity_id") or ""
                if ent in sat_rules:
                    deltas.append(max(0.0, float(sat_rules[ent])))
                elif satellite_area and media.get("area") == satellite_area:
                    deltas.append(default_boost)
                # else: different room, no explicit rule → no effect

        # Legacy fallback only when per-source media contributed nothing,
        # so the two paths never double-count.
        if max(deltas) <= 0.0:
            if await self.is_tv_playing(satellite_area or satellite_id):
                deltas.append(default_boost)

        return max(deltas)

    # ------------------------------------------------------------------
    # STT backend selection (upstream Wyoming vs. Voxtral cloud)
    # ------------------------------------------------------------------

    STT_BACKENDS = ("upstream", "voxtral", "openai")

    def get_stt_backend(self) -> str:
        """Return the active STT backend: 'upstream', 'voxtral' or 'openai'."""
        override = get_setting(self.db, "stt_backend")
        if override and override in self.STT_BACKENDS:
            return override
        return self.settings.stt_backend

    def set_stt_backend(self, value: str) -> None:
        if value not in self.STT_BACKENDS:
            raise ValueError(f"Invalid stt_backend: {value!r}")
        set_setting(self.db, "stt_backend", value)

    def get_mistral_api_key(self) -> str:
        override = get_setting(self.db, "mistral_api_key")
        if override is not None:
            return override
        return self.settings.mistral_api_key or ""

    def has_mistral_api_key(self) -> bool:
        return bool(self.get_mistral_api_key())

    def set_mistral_api_key(self, value: str) -> None:
        set_setting(self.db, "mistral_api_key", value or "")

    def get_mistral_model(self) -> str:
        override = get_setting(self.db, "mistral_model")
        if override:
            return override
        return self.settings.mistral_model

    def set_mistral_model(self, value: str) -> None:
        set_setting(self.db, "mistral_model", (value or "").strip())

    def get_voxtral_backend(self):
        """Return a configured VoxtralBackend instance, or None if not ready."""
        from .stt_backend import VoxtralBackend
        api_key = self.get_mistral_api_key()
        if not api_key:
            return None
        return VoxtralBackend(
            api_key=api_key,
            model=self.get_mistral_model(),
        )

    # ------------------------------------------------------------------
    # Transcript quality tiers: vocabulary, dictionary, dual transcript
    # ------------------------------------------------------------------

    def get_enable_stt_vocabulary(self) -> bool:
        override = get_setting(self.db, "enable_stt_vocabulary")
        if override is not None:
            return override.lower() in ("1", "true", "yes", "on")
        return self.settings.enable_stt_vocabulary

    def set_enable_stt_vocabulary(self, enabled: bool) -> None:
        set_setting(
            self.db, "enable_stt_vocabulary", "true" if enabled else "false"
        )

    def get_stt_vocabulary(self) -> str:
        override = get_setting(self.db, "stt_vocabulary")
        if override is not None:
            return override
        return self.settings.stt_vocabulary

    def set_stt_vocabulary(self, value: str) -> None:
        set_setting(self.db, "stt_vocabulary", value or "")

    def get_vocabulary_store(self):
        """Lazily created store for HA registry vocabulary snapshots."""
        store = getattr(self, "_vocab_store", None)
        if store is None:
            from murdock.core.vocabulary_store import VocabularyStore

            store = VocabularyStore(self.db)
            self._vocab_store = store
        return store

    def get_vocab_selection(self) -> Optional[List[str]]:
        """Explicitly chosen mirrored terms, or None for automatic.

        None means "take the first N by priority". A stored list — even an
        empty one — means the user curated it, and only those terms go to
        the engine.
        """
        raw = get_setting(self.db, "stt_vocab_selection")
        if raw is None:
            return None
        try:
            data = json.loads(raw)
        except (TypeError, ValueError):
            return None
        if not isinstance(data, list):
            return None
        return [str(t) for t in data]

    def set_vocab_selection(self, terms: Optional[List[str]]) -> None:
        """Store a curated selection, or None to return to automatic."""
        if terms is None:
            self.db.execute(
                "DELETE FROM settings WHERE key = 'stt_vocab_selection'"
            )
            self.db.commit()
            return
        set_setting(
            self.db,
            "stt_vocab_selection",
            json.dumps([str(t) for t in terms], ensure_ascii=False),
        )

    def get_selected_ha_terms(self) -> List[str]:
        """Mirrored terms that actually go into the bias prompt."""
        try:
            available = self.get_vocabulary_store().terms()
        except Exception:
            _LOGGER.debug("Vocabulary snapshot unavailable", exc_info=True)
            return []
        selection = self.get_vocab_selection()
        if selection is None:
            return available[:_HA_VOCAB_TERM_CAP]
        # Keep snapshot order, drop entries that no longer exist, and
        # still respect the cap so a huge selection can't bloat the prompt.
        chosen = {t.casefold() for t in selection}
        return [t for t in available if t.casefold() in chosen][
            :_HA_VOCAB_TERM_CAP
        ]

    def get_effective_vocabulary(self) -> str:
        """Vocabulary prompt for the STT backends ("" when disabled).

        Manual terms first, then the selected registry terms. Capped
        because a bias prompt has a length budget (plan §10: an oversized
        list raises false replacements on noisy input) — the canonicalizer
        has no such limit and uses the full list.
        """
        if not self.get_enable_stt_vocabulary():
            return ""
        manual = self.get_stt_vocabulary().strip()
        ha_terms = self.get_selected_ha_terms()
        if not ha_terms:
            return manual
        manual_lower = manual.lower()
        extra = [t for t in ha_terms if t.lower() not in manual_lower]
        if not extra:
            return manual
        joined = ", ".join(extra)
        return f"{manual}, {joined}" if manual else joined

    def active_backend_supports_prompt(self) -> bool:
        """Whether the *active* primary backend will use the bias prompt.

        The local Wyoming upstream has no prompt field, Voxtral doesn't
        document one, and the OpenRouter request shape skips it — so on
        those the vocabulary is stored but never sent. Surfacing that
        beats showing a prompt that goes nowhere.
        """
        kind = self.get_stt_backend()
        if kind == "openai":
            base = (self.get_openai_base_url() or "").lower()
            return "openrouter.ai" not in base
        return False

    # ------------------------------------------------------------------
    # Canonicalization: map the transcript onto real entity names
    # ------------------------------------------------------------------

    # ------------------------------------------------------------------
    # Whisper detection (experimental)
    # ------------------------------------------------------------------

    def get_enable_whisper_detection(self) -> bool:
        override = get_setting(self.db, "enable_whisper_detection")
        if override is not None:
            return override.lower() in ("1", "true", "yes", "on")
        return False

    def set_enable_whisper_detection(self, enabled: bool) -> None:
        set_setting(
            self.db, "enable_whisper_detection", "true" if enabled else "false"
        )

    def get_whisper_threshold(self) -> float:
        from murdock.core.whisper_detect import DEFAULT_THRESHOLD

        override = get_setting(self.db, "whisper_threshold")
        if override:
            try:
                return float(override)
            except ValueError:
                pass
        return DEFAULT_THRESHOLD

    def set_whisper_threshold(self, value: float) -> None:
        set_setting(self.db, "whisper_threshold", f"{float(value):.4f}")

    def detect_whisper(self, audio_16k: bytes):
        """Whisper features for an utterance, or None when disabled.

        Never raises: a detector hiccup must not cost the utterance.
        """
        if not audio_16k or not self.get_enable_whisper_detection():
            return None
        try:
            from murdock.core.whisper_detect import analyze_pcm

            return analyze_pcm(audio_16k, self.get_whisper_threshold())
        except Exception:
            _LOGGER.debug("Whisper detection failed", exc_info=True)
            return None

    def get_enable_canonicalizer(self) -> bool:
        override = get_setting(self.db, "enable_canonicalizer")
        if override is not None:
            return override.lower() in ("1", "true", "yes", "on")
        return False

    def set_enable_canonicalizer(self, enabled: bool) -> None:
        set_setting(
            self.db, "enable_canonicalizer", "true" if enabled else "false"
        )

    def get_canonicalizer_min_score(self) -> float:
        from murdock.core.canonicalize import DEFAULT_MIN_SCORE

        override = get_setting(self.db, "canonicalizer_min_score")
        if override:
            try:
                return float(override)
            except ValueError:
                pass
        return DEFAULT_MIN_SCORE

    def set_canonicalizer_min_score(self, value: float) -> None:
        set_setting(self.db, "canonicalizer_min_score", f"{float(value):.4f}")

    def get_canonicalizer_min_margin(self) -> float:
        from murdock.core.canonicalize import DEFAULT_MIN_MARGIN

        override = get_setting(self.db, "canonicalizer_min_margin")
        if override:
            try:
                return float(override)
            except ValueError:
                pass
        return DEFAULT_MIN_MARGIN

    def set_canonicalizer_min_margin(self, value: float) -> None:
        set_setting(self.db, "canonicalizer_min_margin", f"{float(value):.4f}")

    def get_manual_vocabulary_terms(self) -> List[str]:
        """The manually maintained terms, split into individual entries."""
        raw = self.get_stt_vocabulary()
        return [t.strip() for t in raw.split(",") if t.strip()]

    def get_canonicalizer_terms(self) -> List[str]:
        """Every valid name a transcript may be mapped onto.

        Uncapped on purpose: the 25-term limit exists because a bias
        prompt has a length budget, while a local index does not. Manual
        terms are included — they are names the user cares about by
        definition.
        """
        terms = list(self.get_manual_vocabulary_terms())
        try:
            terms.extend(self.get_vocabulary_store().terms())
        except Exception:
            _LOGGER.debug("Vocabulary snapshot unavailable", exc_info=True)
        return terms

    def canonicalize_transcript(self, transcript: str):
        """Correct near-misses of known names. Returns (text, replacements).

        Never raises: a transcript must reach the satellite even if this
        step has a bad day.
        """
        if not transcript or not self.get_enable_canonicalizer():
            return transcript, []
        try:
            from murdock.core.canonicalize import canonicalize

            terms = self.get_canonicalizer_terms()
            if not terms:
                return transcript, []
            corrected, replacements = canonicalize(
                transcript,
                terms,
                min_score=self.get_canonicalizer_min_score(),
                min_margin=self.get_canonicalizer_min_margin(),
            )
            if replacements:
                self.record_canonicalizer_hits(replacements)
            return corrected, replacements
        except Exception:
            _LOGGER.exception("Canonicalization failed — transcript unchanged")
            return transcript, []

    def record_canonicalizer_hits(self, replacements) -> None:
        """Count corrections so recurring ones can become explicit rules."""
        try:
            from murdock.core.canonicalizer_hits import CanonicalizerHits

            store = getattr(self, "_canon_hits", None)
            if store is None:
                store = CanonicalizerHits(self.db)
                self._canon_hits = store
            store.record(replacements)
        except Exception:
            _LOGGER.debug("Could not record canonicalizer hits", exc_info=True)

    def get_canonicalizer_hits(self, limit: int = 20):
        try:
            from murdock.core.canonicalizer_hits import CanonicalizerHits

            store = getattr(self, "_canon_hits", None)
            if store is None:
                store = CanonicalizerHits(self.db)
                self._canon_hits = store
            return store.top(limit=limit)
        except Exception:
            _LOGGER.debug("Could not read canonicalizer hits", exc_info=True)
            return []

    def get_enable_stt_dictionary(self) -> bool:
        override = get_setting(self.db, "enable_stt_dictionary")
        if override is not None:
            return override.lower() in ("1", "true", "yes", "on")
        return self.settings.enable_stt_dictionary

    def set_enable_stt_dictionary(self, enabled: bool) -> None:
        set_setting(
            self.db, "enable_stt_dictionary", "true" if enabled else "false"
        )

    def get_stt_dictionary(self) -> str:
        override = get_setting(self.db, "stt_dictionary")
        if override is not None:
            return override
        return self.settings.stt_dictionary

    def set_stt_dictionary(self, value: str) -> None:
        set_setting(self.db, "stt_dictionary", value or "")

    def get_dictionary_entries(self) -> list:
        """Parsed dictionary entries; empty when the tier is disabled."""
        if not self.get_enable_stt_dictionary():
            return []
        from .transcript_tools import parse_correction_dictionary
        return parse_correction_dictionary(self.get_stt_dictionary())

    def get_enable_dual_transcript(self) -> bool:
        override = get_setting(self.db, "enable_dual_transcript")
        if override is not None:
            return override.lower() in ("1", "true", "yes", "on")
        return self.settings.enable_dual_transcript

    def set_enable_dual_transcript(self, enabled: bool) -> None:
        set_setting(
            self.db, "enable_dual_transcript", "true" if enabled else "false"
        )

    def dual_transcript_active(self) -> bool:
        """Dual mode requires the toggle AND a configured shadow engine."""
        return (
            self.get_enable_dual_transcript()
            and self.get_shadow_stt_backend() != "none"
        )

    # ------------------------------------------------------------------
    # OpenAI-compatible STT backend (OpenAI, Groq, local speaches, …)
    # ------------------------------------------------------------------

    def get_openai_base_url(self) -> str:
        override = get_setting(self.db, "openai_base_url")
        if override:
            return override
        return self.settings.openai_base_url

    def set_openai_base_url(self, value: str) -> None:
        set_setting(self.db, "openai_base_url", (value or "").strip())

    def get_openai_api_key(self) -> str:
        override = get_setting(self.db, "openai_api_key")
        if override is not None:
            return override
        return self.settings.openai_api_key or ""

    def has_openai_api_key(self) -> bool:
        return bool(self.get_openai_api_key())

    def set_openai_api_key(self, value: str) -> None:
        set_setting(self.db, "openai_api_key", value or "")

    def get_openai_model(self) -> str:
        override = get_setting(self.db, "openai_model")
        if override:
            return override
        return self.settings.openai_model

    def set_openai_model(self, value: str) -> None:
        set_setting(self.db, "openai_model", (value or "").strip())

    def get_openai_backend(self):
        """Return a configured OpenAI-compatible backend, or None.

        Unlike Voxtral, an empty API key is allowed — self-hosted
        OpenAI-compatible servers (speaches, LocalAI) usually run without
        auth. A model name is required.
        """
        from .stt_backend import OpenAICompatibleBackend
        model = self.get_openai_model()
        if not model:
            return None
        return OpenAICompatibleBackend(
            api_key=self.get_openai_api_key(),
            model=model,
            base_url=self.get_openai_base_url(),
            prompt=self.get_effective_vocabulary() or None,
        )

    def get_active_cloud_backend(self):
        """Return the backend for the active cloud stt_backend, or None."""
        backend = self.get_stt_backend()
        if backend == "voxtral":
            return self.get_voxtral_backend()
        if backend == "openai":
            return self.get_openai_backend()
        return None

    # ------------------------------------------------------------------
    # Local fallback + A/B shadow engine
    # ------------------------------------------------------------------

    def get_stt_local_fallback(self) -> bool:
        override = get_setting(self.db, "stt_local_fallback")
        if override is not None:
            return override.lower() in ("1", "true", "yes", "on")
        return self.settings.stt_local_fallback

    def set_stt_local_fallback(self, enabled: bool) -> None:
        set_setting(
            self.db, "stt_local_fallback", "true" if enabled else "false"
        )

    SHADOW_BACKENDS = ("none", "upstream", "voxtral", "openai")

    def get_shadow_stt_backend(self) -> str:
        override = get_setting(self.db, "shadow_stt_backend")
        if override in self.SHADOW_BACKENDS:
            return override
        if self.settings.shadow_stt_backend in self.SHADOW_BACKENDS:
            return self.settings.shadow_stt_backend
        return "none"

    def set_shadow_stt_backend(self, value: str) -> None:
        if value not in self.SHADOW_BACKENDS:
            raise ValueError(f"Invalid shadow_stt_backend: {value!r}")
        set_setting(self.db, "shadow_stt_backend", value)

    def get_shadow_upstream_uri(self) -> str:
        override = get_setting(self.db, "shadow_upstream_uri")
        if override:
            return _normalize_wyoming_uri(override)
        return _normalize_wyoming_uri(self.settings.shadow_upstream_uri or "")

    def set_shadow_upstream_uri(self, value: str) -> None:
        set_setting(self.db, "shadow_upstream_uri", (value or "").strip())

    def get_shadow_mistral_model(self) -> str:
        override = get_setting(self.db, "shadow_mistral_model")
        if override:
            return override
        return self.settings.shadow_mistral_model

    def set_shadow_mistral_model(self, value: str) -> None:
        set_setting(self.db, "shadow_mistral_model", (value or "").strip())

    def get_shadow_mistral_api_key(self) -> str:
        """Shadow Voxtral key; empty falls back to the primary mistral key."""
        override = get_setting(self.db, "shadow_mistral_api_key")
        if override:
            return override
        return self.settings.shadow_mistral_api_key or self.get_mistral_api_key()

    def has_shadow_mistral_api_key(self) -> bool:
        return bool(get_setting(self.db, "shadow_mistral_api_key"))

    def set_shadow_mistral_api_key(self, value: str) -> None:
        set_setting(self.db, "shadow_mistral_api_key", value or "")

    def get_shadow_openai_base_url(self) -> str:
        override = get_setting(self.db, "shadow_openai_base_url")
        if override:
            return override
        return self.settings.shadow_openai_base_url or self.get_openai_base_url()

    def set_shadow_openai_base_url(self, value: str) -> None:
        set_setting(self.db, "shadow_openai_base_url", (value or "").strip())

    def get_shadow_openai_api_key(self) -> str:
        """Shadow key; empty falls back to the primary OpenAI key."""
        override = get_setting(self.db, "shadow_openai_api_key")
        if override:
            return override
        return self.settings.shadow_openai_api_key or self.get_openai_api_key()

    def has_shadow_openai_api_key(self) -> bool:
        return bool(get_setting(self.db, "shadow_openai_api_key"))

    def set_shadow_openai_api_key(self, value: str) -> None:
        set_setting(self.db, "shadow_openai_api_key", value or "")

    def get_shadow_openai_model(self) -> str:
        override = get_setting(self.db, "shadow_openai_model")
        if override:
            return override
        return self.settings.shadow_openai_model

    def set_shadow_openai_model(self, value: str) -> None:
        set_setting(self.db, "shadow_openai_model", (value or "").strip())

    def get_shadow_backend(self):
        """Return the HTTP backend for a cloud shadow kind, or None.

        The "upstream" shadow kind is handled by the caller via
        :func:`murdock.core.stt_backend.transcribe_via_wyoming` with
        :meth:`get_shadow_upstream_uri`.
        """
        from .stt_backend import OpenAICompatibleBackend, VoxtralBackend

        kind = self.get_shadow_stt_backend()
        if kind == "voxtral":
            api_key = self.get_shadow_mistral_api_key()
            if not api_key:
                return None
            return VoxtralBackend(
                api_key=api_key, model=self.get_shadow_mistral_model()
            )
        if kind == "openai":
            model = self.get_shadow_openai_model()
            if not model:
                return None
            return OpenAICompatibleBackend(
                api_key=self.get_shadow_openai_api_key(),
                model=model,
                base_url=self.get_shadow_openai_base_url(),
                name="shadow-openai",
                prompt=self.get_effective_vocabulary() or None,
            )
        return None

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
        emotion_entity=get_setting(db, "ha_emotion_entity") or None,
    )

    # MQTT client: DB override first, env (addon-injected / compose) fallback.
    _mqtt_host = get_setting(db, "mqtt_host")
    _mqtt_user = get_setting(db, "mqtt_username")
    _mqtt_pass = get_setting(db, "mqtt_password")
    _mqtt_port = get_setting(db, "mqtt_port")
    mqtt = MQTTClient(
        host=_mqtt_host if _mqtt_host is not None else (settings.mqtt_host or ""),
        port=int(_mqtt_port) if _mqtt_port else settings.mqtt_port,
        username=_mqtt_user if _mqtt_user is not None else (settings.mqtt_username or ""),
        password=_mqtt_pass if _mqtt_pass is not None else (settings.mqtt_password or ""),
        topic_prefix=get_setting(db, "mqtt_topic_prefix") or settings.mqtt_topic_prefix,
        discovery_prefix=get_setting(db, "mqtt_discovery_prefix") or settings.mqtt_discovery_prefix,
    )

    _CONTEXT = AppContext(
        settings=settings,
        db=db,
        embedder=embedder,
        vad=vad,
        speakers=speakers,
        unknown=unknown,
        ha=ha,
        mqtt=mqtt,
        recognition=recognition,
        emotion=emotion,
    )
    # Apply the persisted threshold override, if any.
    _CONTEXT.speakers.threshold = _CONTEXT.get_verify_threshold()
    # Apply persisted quality-weight override, if any.
    if _CONTEXT.get_quality_weights_source() == "override":
        _CONTEXT.speakers.set_quality_weights(_CONTEXT.get_quality_weights())
    # Load any persisted calibration so confidence is calibrated from the
    # first recognition after a restart (no need to refit on every boot).
    _CONTEXT.load_calibration()
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
