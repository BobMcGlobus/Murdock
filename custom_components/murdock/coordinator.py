"""In-memory speaker state per satellite (plan §7).

Recognition in, in-memory out: the coordinator keeps the latest
:class:`SpeakerState` per satellite and never routes reads through entity
state (the sensors are a dashboard by-product, see sensor.py).

Murdock can deliver a recognition over **either** of two paths, so both
are consumed here:

* the **HA event bus** (``speaker_recognition_detected``), fired by
  Murdock's REST push — needs a long-lived token;
* the **MQTT topic** ``<prefix>/event/recognition``, published by
  Murdock's own MQTT client. This is the recommended, token-free setup,
  and an MQTT message is *not* an HA event — it has to be subscribed
  explicitly or the integration sees nothing at all.

Both carry the identical payload, and duplicates (someone running both)
are dropped by (satellite, timestamp).
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Any

from homeassistant.config_entries import ConfigEntry
from homeassistant.core import Event, HomeAssistant, callback
from homeassistant.helpers import device_registry as dr, entity_registry as er
from homeassistant.helpers.dispatcher import async_dispatcher_send
from homeassistant.helpers.event import async_track_time_interval
from homeassistant.util import dt as dt_util

from .api import MurdockApiClient, MurdockApiError
from .const import (
    CONF_BASE_URL,
    CONF_FRESHNESS_WINDOW,
    CONF_MQTT_PREFIX,
    CONF_SATELLITE_ENTITY,
    CONF_SATELLITE_ID,
    CONF_SATELLITES,
    CONF_TOKEN,
    DEFAULT_FRESHNESS_WINDOW,
    DEFAULT_MQTT_PREFIX,
    EVENT_SPEAKER_RECOGNITION,
    HEALTH_POLL_INTERVAL_SECONDS,
    MQTT_EVENT_SUFFIX,
    SIGNAL_SPEAKER_UPDATE,
)

_LOGGER = logging.getLogger(__name__)


@dataclass
class Ambiguity:
    """A second possible reading of a transcript span (plan §13)."""

    original: str
    alternative: str
    kind: str = "alternative"


def _parse_ambiguities(raw: Any) -> list[Ambiguity]:
    """Read the event's sidecar hints, ignoring anything malformed."""
    if not isinstance(raw, list):
        return []
    out: list[Ambiguity] = []
    for item in raw:
        if not isinstance(item, dict):
            continue
        alternative = str(item.get("alternative") or "").strip()
        if not alternative:
            continue
        out.append(
            Ambiguity(
                original=str(item.get("original") or "").strip(),
                alternative=alternative,
                kind=str(item.get("kind") or "alternative"),
            )
        )
    return out


@dataclass
class SpeakerState:
    """Latest recognition for one satellite."""

    speaker: str | None          # None = unknown / not recognised
    confidence: float
    distance: float | None
    nearest: str | None
    nearest_distance: float | None   # margin gate input
    role: str | None
    weight: float                    # plan §11
    satellite_id: str
    recognized_at: datetime
    uncertain: bool = False          # margin gate said "too close to call"
    whisper: bool = False            # the utterance was whispered
    whisper_score: float | None = None
    reason: str | None = None        # why speaker is None
    ambiguities: list[Ambiguity] = field(default_factory=list)
    context_prior: float = 0.0       # plan §12, frozen at wake (phase 2)


class MurdockCoordinator:
    """Holds per-satellite speaker state and the proxy connection."""

    def __init__(self, hass: HomeAssistant, entry: ConfigEntry) -> None:
        self.hass = hass
        self.entry = entry
        self.api = MurdockApiClient(
            hass,
            entry.data[CONF_BASE_URL],
            entry.data.get(CONF_TOKEN),
        )
        self.available: bool = False
        self.proxy_version: str | None = None
        self.vocabulary_version: str | None = None
        self.last_event_at: datetime | None = None
        self._states: dict[str, SpeakerState] = {}
        self._unsubs: list = []
        # (satellite_id, timestamp) of the last applied payload, so a
        # recognition delivered over both paths is only applied once.
        self._last_seen: tuple[str, Any] | None = None
        self.mqtt_subscribed: bool = False

    # ------------------------------------------------------------------
    # Config accessors
    # ------------------------------------------------------------------

    @property
    def _option(self):
        return {**self.entry.data, **self.entry.options}

    @property
    def freshness_window(self) -> timedelta:
        seconds = self._option.get(
            CONF_FRESHNESS_WINDOW, DEFAULT_FRESHNESS_WINDOW
        )
        return timedelta(seconds=float(seconds))

    @property
    def satellites(self) -> list[dict[str, Any]]:
        """Configured satellite mappings ({satellite_id, satellite_entity})."""
        return list(self._option.get(CONF_SATELLITES, []))

    @property
    def mqtt_topic(self) -> str:
        """Recognition topic, or "" when MQTT consumption is disabled."""
        prefix = str(
            self._option.get(CONF_MQTT_PREFIX, DEFAULT_MQTT_PREFIX) or ""
        ).strip().strip("/")
        return f"{prefix}/{MQTT_EVENT_SUFFIX}" if prefix else ""

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    async def async_setup(self) -> None:
        self._unsubs.append(
            self.hass.bus.async_listen(
                EVENT_SPEAKER_RECOGNITION, self._handle_event
            )
        )
        await self._async_subscribe_mqtt()
        self._unsubs.append(
            async_track_time_interval(
                self.hass,
                self._async_health_poll,
                timedelta(seconds=HEALTH_POLL_INTERVAL_SECONDS),
            )
        )
        await self.async_initial_sync()

    async def _async_subscribe_mqtt(self) -> None:
        """Subscribe to Murdock's MQTT recognition topic, if MQTT is set up.

        Best-effort: no broker, no MQTT integration, or a bad topic just
        means this path stays unused — the event bus still works.
        """
        topic = self.mqtt_topic
        if not topic:
            return
        if "mqtt" not in self.hass.config.components:
            _LOGGER.debug(
                "MQTT integration not loaded — relying on the event bus"
            )
            return
        try:
            from homeassistant.components import mqtt

            unsub = await mqtt.async_subscribe(
                self.hass, topic, self._handle_mqtt_message
            )
        except Exception:
            _LOGGER.warning(
                "Could not subscribe to %s — relying on the event bus",
                topic, exc_info=True,
            )
            return
        self._unsubs.append(unsub)
        self.mqtt_subscribed = True
        _LOGGER.debug("Subscribed to %s", topic)

    async def async_shutdown(self) -> None:
        for unsub in self._unsubs:
            unsub()
        self._unsubs.clear()
        self.mqtt_subscribed = False

    async def async_initial_sync(self) -> None:
        """Rebuild state from the proxy after an HA restart (plan §7)."""
        try:
            self.proxy_version = await self.api.get_version()
            rows = await self.api.get_state()
            self.available = True
        except MurdockApiError as exc:
            _LOGGER.warning("Murdock initial sync failed: %s", exc)
            self.available = False
            async_dispatcher_send(self.hass, SIGNAL_SPEAKER_UPDATE, None)
            return
        for row in rows:
            sat = row.get("satellite_id")
            if not sat:
                continue
            recognized_at = dt_util.utc_from_timestamp(
                float(row.get("recognized_at") or 0)
            )
            self._states[sat] = SpeakerState(
                speaker=row.get("speaker"),
                confidence=0.0,
                distance=row.get("distance"),
                nearest=None,
                nearest_distance=None,
                role=None,
                weight=float(row.get("weight") or 0.0),
                satellite_id=sat,
                recognized_at=recognized_at,
                reason=None if row.get("speaker") else row.get("outcome"),
            )
        await self._async_refresh_vocabulary_meta()
        async_dispatcher_send(self.hass, SIGNAL_SPEAKER_UPDATE, None)

    async def _async_health_poll(self, _now=None) -> None:
        was_available = self.available
        try:
            self.proxy_version = await self.api.get_version()
            self.available = True
        except MurdockApiError:
            self.available = False
        if self.available:
            await self._async_refresh_vocabulary_meta()
        if was_available != self.available:
            async_dispatcher_send(self.hass, SIGNAL_SPEAKER_UPDATE, None)

    async def _async_refresh_vocabulary_meta(self) -> None:
        try:
            meta = await self.api.get_vocabulary_meta()
            self.vocabulary_version = (
                str(meta.get("version")) if meta.get("available") else None
            )
        except MurdockApiError:
            pass

    # ------------------------------------------------------------------
    # Event ingestion
    # ------------------------------------------------------------------

    @callback
    def _handle_event(self, event: Event) -> None:
        """Recognition from the HA event bus (Murdock's REST push)."""
        self._apply_payload(event.data or {}, source="event")

    @callback
    def _handle_mqtt_message(self, msg) -> None:
        """Recognition from Murdock's MQTT topic (token-free path)."""
        try:
            data = json.loads(msg.payload)
        except (TypeError, ValueError):
            _LOGGER.debug("Ignoring non-JSON payload on %s", msg.topic)
            return
        if not isinstance(data, dict):
            return
        self._apply_payload(data, source="mqtt")

    @callback
    def _apply_payload(self, data: dict[str, Any], *, source: str) -> None:
        sat = data.get("satellite_id")
        if not sat:
            return
        # Running both paths delivers the same recognition twice.
        marker = (sat, data.get("timestamp"))
        if marker[1] is not None and marker == self._last_seen:
            _LOGGER.debug("Duplicate recognition via %s — ignored", source)
            return
        self._last_seen = marker
        state = SpeakerState(
            speaker=data.get("speaker") if data.get("is_known") else None,
            confidence=float(data.get("confidence") or 0.0),
            distance=data.get("distance"),
            nearest=data.get("nearest_speaker"),
            nearest_distance=data.get("nearest_distance"),
            role=data.get("role"),
            weight=float(data.get("weight") or 0.0),
            satellite_id=sat,
            recognized_at=dt_util.utcnow(),
            uncertain=bool(data.get("uncertain")),
            whisper=bool(data.get("whisper")),
            whisper_score=data.get("whisper_score"),
            reason=data.get("reason"),
            ambiguities=_parse_ambiguities(data.get("ambiguities")),
        )
        self._states[sat] = state
        self.available = True
        self.last_event_at = state.recognized_at
        async_dispatcher_send(self.hass, SIGNAL_SPEAKER_UPDATE, sat)

    # ------------------------------------------------------------------
    # Reads
    # ------------------------------------------------------------------

    def state_for_satellite(self, satellite_id: str) -> SpeakerState | None:
        """Fresh state for a satellite, or None once it has aged out."""
        state = self._states.get(satellite_id)
        if state is None:
            return None
        if dt_util.utcnow() - state.recognized_at > self.freshness_window:
            return None
        return state

    def raw_state_for_satellite(self, satellite_id: str) -> SpeakerState | None:
        """State ignoring the freshness window (sensors show the last event)."""
        return self._states.get(satellite_id)

    def satellite_for_device(self, device_id: str | None) -> str | None:
        """Resolve an LLMContext device_id to a satellite_id (plan §8).

        Explicit mapping only: direct device hit first, then the device's
        area (survives re-registration of the satellite device), never a
        guess.
        """
        if not device_id:
            return None
        ent_reg = er.async_get(self.hass)
        mapped: list[tuple[str, str | None, str | None]] = []
        for sat in self.satellites:
            sat_id = sat.get(CONF_SATELLITE_ID)
            entity_id = sat.get(CONF_SATELLITE_ENTITY)
            if not sat_id or not entity_id:
                continue
            entry = ent_reg.async_get(entity_id)
            if entry is None:
                mapped.append((sat_id, None, None))
                continue
            area_id = entry.area_id
            if area_id is None and entry.device_id:
                dev = dr.async_get(self.hass).async_get(entry.device_id)
                area_id = dev.area_id if dev else None
            mapped.append((sat_id, entry.device_id, area_id))
        # 1. Direct device match.
        for sat_id, dev_id, _area in mapped:
            if dev_id and dev_id == device_id:
                return sat_id
        # 2. Same area as the requesting device.
        dev = dr.async_get(self.hass).async_get(device_id)
        if dev and dev.area_id:
            for sat_id, _dev_id, area_id in mapped:
                if area_id and area_id == dev.area_id:
                    return sat_id
        return None

    def area_name_for_satellite(self, satellite_id: str) -> str | None:
        """Display name of the satellite's area, for the prompt line."""
        ent_reg = er.async_get(self.hass)
        for sat in self.satellites:
            if sat.get(CONF_SATELLITE_ID) != satellite_id:
                continue
            entity_id = sat.get(CONF_SATELLITE_ENTITY)
            entry = ent_reg.async_get(entity_id) if entity_id else None
            if entry is None:
                return None
            area_id = entry.area_id
            if area_id is None and entry.device_id:
                dev = dr.async_get(self.hass).async_get(entry.device_id)
                area_id = dev.area_id if dev else None
            if area_id:
                from homeassistant.helpers import area_registry as ar

                area = ar.async_get(self.hass).async_get_area(area_id)
                return area.name if area else None
            return None
        return None

    def state_for_device(self, device_id: str | None) -> SpeakerState | None:
        sat = self.satellite_for_device(device_id)
        if sat is None:
            return None
        return self.state_for_satellite(sat)
