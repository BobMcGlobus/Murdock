"""Sensors: per-satellite speaker + proxy diagnostics (plan §17).

Dashboard by-product only — automations and other integrations should
use ``helpers.async_get_speaker`` / the dispatcher signal, not these
states (the entity-state detour adds recorder latency for no benefit).
"""

from __future__ import annotations

from datetime import datetime

from homeassistant.components.sensor import (
    SensorDeviceClass,
    SensorEntity,
)
from homeassistant.config_entries import ConfigEntry
from homeassistant.core import HomeAssistant
from homeassistant.helpers.entity import EntityCategory
from homeassistant.helpers.entity_platform import AddEntitiesCallback

from .const import CONF_SATELLITE_ID, DOMAIN
from .coordinator import MurdockCoordinator
from .entity import MurdockEntity, proxy_device_info, satellite_device_info

STATE_UNKNOWN_SPEAKER = "unbekannt"
STATE_UNCERTAIN = "unsicher"


async def async_setup_entry(
    hass: HomeAssistant,
    entry: ConfigEntry,
    async_add_entities: AddEntitiesCallback,
) -> None:
    coordinator: MurdockCoordinator = hass.data[DOMAIN][entry.entry_id]
    entities: list[SensorEntity] = []
    for sat in coordinator.satellites:
        sat_id = sat.get(CONF_SATELLITE_ID)
        if sat_id:
            entities.append(SpeakerSensor(coordinator, sat_id))
    entities.append(LastRecognitionSensor(coordinator))
    entities.append(VocabularyVersionSensor(coordinator))
    entities.append(DeliveryPathSensor(coordinator))
    async_add_entities(entities)


class SpeakerSensor(MurdockEntity, SensorEntity):
    """Who was last heard on this satellite."""

    _attr_translation_key = "speaker"

    def __init__(
        self, coordinator: MurdockCoordinator, satellite_id: str
    ) -> None:
        super().__init__(coordinator)
        self._satellite_id = satellite_id
        self._attr_unique_id = (
            f"{coordinator.api.base_url}:{satellite_id}:speaker"
        )
        self._attr_device_info = satellite_device_info(
            coordinator, satellite_id
        )

    def _handle_update(self, satellite_id: str | None) -> None:
        if satellite_id in (None, self._satellite_id):
            self.async_write_ha_state()

    @property
    def native_value(self) -> str:
        state = self.coordinator.raw_state_for_satellite(self._satellite_id)
        if state is None or not state.speaker:
            if state is not None and state.uncertain:
                return STATE_UNCERTAIN
            return STATE_UNKNOWN_SPEAKER
        # The freshness window applies to reads for context, not to the
        # dashboard: the sensor keeps showing the last recognition.
        return state.speaker

    @property
    def extra_state_attributes(self) -> dict:
        state = self.coordinator.raw_state_for_satellite(self._satellite_id)
        if state is None:
            return {"satellite_id": self._satellite_id}
        margin = None
        if state.distance is not None and state.nearest_distance is not None:
            margin = round(state.nearest_distance - state.distance, 4)
        return {
            "satellite_id": self._satellite_id,
            "confidence": state.confidence,
            "distance": state.distance,
            "nearest_speaker": state.nearest,
            "nearest_distance": state.nearest_distance,
            "margin": margin,
            "weight": state.weight,
            "role": state.role,
            "uncertain": state.uncertain,
            "reason": state.reason,
            "recognized_at": state.recognized_at.isoformat(),
        }


class LastRecognitionSensor(MurdockEntity, SensorEntity):
    """Timestamp of the last recognition event from the proxy."""

    _attr_translation_key = "last_recognition"
    _attr_device_class = SensorDeviceClass.TIMESTAMP
    _attr_entity_category = EntityCategory.DIAGNOSTIC

    def __init__(self, coordinator: MurdockCoordinator) -> None:
        super().__init__(coordinator)
        self._attr_unique_id = f"{coordinator.api.base_url}:last_recognition"
        self._attr_device_info = proxy_device_info(coordinator)

    @property
    def native_value(self) -> datetime | None:
        return self.coordinator.last_event_at


class DeliveryPathSensor(MurdockEntity, SensorEntity):
    """Which path recognitions actually arrive on.

    The single most useful thing to look at when the speaker stays
    `unbekannt`: Murdock's REST push needs a token, its MQTT publish needs
    a broker, and only a subscribed path delivers anything.
    """

    _attr_translation_key = "delivery_path"
    _attr_entity_category = EntityCategory.DIAGNOSTIC

    def __init__(self, coordinator: MurdockCoordinator) -> None:
        super().__init__(coordinator)
        self._attr_unique_id = f"{coordinator.api.base_url}:delivery_path"
        self._attr_device_info = proxy_device_info(coordinator)

    @property
    def native_value(self) -> str:
        got_any = self.coordinator.last_event_at is not None
        if self.coordinator.mqtt_subscribed:
            return "mqtt+event" if got_any else "mqtt (waiting)"
        return "event" if got_any else "event (waiting)"

    @property
    def extra_state_attributes(self) -> dict:
        return {
            "mqtt_subscribed": self.coordinator.mqtt_subscribed,
            "mqtt_topic": self.coordinator.mqtt_topic or None,
            "recognitions_received": self.coordinator.last_event_at is not None,
        }


class VocabularyVersionSensor(MurdockEntity, SensorEntity):
    """Version of the vocabulary snapshot Murdock currently holds."""

    _attr_translation_key = "vocabulary_version"
    _attr_entity_category = EntityCategory.DIAGNOSTIC

    def __init__(self, coordinator: MurdockCoordinator) -> None:
        super().__init__(coordinator)
        self._attr_unique_id = f"{coordinator.api.base_url}:vocabulary_version"
        self._attr_device_info = proxy_device_info(coordinator)

    @property
    def native_value(self) -> str | None:
        return self.coordinator.vocabulary_version
