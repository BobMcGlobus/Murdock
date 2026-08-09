"""Connectivity binary sensor for the proxy (plan §17)."""

from __future__ import annotations

from homeassistant.components.binary_sensor import (
    BinarySensorDeviceClass,
    BinarySensorEntity,
)
from homeassistant.config_entries import ConfigEntry
from homeassistant.core import HomeAssistant
from homeassistant.helpers.entity import EntityCategory
from homeassistant.helpers.entity_platform import AddEntitiesCallback

from .const import CONF_SATELLITE_ID, DOMAIN
from .coordinator import MurdockCoordinator
from .entity import MurdockEntity, proxy_device_info, satellite_device_info


async def async_setup_entry(
    hass: HomeAssistant,
    entry: ConfigEntry,
    async_add_entities: AddEntitiesCallback,
) -> None:
    coordinator: MurdockCoordinator = hass.data[DOMAIN][entry.entry_id]
    entities: list = [ConnectionSensor(coordinator)]
    for sat in coordinator.satellites:
        sat_id = sat.get(CONF_SATELLITE_ID)
        if sat_id:
            entities.append(WhisperSensor(coordinator, sat_id))
    async_add_entities(entities)


class ConnectionSensor(MurdockEntity, BinarySensorEntity):
    """On while the proxy answers (event received or health poll ok)."""

    _attr_translation_key = "connection"
    _attr_device_class = BinarySensorDeviceClass.CONNECTIVITY
    _attr_entity_category = EntityCategory.DIAGNOSTIC

    def __init__(self, coordinator: MurdockCoordinator) -> None:
        super().__init__(coordinator)
        self._attr_unique_id = f"{coordinator.api.base_url}:connection"
        self._attr_device_info = proxy_device_info(coordinator)

    @property
    def is_on(self) -> bool:
        return self.coordinator.available


class WhisperSensor(MurdockEntity, BinarySensorEntity):
    """On while the last utterance on this satellite was whispered.

    Its own entity rather than an attribute, so an automation can duck the
    TTS volume without templating into the speaker sensor.
    """

    _attr_translation_key = "whisper"

    def __init__(self, coordinator: MurdockCoordinator, satellite_id: str) -> None:
        super().__init__(coordinator)
        self._satellite_id = satellite_id
        self._attr_unique_id = f"{coordinator.api.base_url}:{satellite_id}:whisper"
        self._attr_device_info = satellite_device_info(coordinator, satellite_id)

    def _handle_update(self, satellite_id: str | None) -> None:
        if satellite_id in (None, self._satellite_id):
            self.async_write_ha_state()

    @property
    def is_on(self) -> bool:
        state = self.coordinator.raw_state_for_satellite(self._satellite_id)
        return bool(state and state.whisper)

    @property
    def extra_state_attributes(self) -> dict:
        state = self.coordinator.raw_state_for_satellite(self._satellite_id)
        return {
            "whisper_score": state.whisper_score if state else None,
            "satellite_id": self._satellite_id,
        }
