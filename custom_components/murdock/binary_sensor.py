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

from .const import DOMAIN
from .coordinator import MurdockCoordinator
from .entity import MurdockEntity, proxy_device_info


async def async_setup_entry(
    hass: HomeAssistant,
    entry: ConfigEntry,
    async_add_entities: AddEntitiesCallback,
) -> None:
    coordinator: MurdockCoordinator = hass.data[DOMAIN][entry.entry_id]
    async_add_entities([ConnectionSensor(coordinator)])


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
