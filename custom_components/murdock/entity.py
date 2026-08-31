"""Shared entity base for Murdock entities."""

from __future__ import annotations

from homeassistant.core import callback
from homeassistant.helpers.device_registry import DeviceInfo
from homeassistant.helpers.dispatcher import async_dispatcher_connect
from homeassistant.helpers.entity import Entity

from .const import DOMAIN, SIGNAL_SPEAKER_UPDATE
from .coordinator import MurdockCoordinator


class MurdockEntity(Entity):
    """Push-updated entity backed by the coordinator's dispatcher."""

    _attr_has_entity_name = True
    _attr_should_poll = False

    def __init__(self, coordinator: MurdockCoordinator) -> None:
        self.coordinator = coordinator

    async def async_added_to_hass(self) -> None:
        self.async_on_remove(
            async_dispatcher_connect(
                self.hass, SIGNAL_SPEAKER_UPDATE, self._handle_update
            )
        )

    @callback
    def _handle_update(self, _satellite_id: str | None) -> None:
        # Without @callback the dispatcher classifies this as an executor
        # job and runs it in a worker thread, where async_write_ha_state
        # is not safe to call — Home Assistant logs that as a warning
        # about corrupting data, and it is right to.
        if self.hass is None:
            return
        self.async_write_ha_state()


def proxy_device_info(coordinator: MurdockCoordinator) -> DeviceInfo:
    """One diagnostics device for the proxy itself."""
    return DeviceInfo(
        identifiers={(DOMAIN, coordinator.api.base_url)},
        name="Murdock Proxy",
        manufacturer="Murdock",
        sw_version=coordinator.proxy_version,
        configuration_url=coordinator.api.base_url,
    )


def satellite_device_info(
    coordinator: MurdockCoordinator, satellite_id: str
) -> DeviceInfo:
    """One device per mapped satellite."""
    return DeviceInfo(
        identifiers={(DOMAIN, f"{coordinator.api.base_url}:{satellite_id}")},
        name=f"Murdock {satellite_id}",
        manufacturer="Murdock",
        via_device=(DOMAIN, coordinator.api.base_url),
    )
