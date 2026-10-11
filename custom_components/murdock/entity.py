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


def configuration_url(coordinator: MurdockCoordinator) -> str | None:
    """Where "visit" on the device page should go.

    The internal add-on address is not reachable from a browser, so for
    the add-on it opens the panel inside Home Assistant instead.
    """
    if coordinator.addon_slug:
        return f"homeassistant://hassio/ingress/{coordinator.addon_slug}"
    url = coordinator.api.base_url
    host = url.split("://", 1)[-1].split(":", 1)[0].split("/", 1)[0]
    return url if "." in host or host == "localhost" else None


def proxy_device_info(coordinator: MurdockCoordinator) -> DeviceInfo:
    """One diagnostics device for the proxy itself."""
    info = DeviceInfo(
        identifiers={(DOMAIN, coordinator.id_prefix)},
        name="Murdock Proxy",
        manufacturer="Murdock",
        sw_version=coordinator.proxy_version,
    )
    url = configuration_url(coordinator)
    if url:
        info["configuration_url"] = url
    return info


# Home Assistant replaced ``via_device`` (an identifier tuple) with
# ``via_device_id`` (a registry id); the old key stops working in 2027.8.
_HAS_VIA_DEVICE_ID = "via_device_id" in getattr(DeviceInfo, "__annotations__", {})


def satellite_device_info(
    coordinator: MurdockCoordinator, satellite_id: str
) -> DeviceInfo:
    """One device per mapped satellite, hung under the proxy device."""
    info = DeviceInfo(
        identifiers={(DOMAIN, f"{coordinator.id_prefix}:{satellite_id}")},
        name=f"Murdock {satellite_id}",
        manufacturer="Murdock",
    )
    if _HAS_VIA_DEVICE_ID:
        if coordinator.proxy_device_id:
            info["via_device_id"] = coordinator.proxy_device_id
    else:
        info["via_device"] = (DOMAIN, coordinator.id_prefix)
    return info
