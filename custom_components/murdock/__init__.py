"""The Murdock integration — speaker identity for Home Assistant voice.

Bridges the Murdock speaker-recognition proxy into HA: per-turn speaker
context via an LLM API, registry vocabulary mirroring, and per-satellite
speaker sensors. The MQTT/REST paths of the add-on keep working
independently — this integration is additive.
"""

from __future__ import annotations

import logging

from homeassistant.config_entries import ConfigEntry
from homeassistant.const import Platform
from homeassistant.core import HomeAssistant, callback
from homeassistant.helpers import device_registry as dr
from homeassistant.helpers import entity_registry as er
from homeassistant.helpers import llm

from .const import (
    CONF_BASE_URL,
    CONF_MIRROR_VOCABULARY,
    DEFAULT_MIRROR_VOCABULARY,
    DOMAIN,
)
from .coordinator import MurdockCoordinator
from .llm_api import MurdockLLMApi
from .vocabulary import VocabularyMirror

_LOGGER = logging.getLogger(__name__)

PLATFORMS: list[Platform] = [Platform.BINARY_SENSOR, Platform.SENSOR]


async def async_setup_entry(hass: HomeAssistant, entry: ConfigEntry) -> bool:
    """Set up Murdock from a config entry."""
    coordinator = MurdockCoordinator(hass, entry)
    await coordinator.async_setup()
    hass.data.setdefault(DOMAIN, {})[entry.entry_id] = coordinator

    # The proxy device must exist before the satellite devices that hang
    # under it: they point at it by registry id.
    from .entity import proxy_device_info

    proxy = dr.async_get(hass).async_get_or_create(
        config_entry_id=entry.entry_id, **proxy_device_info(coordinator)
    )
    coordinator.proxy_device_id = proxy.id

    # LLM API (plan §18) — registered like Herold's, unregistered on unload.
    entry.async_on_unload(
        llm.async_register_api(hass, MurdockLLMApi(hass, coordinator))
    )

    # Vocabulary mirroring (plan §9), opt-out via options. Deliberately
    # best-effort: the speaker path is the reason this integration
    # exists, so a vocabulary problem must never fail the entry setup.
    options = {**entry.data, **entry.options}
    if options.get(CONF_MIRROR_VOCABULARY, DEFAULT_MIRROR_VOCABULARY):
        mirror = VocabularyMirror(hass, coordinator)
        coordinator.vocabulary_mirror = mirror
        try:
            await mirror.async_setup()
        except Exception:
            _LOGGER.exception(
                "Vocabulary mirroring failed to start — continuing without it"
            )

    await hass.config_entries.async_forward_entry_setups(entry, PLATFORMS)
    entry.async_on_unload(entry.add_update_listener(_async_update_listener))
    return True


async def async_unload_entry(hass: HomeAssistant, entry: ConfigEntry) -> bool:
    """Unload a config entry."""
    unload_ok = await hass.config_entries.async_unload_platforms(entry, PLATFORMS)
    if unload_ok:
        coordinator: MurdockCoordinator = hass.data[DOMAIN].pop(entry.entry_id)
        mirror = getattr(coordinator, "vocabulary_mirror", None)
        if mirror is not None:
            await mirror.async_shutdown()
        await coordinator.async_shutdown()
    return unload_ok


async def _async_update_listener(hass: HomeAssistant, entry: ConfigEntry) -> None:
    """Reload the entry when options change."""
    await hass.config_entries.async_reload(entry.entry_id)


async def async_migrate_entry(hass: HomeAssistant, entry: ConfigEntry) -> bool:
    """Re-key entities and devices from the proxy URL to the entry id.

    Unique ids and device identifiers used to begin with the base URL, so
    pointing the entry at a new address — the add-on's internal hostname
    instead of homeassistant.local — would have orphaned every entity.
    """
    if entry.version > 1:
        return False
    if entry.minor_version < 2:
        old = str(entry.data.get(CONF_BASE_URL, "")).rstrip("/")
        new = entry.entry_id

        def _rekey(value: str) -> str | None:
            if old and (value == old or value.startswith(old + ":")):
                return new + value[len(old):]
            return None

        @callback
        def _entity(reg_entry: er.RegistryEntry) -> dict | None:
            rekeyed = _rekey(reg_entry.unique_id)
            return {"new_unique_id": rekeyed} if rekeyed else None

        await er.async_migrate_entries(hass, entry.entry_id, _entity)

        dev_reg = dr.async_get(hass)
        for device in dr.async_entries_for_config_entry(dev_reg, entry.entry_id):
            identifiers = set()
            changed = False
            for domain, ident in device.identifiers:
                rekeyed = _rekey(ident) if domain == DOMAIN else None
                identifiers.add((domain, rekeyed or ident))
                changed = changed or rekeyed is not None
            if changed:
                dev_reg.async_update_device(device.id, new_identifiers=identifiers)

        hass.config_entries.async_update_entry(entry, minor_version=2)
        _LOGGER.info("Murdock entities re-keyed from %s to the config entry", old)
    return True
