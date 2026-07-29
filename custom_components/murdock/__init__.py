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
from homeassistant.core import HomeAssistant
from homeassistant.helpers import llm

from .const import CONF_MIRROR_VOCABULARY, DEFAULT_MIRROR_VOCABULARY, DOMAIN
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

    # LLM API (plan §18) — registered like Herold's, unregistered on unload.
    entry.async_on_unload(
        llm.async_register_api(hass, MurdockLLMApi(hass, coordinator))
    )

    # Vocabulary mirroring (plan §9), opt-out via options.
    options = {**entry.data, **entry.options}
    if options.get(CONF_MIRROR_VOCABULARY, DEFAULT_MIRROR_VOCABULARY):
        mirror = VocabularyMirror(hass, coordinator)
        await mirror.async_setup()
        coordinator.vocabulary_mirror = mirror

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
