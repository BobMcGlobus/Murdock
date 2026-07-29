"""Registry → Murdock vocabulary mirroring (plan §9).

Only entities exposed to the voice assistant are pushed — the exposure
flag describes exactly the set people talk about; mirroring the whole
registry would inflate the fuzzy index and raise false replacements.

Registry changes arrive as events and are pushed with a debounce so a
bulk rename ends up as one snapshot, not twenty.
"""

from __future__ import annotations

import logging
import time
from typing import Any

from homeassistant.components.homeassistant.exposed_entities import (
    async_should_expose,
)
from homeassistant.core import CALLBACK_TYPE, HomeAssistant, callback
from homeassistant.helpers import (
    area_registry as ar,
    entity_registry as er,
    floor_registry as fr,
)
from homeassistant.helpers.event import async_call_later
from homeassistant.util import dt as dt_util

from .api import MurdockApiError
from .const import VOCABULARY_DEBOUNCE_SECONDS

_LOGGER = logging.getLogger(__name__)

# Exposure assistant domain used by Assist pipelines.
_CONVERSATION = "conversation"


def build_snapshot(hass: HomeAssistant) -> dict[str, Any]:
    """Assemble the vocabulary payload from the registries."""
    ent_reg = er.async_get(hass)
    area_reg = ar.async_get(hass)
    floor_reg = fr.async_get(hass)

    entities: list[dict[str, Any]] = []
    for entry in ent_reg.entities.values():
        if entry.disabled_by or entry.hidden_by:
            continue
        try:
            if not async_should_expose(hass, _CONVERSATION, entry.entity_id):
                continue
        except Exception:
            continue
        name = entry.name or entry.original_name
        if not name:
            continue
        area_id = entry.area_id
        if area_id is None and entry.device_id:
            from homeassistant.helpers import device_registry as dr

            dev = dr.async_get(hass).async_get(entry.device_id)
            area_id = dev.area_id if dev else None
        area = area_reg.async_get_area(area_id) if area_id else None
        floor = (
            floor_reg.async_get_floor(area.floor_id)
            if area and area.floor_id
            else None
        )
        entities.append(
            {
                "entity_id": entry.entity_id,
                "name": name,
                "aliases": sorted(entry.aliases or []),
                "area": area.name if area else None,
                "floor": floor.name if floor else None,
                "domain": entry.domain,
            }
        )

    return {
        # Monotonic enough for "did anything change since" checks.
        "version": int(time.time()),
        "generated_at": dt_util.utcnow().isoformat(),
        "entities": sorted(entities, key=lambda e: e["entity_id"]),
        "areas": [{"name": a.name} for a in area_reg.async_list_areas()],
        "floors": [{"name": f.name} for f in floor_reg.async_list_floors()],
    }


class VocabularyMirror:
    """Watches the registries and pushes debounced snapshots."""

    def __init__(self, hass: HomeAssistant, coordinator) -> None:
        self.hass = hass
        self.coordinator = coordinator
        self._unsubs: list[CALLBACK_TYPE] = []
        self._debounce_unsub: CALLBACK_TYPE | None = None

    async def async_setup(self) -> None:
        for event_type in (
            er.EVENT_ENTITY_REGISTRY_UPDATED,
            ar.EVENT_AREA_REGISTRY_UPDATED,
            fr.EVENT_FLOOR_REGISTRY_UPDATED,
        ):
            self._unsubs.append(
                self.hass.bus.async_listen(event_type, self._schedule_push)
            )
        # Initial mirror so a fresh install is usable immediately.
        await self.async_push()

    @callback
    def _schedule_push(self, _event=None) -> None:
        if self._debounce_unsub is not None:
            self._debounce_unsub()
        self._debounce_unsub = async_call_later(
            self.hass, VOCABULARY_DEBOUNCE_SECONDS, self._debounced
        )

    async def _debounced(self, _now) -> None:
        self._debounce_unsub = None
        await self.async_push()

    async def async_push(self) -> None:
        snapshot = build_snapshot(self.hass)
        try:
            result = await self.coordinator.api.push_vocabulary(snapshot)
        except MurdockApiError as exc:
            # Murdock keeps working from its last stored snapshot.
            _LOGGER.warning("Vocabulary push failed: %s", exc)
            return
        self.coordinator.vocabulary_version = str(snapshot["version"])
        _LOGGER.debug(
            "Vocabulary pushed: %d entities, %s terms",
            len(snapshot["entities"]), result.get("term_count"),
        )

    async def async_shutdown(self) -> None:
        if self._debounce_unsub is not None:
            self._debounce_unsub()
            self._debounce_unsub = None
        for unsub in self._unsubs:
            unsub()
        self._unsubs.clear()
