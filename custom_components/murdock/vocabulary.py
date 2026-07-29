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


def _alias_strings(hass: HomeAssistant, entry) -> list[str]:
    """Resolve an entry's aliases to plain strings.

    ``RegistryEntry.aliases`` is ``list[str | ComputedNameType]`` in
    current Home Assistant: the ``COMPUTED_NAME`` sentinel stands for the
    computed full entity name (device name + entity name) and is only
    expanded by ``async_get_entity_aliases``. Touching the raw list
    directly — sorting it, for instance — blows up on that sentinel.
    Older cores expose a plain ``set[str]``, hence the fallback.
    """
    try:
        return sorted(
            {a.strip() for a in er.async_get_entity_aliases(hass, entry) if a}
        )
    except AttributeError:
        # Pre-sentinel Home Assistant: aliases are already strings.
        return sorted(
            {str(a).strip() for a in (entry.aliases or []) if isinstance(a, str)}
        )
    except Exception:
        _LOGGER.debug(
            "Could not resolve aliases for %s", entry.entity_id, exc_info=True
        )
        return []


def build_snapshot(hass: HomeAssistant) -> dict[str, Any]:
    """Assemble the vocabulary payload from the registries.

    Per-entity failures are skipped rather than aborting the snapshot: one
    odd registry entry must not cost the household its whole vocabulary.
    """
    ent_reg = er.async_get(hass)
    area_reg = ar.async_get(hass)
    floor_reg = fr.async_get(hass)

    entities: list[dict[str, Any]] = []
    for entry in ent_reg.entities.values():
        try:
            if entry.disabled_by or entry.hidden_by:
                continue
            if not async_should_expose(hass, _CONVERSATION, entry.entity_id):
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
                    "name": str(name),
                    "aliases": _alias_strings(hass, entry),
                    "area": area.name if area else None,
                    "floor": floor.name if floor else None,
                    "domain": entry.domain,
                }
            )
        except Exception:
            _LOGGER.debug(
                "Skipping %s in vocabulary snapshot",
                getattr(entry, "entity_id", "?"), exc_info=True,
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
        """Build and push a snapshot. Never raises.

        Vocabulary is an enhancement, not a prerequisite: whatever goes
        wrong here, the speaker path must keep working and Murdock keeps
        using its last stored snapshot.
        """
        try:
            snapshot = build_snapshot(self.hass)
        except Exception:
            _LOGGER.exception("Could not build the vocabulary snapshot")
            return
        try:
            result = await self.coordinator.api.push_vocabulary(snapshot)
        except MurdockApiError as exc:
            # Murdock keeps working from its last stored snapshot.
            _LOGGER.warning("Vocabulary push failed: %s", exc)
            return
        except Exception:
            _LOGGER.exception("Unexpected error pushing the vocabulary")
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
