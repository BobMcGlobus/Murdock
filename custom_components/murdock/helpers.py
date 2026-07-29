"""Public interface for other integrations (plan §19).

Chronist and friends read the speaker through this — never from a tool
parameter the model filled in. The prompt line is for conversational
tone; data integrity has to bypass the model.
"""

from __future__ import annotations

from homeassistant.core import HomeAssistant

from .const import DOMAIN, SIGNAL_SPEAKER_UPDATE  # noqa: F401  (re-export)
from .coordinator import MurdockCoordinator, SpeakerState


def _coordinator(hass: HomeAssistant) -> MurdockCoordinator | None:
    entries = hass.data.get(DOMAIN) or {}
    for coordinator in entries.values():
        return coordinator
    return None


async def async_get_speaker(
    hass: HomeAssistant, *, device_id: str
) -> SpeakerState | None:
    """Fresh speaker state for the satellite mapped to ``device_id``.

    Returns None when the integration is not set up, the device has no
    explicit satellite mapping, or the last recognition has aged out of
    the freshness window. Subscribe to ``SIGNAL_SPEAKER_UPDATE`` via the
    dispatcher to react to speaker changes.
    """
    coordinator = _coordinator(hass)
    if coordinator is None:
        return None
    return coordinator.state_for_device(device_id)
