"""LLM API: the per-turn speaker line (plan §18).

Registered like Herold's API; enable it in the conversation agent's
options (Voice assistants → agent → LLM APIs). HA rebuilds the prompt on
every turn (`async_get_api_instance` is called fresh per request), which
is exactly what makes this reliable where `extra_system_prompt` — set
once per conversation — goes stale.

Phase 1 ships no tools; the value is the prompt contribution alone.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

from homeassistant.helpers import llm

from .const import DOMAIN

if TYPE_CHECKING:
    from homeassistant.core import HomeAssistant

    from .coordinator import MurdockCoordinator

_LOGGER = logging.getLogger(__name__)

_INSTRUCTION = (
    'Die Zeile "Sprecher:" nennt die per Stimmerkennung identifizierte '
    'Person. Steht dort "unbekannt" oder "unsicher", nimm nicht an, dass '
    "es der Hauptnutzer ist — frage nach, bevor du etwas Personenbezogenes "
    "tust oder speicherst."
)

_WHISPER_LINE = (
    "Die Person flüstert. Antworte ebenfalls leise und knapp, und lies das "
    "nicht als Aufforderung, etwas geheim zu halten."
)

_HINT_INSTRUCTION = (
    '"Transkript-Hinweis" nennt eine zweite mögliche Lesart der Äußerung. '
    "Wenn die Zweitlesart besser zu einer existierenden Entität passt, "
    "nutze sie."
)


def build_speaker_line(
    coordinator: MurdockCoordinator, device_id: str | None
) -> str:
    """The one line that must never be missing (plan §18).

    A silent absence reads as "no objections, probably the main user" —
    so unknown/stale/uncertain states are spelled out explicitly.
    """
    state = coordinator.state_for_device(device_id)
    if state is None:
        return "Sprecher: unbekannt"
    if state.uncertain:
        return "Sprecher: unsicher"
    if not state.speaker:
        return "Sprecher: unbekannt"
    area = coordinator.area_name_for_satellite(state.satellite_id)
    location = area or state.satellite_id
    line = (
        f"Sprecher: {state.speaker} "
        f"(Konfidenz {state.confidence:.2f}, Satellit {location})"
    )
    if state.role:
        line += f" — Rolle: {state.role}"
    return line


def build_api_prompt(
    coordinator: MurdockCoordinator, llm_context: llm.LLMContext
) -> str:
    parts = [
        build_speaker_line(coordinator, llm_context.device_id),
        _INSTRUCTION,
    ]
    state = coordinator.state_for_device(llm_context.device_id)
    # Whispering is worth telling the agent even when the speaker is
    # unknown: the useful reaction is a quieter, shorter answer, and that
    # does not depend on knowing who it was.
    if state and state.whisper:
        parts.append(_WHISPER_LINE)
    if state and state.ambiguities:
        hints = "; ".join(_render_hint(a) for a in state.ambiguities)
        parts.append(f"Transkript-Hinweis: {hints}")
        parts.append(_HINT_INSTRUCTION)
    return "\n".join(parts)


def _render_hint(hint) -> str:
    """Phrase one ambiguity for the prompt."""
    if hint.kind == "additional" or not hint.original:
        return f'eine zweite Engine hörte zusätzlich "{hint.alternative}"'
    if hint.kind == "reading":
        return f'die Äußerung könnte auch lauten: "{hint.alternative}"'
    return f'"{hint.original}" könnte auch "{hint.alternative}" heißen'


class MurdockLLMApi(llm.API):
    """Prompt-only LLM API exposing the current speaker."""

    def __init__(
        self, hass: HomeAssistant, coordinator: MurdockCoordinator
    ) -> None:
        super().__init__(hass=hass, id=DOMAIN, name="Murdock")
        self.coordinator = coordinator

    async def async_get_api_instance(
        self, llm_context: llm.LLMContext
    ) -> llm.APIInstance:
        """Build the per-turn prompt. Tools follow in phase 2."""
        return llm.APIInstance(
            api=self,
            api_prompt=build_api_prompt(self.coordinator, llm_context),
            llm_context=llm_context,
            tools=[],
        )
