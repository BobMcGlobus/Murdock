"""Home Assistant REST + Event integration.

Keeps the handler loosely coupled to HA by sending fire-and-forget
requests in a background task. If HA is unreachable, failures are logged
but do not affect the Wyoming pipeline.
"""

from __future__ import annotations

import asyncio
import logging
from typing import Optional

import httpx

_LOGGER = logging.getLogger("voiceid.ha")


class HomeAssistantClient:
    """Minimal HA client for setting input_text values and firing events."""

    def __init__(
        self,
        base_url: Optional[str],
        token: Optional[str],
        input_text_entity: str = "input_text.current_speaker",
        tv_entity: Optional[str] = None,
        timeout: float = 3.0,
    ) -> None:
        self.base_url = base_url.rstrip("/") if base_url else None
        self.token = token
        self.input_text_entity = input_text_entity
        self.tv_entity = tv_entity
        self.timeout = timeout
        self._client: Optional[httpx.AsyncClient] = None

    @property
    def configured(self) -> bool:
        return bool(self.base_url and self.token)

    def _get_client(self) -> httpx.AsyncClient:
        if self._client is None:
            self._client = httpx.AsyncClient(
                base_url=self.base_url or "",
                headers={
                    "Authorization": f"Bearer {self.token or ''}",
                    "Content-Type": "application/json",
                },
                timeout=self.timeout,
            )
        return self._client

    async def close(self) -> None:
        if self._client is not None:
            await self._client.aclose()
            self._client = None

    async def set_current_speaker(self, speaker_name: str) -> None:
        if not self.configured:
            return
        try:
            client = self._get_client()
            response = await client.post(
                "/api/services/input_text/set_value",
                json={"entity_id": self.input_text_entity, "value": speaker_name},
            )
            if response.status_code >= 400:
                _LOGGER.warning(
                    "HA set_value failed (%d): %s",
                    response.status_code, response.text[:200],
                )
        except Exception as exc:
            _LOGGER.warning("Could not set current speaker in HA: %s", exc)

    async def fire_speaker_event(
        self,
        speaker: str,
        confidence: float,
        satellite_id: Optional[str],
        is_known: bool,
    ) -> None:
        if not self.configured:
            return
        try:
            client = self._get_client()
            response = await client.post(
                "/api/events/speaker_recognition_detected",
                json={
                    "speaker": speaker,
                    "confidence": confidence,
                    "satellite_id": satellite_id,
                    "is_known": is_known,
                },
            )
            if response.status_code >= 400:
                _LOGGER.warning(
                    "HA event fire failed (%d): %s",
                    response.status_code, response.text[:200],
                )
        except Exception as exc:
            _LOGGER.warning("Could not fire speaker event in HA: %s", exc)

    async def is_tv_playing(self) -> bool:
        """Return True if the configured media player entity is in 'playing' state."""
        if not self.configured or not self.tv_entity:
            return False
        try:
            client = self._get_client()
            response = await client.get(f"/api/states/{self.tv_entity}")
            if response.status_code != 200:
                return False
            data = response.json()
            return data.get("state") == "playing"
        except Exception as exc:
            _LOGGER.debug("Could not read TV state: %s", exc)
            return False

    def publish_async(self, coro) -> None:
        """Schedule a fire-and-forget coroutine on the running loop."""
        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:
            _LOGGER.debug("No event loop; dropping HA call")
            return
        loop.create_task(coro)
