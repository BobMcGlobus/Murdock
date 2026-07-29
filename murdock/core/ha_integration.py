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

from murdock.core.event_payload import build_recognition_payload

_LOGGER = logging.getLogger("murdock.ha")


class HomeAssistantClient:
    """Minimal HA client for setting input_text values and firing events."""

    def __init__(
        self,
        base_url: Optional[str],
        token: Optional[str],
        input_text_entity: str = "input_text.current_speaker",
        tv_entity: Optional[str] = None,
        confidence_entity: Optional[str] = None,
        distance_entity: Optional[str] = None,
        nearest_entity: Optional[str] = None,
        role_entity: Optional[str] = None,
        emotion_entity: Optional[str] = None,
        timeout: float = 3.0,
    ) -> None:
        self.base_url = base_url.rstrip("/") if base_url else None
        self.token = token
        self.input_text_entity = input_text_entity
        self.tv_entity = tv_entity
        self.confidence_entity = confidence_entity
        self.distance_entity = distance_entity
        self.nearest_entity = nearest_entity
        self.role_entity = role_entity
        self.emotion_entity = emotion_entity
        self.timeout = timeout
        self._client: Optional[httpx.AsyncClient] = None

    @property
    def configured(self) -> bool:
        return bool(self.base_url and self.token)

    async def reconfigure(
        self,
        base_url: Optional[str] = None,
        token: Optional[str] = None,
        input_text_entity: Optional[str] = None,
        tv_entity: Optional[str] = None,
        confidence_entity: Optional[str] = None,
        distance_entity: Optional[str] = None,
        nearest_entity: Optional[str] = None,
        role_entity: Optional[str] = None,
        emotion_entity: Optional[str] = None,
    ) -> None:
        """Update connection parameters live and drop the cached client."""
        if base_url is not None:
            self.base_url = base_url.rstrip("/") if base_url else None
        if token is not None:
            self.token = token or None
        if input_text_entity is not None:
            self.input_text_entity = input_text_entity
        if tv_entity is not None:
            self.tv_entity = tv_entity if tv_entity else None
        if confidence_entity is not None:
            self.confidence_entity = confidence_entity or None
        if distance_entity is not None:
            self.distance_entity = distance_entity or None
        if nearest_entity is not None:
            self.nearest_entity = nearest_entity or None
        if role_entity is not None:
            self.role_entity = role_entity or None
        if emotion_entity is not None:
            self.emotion_entity = emotion_entity or None
        # Drop the cached httpx client so the next call picks up the new
        # base_url / token.
        await self.close()

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
        distance: Optional[float] = None,
        threshold: Optional[float] = None,
        nearest_speaker: Optional[str] = None,
        nearest_distance: Optional[float] = None,
        weight: Optional[float] = None,
        margin: Optional[float] = None,
        uncertain: bool = False,
        reason: Optional[str] = None,
        role: Optional[str] = None,
        emotion: Optional[str] = None,
        emotion_confidence: Optional[float] = None,
    ) -> None:
        if not self.configured:
            return
        try:
            client = self._get_client()
            payload = build_recognition_payload(
                speaker=speaker,
                is_known=is_known,
                confidence=confidence,
                satellite_id=satellite_id,
                distance=distance,
                threshold=threshold,
                nearest_speaker=nearest_speaker,
                nearest_distance=nearest_distance,
                weight=weight,
                margin=margin,
                uncertain=uncertain,
                reason=reason,
                role=role,
                emotion=emotion,
                emotion_confidence=emotion_confidence,
            )
            response = await client.post(
                "/api/events/speaker_recognition_detected",
                json=payload,
            )
            if response.status_code >= 400:
                _LOGGER.warning(
                    "HA event fire failed (%d): %s",
                    response.status_code, response.text[:200],
                )
        except Exception as exc:
            _LOGGER.warning("Could not fire speaker event in HA: %s", exc)

    async def _set_input_number(self, entity_id: str, value: float) -> None:
        """Set an input_number helper in HA."""
        if not self.configured or not entity_id:
            return
        try:
            client = self._get_client()
            response = await client.post(
                "/api/services/input_number/set_value",
                json={"entity_id": entity_id, "value": round(value, 4)},
            )
            if response.status_code >= 400:
                _LOGGER.warning(
                    "HA input_number set failed (%d): %s",
                    response.status_code, response.text[:200],
                )
        except Exception as exc:
            _LOGGER.warning("Could not set %s in HA: %s", entity_id, exc)

    async def _set_input_text(self, entity_id: str, value: str) -> None:
        """Set an input_text helper in HA."""
        if not self.configured or not entity_id:
            return
        try:
            client = self._get_client()
            response = await client.post(
                "/api/services/input_text/set_value",
                json={"entity_id": entity_id, "value": value or ""},
            )
            if response.status_code >= 400:
                _LOGGER.warning(
                    "HA input_text set failed (%d): %s",
                    response.status_code, response.text[:200],
                )
        except Exception as exc:
            _LOGGER.warning("Could not set %s in HA: %s", entity_id, exc)

    async def push_recognition(
        self,
        speaker: str,
        confidence: float,
        satellite_id: Optional[str],
        is_known: bool,
        distance: Optional[float] = None,
        threshold: Optional[float] = None,
        nearest_speaker: Optional[str] = None,
        nearest_distance: Optional[float] = None,
        weight: Optional[float] = None,
        margin: Optional[float] = None,
        uncertain: bool = False,
        reason: Optional[str] = None,
        role: Optional[str] = None,
        emotion: Optional[str] = None,
        emotion_confidence: Optional[float] = None,
    ) -> None:
        """Push all recognition data to HA in one go.

        Sets input_text/input_number entities and fires the event.
        All calls are fire-and-forget so they don't block the pipeline.
        """
        if not self.configured:
            return
        # Speaker name
        await self.set_current_speaker(speaker)
        # Optional entity pushes
        if self.confidence_entity:
            await self._set_input_number(self.confidence_entity, confidence)
        if self.distance_entity and distance is not None:
            await self._set_input_number(self.distance_entity, distance)
        if self.nearest_entity:
            await self._set_input_text(self.nearest_entity, nearest_speaker or "")
        if self.role_entity:
            await self._set_input_text(self.role_entity, role or "")
        # Emotion gets its own optional input_text. An empty string is
        # pushed when the classifier had no opinion, so a stale "happy"
        # from the previous utterance doesn't linger in the dashboard.
        if self.emotion_entity:
            await self._set_input_text(self.emotion_entity, emotion or "")
        # Event with full payload
        await self.fire_speaker_event(
            speaker=speaker,
            confidence=confidence,
            satellite_id=satellite_id,
            is_known=is_known,
            distance=distance,
            threshold=threshold,
            nearest_speaker=nearest_speaker,
            nearest_distance=nearest_distance,
            weight=weight,
            margin=margin,
            uncertain=uncertain,
            reason=reason,
            role=role,
            emotion=emotion,
            emotion_confidence=emotion_confidence,
        )

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
