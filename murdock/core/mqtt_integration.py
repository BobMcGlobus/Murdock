"""MQTT integration with Home Assistant Auto-Discovery.

Bidirectional MQTT bridge for Murdock:

  * **Publish** — recognition results to state topics. When HA's MQTT
    integration is active, entities appear automatically via the
    discovery protocol; no manual helper creation required.
  * **Subscribe** — context signals (TV state, presence) that HA *pushes*
    onto retained context topics. This inverts the old REST flow: instead
    of Murdock pulling HA state with a long-lived token, HA publishes the
    state and Murdock caches it. No token required — just broker creds.

Topic layout (prefix defaults to ``murdock``)::

    murdock/status                              online | offline (LWT)
    murdock/sensor/<name>/state                 published recognition state
    murdock/binary_sensor/<name>/state          published recognition state
    murdock/event/recognition                   full JSON event
    murdock/context/<room>/tv                    HA → {"playing": true}   (retained)
    murdock/context/<room>/presence              HA → {"present": true}   (retained)
    murdock/context/global/tv                    fallback when no room match

Architecture:
    - On connect: publish discovery configs + online status, subscribe to
      the context tree, then iterate incoming messages into the cache.
    - On disconnect: LWT fires "offline" automatically (broker-side); a
      clean shutdown publishes "offline" explicitly first.
    - Reconnect loop handles transient broker outages with backoff. All
      context topics are retained, so Murdock pulls the last known state
      immediately on (re)connect — never blind after a restart.
"""

from __future__ import annotations

import asyncio
import json
import logging
import time
from typing import Optional

from murdock import __version__

_LOGGER = logging.getLogger("murdock.mqtt")

# Default topic prefix — all state/discovery/context topics hang off this.
DEFAULT_PREFIX = "murdock"
DEFAULT_DISCOVERY_PREFIX = "homeassistant"

# Context entries older than this (seconds) are treated as stale and
# ignored. Retained messages have no age, so we stamp arrival time and
# let very old presence signals decay rather than acting on a value HA
# stopped refreshing hours ago. TV/presence republish on every change,
# so a generous window is safe.
_CONTEXT_TTL_SECONDS = 3600.0


class MQTTClient:
    """Async MQTT client: publishes recognition, subscribes to context."""

    def __init__(
        self,
        host: str = "",
        port: int = 1883,
        username: str = "",
        password: str = "",
        topic_prefix: str = DEFAULT_PREFIX,
        discovery_prefix: str = DEFAULT_DISCOVERY_PREFIX,
        node_id: str = "murdock",
    ) -> None:
        self.host = host
        self.port = port
        self.username = username
        self.password = password
        self.topic_prefix = topic_prefix
        self.discovery_prefix = discovery_prefix
        self.node_id = node_id

        self._client = None
        self._connected = False
        self._connect_task: Optional[asyncio.Task] = None
        self._stop_event = asyncio.Event()

        # Context cache: {(room, key): (value: dict, arrived_at: float)}.
        # Populated from retained `murdock/context/<room>/<key>` messages.
        self._context: dict[tuple[str, str], tuple[dict, float]] = {}

    @property
    def configured(self) -> bool:
        """True when at least a host is set."""
        return bool(self.host)

    @property
    def connected(self) -> bool:
        return self._connected

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    async def start(self) -> None:
        """Start the MQTT connection loop (non-blocking)."""
        if not self.configured:
            _LOGGER.debug("MQTT not configured — skipping start")
            return
        if self._connect_task and not self._connect_task.done():
            return
        self._stop_event.clear()
        self._connect_task = asyncio.create_task(self._connection_loop())
        _LOGGER.info("MQTT client starting → %s:%d", self.host, self.port)

    async def stop(self) -> None:
        """Gracefully disconnect and stop the connection loop."""
        self._stop_event.set()
        # Best-effort explicit offline before tearing down (LWT is the
        # fallback if the disconnect isn't clean).
        if self._connected and self._client is not None:
            try:
                await self._publish_availability(self._client, "offline")
            except Exception:
                pass
        if self._connect_task and not self._connect_task.done():
            self._connect_task.cancel()
            try:
                await self._connect_task
            except (asyncio.CancelledError, Exception):
                pass
        self._connected = False
        self._client = None
        self._context.clear()
        _LOGGER.info("MQTT client stopped")

    async def reconfigure(
        self,
        host: Optional[str] = None,
        port: Optional[int] = None,
        username: Optional[str] = None,
        password: Optional[str] = None,
        topic_prefix: Optional[str] = None,
        discovery_prefix: Optional[str] = None,
    ) -> None:
        """Update settings and reconnect."""
        if host is not None:
            self.host = host.strip()
        if port is not None:
            self.port = port
        if username is not None:
            self.username = username
        if password is not None:
            self.password = password
        if topic_prefix is not None:
            self.topic_prefix = topic_prefix.strip() or DEFAULT_PREFIX
        if discovery_prefix is not None:
            self.discovery_prefix = discovery_prefix.strip() or DEFAULT_DISCOVERY_PREFIX
        # Restart connection with new settings.
        await self.stop()
        await self.start()

    # ------------------------------------------------------------------
    # Connection loop with auto-reconnect + message handling
    # ------------------------------------------------------------------

    async def _connection_loop(self) -> None:
        """Maintain MQTT connection with exponential backoff on failure.

        Once connected, this also serves as the inbound message pump:
        we iterate ``client.messages`` to fold context updates into the
        cache. Publishing happens concurrently from other tasks via the
        shared ``self._client`` handle.
        """
        try:
            import aiomqtt
        except ImportError:
            _LOGGER.error(
                "aiomqtt not installed — MQTT integration unavailable. "
                "Install with: pip install aiomqtt"
            )
            return

        backoff = 1.0
        max_backoff = 60.0

        while not self._stop_event.is_set():
            try:
                will = aiomqtt.Will(
                    topic=f"{self.topic_prefix}/status",
                    payload="offline",
                    qos=1,
                    retain=True,
                )

                kwargs = {
                    "hostname": self.host,
                    "port": self.port,
                    "will": will,
                    "keepalive": 60,
                }
                if self.username:
                    kwargs["username"] = self.username
                if self.password:
                    kwargs["password"] = self.password

                async with aiomqtt.Client(**kwargs) as client:
                    self._client = client
                    self._connected = True
                    backoff = 1.0
                    _LOGGER.info("MQTT connected to %s:%d", self.host, self.port)

                    # Publish online status + discovery configs.
                    await self._publish_availability(client, "online")
                    await self._publish_discovery(client)

                    # Subscribe to the context tree — HA pushes TV /
                    # presence here. Retained messages arrive immediately.
                    await client.subscribe(f"{self.topic_prefix}/context/#")

                    # Inbound pump: blocks until the connection drops or
                    # the task is cancelled on stop().
                    async for message in client.messages:
                        self._handle_context_message(message)

            except asyncio.CancelledError:
                break
            except Exception as exc:
                self._connected = False
                self._client = None
                _LOGGER.warning(
                    "MQTT connection failed (%s), retrying in %.0fs",
                    exc, backoff,
                )
                try:
                    await asyncio.wait_for(
                        self._stop_event.wait(), timeout=backoff
                    )
                    break  # stop was requested during backoff
                except asyncio.TimeoutError:
                    pass
                backoff = min(backoff * 2, max_backoff)

        self._connected = False
        self._client = None

    # ------------------------------------------------------------------
    # Inbound context handling
    # ------------------------------------------------------------------

    def _handle_context_message(self, message) -> None:
        """Fold an inbound `murdock/context/<room>/<key>` message into the cache.

        Payloads are JSON objects (e.g. ``{"playing": true}``). A bare
        ``true``/``false`` or string is also accepted and wrapped. An
        empty payload clears that context entry (retained-message
        deletion).
        """
        topic = str(message.topic)
        prefix = f"{self.topic_prefix}/context/"
        if not topic.startswith(prefix):
            return
        rest = topic[len(prefix):]
        parts = rest.split("/")
        if len(parts) != 2:
            _LOGGER.debug("MQTT context topic ignored (bad shape): %s", topic)
            return
        room, key = parts[0], parts[1]

        raw = message.payload
        if isinstance(raw, (bytes, bytearray)):
            raw = raw.decode("utf-8", errors="replace")
        raw = (raw or "").strip()

        if not raw:
            # Empty retained payload → clear the entry.
            self._context.pop((room, key), None)
            _LOGGER.debug("MQTT context cleared: %s/%s", room, key)
            return

        try:
            value = json.loads(raw)
        except (ValueError, TypeError):
            # Tolerate bare scalars: "true", "playing", "1".
            low = raw.lower()
            if low in ("true", "false"):
                value = {"value": low == "true"}
            else:
                value = {"value": raw}

        if not isinstance(value, dict):
            value = {"value": value}

        self._context[(room, key)] = (value, time.monotonic())
        _LOGGER.debug("MQTT context update: %s/%s = %s", room, key, value)

    def _context_get(self, room: str, key: str) -> Optional[dict]:
        """Return a fresh context value dict, or None if missing/stale."""
        entry = self._context.get((room, key))
        if entry is None:
            return None
        value, arrived = entry
        if time.monotonic() - arrived > _CONTEXT_TTL_SECONDS:
            return None
        return value

    @staticmethod
    def _coerce_bool(value: dict, *keys: str) -> Optional[bool]:
        """Pull a boolean out of a context dict by trying several keys."""
        for k in (*keys, "value", "state"):
            if k in value:
                v = value[k]
                if isinstance(v, bool):
                    return v
                if isinstance(v, str):
                    return v.lower() in ("on", "true", "playing", "home", "1", "yes")
                if isinstance(v, (int, float)):
                    return bool(v)
        return None

    def is_tv_playing(self, room: Optional[str] = None) -> Optional[bool]:
        """Return TV state from the context cache.

        Checks the room-specific topic first (``context/<room>/tv``),
        then falls back to ``context/global/tv``. Returns None when no
        context has been received at all, so callers can fall back to the
        legacy REST path.
        """
        if room:
            value = self._context_get(room, "tv")
            if value is not None:
                result = self._coerce_bool(value, "playing")
                if result is not None:
                    return result
        value = self._context_get("global", "tv")
        if value is not None:
            return self._coerce_bool(value, "playing")
        return None

    def is_present(self, room: Optional[str] = None) -> Optional[bool]:
        """Return room/global presence from the context cache, or None."""
        if room:
            value = self._context_get(room, "presence")
            if value is not None:
                result = self._coerce_bool(value, "present", "occupied")
                if result is not None:
                    return result
        value = self._context_get("global", "presence")
        if value is not None:
            return self._coerce_bool(value, "present", "occupied")
        return None

    def context_snapshot(self) -> dict:
        """Return a JSON-serialisable view of the context cache (for the UI)."""
        out: dict = {}
        now = time.monotonic()
        for (room, key), (value, arrived) in self._context.items():
            if now - arrived > _CONTEXT_TTL_SECONDS:
                continue
            out.setdefault(room, {})[key] = {
                "value": value,
                "age_seconds": round(now - arrived, 1),
            }
        return out

    # ------------------------------------------------------------------
    # Discovery: entities auto-register in HA
    # ------------------------------------------------------------------

    def _device_info(self) -> dict:
        """HA device registry entry — groups all entities under one device."""
        return {
            "identifiers": [self.node_id],
            "name": "Murdock",
            "model": "Speaker Recognition Proxy",
            "manufacturer": "Murdock",
            "sw_version": __version__,
        }

    def _discovery_configs(self) -> list[tuple[str, str, dict]]:
        """Return (component, object_id, config) tuples for all entities."""
        device = self._device_info()
        avail = {
            "availability_topic": f"{self.topic_prefix}/status",
            "payload_available": "online",
            "payload_not_available": "offline",
        }
        prefix = self.topic_prefix

        return [
            # --- Sensors ---
            ("sensor", "current_speaker", {
                "name": "Current Speaker",
                "unique_id": f"{self.node_id}_current_speaker",
                "state_topic": f"{prefix}/sensor/current_speaker/state",
                "icon": "mdi:account-voice",
                "device": device,
                **avail,
            }),
            ("sensor", "confidence", {
                "name": "Speaker Confidence",
                "unique_id": f"{self.node_id}_confidence",
                "state_topic": f"{prefix}/sensor/confidence/state",
                "unit_of_measurement": "%",
                "icon": "mdi:percent-circle",
                "device": device,
                **avail,
            }),
            ("sensor", "distance", {
                "name": "Speaker Distance",
                "unique_id": f"{self.node_id}_distance",
                "state_topic": f"{prefix}/sensor/distance/state",
                "icon": "mdi:vector-difference",
                "device": device,
                **avail,
            }),
            ("sensor", "nearest_speaker", {
                "name": "Nearest Speaker",
                "unique_id": f"{self.node_id}_nearest_speaker",
                "state_topic": f"{prefix}/sensor/nearest_speaker/state",
                "icon": "mdi:account-search",
                "device": device,
                **avail,
            }),
            ("sensor", "role", {
                "name": "Speaker Role",
                "unique_id": f"{self.node_id}_role",
                "state_topic": f"{prefix}/sensor/role/state",
                "icon": "mdi:badge-account",
                "device": device,
                **avail,
            }),
            ("sensor", "emotion", {
                "name": "Speaker Emotion",
                "unique_id": f"{self.node_id}_emotion",
                "state_topic": f"{prefix}/sensor/emotion/state",
                "icon": "mdi:emoticon-outline",
                "device": device,
                **avail,
            }),
            ("sensor", "satellite", {
                "name": "Active Satellite",
                "unique_id": f"{self.node_id}_satellite",
                "state_topic": f"{prefix}/sensor/satellite/state",
                "icon": "mdi:satellite-uplink",
                "device": device,
                **avail,
            }),
            # --- Binary sensor ---
            ("binary_sensor", "speaker_recognized", {
                "name": "Speaker Recognized",
                "unique_id": f"{self.node_id}_speaker_recognized",
                "state_topic": f"{prefix}/binary_sensor/speaker_recognized/state",
                "payload_on": "ON",
                "payload_off": "OFF",
                "icon": "mdi:account-check",
                "device_class": "occupancy",
                "device": device,
                **avail,
            }),
        ]

    async def _publish_discovery(self, client) -> None:
        """Publish all discovery configs (retained)."""
        configs = self._discovery_configs()
        for component, object_id, config in configs:
            topic = (
                f"{self.discovery_prefix}/{component}/{self.node_id}"
                f"/{object_id}/config"
            )
            payload = json.dumps(config)
            await client.publish(topic, payload, qos=1, retain=True)
        _LOGGER.info("MQTT discovery published (%d entities)", len(configs))

    async def _publish_availability(self, client, status: str) -> None:
        """Publish online/offline status (retained)."""
        await client.publish(
            f"{self.topic_prefix}/status", status, qos=1, retain=True
        )

    # ------------------------------------------------------------------
    # State publishing
    # ------------------------------------------------------------------

    async def _publish_state(self, topic: str, value: str) -> None:
        """Publish a single state value. Swallows errors."""
        if not self._connected or self._client is None:
            return
        try:
            await self._client.publish(topic, value, qos=0, retain=True)
        except Exception as exc:
            _LOGGER.debug("MQTT publish failed for %s: %s", topic, exc)

    async def publish_recognition(
        self,
        speaker: str,
        confidence: float,
        satellite_id: Optional[str],
        is_known: bool,
        distance: Optional[float] = None,
        threshold: Optional[float] = None,
        nearest_speaker: Optional[str] = None,
        role: Optional[str] = None,
        emotion: Optional[str] = None,
        emotion_confidence: Optional[float] = None,
    ) -> None:
        """Push a recognition result to all state topics."""
        if not self._connected or self._client is None:
            return

        prefix = self.topic_prefix
        confidence_pct = round(confidence * 100, 1)

        await asyncio.gather(
            self._publish_state(
                f"{prefix}/sensor/current_speaker/state", speaker
            ),
            self._publish_state(
                f"{prefix}/sensor/confidence/state", str(confidence_pct)
            ),
            self._publish_state(
                f"{prefix}/sensor/distance/state",
                f"{distance:.4f}" if distance is not None else "",
            ),
            self._publish_state(
                f"{prefix}/sensor/nearest_speaker/state",
                nearest_speaker or "",
            ),
            self._publish_state(
                f"{prefix}/sensor/role/state", role or ""
            ),
            self._publish_state(
                f"{prefix}/sensor/emotion/state", emotion or ""
            ),
            self._publish_state(
                f"{prefix}/sensor/satellite/state", satellite_id or ""
            ),
            self._publish_state(
                f"{prefix}/binary_sensor/speaker_recognized/state",
                "ON" if is_known else "OFF",
            ),
            return_exceptions=True,
        )

        # Also publish a JSON event for advanced automations.
        event_payload = {
            "speaker": speaker,
            "confidence": round(confidence, 4),
            "is_known": is_known,
            "satellite_id": satellite_id,
        }
        if distance is not None:
            event_payload["distance"] = round(distance, 4)
        if threshold is not None:
            event_payload["threshold"] = round(threshold, 4)
        if nearest_speaker:
            event_payload["nearest_speaker"] = nearest_speaker
        if role:
            event_payload["role"] = role
        if emotion:
            event_payload["emotion"] = emotion
            if emotion_confidence is not None:
                event_payload["emotion_confidence"] = round(emotion_confidence, 4)

        await self._publish_state(
            f"{prefix}/event/recognition", json.dumps(event_payload)
        )

    def publish_async(self, coro) -> None:
        """Schedule a fire-and-forget coroutine on the running loop."""
        if not self._connected:
            return
        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:
            return
        loop.create_task(coro)

    # ------------------------------------------------------------------
    # Cleanup: remove discovery on explicit disable
    # ------------------------------------------------------------------

    async def remove_discovery(self) -> None:
        """Publish empty payloads to discovery topics (un-registers entities)."""
        if not self._connected or self._client is None:
            return
        for component, object_id, _ in self._discovery_configs():
            topic = (
                f"{self.discovery_prefix}/{component}/{self.node_id}"
                f"/{object_id}/config"
            )
            await self._client.publish(topic, "", qos=1, retain=True)
        _LOGGER.info("MQTT discovery removed")
