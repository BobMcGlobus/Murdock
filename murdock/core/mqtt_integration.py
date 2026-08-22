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
from murdock.core.event_payload import build_recognition_payload

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

# How long an "active satellite" signal stays valid. HA publishes which
# satellite started listening just before the STT request reaches us; by
# the time we verify (a few seconds later) it's still fresh. A short TTL
# means a stale signal from a much earlier utterance is never misapplied.
_ACTIVE_SAT_TTL_SECONDS = 30.0


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

        # Friendly names announced by the active-satellite automation,
        # so the UI can stop showing raw entity ids. A callback lets the
        # context persist them; without one they stay in memory.
        self.satellite_names: dict = {}
        self.on_satellite_name = None

        self._client = None
        self._connected = False
        self._connect_task: Optional[asyncio.Task] = None
        self._stop_event = asyncio.Event()

        # Context cache: {(room, key): (value: dict, arrived_at: float)}.
        # Populated from retained `murdock/context/<room>/<key>` messages.
        self._context: dict[tuple[str, str], tuple[dict, float]] = {}

        # Most-recently-announced active satellite (id, area, arrived_at),
        # from `murdock/active_satellite`. HA's pipeline doesn't pass the
        # device to the STT stage, so this is how Murdock learns which
        # satellite a recognition came from (and, optionally, its room so
        # media playing in that room can tighten the threshold).
        self._active_satellite: Optional[tuple[str, Optional[str], float]] = None

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
        self._active_satellite = None
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
                    # …and the active-satellite signal (which device is
                    # currently running a pipeline).
                    await client.subscribe(f"{self.topic_prefix}/active_satellite")

                    # Inbound pump: blocks until the connection drops or
                    # the task is cancelled on stop().
                    active_topic = f"{self.topic_prefix}/active_satellite"
                    async for message in client.messages:
                        if str(message.topic) == active_topic:
                            self._handle_active_satellite(message)
                        else:
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

    def _handle_active_satellite(self, message) -> None:
        """Record which satellite HA says is currently active.

        Payload is either a bare id string (e.g. an entity id or room) or
        a JSON object ``{"id": "...", "area": "...", "name": "..."}``. The
        optional area lets media playing in that room tighten the
        threshold; the optional name is what the UI shows instead of the
        entity id. An empty payload clears it.
        """
        raw = message.payload
        if isinstance(raw, (bytes, bytearray)):
            raw = raw.decode("utf-8", errors="replace")
        raw = (raw or "").strip()
        if not raw:
            self._active_satellite = None
            return
        sat_id = raw
        area: Optional[str] = None
        if raw.startswith("{"):
            try:
                obj = json.loads(raw)
            except (ValueError, TypeError):
                obj = None
            if isinstance(obj, dict):
                sat_id = str(obj.get("id") or obj.get("satellite") or "").strip() or raw
                a = obj.get("area")
                area = str(a).strip() if a else None
                n = obj.get("name") or obj.get("friendly_name")
                name = str(n).strip() if n else ""
                if sat_id and name and self.satellite_names.get(sat_id) != name:
                    self.satellite_names[sat_id] = name
                    if self.on_satellite_name is not None:
                        try:
                            self.on_satellite_name(sat_id, name)
                        except Exception:
                            _LOGGER.debug(
                                "Storing satellite name failed", exc_info=True
                            )
        if not sat_id:
            self._active_satellite = None
            return
        self._active_satellite = (sat_id, area, time.monotonic())
        _LOGGER.debug("MQTT active satellite: %s (area=%s)", sat_id, area)

    def _fresh_active_satellite(
        self, max_age: float
    ) -> Optional[tuple[str, Optional[str]]]:
        if self._active_satellite is None:
            return None
        sat_id, area, arrived = self._active_satellite
        if time.monotonic() - arrived > max_age:
            return None
        return (sat_id, area)

    def get_active_satellite(
        self, max_age: float = _ACTIVE_SAT_TTL_SECONDS
    ) -> Optional[str]:
        """Return the satellite id HA last announced, if still fresh."""
        fresh = self._fresh_active_satellite(max_age)
        return fresh[0] if fresh else None

    def get_active_satellite_area(
        self, max_age: float = _ACTIVE_SAT_TTL_SECONDS
    ) -> Optional[str]:
        """Return the active satellite's room/area, if announced and fresh."""
        fresh = self._fresh_active_satellite(max_age)
        return fresh[1] if fresh else None

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

    def known_media(self) -> list[dict]:
        """Return every media player Murdock has heard about (fresh only).

        Each entry: ``{"entity_id", "area", "playing"}``. Sourced from the
        ``context/media/<entity_id>`` topics one HA automation publishes
        for all TVs/radios/speakers.
        """
        now = time.monotonic()
        out: list[dict] = []
        for (room, key), (value, arrived) in self._context.items():
            if room != "media":
                continue
            if now - arrived > _CONTEXT_TTL_SECONDS:
                continue
            out.append({
                "entity_id": key,
                "area": value.get("area"),
                "playing": bool(self._coerce_bool(value, "playing")),
            })
        return out

    def playing_media(self) -> list[dict]:
        """Subset of :meth:`known_media` that is currently playing."""
        return [m for m in self.known_media() if m["playing"]]

    def is_tv_playing(self, room: Optional[str] = None) -> Optional[bool]:
        """Return the legacy single-TV signal from the context cache.

        Checks ``context/<room>/tv`` first, then ``context/global/tv``.
        Per-media-player state is handled separately (``known_media`` /
        ``playing_media`` and the restriction matrix in the context
        layer). Returns None when there's no TV context at all, so callers
        can fall back to the legacy REST poll.
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
            ("sensor", "voice_style", {
                "name": "Voice style",
                "unique_id": f"{self.node_id}_voice_style",
                "state_topic": f"{prefix}/sensor/voice_style/state",
                "icon": "mdi:tune-vertical",
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
            ("sensor", "whisper_score", {
                "name": "Whisper score",
                "unique_id": f"{self.node_id}_whisper_score",
                "state_topic": f"{prefix}/sensor/whisper_score/state",
                "icon": "mdi:waveform",
                "device": device,
                **avail,
            }),
            ("binary_sensor", "whisper", {
                "name": "Whispering",
                "unique_id": f"{self.node_id}_whisper",
                "state_topic": f"{prefix}/binary_sensor/whisper/state",
                "payload_on": "ON",
                "payload_off": "OFF",
                "icon": "mdi:account-voice-off",
                "device": device,
                **avail,
            }),
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
        nearest_distance: Optional[float] = None,
        weight: Optional[float] = None,
        margin: Optional[float] = None,
        uncertain: bool = False,
        reason: Optional[str] = None,
        role: Optional[str] = None,
        ambiguities: Optional[list] = None,
        whisper: bool = False,
        whisper_score: Optional[float] = None,
        voice_style: Optional[str] = None,
        speakers: Optional[list] = None,
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
                f"{prefix}/sensor/voice_style/state", voice_style or "normal"
            ),
            self._publish_state(
                f"{prefix}/sensor/satellite/state", satellite_id or ""
            ),
            self._publish_state(
                f"{prefix}/binary_sensor/speaker_recognized/state",
                "ON" if is_known else "OFF",
            ),
            # Whispering is published as its own entity, so an automation
            # can duck the TTS volume without parsing the JSON event.
            self._publish_state(
                f"{prefix}/binary_sensor/whisper/state",
                "ON" if whisper else "OFF",
            ),
            self._publish_state(
                f"{prefix}/sensor/whisper_score/state",
                f"{whisper_score:.3f}" if whisper_score is not None else "",
            ),
            return_exceptions=True,
        )

        # Also publish a JSON event for advanced automations. Same shape
        # as the HA event (speaker=null on non-recognition, see
        # murdock/core/event_payload.py).
        event_payload = build_recognition_payload(
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
            ambiguities=ambiguities,
            whisper=whisper,
            speakers=speakers,
        )

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
