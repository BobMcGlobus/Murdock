"""Constants for the Murdock integration."""

from __future__ import annotations

DOMAIN = "murdock"

# Config entry keys
CONF_BASE_URL = "base_url"
CONF_TOKEN = "token"
CONF_SATELLITES = "satellites"
CONF_SATELLITE_ID = "satellite_id"
CONF_SATELLITE_ENTITY = "satellite_entity"

# Options
CONF_CONTEXT_MODE = "context_mode"
CONF_FRESHNESS_WINDOW = "freshness_window"
CONF_MIN_CONFIDENCE = "min_confidence"
CONF_MIRROR_VOCABULARY = "mirror_vocabulary"
CONF_MQTT_PREFIX = "mqtt_topic_prefix"

CONTEXT_MODE_CLEAN = "clean"
CONTEXT_MODE_INLINE = "inline"
CONTEXT_MODE_SIDECAR = "sidecar"
CONTEXT_MODE_AUTO = "auto"
CONTEXT_MODES = [
    CONTEXT_MODE_CLEAN,
    CONTEXT_MODE_INLINE,
    CONTEXT_MODE_SIDECAR,
    CONTEXT_MODE_AUTO,
]

DEFAULT_CONTEXT_MODE = CONTEXT_MODE_SIDECAR
DEFAULT_FRESHNESS_WINDOW = 30  # seconds
DEFAULT_MIN_CONFIDENCE = 0.0   # prompt line names the speaker above this
DEFAULT_MIRROR_VOCABULARY = True
DEFAULT_MQTT_PREFIX = "murdock"

# Two independent delivery paths, because Murdock has two:
#   * HA event bus — fed by Murdock's REST push (needs a token)
#   * MQTT topic   — fed by Murdock's MQTT client (the recommended,
#                    token-free setup); an MQTT message is NOT an HA
#                    event, so it has to be subscribed explicitly.
EVENT_SPEAKER_RECOGNITION = "speaker_recognition_detected"
MQTT_EVENT_SUFFIX = "event/recognition"

# Dispatcher signal: fired with the satellite_id on every state change.
SIGNAL_SPEAKER_UPDATE = f"{DOMAIN}_speaker_update"

# Debounce for registry → vocabulary pushes (plan §9).
VOCABULARY_DEBOUNCE_SECONDS = 5.0

# Slow poll keeping the connection sensor honest between events.
HEALTH_POLL_INTERVAL_SECONDS = 300
