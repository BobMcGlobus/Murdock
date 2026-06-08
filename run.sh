#!/usr/bin/env bash
# Murdock Home Assistant add-on entrypoint.
#
# HA's Supervisor writes the user's addon options to /data/options.json
# on every container start. We parse that with jq instead of bashio —
# keeping the base image free of HA's Alpine s6/bashio stack so we can
# use plain python:3.11-slim (needed for the onnxruntime wheels).
#
# The env vars we export here are the same ones Murdock already reads
# in plain docker-compose deployments, so the Python side doesn't need
# any addon-specific branches.
set -euo pipefail

log() { printf '[run] %s\n' "$*"; }

OPTIONS="/data/options.json"
if [ ! -f "$OPTIONS" ]; then
    echo "ERROR: $OPTIONS not found. Is this really running as a HA addon?" >&2
    exit 1
fi

cfg() { jq -r ".${1} // \"\"" "$OPTIONS"; }

STT_BACKEND="$(cfg stt_backend)"
UPSTREAM_URI="$(cfg upstream_uri)"
MISTRAL_API_KEY="$(cfg mistral_api_key)"
MISTRAL_MODEL="$(cfg mistral_model)"
LOG_LEVEL="$(cfg log_level)"
ADVERTISED="$(cfg advertised_languages)"
MQTT_ENABLED="$(cfg mqtt_enabled)"

# Validate: upstream mode needs a URI, voxtral mode needs an API key.
if [ "${STT_BACKEND:-upstream}" = "upstream" ] && [ -z "$UPSTREAM_URI" ]; then
    echo "ERROR: upstream_uri is required when stt_backend is 'upstream'." >&2
    exit 1
fi

export LISTEN_URI="tcp://0.0.0.0:10350"
export STT_BACKEND="${STT_BACKEND:-upstream}"
export UPSTREAM_URI="${UPSTREAM_URI:-tcp://localhost:10300}"
export LOG_LEVEL="${LOG_LEVEL:-info}"
export WEB_HOST="0.0.0.0"
export WEB_PORT="8099"

# Voxtral / Mistral Cloud settings
if [ -n "$MISTRAL_API_KEY" ]; then
    export MISTRAL_API_KEY
fi
if [ -n "$MISTRAL_MODEL" ]; then
    export MISTRAL_MODEL
fi

# Empty means "auto-detect from upstream" — same semantics as the
# compose deployment, so users aren't surprised moving between them.
if [ -n "$ADVERTISED" ]; then
    export ADVERTISED_LANGUAGES="$ADVERTISED"
fi

# --- Home Assistant hand-off ----------------------------------------------
#
# Addons get SUPERVISOR_TOKEN injected with core-api scope. Using it by
# default means the user doesn't have to mint a long-lived token just
# to light up the HA integration. The Web UI can still override with a
# different token later — this is a default, not a lock-in.
if [ -n "${SUPERVISOR_TOKEN:-}" ]; then
    export HA_URL="${HA_URL:-http://supervisor/core}"
    export HA_TOKEN="${HA_TOKEN:-${SUPERVISOR_TOKEN}}"
    log "HA API wired via supervisor token."
fi

# --- MQTT auto-wiring (services: mqtt:want) -------------------------------
#
# When a Mosquitto broker is installed, the Supervisor exposes its host,
# port and credentials at /services/mqtt. We pull them with the injected
# token so MQTT discovery works out of the box — the user never types a
# broker address. If no broker is published the call 404s and we leave
# MQTT untouched (the Web UI can still configure it manually).
if [ "${MQTT_ENABLED:-true}" = "true" ] && [ -n "${SUPERVISOR_TOKEN:-}" ]; then
    MQTT_JSON="$(curl -sf \
        -H "Authorization: Bearer ${SUPERVISOR_TOKEN}" \
        "http://supervisor/services/mqtt" 2>/dev/null || true)"
    if [ -n "$MQTT_JSON" ] && [ "$(echo "$MQTT_JSON" | jq -r '.result // ""')" = "ok" ]; then
        export MQTT_ENABLED="true"
        export MQTT_HOST="$(echo "$MQTT_JSON" | jq -r '.data.host // ""')"
        export MQTT_PORT="$(echo "$MQTT_JSON" | jq -r '.data.port // 1883')"
        _mqtt_user="$(echo "$MQTT_JSON" | jq -r '.data.username // ""')"
        _mqtt_pass="$(echo "$MQTT_JSON" | jq -r '.data.password // ""')"
        [ -n "$_mqtt_user" ] && export MQTT_USERNAME="$_mqtt_user"
        [ -n "$_mqtt_pass" ] && export MQTT_PASSWORD="$_mqtt_pass"
        log "MQTT auto-wired from Mosquitto service: ${MQTT_HOST}:${MQTT_PORT}"
    else
        log "MQTT enabled but no broker service published — configure host in Web UI."
    fi
fi

# Persistent storage: /data is mounted by the supervisor and survives
# addon updates. Models live there too so the ~30 MB download happens
# once, not on every version bump.
export DATA_DIR="/data"
export MODEL_DIR="/data/models"
mkdir -p "$DATA_DIR" "$MODEL_DIR"

log "STT backend:  $STT_BACKEND"
log "Listen URI:   $LISTEN_URI"
if [ "$STT_BACKEND" = "upstream" ]; then
    log "Upstream URI: $UPSTREAM_URI"
else
    log "Mistral model: ${MISTRAL_MODEL:-voxtral-mini-latest}"
    log "API key:       ${MISTRAL_API_KEY:+set}"
fi
log "Log level:    $LOG_LEVEL"
log "Advertise:    ${ADVERTISED_LANGUAGES:-<auto>}"
log "Data dir:     $DATA_DIR"

exec /app/scripts/entrypoint.sh
