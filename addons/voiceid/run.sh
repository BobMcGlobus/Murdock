#!/usr/bin/env bash
# VoiceID Home Assistant add-on entrypoint.
#
# HA's Supervisor writes the user's addon options to /data/options.json
# on every container start. We parse that with jq instead of bashio —
# keeping the base image free of HA's Alpine s6/bashio stack so we can
# use plain python:3.11-slim (needed for the onnxruntime wheels).
#
# The env vars we export here are the same ones VoiceID already reads
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

UPSTREAM_URI="$(cfg upstream_uri)"
LOG_LEVEL="$(cfg log_level)"
ADVERTISED="$(cfg advertised_languages)"

if [ -z "$UPSTREAM_URI" ]; then
    echo "ERROR: upstream_uri is required in the addon options." >&2
    exit 1
fi

export LISTEN_URI="tcp://0.0.0.0:10350"
export UPSTREAM_URI
export LOG_LEVEL="${LOG_LEVEL:-info}"
export WEB_HOST="0.0.0.0"
export WEB_PORT="8099"

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

# Persistent storage: /data is mounted by the supervisor and survives
# addon updates. Models live there too so the ~30 MB download happens
# once, not on every version bump.
export DATA_DIR="/data"
export MODEL_DIR="/data/models"
mkdir -p "$DATA_DIR" "$MODEL_DIR"

log "Listen URI:   $LISTEN_URI"
log "Upstream URI: $UPSTREAM_URI"
log "Log level:    $LOG_LEVEL"
log "Advertise:    ${ADVERTISED_LANGUAGES:-<auto>}"
log "Data dir:     $DATA_DIR"

exec /app/scripts/entrypoint.sh
