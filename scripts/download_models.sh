#!/usr/bin/env bash
# Download the CAM++ speaker embedder and Silero VAD ONNX models.
#
# Each download is validated as an actual ONNX/protobuf file before being
# moved into place — many model hubs (looking at you, ModelScope) silently
# return an HTML landing page on direct file links, which then explodes at
# runtime with "Protobuf parsing failed".
#
# Behaviour flags (set via env):
#   SKIP_MODEL_DOWNLOAD=1  — print a warning and exit 0 if a download
#                            fails. Useful during `docker build` when
#                            Hugging Face is rate-limited; the container
#                            can then pick up the model from a mounted
#                            volume at startup instead.
set -uo pipefail

MODEL_DIR="${1:-./models}"
mkdir -p "$MODEL_DIR"

# CAM++ (3D-Speaker iic checkpoint, 192-dim, ~28 MB).
# Multiple mirrors because HF occasionally returns 500/503 under load and
# we can't afford to fail the build for a transient CDN hiccup.
CAMPP_URLS=(
    "https://huggingface.co/model-scope/CosyVoice-300M/resolve/main/campplus.onnx"
    "https://huggingface.co/gpustack/CosyVoice-300M-Instruct/resolve/main/campplus.onnx"
    "https://huggingface.co/FunAudioLLM/CosyVoice2-0.5B/resolve/main/campplus.onnx"
    "https://huggingface.co/mradermacher/CosyVoice-300M-GGUF/resolve/main/campplus.onnx"
    "https://hf-mirror.com/model-scope/CosyVoice-300M/resolve/main/campplus.onnx"
    "https://hf-mirror.com/gpustack/CosyVoice-300M-Instruct/resolve/main/campplus.onnx"
)

# Emotion (emotion2vec+ base) — OPT-IN, ~356 MB, only fetched when
# DOWNLOAD_EMOTION_MODEL=1. Two files: the ONNX is the feature extractor
# and emits frame-level 768-dim features; the tiny .bin is the 9-class
# linear head. Without both there is no classifier, only features.
EMOTION_URLS=(
    "https://huggingface.co/ykevinc/emotion2vec-plus-base-onnx/resolve/main/model.onnx"
)
EMOTION_HEAD_URLS=(
    "https://huggingface.co/ykevinc/emotion2vec-plus-base-onnx/resolve/main/classifier.bin"
)

# Silero VAD v5 — ~2 MB, hosted directly in the upstream GitHub repo.
SILERO_URLS=(
    "https://github.com/snakers4/silero-vad/raw/master/src/silero_vad/data/silero_vad.onnx"
    "https://raw.githubusercontent.com/snakers4/silero-vad/master/src/silero_vad/data/silero_vad.onnx"
)

# Validate that a file looks like a real ONNX (protobuf) blob.
# Real ONNX files start with the bytes 08 ?? 12 (ir_version field, varint),
# but the cheap and very reliable check is "not text/HTML and >1 MB".
validate_onnx() {
    local path="$1"
    local min_bytes="$2"
    if [ ! -s "$path" ]; then
        return 1
    fi
    local size
    size=$(stat -c %s "$path" 2>/dev/null || stat -f %z "$path")
    if [ "$size" -lt "$min_bytes" ]; then
        echo "  -> file too small ($size bytes < $min_bytes)" >&2
        return 1
    fi
    # Reject any file that starts with HTML / XML / JSON.
    local head
    head=$(head -c 16 "$path" | tr -d '\0' | tr '[:upper:]' '[:lower:]')
    case "$head" in
        *"<!doctype"*|*"<html"*|*"<?xml"*|"{"*)
            echo "  -> downloaded file looks like text, not ONNX" >&2
            return 1
            ;;
    esac
    return 0
}

# Try one URL, with aggressive curl-level retries. The outer loop in
# download_with_fallback handles per-URL failures; this handles transient
# 5xx responses within a single mirror.
try_download() {
    local url="$1"
    local tmp="$2"
    # --retry 6 + --retry-delay 5 gives us up to ~30s of patience on a
    # single mirror, which is usually enough to ride out HF hiccups.
    curl -fsSL \
        --connect-timeout 15 \
        --max-time 300 \
        --retry 6 \
        --retry-delay 5 \
        --retry-all-errors \
        -o "$tmp" \
        "$url"
}

download_with_fallback() {
    local target="$1"
    local min_bytes="$2"
    shift 2
    local urls=("$@")

    if [ -f "$target" ] && validate_onnx "$target" "$min_bytes"; then
        echo "$(basename "$target") already present and valid."
        return 0
    fi

    # Two full passes through the mirror list so transient 5xx across
    # ALL mirrors (HF CDN-wide blip) don't doom the build on the first
    # sweep. Total worst-case wait: ~5 min, still inside Docker's
    # default build timeout.
    local pass
    for pass in 1 2; do
        for url in "${urls[@]}"; do
            echo "[pass $pass] Downloading $(basename "$target") from $url"
            local tmp="${target}.partial"
            if try_download "$url" "$tmp"; then
                if validate_onnx "$tmp" "$min_bytes"; then
                    mv "$tmp" "$target"
                    local sz
                    sz=$(stat -c %s "$target" 2>/dev/null || stat -f %z "$target")
                    echo "  ok ($sz bytes)"
                    return 0
                else
                    echo "  validation failed, trying next mirror"
                    rm -f "$tmp"
                fi
            else
                echo "  curl failed, trying next mirror"
                rm -f "$tmp"
            fi
        done
        if [ "$pass" -lt 2 ]; then
            echo "All mirrors failed on pass $pass — waiting 15s before second pass…"
            sleep 15
        fi
    done

    echo "ERROR: could not download $(basename "$target") from any mirror" >&2
    return 1
}

missing=0
download_with_fallback "$MODEL_DIR/campplus.onnx" 10000000 "${CAMPP_URLS[@]}" \
    || missing=1
download_with_fallback "$MODEL_DIR/silero_vad.onnx" 500000 "${SILERO_URLS[@]}" \
    || missing=1

# Emotion detection is opt-in and large, so it is never part of the
# required set — a failure here must not block startup.
if [ "${DOWNLOAD_EMOTION_MODEL:-0}" = "1" ]; then
    echo
    echo "Emotion model requested (~356 MB) — this takes a while."
    if download_with_fallback "$MODEL_DIR/emotion.onnx" 100000000 "${EMOTION_URLS[@]}"; then
        # The head is ~27 KB, far below the ONNX size floor, so validate
        # it by exact size: 8-byte header + 9x768 float32 weights + 9 bias.
        if curl -fsSL --retry 3 -o "$MODEL_DIR/emotion_head.bin" "${EMOTION_HEAD_URLS[0]}"; then
            head_size=$(stat -c %s "$MODEL_DIR/emotion_head.bin" 2>/dev/null                 || stat -f %z "$MODEL_DIR/emotion_head.bin")
            if [ "$head_size" -ne 27692 ]; then
                echo "  emotion head has $head_size bytes, expected 27692 — discarding" >&2
                rm -f "$MODEL_DIR/emotion_head.bin" "$MODEL_DIR/emotion.onnx"
            else
                echo "  emotion classifier head OK ($head_size bytes)"
            fi
        else
            echo "  could not fetch the emotion head — removing the ONNX too," >&2
            echo "  since the extractor alone cannot classify anything." >&2
            rm -f "$MODEL_DIR/emotion.onnx"
        fi
    fi
fi

echo
echo "Models in $MODEL_DIR:"
ls -lh "$MODEL_DIR" 2>/dev/null || true

if [ "$missing" -ne 0 ]; then
    if [ "${SKIP_MODEL_DOWNLOAD:-0}" = "1" ]; then
        echo
        echo "WARNING: at least one model is missing, but SKIP_MODEL_DOWNLOAD=1" >&2
        echo "         is set — continuing anyway. Murdock will try to fetch" >&2
        echo "         the missing files at container startup, or you can" >&2
        echo "         mount a pre-populated models directory via a volume." >&2
        exit 0
    fi
    exit 1
fi
