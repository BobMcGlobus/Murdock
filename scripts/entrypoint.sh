#!/usr/bin/env bash
# Container entrypoint.
#
# Responsibility: make sure the ONNX models exist before handing off to
# the Python app. The models are normally downloaded at build time, but
# if Hugging Face was flaky during the build (SKIP_MODEL_DOWNLOAD=1)
# the files can be missing. We try to recover in three ways:
#
#   1. Nothing to do — files already in /app/models (e.g. mounted volume)
#   2. Fetch them now via scripts/download_models.sh
#   3. Print a clear error and refuse to start
#
# This keeps the fast path cheap (milliseconds to stat two files) while
# giving users a safety net for HF outages.
set -uo pipefail

MODEL_DIR="${MODEL_DIR:-/app/models}"
CAMPP="$MODEL_DIR/campplus.onnx"
SILERO="$MODEL_DIR/silero_vad.onnx"

need_download=0
if [ ! -s "$CAMPP" ]; then
    echo "entrypoint: $CAMPP is missing — will try to download"
    need_download=1
fi
if [ ! -s "$SILERO" ]; then
    echo "entrypoint: $SILERO is missing — will try to download"
    need_download=1
fi

if [ "$need_download" -eq 1 ]; then
    mkdir -p "$MODEL_DIR"
    if ! bash /app/scripts/download_models.sh "$MODEL_DIR"; then
        echo
        echo "====================================================================="
        echo " ERROR: Murdock couldn't download the required ONNX models."
        echo
        echo " Hugging Face is probably rate-limiting or down. Options:"
        echo
        echo "  1. Wait a few minutes and restart the container:"
        echo "       docker compose restart murdock"
        echo
        echo "  2. Download the models on your host and mount them into"
        echo "     the container. Add this volume to docker-compose.yml:"
        echo
        echo "       volumes:"
        echo "         - ./models:/app/models"
        echo
        echo "     Then on the host:"
        echo "       mkdir -p ./models"
        echo "       curl -L -o ./models/campplus.onnx \\"
        echo "         https://huggingface.co/model-scope/CosyVoice-300M/resolve/main/campplus.onnx"
        echo "       curl -L -o ./models/silero_vad.onnx \\"
        echo "         https://github.com/snakers4/silero-vad/raw/master/src/silero_vad/data/silero_vad.onnx"
        echo "       docker compose up -d"
        echo "====================================================================="
        exit 1
    fi
fi

exec python -m wyoming_murdock "$@"
