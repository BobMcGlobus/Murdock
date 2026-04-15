# VoiceID Home Assistant Add-on image.
#
# We use python:3.11-slim instead of HA's alpine-based base image because
# onnxruntime does not publish musl wheels, and compiling it on Alpine
# inside the addon build sandbox is painfully slow (and sometimes OOMs).
# python:3.11-slim is multi-arch (amd64 + aarch64) and matches what our
# normal docker-compose deployment already uses, so one codebase covers
# both delivery paths.
#
# The BUILD_FROM arg is accepted-but-ignored for compatibility with HA's
# builder — the builder passes it on every arch and errors out if the
# Dockerfile doesn't declare it.
ARG BUILD_FROM
FROM python:3.11-slim

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    DATA_DIR=/data \
    MODEL_DIR=/data/models

# ffmpeg for non-WAV uploads (WebM/Opus from the browser recorder, MP3/
# M4A enrolls, …). libsndfile1 for soundfile edge cases. jq to parse
# HA's /data/options.json at startup without pulling in bashio.
RUN apt-get update \
    && apt-get install -y --no-install-recommends \
        bash \
        ca-certificates \
        curl \
        ffmpeg \
        jq \
        libsndfile1 \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Install Python deps first so application edits don't bust the layer.
COPY requirements.txt /app/requirements.txt
RUN pip install --no-cache-dir -r /app/requirements.txt

# Project code — paths relative to the repo root (build context).
COPY scripts/ /app/scripts/
COPY voiceid/ /app/voiceid/
COPY wyoming_voiceid/ /app/wyoming_voiceid/

COPY addons/voiceid/run.sh /run.sh
RUN chmod +x /run.sh /app/scripts/entrypoint.sh /app/scripts/download_models.sh

# Wyoming protocol for satellites, :8099 is the Web UI (via HA ingress
# or direct port access).
EXPOSE 10350 8099

CMD ["/run.sh"]
