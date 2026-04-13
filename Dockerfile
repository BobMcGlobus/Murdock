FROM python:3.11-slim

LABEL description="VoiceID v2 — Wyoming speaker-recognition proxy for Home Assistant"

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
        curl \
        libsndfile1 \
        ffmpeg \
        ca-certificates \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY scripts/ scripts/
COPY voiceid/ voiceid/
COPY wyoming_voiceid/ wyoming_voiceid/

# Download ONNX models at build time so the runtime image is self-contained.
# SKIP_MODEL_DOWNLOAD=1 makes the build tolerant of transient Hugging Face
# 5xx errors: if the download fails, the container will retry on startup
# and/or pick up the files from a mounted ./models volume.
RUN mkdir -p /app/models \
    && SKIP_MODEL_DOWNLOAD=1 bash scripts/download_models.sh /app/models

RUN mkdir -p /app/data

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    DATA_DIR=/app/data \
    MODEL_DIR=/app/models

EXPOSE 10350 8099

ENTRYPOINT ["bash", "/app/scripts/entrypoint.sh"]
