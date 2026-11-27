# Voice Support (Deutsch): the OHF Voice Support app, built with a German
# acoustic model and wake word configuration.
#
# Everything except wake.de.yaml, the model download and the start script
# is the upstream app (https://github.com/OHF-Voice/app-voice-support,
# Apache-2.0) at the tag pinned below, built the way its own Dockerfile
# builds it: graphs and the resident decoder in a builder, a pruned runtime.
ARG BUILD_FROM=ghcr.io/home-assistant/base-debian:trixie-2026.11.0

# ---------------------------------------------------------------------------
# Builder
# ---------------------------------------------------------------------------
FROM ${BUILD_FROM} AS builder
ARG BUILD_ARCH=amd64
ARG APP_VERSION=0.2.1
ARG RHASSPY_SPEECH_REVISION=9e6bcf3af07909f39f71674f62372508d00dd4a9
ARG RHASSPY_SPEECH_URL=https://huggingface.co/datasets/rhasspy/rhasspy-speech/resolve

SHELL ["/bin/bash", "-o", "pipefail", "-c"]

RUN \
    apt-get update \
    && apt-get install -y --no-install-recommends \
        binutils \
        ca-certificates \
        curl \
        perl \
        python3 \
        python3-pip \
        python3-venv \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# The upstream app, source and all, at the pinned tag.
WORKDIR /usr/src
RUN \
    curl -fsSL "https://github.com/OHF-Voice/app-voice-support/archive/refs/tags/${APP_VERSION}.tar.gz" \
        | tar -xz \
    && mv "app-voice-support-${APP_VERSION}/voice_support" /usr/src/voice \
    && rm -rf "app-voice-support-${APP_VERSION}"

WORKDIR /usr/src/voice
RUN \
    python3 -m venv /usr/src/.venv \
    && /usr/src/.venv/bin/pip3 install --no-cache-dir -r requirements.txt
ENV PATH="/usr/src/.venv/bin:${PATH}"

# Kaldi/OpenFST tools as upstream, and the German model instead of the
# English one. The Supervisor passes BUILD_ARCH; the tools are named after
# Docker's arch names.
RUN \
    case "${BUILD_ARCH}" in \
        amd64) TOOLS_ARCH=amd64 ;; \
        aarch64) TOOLS_ARCH=arm64 ;; \
        *) echo "Unsupported architecture ${BUILD_ARCH}" >&2; exit 1 ;; \
    esac \
    && mkdir -p local/wake-kaldi-tools local/wake-kaldi-models \
    && curl -fsSL \
        "${RHASSPY_SPEECH_URL}/${RHASSPY_SPEECH_REVISION}/tools/rhasspy-speech_${TOOLS_ARCH}.tar.gz" \
        | tar -xz -C local/wake-kaldi-tools \
    && curl -fsSL \
        "${RHASSPY_SPEECH_URL}/${RHASSPY_SPEECH_REVISION}/models/de_DE-zamia.tar.gz" \
        | tar -xz -C local/wake-kaldi-models \
    && test -d local/wake-kaldi-tools/kaldi \
    && test -d local/wake-kaldi-models/de_DE-zamia

COPY wake.de.yaml ./wake.yaml

# The resident decoder, exactly as upstream builds it.
RUN \
    apt-get update \
    && apt-get install -y --no-install-recommends \
        libgfortran5 \
        libopenblas0 \
    && apt-get install -y --no-install-recommends \
        g++ \
        libopenblas-dev \
    && bash voice/kaldi/decoderd/build.sh "$PWD/local/wake-kaldi-tools" \
    && rm -rf voice/kaldi/decoderd/.fst_shadow_include \
    && apt-get purge -y --auto-remove g++ libopenblas-dev \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/* \
    && ldd voice/kaldi/decoderd/verify_decoderd > /dev/null

# The German graphs. A phrase or pronunciation the model cannot express
# fails the build here, naming the word.
RUN python3 -m voice --build-graphs --no-ctc

RUN python3 script/prune_runtime local /usr/src/lean-local

# Upstream's s6 services without the discovery one: this app is reached
# through Murdock, never announced to Home Assistant itself.
RUN \
    rm -rf rootfs/etc/s6-overlay/s6-rc.d/discovery \
        rootfs/etc/s6-overlay/s6-rc.d/user/contents.d/discovery \
    && rm -f rootfs/etc/s6-overlay/user-bundles.d/user/contents.d/discovery

# ---------------------------------------------------------------------------
# Runtime: no compiler, no Kaldi binaries, no training FSTs.
# ---------------------------------------------------------------------------
FROM ${BUILD_FROM}

SHELL ["/bin/bash", "-o", "pipefail", "-c"]
WORKDIR /usr/src/voice

RUN \
    apt-get update \
    && apt-get install -y --no-install-recommends \
        ca-certificates \
        curl \
        libgfortran5 \
        libopenblas0 \
        python3 \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

COPY --from=builder /usr/src/.venv /usr/src/.venv
ENV PATH="/usr/src/.venv/bin:${PATH}"
ENV OPENBLAS_NUM_THREADS=1

COPY --from=builder /usr/src/lean-local/ ./local/
COPY --from=builder /usr/src/voice/var/ ./var/
COPY --from=builder /usr/src/voice/voice/ ./voice/
COPY --from=builder /usr/src/voice/wake.yaml /usr/src/voice/config.yaml ./

RUN ldd voice/kaldi/decoderd/verify_decoderd > /dev/null

WORKDIR /
COPY --from=builder /usr/src/voice/rootfs/ /
COPY run /etc/s6-overlay/s6-rc.d/voice/run
RUN chmod a+x /etc/s6-overlay/s6-rc.d/voice/run /etc/s6-overlay/s6-rc.d/voice/finish

HEALTHCHECK --start-period=2m \
    CMD curl -sf http://127.0.0.1:10900/health > /dev/null || exit 1
