# Murdock v2

[![CI](https://github.com/BobMcGlobus/Murdock/actions/workflows/ci.yml/badge.svg)](https://github.com/BobMcGlobus/Murdock/actions/workflows/ci.yml)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

Self-hosted **speaker-recognition Wyoming proxy** for Home Assistant. Sits
between a voice satellite (e.g. Home Assistant Voice PE) and a downstream
STT engine (faster-whisper, Parakeet, etc.) and acts as a *gatekeeper*:
it decides **who** is speaking, blocks unknown voices (including TV
audio), and injects speaker context back into Home Assistant so the
conversation agent knows whose command it's handling.

> Reference base: [jxlarrea/wyoming-voice-match](https://github.com/jxlarrea/wyoming-voice-match).
> Murdock replaces the ECAPA-TDNN embedder with WeSpeaker **CAM++ (ONNX)**,
> adds a **Web UI**, **sqlite-vec**-backed storage, **unknown-voice logging**,
> **liveness heuristics**, **sample quality scoring**, **voice clustering**,
> and direct **HA REST integration**.

## Architecture

```
Voice Satellite → Wake Word → [Murdock Proxy] → STT Engine → HA Assist Pipeline
                                    │
                                    ├─ CAM++ embedding (CPU, onnxruntime)
                                    ├─ sqlite-vec KNN (cosine distance)
                                    ├─ Silero VAD (enrollment QC)
                                    ├─ Liveness heuristics (TV / background rejection)
                                    ├─ Sample quality scoring (SNR, consistency, centroid fit)
                                    ├─ Per-satellite threshold overrides
                                    └─ HA REST (input_text + event bus)
```

**Known speaker** → audio is forwarded to upstream STT, speaker name is
pushed into `input_text.current_speaker` (plus optional confidence,
distance, nearest-speaker and role entities), and a
`speaker_recognition_detected` event is fired.

**Unknown speaker** → empty transcript (command blocked). Audio is
optionally logged to an "enrollment postbox" with TTL so you can review,
cluster, and tag the sample later.

## Features

### Core recognition
- **Wyoming proxy** — registers as an ASR provider; HA auto-discovers it.
- **CAM++ ONNX** — 7 M params, 192-dim embeddings, <20 ms inference on
  CPU (no GPU needed).
- **sqlite-vec** — single-file database with SIMD-accelerated KNN.
- **Silero VAD** — rejects enrollment samples with too little speech.
- **Adaptive speaker extraction** — when an utterance contains more than
  one speech region (target + TV, or a second person), each region is
  embedded and scored, and only the dominant enrolled speaker's regions
  are kept before verification — so a background voice no longer drags
  the embedding into "unknown". A fast-path skips single-region clips
  entirely, so the common case adds only one cheap VAD pass.
- **Liveness heuristic** — spectral rolloff / crest factor / HF ratio
  label samples as "likely live" or "likely TV".
- **Auto-enroll with smart replacement** — high-confidence matches add
  fresh embeddings to the profile, replacing the lowest-quality sample
  when the cap is reached so the profile ages with the user's voice.

### Quality & tuning
- **Confidence calibration (Platt scaling)** — maps the raw cosine
  distance to a calibrated *probability that it's the same speaker*,
  fitted automatically from your enrolled samples (genuine =
  leave-one-out centroid distances, impostor = cross-speaker distances).
  The confidence reported to HA/MQTT becomes meaningful instead of a bare
  `1 - distance`. Refits in the background on enrollment changes; gating
  still uses the distance threshold. Groundwork for adaptive thresholds
  and context fusion.
- **Sample quality scoring** — composite 0–1 score from speech ratio,
  SNR, liveness, embedding consistency and centroid fit. Per-speaker
  "training quality" badge. Component weights are tunable from the UI.
- **Per-satellite thresholds** — override the verify threshold per
  satellite ID (e.g. a noisy kitchen vs. a quiet study) without
  touching the global default.
- **Voice-sample clustering** — greedy cosine clustering of untagged
  unknown samples, with bulk-assign so labelling the same voice five
  times is one click instead of five.

### UI & operations
- **Web UI** — speaker enrollment (browser mic or WAV/MP3/M4A/OGG/
  FLAC/WebM upload), sample review with playback, unknown-voice
  review, cluster review, recognition event log, runtime settings.
- **Satellite tagging** — samples remember which satellite recorded
  them, visible in the UI.
- **Backup/restore** — ZIP export of all speakers, samples and
  metadata; import with merge or replace mode.
- **DE / EN UI** — full translation for both locales.

### Home Assistant integration
- **MQTT auto-discovery (recommended)** — publishes recognition results
  over MQTT; entities (`sensor.murdock_current_speaker`,
  `binary_sensor.murdock_speaker_recognized`, confidence, distance,
  nearest speaker, role, emotion, satellite) appear automatically. **No
  token, no manual helpers.** In the add-on the broker is auto-wired from
  the Mosquitto service — zero config.
- **Context push (token-free)** — HA publishes TV state / presence onto
  retained `murdock/context/<room>/<key>` topics; Murdock subscribes and
  tightens its threshold while the TV is on. The data flow is inverted
  vs. the old REST path, so Murdock never needs a long-lived token.
- **REST + token (legacy)** — the original push-via-`input_text`/event
  path is still available for users who can't run a broker.

### Deployment
- **Home Assistant add-on** — one-click install from this repo as a
  custom repository; Web UI served through HA ingress (no port
  forwarding needed); MQTT broker auto-wired from Mosquitto.
- **docker-compose** — same image, standalone deployment.

## Quick start

### Option A — Home Assistant add-on (recommended for HA users)

1. **Settings → Add-ons → Add-on store → ⋮ → Repositories** and add
   this repository's URL.
2. Refresh, find **Murdock**, **Install**.
3. **Configuration** tab: set `upstream_uri` to your existing STT
   server (e.g. `tcp://core-whisper:10300` if faster-whisper runs on
   the same HA host).
4. Start the addon, click **Open Web UI**.

See [`DOCS.md`](DOCS.md) for details.

### Option B — docker-compose

```bash
# 1. Build
docker compose build

# 2. Start (upstream STT config is in docker-compose.yml)
docker compose up -d

# 3. Open the Web UI
open http://localhost:8099
```

All integration settings (MQTT broker, HA connection, entities) are
configured **in the Web UI** (Settings tab) — no `.env` file required.

### Home Assistant setup

1. **Voice → Devices & Services**: Home Assistant should auto-discover
   Murdock as a Wyoming ASR provider at `tcp://<host>:10350`.
2. **Assist Pipeline**: select `murdock-proxy` as the speech-to-text engine.
3. **Integration** — pick one:
   - **MQTT (recommended)**: enable it in **Web UI → MQTT**. In the
     add-on the broker is auto-wired from Mosquitto; standalone, enter
     your broker host/credentials. Entities appear automatically under a
     **Murdock** device — no helpers to create.
   - **REST (legacy)**: **Web UI → Home Assistant** tab, enter HA URL +
     long-lived token, then **Copy HA template** for a ready-made
     `configuration.yaml` snippet with all helper entities.
4. **Conversation agent**: add to `extra_system_prompt`:

   ```
   The current speaker is: {{ states('sensor.murdock_current_speaker') }}
   ```

### MQTT topics

| Topic | Direction | Payload |
|---|---|---|
| `homeassistant/<comp>/murdock/<id>/config` | Murdock → HA | discovery (retained) |
| `murdock/status` | Murdock → HA | `online` / `offline` (LWT, retained) |
| `murdock/sensor/<name>/state` | Murdock → HA | recognition state (retained) |
| `murdock/binary_sensor/speaker_recognized/state` | Murdock → HA | `ON` / `OFF` |
| `murdock/event/recognition` | Murdock → HA | full JSON event |
| `murdock/context/<room>/tv` | HA → Murdock | `{"playing": true}` (retain!) |
| `murdock/context/<room>/presence` | HA → Murdock | `{"present": true}` (retain!) |

Context topics **must** be published with `retain: true` so Murdock pulls
the last known state immediately on (re)connect instead of starting blind.
The Web UI's **MQTT → Context push** section generates a ready-to-paste HA
automation for the TV topic. `<room>` should match the satellite name
Murdock sees; `global` is the fallback when no room-specific topic exists.

### Enrolling speakers

**Web UI (recommended):** record 3–5 samples of 5 s each per speaker
directly in your browser, or upload audio files (WAV, MP3, M4A, OGG,
FLAC, WebM). The quality score on each sample tells you whether the
recording is actually training-worthy.

**CLI:**

```bash
docker compose exec murdock python -m scripts.enroll \
    --speaker jonas \
    /app/data/samples/jonas_1.wav \
    /app/data/samples/jonas_2.wav \
    /app/data/samples/jonas_3.wav
```

## Configuration

**Everything except the bootstrap knobs is configured in the Web UI**
and persisted in the database. The env vars below are only needed on
first start (docker-compose deployment).

| Variable | Default | Description |
|---|---|---|
| `LISTEN_URI` | `tcp://0.0.0.0:10350` | Wyoming proxy bind address |
| `UPSTREAM_URI` | `tcp://localhost:10300` | Upstream STT engine |
| `WEB_PORT` | `8099` | Web UI port |
| `VERIFY_THRESHOLD` | `0.30` | Max cosine distance to accept a match |
| `ADVERTISED_LANGUAGES` | — | Force languages in Wyoming Info (e.g. `de,en`) |
| `LOG_LEVEL` | `info` | `debug` / `info` / `warning` / `error` |

All runtime knobs — verify threshold (global and per-satellite), unknown
logging, require-match, auto-enroll, quality weights, HA connection,
TV entity — live in the UI and survive restarts.

## Project layout

```
.
├── Dockerfile                    # HA addon image (also used by HA builder)
├── Dockerfile.standalone         # docker-compose image
├── config.yaml                   # HA addon options, ports, ingress
├── build.yaml                    # HA addon build config
├── run.sh                        # HA addon entrypoint (options.json → env)
├── repository.yaml               # HA "Add repository" metadata
├── docker-compose.yml
├── docker-compose.prod.yml       # Production (Unraid, NAS)
├── requirements.txt
├── DOCS.md                       # HA addon documentation tab
├── CHANGELOG.md
├── wyoming_murdock/              # Wyoming proxy handler + entry point
├── murdock/                      # Core Python package
│   ├── config.py                 # Env-based settings
│   ├── core/
│   │   ├── audio.py              # PCM/WAV utilities, ffmpeg decode
│   │   ├── fbank.py              # Kaldi-compatible log-mel filterbank
│   │   ├── embeddings.py         # CAM++ ONNX inference
│   │   ├── vad.py                # Silero VAD wrapper
│   │   ├── db.py                 # SQLite + sqlite-vec schema + migrations
│   │   ├── speaker_store.py      # Enrollment, verification, CRUD
│   │   ├── unknown_store.py      # Unknown-sample logging + TTL cleanup
│   │   ├── unknown_cluster.py    # Greedy cosine clustering of unknowns
│   │   ├── sample_quality.py     # Composite quality scoring
│   │   ├── liveness.py           # Spectral liveness heuristic
│   │   ├── extraction.py         # Adaptive target-speaker region extraction
│   │   ├── calibration.py        # Platt-scaling confidence calibration
│   │   ├── ha_integration.py     # HA REST client (legacy push path)
│   │   ├── mqtt_integration.py   # MQTT discovery + context subscribe (recommended)
│   │   ├── recognition_log.py    # Event log store
│   │   ├── info_cache.py         # Wyoming Info cache + upstream describe
│   │   └── context.py            # Shared app context
│   ├── api/                      # FastAPI Web UI backend
│   │   ├── app.py
│   │   ├── routes_speakers.py
│   │   ├── routes_unknown.py     # + clustering + bulk-assign
│   │   ├── routes_settings.py    # + per-satellite thresholds
│   │   ├── routes_recognition.py
│   │   └── routes_backup.py
│   └── ui/static/                # HTML/CSS/JS frontend (i18n: DE/EN)
├── scripts/
│   ├── enroll.py                 # CLI enrollment helper
│   ├── entrypoint.sh             # Container entrypoint (model download)
│   └── download_models.sh        # Fetch CAM++ + Silero ONNX models
└── data/                         # Persistent volume (murdock.db lives here)
```

## Post-MVP roadmap

- Fold trustworthy recognition events into the calibration fit (not just
  enrollment-derived pairs)
- Adaptive thresholds per speaker, in calibrated-probability space
- Parakeet STT bridge / integration notes
- ML-trained liveness classifier (using gathered TV samples)
- Multi-modal identity fusion (voice + phone presence + room + mmWave)

## License

[MIT](LICENSE) © 2026 BobMcGlobus. The upstream reference
`wyoming-voice-match` is also MIT-licensed; Murdock is a rewrite, not a
direct fork, but borrows Wyoming plumbing patterns from it.
