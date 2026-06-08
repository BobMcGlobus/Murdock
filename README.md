# Murdock v2

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
- **Liveness heuristic** — spectral rolloff / crest factor / HF ratio
  label samples as "likely live" or "likely TV".
- **Auto-enroll with smart replacement** — high-confidence matches add
  fresh embeddings to the profile, replacing the lowest-quality sample
  when the cap is reached so the profile ages with the user's voice.

### Quality & tuning
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

### Deployment
- **Home Assistant add-on** — one-click install from this repo as a
  custom repository; Web UI served through HA ingress (no port
  forwarding needed); supervisor token auto-wired so the HA
  integration works out of the box without a long-lived token.
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

See [`addons/murdock/DOCS.md`](addons/murdock/DOCS.md) for details.

### Option B — docker-compose

```bash
# 1. Build
docker compose build

# 2. Start (upstream STT config is in docker-compose.yml)
docker compose up -d

# 3. Open the Web UI
open http://localhost:8099
```

Home Assistant URL, token and entities are configured **in the Web UI**
(Settings → Home Assistant tab) — no `.env` file required.

### Home Assistant setup

1. **Voice → Devices & Services**: Home Assistant should auto-discover
   Murdock as a Wyoming ASR provider at `tcp://<host>:10350`.
2. **Assist Pipeline**: select `murdock-proxy` as the speech-to-text engine.
3. **Web UI → Home Assistant tab**: enter your HA URL + long-lived
   token (the add-on auto-wires this), and click **Copy HA template**
   for a ready-made `configuration.yaml` snippet with all helper
   entities.
4. **Conversation agent**: add to `extra_system_prompt`:

   ```
   The current speaker is: {{ states('input_text.current_speaker') }}
   ```

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
├── Dockerfile                    # docker-compose image
├── docker-compose.yml
├── requirements.txt
├── addons/                       # Home Assistant add-on
│   ├── repository.yaml           # HA "Add repository" metadata
│   └── murdock/
│       ├── config.yaml           # Addon options, ports, ingress
│       ├── Dockerfile            # addon image (python:3.11-slim)
│       ├── run.sh                # /data/options.json → env bootstrap
│       ├── build.yaml
│       ├── DOCS.md               # shown in HA on the Documentation tab
│       └── README.md
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
│   │   ├── ha_integration.py     # HA REST client
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

- Emotion detection on verified speech
- Parakeet STT bridge / integration notes
- Platt-scaling confidence calibration
- Adaptive thresholds per speaker (beyond per-satellite)
- ML-trained liveness classifier (using gathered TV samples)
- MQTT output as an alternative to HA REST

## License

TBD — the upstream reference `wyoming-voice-match` is MIT-licensed;
Murdock is a rewrite, not a direct fork, but borrows Wyoming plumbing
patterns from it.
