# VoiceID v2

Self-hosted **speaker-recognition Wyoming proxy** for Home Assistant. Sits
between a voice satellite (e.g. Home Assistant Voice PE) and a downstream
STT engine (faster-whisper, Parakeet, etc.) and acts as a *gatekeeper*:
it decides **who** is speaking, blocks unknown voices (including TV
audio), and injects speaker context back into Home Assistant so the
conversation agent knows whose command it's handling.

> Reference base: [jxlarrea/wyoming-voice-match](https://github.com/jxlarrea/wyoming-voice-match).
> VoiceID replaces the ECAPA-TDNN embedder with WeSpeaker **CAM++ (ONNX)**,
> adds a **Web UI**, **sqlite-vec**-backed storage, **unknown-voice logging**,
> **liveness heuristics**, and direct **HA REST integration**.

## Architecture

```
Voice Satellite → Wake Word → [VoiceID Proxy] → STT Engine → HA Assist Pipeline
                                    │
                                    ├─ CAM++ embedding (CPU, onnxruntime)
                                    ├─ sqlite-vec KNN (cosine distance)
                                    ├─ Silero VAD (enrollment QC)
                                    ├─ Liveness heuristics
                                    └─ HA REST (input_text + event bus)
```

**Known speaker** → audio is forwarded to upstream STT, speaker name is
pushed into `input_text.current_speaker`, and a
`speaker_recognition_detected` event is fired.

**Unknown speaker** → empty transcript (command blocked). Audio is
optionally logged to an "enrollment postbox" with TTL so you can review
and tag the sample later.

## Features

- **Wyoming proxy** — registers itself as an ASR provider; Home Assistant
  auto-discovers it.
- **CAM++ ONNX** — 7 M params, 192-dim embeddings, <20 ms inference on
  CPU (no GPU needed).
- **sqlite-vec** — single-file database with SIMD-accelerated KNN. No
  more `.npy` files to manage.
- **Web UI** — speaker enrollment (browser mic or WAV upload), sample
  review with playback, unknown-voice review, runtime settings.
- **Silero VAD** — rejects enrollment samples with too little speech.
- **Unknown logging** — untagged samples auto-expire after 48 h (configurable).
- **Privacy toggle** — disable unknown logging entirely with one click.
- **Liveness heuristic** — spectral rolloff / crest factor / HF ratio
  label samples as "likely live" or "likely TV".
- **HA integration** — REST call to set `input_text.current_speaker`,
  and `speaker_recognition_detected` events on the HA event bus.

## Quick start

```bash
# 1. Build
docker compose build

# 2. Configure environment
cat > .env <<EOF
HA_URL=http://192.168.x.x:8123
HA_TOKEN=your_long_lived_token
HA_TV_ENTITY=media_player.living_room_tv
EOF

# 3. Run
docker compose up -d

# 4. Open the Web UI
open http://localhost:8099
```

Make sure your upstream ASR (e.g. `wyoming-faster-whisper`) is reachable
at the address in `UPSTREAM_URI` in `docker-compose.yml`.

### Home Assistant setup

1. **Voice → Devices & Services**: Home Assistant should auto-discover
   VoiceID as a Wyoming ASR provider at `tcp://<host>:10350`.
2. **Assist Pipeline**: select `voiceid-proxy` as the speech-to-text engine.
3. **Helper**: create `input_text.current_speaker` (or use a different
   entity and set `HA_INPUT_TEXT_ENTITY`).
4. **Conversation agent**: add a template to your `extra_system_prompt`:

   ```
   The current speaker is: {{ states('input_text.current_speaker') }}
   ```

### Enrolling speakers

**Web UI (recommended):** record 3–5 samples of 5 s each per speaker
directly in your browser, or upload WAV files.

**CLI:**

```bash
docker compose exec voiceid python -m scripts.enroll \
    --speaker jonas \
    /app/data/samples/jonas_1.wav \
    /app/data/samples/jonas_2.wav \
    /app/data/samples/jonas_3.wav
```

## Configuration

All settings are environment variables (see `docker-compose.yml` for the
full list). The key ones:

| Variable | Default | Description |
|---|---|---|
| `LISTEN_URI` | `tcp://0.0.0.0:10350` | Wyoming proxy bind address |
| `UPSTREAM_URI` | `tcp://localhost:10300` | Upstream STT engine |
| `WEB_PORT` | `8099` | Web UI port |
| `VERIFY_THRESHOLD` | `0.30` | Max cosine distance to accept a match |
| `TV_THRESHOLD_BOOST` | `0.05` | Tighten threshold when TV is playing |
| `UNKNOWN_LOGGING` | `true` | Log rejected audio for later review |
| `UNKNOWN_TTL_HOURS` | `48` | TTL for untagged unknown samples |
| `REQUIRE_SPEAKER_MATCH` | `true` | Block unknowns (vs forward anyway) |
| `HA_URL` | — | Base URL of your Home Assistant instance |
| `HA_TOKEN` | — | Long-lived access token |
| `HA_TV_ENTITY` | — | Optional media player for TV-aware thresholding |

Thresholds, privacy toggle and require-match can also be changed at
runtime from the Web UI (stored in the database).

## Project layout

```
voiceid/
├── Dockerfile
├── docker-compose.yml
├── requirements.txt
├── wyoming_voiceid/          # Wyoming proxy handler + entry point
├── voiceid/
│   ├── config.py             # Env-based settings
│   ├── core/
│   │   ├── audio.py          # PCM/WAV utilities
│   │   ├── fbank.py          # Kaldi-compatible log-mel filterbank
│   │   ├── embeddings.py     # CAM++ ONNX inference
│   │   ├── vad.py            # Silero VAD wrapper
│   │   ├── db.py             # SQLite + sqlite-vec schema
│   │   ├── speaker_store.py  # Enrollment, verification, CRUD
│   │   ├── unknown_store.py  # Unknown-sample logging + TTL cleanup
│   │   ├── liveness.py       # Spectral liveness heuristic
│   │   ├── ha_integration.py # HA REST client
│   │   └── context.py        # Shared app context
│   ├── api/                  # FastAPI Web UI backend
│   │   ├── app.py
│   │   ├── routes_speakers.py
│   │   ├── routes_unknown.py
│   │   └── routes_settings.py
│   └── ui/static/            # HTML/CSS/JS frontend
├── scripts/
│   ├── enroll.py             # CLI enrollment helper
│   └── download_models.sh    # Fetch CAM++ + Silero ONNX models
└── data/                     # Persistent volume (voiceid.db lives here)
```

## Post-MVP roadmap

- Embedding-based clustering of unknown samples
- Platt-scaling confidence calibration
- Adaptive thresholds per speaker
- ML-trained liveness classifier (using gathered TV samples)
- Continuous learning (opt-in re-enrollment)
- MQTT output as an alternative to HA REST

## License

TBD — the upstream reference `wyoming-voice-match` is MIT-licensed;
VoiceID is a rewrite, not a direct fork, but borrows Wyoming plumbing
patterns from it.
