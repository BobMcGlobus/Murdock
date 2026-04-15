# VoiceID

Speaker-recognition proxy that sits between your Wyoming satellites and
your existing STT engine (faster-whisper, wyoming-whisper, Parakeet, …).
VoiceID transparently transcribes via the upstream and additionally
identifies *who* is speaking, pushing the result to Home Assistant so
your automations can personalise responses.

## How it works

```
 ┌──────────────┐   Wyoming   ┌──────────┐   Wyoming    ┌──────────────┐
 │  Satellite   │ ─────────►  │ VoiceID  │ ──────────►  │   STT        │
 │ (ESPHome,    │             │ (addon)  │              │ (whisper …)  │
 │  wyoming-sat)│ ◄─────────  │          │ ◄──────────  │              │
 └──────────────┘  transcript └──────────┘  transcript  └──────────────┘
                                   │
                                   │ state push
                                   ▼
                           ┌──────────────┐
                           │ Home         │
                           │ Assistant    │
                           └──────────────┘
```

Every voice pipeline utterance flows through VoiceID. It passes the
audio to your STT upstream unchanged, but in parallel runs a speaker
embedder to decide whether the voice belongs to one of your enrolled
speakers. The recognised name (plus confidence, nearest speaker,
distance, role) is published to configurable Home Assistant entities.

## Installation

1. In **Settings → Add-ons → Add-on store**, click the three-dot menu →
   **Repositories** and add the VoiceID repository URL.
2. Refresh the store, find **VoiceID**, click **Install**.
3. On the **Configuration** tab set `upstream_uri` to the address of
   your existing STT server (e.g. `tcp://core-whisper:10300` if you run
   faster-whisper on the same HA host).
4. Start the addon, then open it — the Web UI is served through HA's
   ingress so you get it at **Settings → Add-ons → VoiceID → OPEN WEB UI**.

First start downloads the CAM++ embedding model (~27 MB) and Silero VAD
(~2 MB) into the addon's persistent `/data/models` — subsequent restarts
are instant.

## Configuration

| Option | Description | Default |
|---|---|---|
| `upstream_uri` | Wyoming URI of the STT server VoiceID forwards audio to. `host:port` is auto-prefixed with `tcp://`. | `tcp://core-whisper:10300` |
| `log_level` | `debug` / `info` / `warning` / `error`. | `info` |
| `advertised_languages` | Comma-separated language codes for the Wyoming Info event (e.g. `de,en`). Empty = auto-detect from upstream. | `de,en` |

**Everything else** — verify threshold, speaker enrollment, HA entity
mapping, auto-enroll, per-satellite thresholds, sample quality weights,
voice cluster review — is configured through the Web UI and persisted
to the addon's `/data` volume. Addon updates don't wipe your speakers.

## Enrolling speakers

Open the Web UI → **Speakers** tab, click **Start recording**, say a
sentence or two, enter a name and hit **Enroll**. Repeat 3–5 times with
different moods/sentences to train a robust voice profile.

You can also upload pre-recorded audio (WAV, MP3, M4A, OGG, FLAC, WebM)
if you already have samples lying around.

## Wiring it into automations

The **Home Assistant** tab has one-click generators for the `input_text`
helpers VoiceID writes into. The `Copy HA template` button produces a
YAML blob you can drop into `configuration.yaml`. Example automation:

```yaml
automation:
  - alias: "Morning greeting by speaker"
    trigger:
      - platform: state
        entity_id: input_text.current_speaker
    condition:
      - "{{ trigger.to_state.state not in ['', 'unknown', 'Unknown'] }}"
    action:
      - service: tts.speak
        data:
          message: "Good morning, {{ states('input_text.current_speaker') }}"
```

## Ingress

The Web UI is exposed through HA's ingress, so you don't need to open
port 8099 to the LAN. If you prefer direct access (browser bookmarks
etc.), the optional `8099/tcp` port can be mapped on the addon's
**Network** tab.

## Troubleshooting

- **"Upstream unreachable"** in the Settings tab: the `upstream_uri`
  host isn't resolvable from inside the addon. Use `core-whisper` (the
  hostname of another addon) or an IP — `host.docker.internal` does
  not work inside HA addons.
- **Always "unknown"**: run **Debug VAD** in the Speakers tab on one of
  your enrollments. If the speech ratio is 0 % the upstream audio
  isn't reaching the embedder; check that the satellite actually
  streams audio and the `skip_leading_seconds` setting isn't trimming
  everything.
- **HA entities not updating**: check the Home Assistant tab in the Web
  UI. The **Test** button verifies the addon can reach HA; if it fails,
  the supervisor token wasn't injected — restart the addon.

## Uninstall

Stopping and removing the addon is safe — your speakers are stored in
`/data/voiceid.db` and will still be there if you reinstall. To wipe
everything, delete the addon's data volume (Supervisor CLI:
`ha addons uninstall local_voiceid --remove-data`).
