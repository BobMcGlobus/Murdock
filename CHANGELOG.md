# Changelog

## 0.3.0

Satellite identity and media-aware gating.

- **Satellite identification over MQTT** — HA's pipeline never tells the
  STT stage which device is speaking, so per-satellite features stayed
  dark. A small HA automation now publishes the active satellite (and its
  room) on `murdock/active_satellite` when it starts listening; Murdock
  attributes the recognition to it. The Web UI generates the automation
  (MQTT → Satellite identification).
- **Media context, one automation for all** — a single HA automation
  publishes every TV/radio/speaker's playing-state + room to
  `murdock/context/media/<entity>`. Murdock tightens the threshold when
  something plays in the active satellite's room (MQTT → Media context).
- **Per-satellite × per-source restriction matrix** — set how much each
  media source tightens each satellite while playing (living-room TV
  restricts the living-room satellite strongly; a bedroom radio not at
  all). The strongest playing source wins; an explicit `0` disables a
  source. New "Media restrictions" card in the settings tab.
- Browser cache-busting for CSS/JS so add-on updates no longer leave a
  stale UI (no more "settings won't save / no translations" after an
  update). README brought fully up to date.

## 0.2.3

- **Clarify the "Test upstream" result** — it lists every language the
  *upstream* STT supports (often ~100 for Whisper), which looked like
  Murdock was advertising all of them. It now reads "Upstream supports N
  languages: …" with a note that Murdock only advertises the languages
  you configure. No functional change — the advertised-languages override
  was always respected (verified: with `de,en` set, Murdock's Wyoming
  Describe returns exactly `de, en`).

## 0.2.2

- **Satellite identification over MQTT** — HA's Assist pipeline doesn't
  pass the originating device to the STT stage, so `Transcribe.name`
  arrives empty and per-satellite features stayed dark. Murdock now
  subscribes to a `murdock/active_satellite` topic; a small HA automation
  (generated for you under **MQTT → Satellite identification**) publishes
  the active satellite's room when it starts listening, and Murdock
  attributes the recognition to it. The value is used only when fresh
  (≤ 30 s) and when no name came in over Wyoming, so a directly-connected
  satellite with its own name still wins. This lights up per-satellite
  thresholds and room-based TV context.

## 0.2.1

- **Fix stale UI after updates** — CSS/JS are now cache-busted with the
  running version (`?v=<version>`), so an add-on update no longer leaves
  the browser running old `app.js` / `i18n.js`. This was the cause of the
  MQTT settings appearing unsaved, the Test button doing nothing, and
  missing translations right after updating to 0.2.0. (After updating to
  this version, do one hard refresh — Ctrl/Cmd+Shift+R — to clear the
  pre-fix cache; subsequent updates refresh automatically.)

## 0.2.0

### Home Assistant integration
- **MQTT with auto-discovery (recommended)** — publishes recognition
  results over MQTT; entities appear automatically under a *Murdock*
  device, no token and no manual helpers. In the add-on the broker is
  auto-wired from the Mosquitto service (`services: mqtt:want`).
- **Token-free context push** — Home Assistant publishes TV state /
  presence onto retained `murdock/context/<room>/<key>` topics; Murdock
  subscribes and tightens its threshold while the TV is on. The HA REST +
  long-lived-token path is kept as a legacy option.

### Recognition quality
- **Adaptive target-speaker extraction** — multi-speaker utterances
  (target + TV, or a second person) are split per region; only the
  dominant enrolled speaker's regions are kept before verification, so a
  background voice no longer drags the embedding to "unknown". A
  fast-path skips single-region clips (no added latency).
- **Confidence calibration (Platt scaling)** — the confidence reported to
  HA is now a calibrated *probability that it's the same speaker*, fitted
  automatically from your enrolled samples and refitted in the background
  on enrollment changes.

### Training over the voice satellite
- Every utterance is captured to the **Unknown voices** tab (even with no
  speakers enrolled), so you can train entirely over the voice pipeline.
- Blocked entries in the recognition log get an **Add to speaker** button.
- **Create speaker (no samples)** to set up a profile and fill it later.
- The browser microphone is gracefully disabled in the add-on (HA ingress
  isn't a secure context, so `getUserMedia` is unavailable) with guidance
  toward upload / satellite training.

### Project
- MIT license, CI (pytest) and multi-arch GHCR publishing.

## 0.1.0 — Initial addon release

- Package Murdock as a Home Assistant add-on.
- Web UI served through HA ingress (no port forwarding needed).
- Supervisor token auto-injected so the HA integration works out of
  the box without the user minting a long-lived token.
- Persistent storage in `/data` — speakers, ONNX models, and recognition
  history survive addon updates and reinstalls.
- Multi-arch build for amd64 and aarch64.

Murdock features at first release:

- Wyoming-protocol proxy for faster-whisper / wyoming-whisper / Parakeet.
- CAM++ speaker embeddings + Silero VAD + liveness scoring.
- Sample quality scoring with configurable component weights.
- Auto-enroll with smart replacement of the lowest-quality sample.
- Per-satellite verify threshold overrides.
- Unknown-voice clustering and bulk-assign.
- Speaker backup / restore via ZIP.
- Recognition event log and stats.
- DE / EN UI.
