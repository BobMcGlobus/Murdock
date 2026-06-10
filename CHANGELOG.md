# Changelog

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
