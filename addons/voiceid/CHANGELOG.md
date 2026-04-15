# Changelog

## 0.1.0 — Initial addon release

- Package VoiceID as a Home Assistant add-on.
- Web UI served through HA ingress (no port forwarding needed).
- Supervisor token auto-injected so the HA integration works out of
  the box without the user minting a long-lived token.
- Persistent storage in `/data` — speakers, ONNX models, and recognition
  history survive addon updates and reinstalls.
- Multi-arch build for amd64 and aarch64.

VoiceID features at first release:

- Wyoming-protocol proxy for faster-whisper / wyoming-whisper / Parakeet.
- CAM++ speaker embeddings + Silero VAD + liveness scoring.
- Sample quality scoring with configurable component weights.
- Auto-enroll with smart replacement of the lowest-quality sample.
- Per-satellite verify threshold overrides.
- Unknown-voice clustering and bulk-assign.
- Speaker backup / restore via ZIP.
- Recognition event log and stats.
- DE / EN UI.
