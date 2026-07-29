# Changelog

## Unreleased

**Sidecar mode for ambiguity markers** — the `[oder: …]` markers from the
correction dictionary and the dual transcript no longer have to live in
the transcript. New setting under *Settings → Speaker context*:

- `inline` (default, unchanged) — markers stay in the text.
- `sidecar` — the transcript stays clean and the ambiguities travel in
  the recognition event's new `ambiguities` field. The HA integration
  turns them into a `Transkript-Hinweis` prompt line. This is the
  combination that was previously impossible: hints for the LLM *and*
  working local intent matching.
- `clean` — drop the markers entirely.
- `auto` — sidecar when an event sink (MQTT or HA REST) is wired,
  otherwise inline, so hints are never silently lost.

**Decide before marking.** Before an ambiguity is passed on, it's checked
against the mirrored vocabulary: if exactly one of the two readings is a
real entity name, that reading wins outright and no hint is emitted —
"Bad-Lightstrip [oder: Bed-Lightstrip]" simply becomes "Bed-Lightstrip"
when that's the entity you actually own. Only genuine ambiguity (both or
neither known) reaches the agent.

- The paths that publish the event before the transcript exists (early
  match, unknown-forwarded) fire one additional event carrying the hints;
  with no hints there's no second event, so the common case is unchanged.
- 21 new tests (197 total).

## 0.8.0

**New: a Home Assistant custom integration** (`custom_components/murdock`,
version 0.1.0) — the speaker now reaches your conversation agent *per
turn* instead of once per conversation. Ships alongside the add-on;
MQTT, REST and the token path all keep working exactly as before, the
integration is purely additive.

- **LLM API "Murdock"** — enable it under *Settings → Voice assistants →
  your agent → LLM APIs*. Contributes one line to every turn:
  `Sprecher: Jonas (Konfidenz 0.94, Satellit Wohnzimmer)`. The line is
  never omitted — an unknown or ambiguous voice says so explicitly, so
  the agent can't quietly assume the main user. Home Assistant rebuilds
  the prompt on every turn, which is what makes this reliable where
  `extra_system_prompt` went stale mid-conversation.
- **Explicit satellite mapping** — each Murdock satellite ID is assigned
  to its `assist_satellite` entity in the config flow; resolution is
  device hit first, then area. Never guessed: a wrong mapping would mean
  right speaker, wrong room.
- **Vocabulary mirroring** — entity, area and floor names (including
  aliases) of *exposed* entities are pushed to Murdock with a 5 s
  debounce and feed the STT bias prompt. Murdock stores each push as a
  versioned snapshot and keeps using the last one when HA is
  unreachable — source, not dependency.
- **Entities** — `sensor.murdock_<satellite>_speaker` (with confidence,
  distance, nearest speaker, margin, weight, role) plus proxy
  diagnostics: connection, last recognition, vocabulary version.
- **Public helper** `async_get_speaker(hass, device_id=…)` for other
  integrations, so speaker attribution can bypass the model entirely.

**Margin gate** (off by default) — a match is only trusted when the
second-best speaker is a minimum distance behind. Two voices that are
plausibly the same now yield `unsicher` instead of a coin flip, both in
the prompt line and the sensor. Configurable globally and per satellite.

**Recognition events are now complete and honest** — the payload gained
`nearest_distance`, `weight`, `margin`, `uncertain`, `reason` and
`timestamp`, and it fires on **every** utterance including
non-recognitions (`speaker: null`). Previously a blocked or unrecognised
utterance left the last speaker standing.

- Breaking for hand-written automations: on non-recognition `speaker` is
  now `null` with the sentinel moved to `reason` (`unknown`, `uncertain`,
  `tv-noise`, `early-reject`, `short`, `no-speakers`, `embed-failed`).
  MQTT sensor states are unchanged.
- New endpoints for the integration: `GET /api/version`,
  `GET /api/satellites`, `GET /api/state`, `POST`/`GET /api/vocabulary`.
- `recognition_events` records the speaker weight and margin, so "why
  didn't that rule fire" stays answerable later.
- 24 new tests (176 total).

**Installing the integration:** copy `custom_components/murdock` into
your HA `/config/custom_components/` (or unpack the ZIP attached to the
GitHub release), restart HA, then add *Murdock* under Devices &
Services. The integration talks to Murdock's REST API, so the add-on
needs its Web UI port reachable: *Murdock add-on → Configuration →
Network* → set `8099` and restart the add-on. Ingress alone is not
enough.

## 0.7.0

Transcript quality tiers — three independently toggleable weapons
against systematic STT misrecognitions (all default off, under
*Settings → STT backend → Transcript quality*):

- **Custom vocabulary** — your terms ("Fehenlichter, Bed-Lightstrip")
  are sent to whisper-family cloud backends as context, so custom names
  are recognised at the source. Automatically skipped where the field
  isn't supported (Voxtral, OpenRouter) — never risks a failed request.
- **Correction dictionary** — one entry per line:
  `fehlende Lichter -> Fehenlichter` replaces deterministically (keeps
  HA's local intent matching working), `Bad-Lightstrip ~> Bed-Lightstrip`
  annotates with `[oder: …]` for an LLM agent. Case-insensitive,
  word-boundary matching, `#` comments, longer phrases win.
- **Dual transcript** — requires a configured shadow engine: the shadow
  runs blocking in parallel and both readings are merged, marking real
  disagreements inline as `primary [oder: shadow]` (shadow-only words as
  `[oder zusätzlich: …]`). Casing/punctuation never fake a disagreement;
  the shadow is capped at 6 s, then the primary answers alone; the
  result is reused for the recognition log (no double transcription).
  The dictionary applies to both sides *before* the merge. LLM-only —
  breaks rigid intent matching.

## 0.6.2

- **Shadow STT: full settings parity** — the A/B shadow engine now has
  its own Mistral API key (empty = fall back to the primary key, as
  before). Every shadow backend is thereby fully independently
  configurable: Wyoming (own URI), Voxtral (own model + own key),
  OpenAI-compatible (own base URL + key + model) — e.g. for a second
  account or billing separation.

## 0.6.1

- **OpenRouter support** in the OpenAI-compatible STT backend. OpenRouter
  offers `whisper-large-v3-turbo` and routes to Groq's fast inference —
  usable without a Groq account. Their endpoint deviates from the OpenAI
  shape (JSON body with base64 audio, provider-prefixed model slugs, API
  root under `/api`); Murdock now detects `openrouter.ai` in the base URL
  and adapts automatically. Preset: base URL `https://openrouter.ai`,
  model `openai/whisper-large-v3-turbo` (note the prefix). All other
  endpoints keep the standard multipart format.

## 0.6.0

Pluggable STT — pick, fall back, and A/B-test your transcription engine.

- **OpenAI-compatible backend** — new `stt_backend` option "openai"
  talks to any `/v1/audio/transcriptions` endpoint: OpenAI
  (`https://api.openai.com` + `gpt-4o-transcribe`), Groq
  (`https://api.groq.com/openai` + `whisper-large-v3-turbo`) or a local
  OpenAI-compatible server such as speaches (empty API key allowed).
- **Voxtral model selectable** — switch e.g. to `voxtral-small-latest`
  (more accurate than mini) straight from the UI or add-on options.
- **Local fallback (opt-in)** — when the cloud STT fails (internet down,
  provider outage), the buffered audio is transcribed one-shot over the
  Wyoming upstream instead of returning an empty transcript.
- **A/B shadow engine** — transcribe every utterance with a *second*
  engine in the background (another Wyoming server, another Voxtral
  model, or another OpenAI-compatible endpoint). Never returned to Home
  Assistant and adds zero latency — the shadow transcript appears next
  to the primary one in the recognition log with differences
  highlighted, so you can compare engines on your real commands before
  switching.
- Backends now signal failures explicitly instead of silently returning
  an empty transcript (which the fallback builds on).

## 0.5.2

- **Clarified speaker-context modes** — MQTT was never switched off by
  the delivery mode (the mode only controls the transcript hand-off),
  but the old option label "MQTT / system prompt" suggested otherwise.
  Modes are now labelled "Transcript untouched (via system prompt)" and
  "Augment transcript"; the UI, README and wiki state explicitly that
  the MQTT sensor entities keep publishing in both modes whenever MQTT
  is enabled. Labels/docs only — no behaviour change.

## 0.5.1

- **Speaker context delivery mode** — a dropdown under *Settings →
  Speaker context to the conversation agent* picks how the recognised
  speaker reaches your agent:
  - **MQTT / system prompt** (default): transcript untouched, speaker via
    sensors; HA's local intent matching keeps working.
  - **Transcript augmentation**: inject the recognition context straight
    into the returned transcript (known/unknown templates with
    `{{ speaker }}`, `{{ role }}`, `{{ confidence }}`, `{{ nearest }}`, …)
    so it's fresh on every utterance with no MQTT and no system-prompt
    cache staleness — at the cost of HA's local intent matching, so it's
    for LLM-driven setups.

  Each mode is explained inline in the UI and on the new
  [Speaker Context](https://github.com/BobMcGlobus/Murdock/wiki/Speaker-Context)
  wiki page. The dropdown leaves room for a future third mode.

## 0.5.0

Smarter gating: microphone-aware profiles, per-speaker thresholds, and
an opt-in early reject.

- **Per-satellite voice profiles** — an extra voiceprint per (speaker,
  satellite) built from same-mic samples (≥3 tagged); verification
  scores against the better of global/same-mic. Removes systematic
  microphone bias between satellites (e.g. fewer mics, no beamforming).
  Auto-maintained on enroll, auto-enroll and sample deletion. Default on.
- **Adaptive per-speaker thresholds** — each speaker's gate derives from
  their own genuine/impostor score distributions, recomputed on every
  calibration refit and bounded to ±0.08 around the global threshold.
  Per-satellite overrides keep applying as a delta on top; media
  tightening still subtracts. Values are shown in the calibration card.
  Default on.
- **Early reject (opt-in, default off)** — after ~1.5 s of clean voice,
  sessions catastrophically far from every profile (distance ≥ threshold
  + margin, default 0.25) are dropped: STT forwarding stops immediately
  (no CPU wasted, unknown audio stops leaving the house on cloud STT)
  and the satellite gets its empty transcript instantly at stream end.
  Media playing in the room (MQTT context) halves the margin. Rejected
  audio still lands in Unknown voices for one-click training. New
  recognition-log outcome "blocked-early-reject".
- **Default change:** "Require speaker match" is now OFF by default —
  the hard gate made fresh installs unusable before any speakers were
  enrolled. Existing installs keep their stored setting. Combine with
  early reject to drop TV/radio while unknown humans still pass.

## 0.4.1

- **Logo** — Murdock has a face: a crimson devil-horned head with a
  voice-waveform inside (a nod to the namesake — Daredevil perceives
  through sound, Murdock identifies through it). Shows as the add-on
  icon/logo in the store, as the Web UI favicon and next to the header
  wordmark. Geometry lives in `scripts/gen_logo.py`.
- **CI** — workflows moved to Node 24 actions (checkout v5,
  setup-python v6, `FORCE_JAVASCRIPT_ACTIONS_TO_NODE24`) ahead of
  GitHub's 2026-06-16 flag day.

## 0.4.0

Diagnostics and quality-of-life.

- **Voice map** — 2-D PCA projection of the whole embedding space in the
  Speakers tab: one color per speaker, rings for centroids, gray crosses
  for unknown samples. **Click any point** to play it, delete it, or
  (for unknowns) assign it to a speaker — the map re-renders after each
  action.
- **Speaker health** — per-speaker panel with each sample's drift from
  the centroid, age and quality, plus a quality trend (newest vs. oldest
  half). Drifted samples are flagged for pruning.
- **Threshold recommendation** — "Suggest from log" next to the verify
  threshold analyses the recognition log (match distances vs.
  blocked/unknown distances) and proposes an empirically grounded
  threshold, with a one-click Apply. Warns when the distributions
  overlap.
- **Full backup** — the backup ZIP now includes all settings
  (thresholds, MQTT/HA config, media matrix, calibration parameters).
  Restore re-applies everything live, no restart needed. Note: the
  archive contains your configured credentials — store it safely.
- **Collapsible settings** — every settings card folds to its heading;
  state is remembered per card.
- New [project wiki](https://github.com/BobMcGlobus/Murdock/wiki) with
  installation, MQTT, training, tuning, backup and architecture guides.

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
