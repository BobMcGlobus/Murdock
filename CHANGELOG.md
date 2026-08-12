# Changelog

## 0.8.8

**The language hint is normalised, and there is a fallback when Home
Assistant sends none.** Both failures produce the same symptom, and it
is a nasty one: a German utterance comes back as fluent English
nonsense rather than as an error, because a transcription endpoint that
cannot place the language does not complain — it guesses, and nearly
all of them guess English.

- Home Assistant forwards whatever the pipeline is configured with,
  commonly a full tag like `de-DE`. OpenAI documents ISO-639-1, and a
  local Parakeet server that cannot match the tag falls back to its own
  default. Tags are now reduced to their primary subtag where the
  request is built, so every caller gets the guarantee.
- A new **STT language** setting fills in when Home Assistant sends no
  language at all. Default `de`; empty means "let the endpoint decide",
  and the backend logs a warning when it comes to that.

Verified against a real `parakeet-tdt-0.6b-v3` ONNX server: `de-DE` and
an absent hint both arrive as `language=de`, where an unset hint had the
server logging `language=en`.

## 0.8.7

**Transcript latency.** Speaker verification was never the slow part —
it runs in 70–150 ms — but the cloud transcript could take anywhere from
0.8 to 9 seconds, and nothing in the log could say why.

- **The language hint was being dropped on the OpenRouter path.** Home
  Assistant sends it and the multipart branch honoured it, but the
  OpenRouter JSON body was built without it, so the primary engine
  re-detected the language on every single utterance.
- **No temperature was ever sent.** That leaves Whisper's fallback
  cascade live: when its own compression-ratio and log-prob checks fail
  it re-decodes the whole clip at 0.2 through 1.0 — up to six full
  passes. It is now pinned to 0. This is the shape of the multi-second
  outliers on otherwise sub-second audio, and it matches the A/B log,
  where the slow transcripts are the ones the two engines disagree
  about.
- **The request is timed as TTFB plus body download**, not one number,
  and the breakdown is persisted with the event and shown in the
  recognition log. "The model thought hard" and "the upload crawled" are
  now different-looking problems. Recorded on failure too.
- **Audio is conditioned before upload**: leading and trailing silence
  trimmed, level normalised. Models hallucinate in silence and the
  invented text is what trips the re-decode. Only the ends are trimmed —
  splicing internal pauses risks clipped word onsets. A VAD that finds
  only a sliver of speech is distrusted outright, because Silero
  under-detects whispering.
- **The request timeout is configurable**, defaulting to 8 s instead of
  the hard-coded 30 s that let a stalled provider hold the assistant for
  half a minute. The existing local Wyoming fallback covers the expiry.

The speaker embedding is unaffected by any of this: it is computed from
the untouched audio, and fbank does its own per-utterance mean
normalisation.

## 0.8.6

**Whispering now leaves Murdock.** The detector has been reporting a
score since 0.8.5, but nothing outside the web UI could see it, and
assigning a whispered clip to a speaker filed it as normal speech —
quietly degrading that speaker's ordinary voiceprint.

- **Assignment respects the speaking style.** A clip assigned from
  *Unknown* or the recognition log is filed under the style it was
  *spoken* in, decided from the clip's own whisper score. A whispered
  clip lands in the whisper profile and leaves the normal voiceprint
  alone. The dialog can still override it explicitly.
- **The samples list badges whispered samples** with `geflüstert`, so a
  whisper profile can be curated without playing every clip. The samples
  API had been dropping the style column outright; it is now part of the
  response.
- **MQTT gains `binary_sensor.murdock_whispering` and
  `sensor.murdock_whisper_score`**, both auto-discovered under the
  existing Murdock device.
- **The HA integration (0.3.0) gains a per-satellite whisper sensor**,
  whisper attributes on the speaker sensor, and a prompt line telling
  the conversation agent to answer quietly — and explicitly not to read
  whispering as an instruction to keep something secret.

Whispering is reported even when the speaker is `unbekannt`: answering
more quietly does not depend on knowing who asked.

## 0.8.5

**Fixes a regression in 0.8.4 that emptied the recognition log.** A
search-and-replace in the 0.8.4 multi-speaker work put a publish-only
keyword into the call that writes the audit row. Every insert raised
TypeError, the handler swallowed it at debug level, and no utterance has
been logged since. Two changes so this cannot repeat quietly:

- the argument is now real — the speaker roster is persisted with the
  event and shown in the log;
- a failed audit write logs at **warning** level. A swallowed failure
  here is indistinguishable from "nothing is happening", which is exactly
  how it went unnoticed for a release.
- New regression tests exercise the handler's call against a real store,
  including a signature check that every forwarded keyword is accepted.
  Nothing covered that seam before.

**The whisper score is visible.** The detector always produced a number;
only a yes/no reached the UI. Now:

- the recognition log shows it — highlighted when over the threshold
  (`geflüstert 0.87`), muted when measured but under it (`Flüstern 0.41`),
  which is what makes the threshold tunable;
- unknown samples carry it too, so a quiet rejected clip is explainable;
- both are also in `GET /api/recognition` and `GET /api/unknown`.

**Whisper voice profiles.** Speakers can now be recognised *while
whispering*. Enrollment has a speaking-style selector; whispered samples
build a second voiceprint.

- Whispered samples **never** enter the normal voiceprint or the
  per-satellite profiles — averaging in a voice with no pitch would make
  ordinary recognition worse for everyone.
- The whisper centroid is only consulted when the detector says the
  utterance was whispered, and the threshold is unchanged. Someone who
  never enrolled a whisper still comes back as unknown, so this stays a
  recognition feature rather than a way past the gate.
- Two whispered samples are enough (whispering varies less than normal
  speech); below that no profile is built.

## 0.8.4

**Profile health.** The Speakers tab now says what would concretely make
recognition better, instead of leaving you to interpret quality scores:
too few samples, poor recording quality, a satellite the speaker uses
often but has no samples from (so no per-satellite profile can be built),
matches passing only just, or a profile drifting worse over time. Every
finding carries the numbers behind it. An empty panel is the normal
state — advice that always has something to say gets ignored.


**Whisper detection (experimental).** Whispering has no vocal-fold
vibration, so there is no fundamental frequency and no harmonic
structure — a far cleaner signal than most voice properties. The detector
scores harmonicity, zero-crossing rate and spectral tilt; loudness only
refines, never decides, so a distant quiet speaker is not mistaken for a
whisper.

- It runs **before** the gates on purpose: whispered speech is quiet and
  spectrally flat, exactly what the TV/playback liveness heuristic and
  early reject throw away. Both are skipped for a detected whisper —
  otherwise Murdock would discard precisely the utterances the feature
  exists to notice.
- The flag travels in the recognition event (`whisper: true`) and is
  marked in the recognition log, so the threshold can be tuned against
  real recordings.
- **Never used to relax verification.** Whispering does wreck speaker
  embeddings, so a whispered command usually comes back as "unknown" —
  that is deliberate. Treating a whisper as proof of identity would turn
  the feature into a way past the gate.

**Emotion detection now actually works.** It was announced but shipped
without a model, and — worse — the classifier papered over the mismatch
by inventing label names like `class_44737`. Both are fixed:

- emotion2vec+ base publishes its ONNX as a **feature extractor**
  (frame-level 768-dim output); the 9-class head ships separately as a
  small binary. The classifier now mean-pools the frames and applies that
  head, which is what upstream does.
- A shape mismatch with no usable head raises instead of guessing. A
  confidently wrong emotion is worse than an absent one.
- `DOWNLOAD_EMOTION_MODEL=1 scripts/download_models.sh` fetches both
  files (~356 MB, opt-in, never part of the required set). The head is
  validated by exact size, and the ONNX is deleted if the head is missing
  — the extractor alone cannot classify anything.

**Multiple speakers per utterance.** Extraction already scored every
speech region against the enrolled speakers to pick the dominant one, and
threw the rest away. The full roster (name + speaking seconds, longest
first) now travels in the event as `speakers` whenever more than one
known voice was heard. The gate still follows the dominant speaker.

## 0.8.3

**Automatic name correction** — the feature the vocabulary should have been
powering all along. Murdock knows exactly which entity names exist; it now
maps misheard spans onto them instead of only trying to bias the engine up
front.

```
"schalte das Bad-Lightstrip ein"  →  "schalte das Bett-Lightstrip ein"
```

- Candidates are indexed by **Kölner Phonetik**. German mishearings are
  phonetic — "Bad" and "Bett" both encode to `12` — and plain edit distance
  ranks exactly those pairs far apart. Phonetics may propose, never decide:
  only the entity list knows which spelling you actually own.
- Scoring blends sequence similarity, character-trigram overlap and a
  phonetic-match bonus. Two ways in: the normal similarity floor (0.82), or
  a floor 0.10 lower when the phonetic codes are identical.
- **The margin decides ties.** The winner must lead the runner-up clearly
  (0.10), so two equally plausible entities are never resolved by coin
  flip — same discipline as the speaker margin gate.
- Never touches already-valid names, spans under four characters, or common
  German words. Runs *after* the explicit dictionary, so your rules win.
- Works with **every** STT backend, and the result is plain text, so HA's
  local intent matching improves rather than breaks.
- Under 50 ms with 400 entities (trigram overlap blocks non-contenders
  before the expensive comparison).
- Default off, under *Settings → Transcription → STT backend*.

**Recurring corrections become rules.** Every applied correction is
counted; frequent ones appear under *Recurring corrections* with a *Make it
a rule* button that writes an exact dictionary entry and enables that tier.
Deliberately a click, not an automatism.

**The bias prompt, demoted to what it is.** It only ever reached
OpenAI-compatible non-OpenRouter backends — never the default Wyoming
upstream, never Voxtral — so on a normal install the mirrored vocabulary
did nothing. The UI now says so when the active backend ignores the prompt,
and the 25 terms are **curatable**: mirrored terms are clickable chips
(filled = sent), your own terms are separate red chips with add/remove, a
counter shows what is in play, and *Automatic selection* returns to the
default first-25.

**HACS support.** The integration installs as a HACS custom repository:
`hacs.json`, `info.md`, entity icons (`icons.json`), and brand assets at
the sizes `home-assistant/brands` specifies. A new `validate` workflow runs
`hassfest` and the HACS action on every push — it immediately caught a URL
in a config-flow string and the missing brand directory, both fixed.

**Experimental tab.** Unproven features moved out of the settings page;
emotion detection lives there now.

248 tests green.

## 0.8.2

**The settings page is navigable again.** It had grown to twelve cards on
one endless scroll. Now four groups — *Recognition*, *Transcription*,
*Home Assistant*, *System* — switched by pills at the top, with the chosen
group remembered across reloads. Inside the recognition card the everyday
knobs (threshold, margin gate, require-match, passthrough, auto-enroll,
unknown logging) stay visible; the rest moved into three folded
*Advanced* blocks: noise & multiple voices, per-satellite profiles & early
reject, upstream & languages. Nothing was removed and no field changed
form, so every setting saves exactly as before — including from a
collapsed block.

**STT A/B comparison now shows speed.** Each engine's wall-clock time
appears as a badge next to its transcript, the slower of the two is
highlighted, and the shadow line spells out the delta ("1.71s faster").
Comparing wording was only half the question — this answers whether the
better transcript is worth its latency.

- `recognition_events` gained `transcript_ms` and `shadow_ms`; both are
  exposed by `GET /api/recognition` (along with `weight` and `margin`).
- Note on reading the numbers: in **upstream** mode the primary streams
  while the audio is still arriving, so its figure is the remaining wait
  and understates the engine's own work. Cloud backends and the shadow
  always transcribe the finished buffer, so those are full request times.


**See the mirrored vocabulary.** The terms the HA integration contributes
were invisible: the Web UI only showed the manual list, and the prompt
that actually reaches the STT engine wasn't exposed anywhere. New panel
under *Settings → STT backend → Transcript quality*:

- every mirrored term as a chip, with the ones **beyond the 25-term cap**
  greyed out — the cap was silently dropping terms before
- snapshot version, entity/term counts and push time
- the **effective prompt**, i.e. manual terms plus capped HA terms exactly
  as sent (or a note when the vocabulary tier is switched off)

`GET /api/vocabulary` gained `terms` (full list), `term_cap`,
`effective_prompt` and `effective_enabled` to back it.

## 0.8.1

Makes the 0.8.0 integration actually work on a token-free (MQTT) setup —
0.8.0's integration could not see recognitions there at all, and even on
the REST path the speaker arrived one turn late. **Update both the add-on
and the integration** (0.2.0, attached to the release).

**Integration 0.2.0 — the MQTT path now works, and the speaker arrives in
time.** Two defects that together made the integration useless on the
recommended setup:

- **MQTT recognitions never reached the integration.** Murdock publishes
  them to `<prefix>/event/recognition`, but an MQTT message is *not* a
  Home Assistant event — the integration only listened on the event bus,
  which is fed by the *legacy* REST/token push. Anyone on the
  recommended, token-free MQTT setup saw `unbekannt` forever. The
  integration now subscribes to the topic as well (prefix configurable,
  blank disables it); running both paths is fine, duplicates are dropped
  by satellite + timestamp.
- **The speaker lost a race it could not win.** Home Assistant starts the
  intent stage — where the agent's prompt is built — within a millisecond
  of receiving the transcript, but Murdock published the recognition
  *after* answering. Even a perfectly delivered event arrived too late
  for the turn that caused it. The match path now publishes and **waits**
  (1 s cap) before answering the satellite, costing a few milliseconds on
  a local network. Emotion, classified after the response, only sends a
  second event when there is an emotion to add.
- New `sensor.murdock_delivery_path` diagnostic shows which transports are
  live (`mqtt+event`, `event (waiting)`, …) — the first thing to check
  when the speaker stays unknown.

**Integration fix 0.1.1 — vocabulary mirroring crashed on HA 2026.7+.**
`RegistryEntry.aliases` became `list[str | ComputedNameType]`, where the
`COMPUTED_NAME` sentinel stands for the computed full entity name and is
only expanded by `async_get_entity_aliases()`. Sorting the raw list threw
`TypeError: '<' not supported between instances of 'str' and
'ComputedNameType'` and, because the initial push was awaited during
setup, took the **whole integration** down with it — no speaker, no LLM
API, no sensors.

- Aliases are now resolved through the official helper, with a fallback
  for older cores where `aliases` is still a plain `set[str]`.
- Vocabulary mirroring is strictly best-effort: per-entity failures are
  skipped, `async_push()` never raises, and a mirror that fails to start
  no longer fails the config entry. The speaker path is the reason this
  integration exists; vocabulary is an enhancement.
- New `scripts/verify_integration_api.py` checks every Home Assistant API
  the integration touches (30 assertions, all modules imported) against a
  real HA install. The repo's pytest suite can't cover
  `custom_components/` — HA needs Python 3.14, the add-on image is on
  3.11 — so this closes that gap. Verified green on HA 2026.7.4.

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
