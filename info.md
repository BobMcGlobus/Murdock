# Murdock — speaker identity for Home Assistant voice

Tells your conversation agent **who is speaking**, freshly on every turn.

Murdock is a Wyoming speaker-recognition proxy that sits between your
voice satellites and your STT engine. This integration is its Home
Assistant half.

## What you get

- **An LLM API** that contributes one line to every turn:
  `Sprecher: Jonas (Konfidenz 0.94, Satellit Wohnzimmer)` — never
  omitted, so an unknown or ambiguous voice says so explicitly instead of
  letting the agent assume the main user.
- **Vocabulary mirroring** — the names and aliases of your exposed
  entities, areas and floors are pushed to Murdock, where they correct
  misheard names ("Bad-Lightstrip" → "Bett-Lightstrip").
- **Per-satellite speaker sensors** with confidence, distance, margin and
  weight, plus proxy diagnostics.
- **`async_get_speaker()`** for other integrations, so speaker attribution
  can bypass the model entirely.

## Requirements

You need the Murdock proxy itself running — as a Home Assistant add-on or
via docker-compose. See the
[repository](https://github.com/BobMcGlobus/Murdock) for setup.

**The proxy's API must be reachable.** In the add-on the Web UI port is
unpublished by default: *Murdock add-on → Configuration → Network* → set
port `8099`, then restart the add-on. Ingress alone is not enough.

## After installing

1. Restart Home Assistant.
2. *Settings → Devices & Services → Add integration → Murdock*, enter
   `http://<ha-host>:8099`.
3. Map each Murdock satellite ID to its `assist_satellite` entity.
4. Enable the API under *Settings → Voice assistants → your agent → LLM
   APIs → Murdock*.

Full documentation: [the wiki](https://github.com/BobMcGlobus/Murdock/wiki/HA-Integration).
