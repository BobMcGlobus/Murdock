"""Cache of the upstream ASR's language list.

Home Assistant's Wyoming integration asks the STT server for a
``Describe`` on every reconnect and filters the pipeline picker by the
languages advertised in the returned ``Info`` event. If VoiceID serves a
stale or empty language list, the STT entity shows up in HA but is
grayed out for any pipeline whose language doesn't match.

This module keeps a lazy, TTL'd cache of the upstream's supported
languages and builds a fresh ``Info`` object on demand, so every
Describe from HA reflects whatever the upstream currently offers — with
an optional hard override sourced either from ``ADVERTISED_LANGUAGES``
or from the live settings table (via an ``override_provider`` callable).
"""

from __future__ import annotations

import asyncio
import logging
import time
from typing import Callable, List, Optional

from wyoming.client import AsyncClient
from wyoming.info import (
    AsrModel,
    AsrProgram,
    Attribution,
    Describe,
    Info,
)

from voiceid import __version__

_LOGGER = logging.getLogger("voiceid.info_cache")


def parse_languages(value: Optional[str]) -> Optional[List[str]]:
    """Parse a comma-separated language string into a clean list.

    Returns ``None`` for empty / missing input so callers can easily
    distinguish "no override" from "empty override".
    """
    if not value:
        return None
    parts = [s.strip() for s in value.split(",")]
    langs = [p for p in parts if p]
    return langs or None


# Backwards-compat alias for any lingering callers.
_parse_languages = parse_languages


class UpstreamInfoCache:
    """Lazy, TTL'd cache of upstream ASR languages.

    Thread/task-safe: refreshes are serialised behind an asyncio Lock so
    concurrent Describe requests only trigger a single upstream query.

    ``override_provider`` is a zero-arg callable that returns either a
    list of language codes (hard override — upstream is never queried)
    or ``None`` (fall through to the live upstream query). The callable
    is evaluated on every access so UI edits to the advertised-language
    setting take effect immediately, without a service restart.
    """

    def __init__(
        self,
        upstream_uri: str,
        *,
        ttl: float = 60.0,
        override_provider: Optional[Callable[[], Optional[List[str]]]] = None,
        query_timeout: float = 5.0,
    ) -> None:
        self.upstream_uri = upstream_uri
        self.ttl = ttl
        self.override_provider = override_provider
        self.query_timeout = query_timeout
        self._languages: List[str] = []
        self._last_fetch: float = 0.0
        self._lock = asyncio.Lock()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def _current_override(self) -> Optional[List[str]]:
        if self.override_provider is None:
            return None
        try:
            value = self.override_provider()
        except Exception:
            _LOGGER.exception("override_provider raised — ignoring")
            return None
        if not value:
            return None
        return list(value)

    async def get_languages(self, *, force: bool = False) -> List[str]:
        override = self._current_override()
        if override:
            return override
        now = time.monotonic()
        fresh = self._languages and (now - self._last_fetch) < self.ttl
        if fresh and not force:
            return list(self._languages)
        async with self._lock:
            # Another waiter may already have refreshed while we waited.
            now = time.monotonic()
            if (
                not force
                and self._languages
                and (now - self._last_fetch) < self.ttl
            ):
                return list(self._languages)
            try:
                langs = await self._query_upstream_once()
            except Exception as exc:
                _LOGGER.warning(
                    "Could not refresh upstream languages from %s: %s",
                    self.upstream_uri, exc,
                )
                return list(self._languages)
            if langs:
                self._languages = langs
                self._last_fetch = now
                _LOGGER.info(
                    "Upstream %s advertises languages: %s",
                    self.upstream_uri, ", ".join(langs),
                )
            else:
                _LOGGER.warning(
                    "Upstream %s returned no languages in Info event",
                    self.upstream_uri,
                )
            return list(self._languages)

    async def prime(self, attempts: int = 5, delay: float = 3.0) -> None:
        """Best-effort eager fetch at startup, so the first HA Describe
        after container boot already has real languages.

        Evaluated each time — if a runtime override is configured via
        the settings UI we skip the upstream query entirely and return
        immediately.
        """
        override = self._current_override()
        if override:
            _LOGGER.info(
                "Advertised-languages override active: %s",
                ", ".join(override),
            )
            return
        for attempt in range(1, attempts + 1):
            langs = await self.get_languages(force=True)
            if langs:
                return
            # Re-check override on each retry — the user may configure
            # it via the UI while we're still waiting for the upstream.
            if self._current_override():
                return
            if attempt < attempts:
                _LOGGER.info(
                    "Upstream %s not ready (attempt %d/%d) — retrying in %.0fs",
                    self.upstream_uri, attempt, attempts, delay,
                )
                await asyncio.sleep(delay)
        _LOGGER.warning(
            "Could not determine upstream languages after %d attempts. "
            "Configure an override in the VoiceID settings UI to unblock HA.",
            attempts,
        )

    def invalidate(self) -> None:
        """Drop the cached upstream languages so the next Describe
        refetches. Called by the REST layer after the admin edits the
        upstream URI or advertised languages."""
        self._languages = []
        self._last_fetch = 0.0

    async def build_info(self) -> Info:
        """Return a fresh Info event with current upstream languages."""
        languages = await self.get_languages()
        if not languages:
            # Last-ditch fallback so HA's pipeline picker doesn't silently
            # gray out every language. Prefer common assistant languages.
            languages = ["de", "en"]
            _LOGGER.debug(
                "No upstream languages available — falling back to %s",
                languages,
            )
        return Info(
            asr=[
                AsrProgram(
                    name="voiceid",
                    description=(
                        f"VoiceID v{__version__} — speaker-gated ASR proxy"
                    ),
                    attribution=Attribution(
                        name="VoiceID",
                        url="https://github.com/bobmcphee/VoiceID",
                    ),
                    installed=True,
                    version=__version__,
                    models=[
                        AsrModel(
                            name="voiceid-proxy",
                            description="CAM++ speaker gate → upstream ASR",
                            languages=languages,
                            attribution=Attribution(
                                name="VoiceID",
                                url="https://github.com/bobmcphee/VoiceID",
                            ),
                            installed=True,
                            version=__version__,
                        )
                    ],
                )
            ]
        )

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    async def _query_upstream_once(self) -> List[str]:
        async with AsyncClient.from_uri(self.upstream_uri) as client:
            await client.write_event(Describe().event())
            while True:
                event = await asyncio.wait_for(
                    client.read_event(), timeout=self.query_timeout
                )
                if event is None:
                    return []
                if Info.is_type(event.type):
                    info = Info.from_event(event)
                    langs: List[str] = []
                    for asr in info.asr:
                        for model in asr.models:
                            langs.extend(model.languages)
                    # Dedup preserving order.
                    seen: set[str] = set()
                    unique: List[str] = []
                    for lang in langs:
                        if lang and lang not in seen:
                            seen.add(lang)
                            unique.append(lang)
                    return unique
