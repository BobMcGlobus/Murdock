"""HTTP client for Murdock's FastAPI (plan §6).

Thin async wrapper — no retries, short timeouts. Murdock being down must
never stall HA: callers treat every error as "proxy unreachable" and
keep working from their last known state (source, not dependency).
"""

from __future__ import annotations

import logging
from typing import Any

import aiohttp

from homeassistant.core import HomeAssistant
from homeassistant.helpers.aiohttp_client import async_get_clientsession

_LOGGER = logging.getLogger(__name__)

_TIMEOUT = aiohttp.ClientTimeout(total=10)


class MurdockApiError(Exception):
    """Raised when the proxy is unreachable or answers with an error."""


class MurdockApiClient:
    """Client against Murdock's REST API."""

    def __init__(
        self, hass: HomeAssistant, base_url: str, token: str | None = None
    ) -> None:
        self._session = async_get_clientsession(hass)
        self._base_url = base_url.rstrip("/")
        self._token = token or None

    @property
    def base_url(self) -> str:
        return self._base_url

    async def _request(
        self, method: str, path: str, json: Any | None = None
    ) -> Any:
        headers = {}
        if self._token:
            headers["Authorization"] = f"Bearer {self._token}"
        url = f"{self._base_url}{path}"
        try:
            async with self._session.request(
                method, url, json=json, headers=headers, timeout=_TIMEOUT
            ) as resp:
                if resp.status >= 400:
                    body = (await resp.text())[:200]
                    raise MurdockApiError(f"{method} {path} → {resp.status}: {body}")
                return await resp.json()
        except MurdockApiError:
            raise
        except Exception as exc:
            raise MurdockApiError(f"{method} {path} failed: {exc}") from exc

    async def get_version(self) -> str:
        data = await self._request("GET", "/api/version")
        return str(data.get("version", ""))

    async def get_satellites(self) -> list[dict[str, Any]]:
        data = await self._request("GET", "/api/satellites")
        return list(data.get("satellites", []))

    async def get_state(self) -> list[dict[str, Any]]:
        data = await self._request("GET", "/api/state")
        return list(data.get("satellites", []))

    async def get_vocabulary_meta(self) -> dict[str, Any]:
        return await self._request("GET", "/api/vocabulary")

    async def push_vocabulary(self, payload: dict[str, Any]) -> dict[str, Any]:
        return await self._request("POST", "/api/vocabulary", json=payload)
