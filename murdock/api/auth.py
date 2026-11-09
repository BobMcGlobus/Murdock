"""Authentication for the web API.

A request is allowed if it comes from one of:

* Home Assistant ingress: under the Supervisor, from the ingress proxy's
  fixed address. Home Assistant has already authenticated an admin.
* ``Authorization: Bearer <token>`` with the API token.
* A session cookie from ``/api/auth/login``. Sessions are stored server
  side, so signing out or rotating the token revokes them. Cookie
  requests that change state also need the ``X-Murdock-UI`` header,
  which a page on another origin can't send without a CORS preflight.

The token comes from ``API_TOKEN`` or is generated into
``<data_dir>/api_token`` on first start.
"""

from __future__ import annotations

import hashlib
import hmac
import json
import logging
import os
import secrets
import time
from pathlib import Path
from typing import NamedTuple, Optional

from starlette.requests import HTTPConnection

_LOGGER = logging.getLogger("murdock.api.auth")

# Fixed address of the Supervisor's ingress proxy.
INGRESS_PROXY_IP = "172.30.32.2"

TOKEN_FILENAME = "api_token"
SESSIONS_FILENAME = "api_sessions.json"
SESSION_COOKIE = "murdock_session"
UI_HEADER = "x-murdock-ui"
SESSION_MAX_AGE = 90 * 24 * 3600
_MIN_PINNED_TOKEN_LENGTH = 16

_SAFE_METHODS = frozenset({"GET", "HEAD", "OPTIONS"})
_PUBLIC_PATHS = frozenset({
    "/",
    "/api/health",
    "/api/auth/status",
    "/api/auth/login",
})
_PUBLIC_PREFIXES = ("/static/",)

VIA_INGRESS = "ingress"
VIA_TOKEN = "token"
VIA_SESSION = "session"


class Verdict(NamedTuple):
    via: Optional[str]
    status: int = 200
    detail: str = ""


def _write_private(path: Path, text: str) -> None:
    """Atomically write ``text`` to ``path`` with mode 0600."""
    tmp = path.with_name(path.name + ".tmp")
    fd = os.open(str(tmp), os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
    try:
        os.write(fd, text.encode("utf-8"))
    finally:
        os.close(fd)
    os.replace(tmp, path)


def _digest(session_id: str) -> str:
    return hashlib.sha256(session_id.encode("utf-8")).hexdigest()


class ApiAuth:
    def __init__(
        self,
        data_dir: Path,
        env_token: Optional[str] = None,
        under_supervisor: Optional[bool] = None,
    ) -> None:
        data_dir = Path(data_dir)
        self._token_path = data_dir / TOKEN_FILENAME
        self._sessions_path = data_dir / SESSIONS_FILENAME
        if under_supervisor is None:
            under_supervisor = bool(os.environ.get("SUPERVISOR_TOKEN"))
        self.under_supervisor = under_supervisor
        env_token = (env_token or "").strip()
        self.pinned = bool(env_token)
        if self.pinned:
            self._token = env_token
            if len(env_token) < _MIN_PINNED_TOKEN_LENGTH:
                _LOGGER.warning(
                    "API_TOKEN is shorter than %d characters; use a long random value",
                    _MIN_PINNED_TOKEN_LENGTH,
                )
        else:
            self._token = self._load_or_create_token()
        self._sessions = self._load_sessions()

    @classmethod
    def from_settings(cls, settings) -> "ApiAuth":
        return cls(data_dir=settings.data_dir, env_token=settings.api_token)

    @property
    def token(self) -> str:
        return self._token

    def _load_or_create_token(self) -> str:
        try:
            token = self._token_path.read_text(encoding="utf-8").strip()
        except FileNotFoundError:
            token = ""
        if token:
            return token
        token = secrets.token_urlsafe(32)
        self._token_path.parent.mkdir(parents=True, exist_ok=True)
        _write_private(self._token_path, token + "\n")
        _LOGGER.info(
            "Generated an API token in %s (shown in the Web UI under "
            "Settings → Home Assistant → API access)", self._token_path,
        )
        return token

    def rotate(self) -> str:
        """Replace the token and revoke every session."""
        if self.pinned:
            raise RuntimeError("API token is set by API_TOKEN")
        self._token = secrets.token_urlsafe(32)
        _write_private(self._token_path, self._token + "\n")
        self._sessions = {}
        self._save_sessions()
        _LOGGER.warning("API token rotated")
        return self._token

    def token_matches(self, candidate: Optional[str]) -> bool:
        if not candidate:
            return False
        return hmac.compare_digest(
            candidate.strip().encode("utf-8"), self._token.encode("utf-8")
        )

    # Sessions are kept as {sha256(session id): expiry}, so the file never
    # holds a usable cookie value.

    def _load_sessions(self) -> dict:
        try:
            stored = json.loads(self._sessions_path.read_text(encoding="utf-8"))
        except (FileNotFoundError, ValueError):
            return {}
        now = time.time()
        return {
            key: expiry for key, expiry in stored.items()
            if isinstance(expiry, (int, float)) and expiry > now
        }

    def _save_sessions(self) -> None:
        self._sessions_path.parent.mkdir(parents=True, exist_ok=True)
        _write_private(self._sessions_path, json.dumps(self._sessions))

    def create_session(self) -> str:
        session_id = secrets.token_urlsafe(32)
        now = time.time()
        self._sessions = {k: v for k, v in self._sessions.items() if v > now}
        self._sessions[_digest(session_id)] = now + SESSION_MAX_AGE
        self._save_sessions()
        return session_id

    def revoke_session(self, session_id: Optional[str]) -> None:
        if session_id and self._sessions.pop(_digest(session_id), None) is not None:
            self._save_sessions()

    def session_valid(self, session_id: Optional[str]) -> bool:
        if not session_id:
            return False
        expiry = self._sessions.get(_digest(session_id))
        return expiry is not None and expiry > time.time()

    def is_ingress(self, scope) -> bool:
        client = scope.get("client")
        return self.under_supervisor and bool(client) and client[0] == INGRESS_PROXY_IP

    def authorize(self, scope) -> Verdict:
        if self.is_ingress(scope):
            return Verdict(VIA_INGRESS)
        conn = HTTPConnection(scope)
        scheme, _, value = conn.headers.get("authorization", "").partition(" ")
        if scheme.lower() == "bearer" and value:
            if self.token_matches(value):
                return Verdict(VIA_TOKEN)
            return Verdict(None, 401, "invalid API token")
        # Browsers can't add headers to a WebSocket handshake, so a cookie
        # alone never opens one.
        if scope["type"] == "http" and self.session_valid(conn.cookies.get(SESSION_COOKIE)):
            if scope["method"] in _SAFE_METHODS or conn.headers.get(UI_HEADER):
                return Verdict(VIA_SESSION)
            return Verdict(None, 403, "missing X-Murdock-UI header")
        return Verdict(None, 401, "authentication required")


def is_public(path: str) -> bool:
    return path in _PUBLIC_PATHS or path.startswith(_PUBLIC_PREFIXES)


class AuthMiddleware:
    """Reject non-public requests that ``ApiAuth`` doesn't allow."""

    def __init__(self, app, auth: ApiAuth) -> None:
        self.app = app
        self.auth = auth

    async def __call__(self, scope, receive, send) -> None:
        if scope["type"] not in ("http", "websocket") or is_public(scope["path"]):
            await self.app(scope, receive, send)
            return
        verdict = self.auth.authorize(scope)
        if verdict.via is not None:
            scope.setdefault("state", {})["auth"] = verdict.via
            await self.app(scope, receive, send)
            return
        if scope["type"] == "websocket":
            await send({"type": "websocket.close", "code": 1008})
            return
        body = json.dumps({"detail": verdict.detail}).encode("utf-8")
        headers = [
            (b"content-type", b"application/json"),
            (b"content-length", str(len(body)).encode("ascii")),
        ]
        if verdict.status == 401:
            headers.append((b"www-authenticate", b'Bearer realm="murdock"'))
        await send({"type": "http.response.start", "status": verdict.status, "headers": headers})
        await send({"type": "http.response.body", "body": body})
