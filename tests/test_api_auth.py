"""Web API authentication: ingress, API token, or browser session."""

from __future__ import annotations

import io
import logging
import stat
import zipfile
from types import SimpleNamespace

import pytest
from fastapi.testclient import TestClient

from murdock.api.app import create_app
from murdock.api.auth import INGRESS_PROXY_IP, SESSION_COOKIE, ApiAuth
from murdock.config import Settings
from murdock.core.context import AppContext
from murdock.core.db import open_db, set_setting

SECRETS = ("HA-LONG-LIVED-TOKEN", "MQTT-PASSWORD", "sk-OPENAI-KEY")
EVIL = "https://evil.example"
UI = {"X-Murdock-UI": "1"}


def _context(tmp_path):
    db = open_db(tmp_path / "m.db")
    set_setting(db, "ha_token", SECRETS[0])
    set_setting(db, "mqtt_password", SECRETS[1])
    ctx = AppContext(
        settings=Settings(data_dir=tmp_path), db=db, embedder=None, vad=None,
        speakers=SimpleNamespace(list_speakers=lambda: []),
        unknown=None, ha=None, mqtt=None, recognition=None,
    )
    ctx.stt_services.create("OpenAI", "openai", {"api_key": SECRETS[2], "model": "m"})
    return ctx


def _client(tmp_path, *, supervisor=False, address="testclient", env_token=None):
    auth = ApiAuth(tmp_path, env_token=env_token, under_supervisor=supervisor)
    app = create_app(_context(tmp_path), auth=auth)
    return TestClient(app, client=(address, 50000)), auth


def _signed_in(tmp_path):
    client, auth = _client(tmp_path)
    client.post("/api/auth/login", json={"token": auth.token})
    return client, auth


def _secrets_in(response) -> list[str]:
    z = zipfile.ZipFile(io.BytesIO(response.content))
    blob = "".join(z.read(n).decode("utf-8", "replace") for n in z.namelist())
    return [s for s in SECRETS if s in blob]


# --- Anonymous and cross-origin access -----------------------------------


def test_no_cors_headers_for_other_origins(tmp_path):
    client, auth = _client(tmp_path)
    anonymous = client.get("/api/backup", headers={"Origin": EVIL})
    assert anonymous.status_code == 401
    authorised = client.get(
        "/api/backup",
        headers={"Origin": EVIL, "Authorization": f"Bearer {auth.token}"},
    )
    assert authorised.status_code == 200
    assert "access-control-allow-origin" not in authorised.headers
    preflight = client.options(
        "/api/backup/restore",
        headers={"Origin": EVIL, "Access-Control-Request-Method": "POST"},
    )
    assert "access-control-allow-origin" not in preflight.headers


@pytest.mark.parametrize("path", [
    "/api/backup", "/api/settings", "/api/speakers", "/api/stt-services",
    "/api/version", "/api/state", "/api/auth/token", "/docs", "/openapi.json",
])
def test_protected_paths_need_auth(tmp_path, path):
    client, _ = _client(tmp_path)
    assert client.get(path).status_code == 401


def test_restore_needs_auth(tmp_path):
    client, _ = _client(tmp_path)
    r = client.post(
        "/api/backup/restore?mode=replace",
        files={"file": ("b.zip", b"PK\x05\x06" + b"\x00" * 18, "application/zip")},
    )
    assert r.status_code == 401


def test_public_paths(tmp_path):
    client, _ = _client(tmp_path)
    assert client.get("/api/health").status_code == 200
    assert client.get("/").status_code == 200
    assert client.get("/static/app.js").status_code == 200
    assert client.get("/api/auth/status").json() == {"authenticated": False, "via": None}


# --- API token -------------------------------------------------------------


def test_bearer_token_is_accepted(tmp_path):
    client, auth = _client(tmp_path)
    r = client.get("/api/backup", headers={"Authorization": f"Bearer {auth.token}"})
    assert r.status_code == 200
    assert _secrets_in(r) == list(SECRETS)


def test_wrong_bearer_token_is_refused(tmp_path):
    client, _ = _client(tmp_path)
    r = client.get("/api/backup", headers={"Authorization": "Bearer nope"})
    assert r.status_code == 401
    assert r.headers["www-authenticate"].startswith("Bearer")


def test_token_is_generated_once_with_private_mode(tmp_path):
    first = ApiAuth(tmp_path, under_supervisor=False)
    path = tmp_path / "api_token"
    assert path.read_text().strip() == first.token
    assert len(first.token) >= 40
    assert stat.S_IMODE(path.stat().st_mode) == 0o600
    assert ApiAuth(tmp_path, under_supervisor=False).token == first.token


def test_pinned_token_is_not_written(tmp_path):
    auth = ApiAuth(tmp_path, env_token="  pinned-token-of-length  ", under_supervisor=False)
    assert auth.token == "pinned-token-of-length" and auth.pinned
    assert not (tmp_path / "api_token").exists()


def test_short_pinned_token_logs_a_warning(tmp_path, caplog):
    with caplog.at_level(logging.WARNING, logger="murdock.api.auth"):
        ApiAuth(tmp_path, env_token="short", under_supervisor=False)
    assert "shorter than" in caplog.text


# --- Home Assistant ingress ----------------------------------------------


def test_ingress_is_trusted_under_the_supervisor(tmp_path):
    client, _ = _client(tmp_path, supervisor=True, address=INGRESS_PROXY_IP)
    assert client.get("/api/backup").status_code == 200
    assert client.get("/api/auth/status").json() == {"authenticated": True, "via": "ingress"}


def test_ingress_ip_ignored_without_supervisor(tmp_path):
    client, _ = _client(tmp_path, supervisor=False, address=INGRESS_PROXY_IP)
    assert client.get("/api/backup").status_code == 401


def test_gateway_address_is_not_ingress(tmp_path):
    # HA Core and port-forwarded LAN traffic arrive from the gateway, .1.
    client, _ = _client(tmp_path, supervisor=True, address="172.30.32.1")
    assert client.get("/api/backup").status_code == 401


def test_forwarded_headers_do_not_grant_ingress(tmp_path):
    client, _ = _client(tmp_path, supervisor=True)
    r = client.get("/api/backup", headers={
        "X-Ingress-Path": "/api/hassio_ingress/abc",
        "X-Forwarded-For": INGRESS_PROXY_IP,
    })
    assert r.status_code == 401


# --- Browser session ------------------------------------------------------


def test_login_with_wrong_token_fails(tmp_path):
    client, _ = _client(tmp_path)
    r = client.post("/api/auth/login", json={"token": "guess"})
    assert r.status_code == 401
    assert SESSION_COOKIE not in r.cookies


def test_login_sets_strict_httponly_session(tmp_path):
    client, auth = _client(tmp_path)
    r = client.post("/api/auth/login", json={"token": auth.token})
    cookie = r.headers["set-cookie"]
    assert cookie.startswith(f"{SESSION_COOKIE}=")
    assert "HttpOnly" in cookie and "SameSite=strict" in cookie
    assert "Secure" not in cookie
    assert auth.token not in cookie
    assert client.get("/api/backup").status_code == 200
    assert client.get("/api/auth/status").json()["via"] == "session"


def test_session_cookie_is_secure_behind_a_tls_proxy(tmp_path):
    client, auth = _client(tmp_path)
    r = client.post(
        "/api/auth/login", json={"token": auth.token},
        headers={"X-Forwarded-Proto": "https"},
    )
    assert "Secure" in r.headers["set-cookie"]


def test_each_login_gets_its_own_session(tmp_path):
    first, auth = _signed_in(tmp_path)
    second = TestClient(first.app)
    second.post("/api/auth/login", json={"token": auth.token})
    assert first.cookies[SESSION_COOKIE] != second.cookies[SESSION_COOKIE]


def test_foreign_malformed_cookies_are_ignored(tmp_path):
    # Cookies are shared across ports, so other services' cookies arrive too.
    client, _ = _signed_in(tmp_path)
    session = client.cookies[SESSION_COOKIE]
    client.cookies.clear()
    for prefix in ("hide_banner; ", "theme=dark mode; ", 'prefs={"a": 1}; '):
        r = client.get("/api/backup", headers={"Cookie": f"{prefix}{SESSION_COOKIE}={session}"})
        assert r.status_code == 200


def test_cookie_alone_cannot_change_state(tmp_path):
    client, _ = _signed_in(tmp_path)
    assert client.post("/api/auth/token/rotate").status_code == 403
    assert client.post("/api/auth/token/rotate", headers=UI).status_code == 200


def test_logout_revokes_the_session(tmp_path):
    client, _ = _signed_in(tmp_path)
    session = client.cookies[SESSION_COOKIE]
    assert client.post("/api/auth/logout", headers=UI).status_code == 200
    replay = TestClient(client.app)
    replay.cookies.set(SESSION_COOKIE, session)
    assert replay.get("/api/auth/token").status_code == 401


def test_logout_needs_the_ui_header(tmp_path):
    client, _ = _signed_in(tmp_path)
    assert client.post("/api/auth/logout").status_code == 403
    assert client.get("/api/backup").status_code == 200


def test_sessions_survive_a_restart(tmp_path):
    client, _ = _signed_in(tmp_path)
    session = client.cookies[SESSION_COOKIE]
    restarted, _ = _client(tmp_path)
    restarted.cookies.set(SESSION_COOKIE, session)
    assert restarted.get("/api/backup").status_code == 200
    assert session not in (tmp_path / "api_sessions.json").read_text()


def test_cookie_does_not_open_a_websocket(tmp_path):
    client, auth = _signed_in(tmp_path)
    scope = {
        "type": "websocket", "path": "/ws", "client": ("testclient", 1),
        "headers": [(b"cookie", f"{SESSION_COOKIE}={client.cookies[SESSION_COOKIE]}".encode())],
    }
    assert auth.authorize(scope).via is None


# --- Showing and rotating the token ---------------------------------------


def test_token_endpoint(tmp_path):
    client, auth = _client(tmp_path, supervisor=True, address=INGRESS_PROXY_IP)
    assert client.get("/api/auth/token").json() == {"token": auth.token, "pinned": False}


def test_rotate_invalidates_old_token_and_other_sessions(tmp_path):
    client, auth = _signed_in(tmp_path)
    other = TestClient(client.app)
    other.post("/api/auth/login", json={"token": auth.token})
    old = auth.token
    new = client.post("/api/auth/token/rotate", headers=UI).json()["token"]
    assert new != old
    assert (tmp_path / "api_token").read_text().strip() == new
    assert client.get("/api/backup").status_code == 200
    assert other.get("/api/backup").status_code == 401
    assert client.get("/api/backup", headers={"Authorization": f"Bearer {old}"}).status_code == 401


def test_pinned_token_cannot_be_rotated(tmp_path):
    client, auth = _client(tmp_path, env_token="pinned-token-of-length")
    r = client.post(
        "/api/auth/token/rotate",
        headers={"Authorization": "Bearer pinned-token-of-length"},
    )
    assert r.status_code == 409
    assert auth.token == "pinned-token-of-length"
