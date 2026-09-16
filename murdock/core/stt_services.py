"""Speech-to-text services as first-class, reusable connections.

Transcription used to be configured as one backend selector plus a
parallel set of *shadow* fields that duplicated every credential, and a
local fallback hard-wired to the upstream URI. Switching engines meant
retyping keys, and adding a third engine to compare was impossible.

Now each connection is a named service — a Wyoming server, an
OpenAI-compatible endpoint, Voxtral, a Home Assistant STT entity — kept
once and referred to by role:

* **main** — exactly one. The only service that can stream while the
  user is still speaking.
* **fallbacks** — an ordered chain, tried when the main fails and,
  optionally, when it hears nothing. A local service at the end is the
  offline fallback: with no internet every remote service fails at
  connect and the chain reaches it. Once a *remote* service fails that
  way, the remaining remote services are skipped rather than each
  burning its own timeout.
* **shadows** — any number, for benchmarking. They run only after the
  answer has gone back, one at a time, so they never compete with the
  main for the network or the CPU and never skew each other's timings.

Roles are stored in the settings table and refer to service ids;
credentials sit in the service's config and never leave through the API.
"""

from __future__ import annotations

import ipaddress
import json
import logging
import sqlite3
import time
from dataclasses import dataclass, field
from threading import RLock
from typing import Dict, List, Optional
from urllib.parse import urlparse

from .db import get_setting, set_setting

_LOGGER = logging.getLogger("murdock.stt_services")

KIND_WYOMING = "wyoming"
KIND_OPENAI = "openai"
KIND_VOXTRAL = "voxtral"
KIND_HA = "ha"
KINDS = (KIND_WYOMING, KIND_OPENAI, KIND_VOXTRAL, KIND_HA)

#: Config keys that hold credentials. Never returned by the API; an
#: empty value on update means "keep what is stored".
SECRET_KEYS = frozenset({"api_key"})

#: Which config keys each kind understands.
KIND_FIELDS: Dict[str, tuple] = {
    KIND_WYOMING: ("uri", "timeout_sec"),
    KIND_OPENAI: ("base_url", "api_key", "model", "timeout_sec"),
    KIND_VOXTRAL: ("api_key", "model", "timeout_sec"),
    KIND_HA: ("entity_id", "timeout_sec"),
}

_KEY_MAIN = "stt_main_service"
_KEY_FALLBACKS = "stt_fallback_services"
_KEY_SHADOWS = "stt_shadow_services"
_KEY_FALLBACK_ON_EMPTY = "stt_fallback_on_empty"

#: Single-label names and these suffixes are treated as the local network.
_LOCAL_SUFFIXES = (".local", ".lan", ".home", ".internal", ".home.arpa", ".localdomain")


def _normalize_wyoming_uri(value: str) -> str:
    v = (value or "").strip()
    if v and "://" not in v:
        v = "tcp://" + v
    return v


def _host_is_local(host: Optional[str]) -> bool:
    """Whether a host is on the local network.

    Unknown or unparsable hosts count as local: the point of the check
    is to *skip* services, and skipping one wrongly is the worse mistake.
    """
    if not host:
        return True
    h = host.strip().lower().strip("[]")
    if h in ("localhost",) or h.endswith(_LOCAL_SUFFIXES):
        return True
    try:
        ip = ipaddress.ip_address(h)
    except ValueError:
        # A bare name without dots is a LAN or container name
        # (core-whisper, murdock); a dotted one is public DNS.
        return "." not in h
    return ip.is_private or ip.is_loopback or ip.is_link_local


@dataclass
class SttService:
    id: int
    name: str
    kind: str
    config: dict = field(default_factory=dict)
    created_at: float = 0.0

    def is_remote(self, ha_base_url: Optional[str] = None) -> bool:
        """Whether reaching this service needs the internet."""
        if self.kind == KIND_VOXTRAL:
            return True
        if self.kind == KIND_WYOMING:
            host = urlparse(_normalize_wyoming_uri(self.config.get("uri", ""))).hostname
            return not _host_is_local(host)
        if self.kind == KIND_OPENAI:
            host = urlparse(self.config.get("base_url") or "https://api.openai.com").hostname
            return not _host_is_local(host)
        if self.kind == KIND_HA:
            # The request goes to Home Assistant; whether *its* engine
            # needs the internet is its business, and a failure there
            # comes back as an error rather than an unreachable host.
            return not _host_is_local(urlparse(ha_base_url or "").hostname)
        return False

    def public(self) -> dict:
        """Everything except credentials, plus which secrets are set."""
        cfg = {k: v for k, v in self.config.items() if k not in SECRET_KEYS}
        return {
            "id": self.id,
            "name": self.name,
            "kind": self.kind,
            "config": cfg,
            "secrets_set": sorted(k for k in SECRET_KEYS if self.config.get(k)),
            "created_at": self.created_at,
        }


@dataclass
class SttRoles:
    main: Optional[int] = None
    fallbacks: List[int] = field(default_factory=list)
    shadows: List[int] = field(default_factory=list)
    fallback_on_empty: bool = True


def _clean_config(kind: str, config: dict) -> dict:
    allowed = KIND_FIELDS[kind]
    out: dict = {}
    for key in allowed:
        if key not in config:
            continue
        value = config[key]
        if key == "timeout_sec":
            if value in (None, ""):
                continue
            out[key] = max(1.0, min(120.0, float(value)))
            continue
        value = "" if value is None else str(value).strip()
        if key == "uri":
            value = _normalize_wyoming_uri(value)
        out[key] = value
    return out


class SttServiceStore:
    """Persistence for services and their roles."""

    def __init__(self, conn: sqlite3.Connection) -> None:
        self.conn = conn
        self._lock = RLock()

    # -- services ------------------------------------------------------

    def _row(self, row) -> SttService:
        try:
            cfg = json.loads(row["config"] or "{}")
        except (TypeError, ValueError):
            cfg = {}
        return SttService(
            id=int(row["id"]), name=row["name"], kind=row["kind"],
            config=cfg if isinstance(cfg, dict) else {},
            created_at=float(row["created_at"] or 0.0),
        )

    def list(self) -> List[SttService]:
        with self._lock:
            rows = self.conn.execute(
                "SELECT id, name, kind, config, created_at FROM stt_services "
                "ORDER BY id"
            ).fetchall()
        return [self._row(r) for r in rows]

    def get(self, service_id: Optional[int]) -> Optional[SttService]:
        if service_id is None:
            return None
        with self._lock:
            row = self.conn.execute(
                "SELECT id, name, kind, config, created_at FROM stt_services "
                "WHERE id = ?", (int(service_id),),
            ).fetchone()
        return self._row(row) if row else None

    def create(self, name: str, kind: str, config: dict) -> SttService:
        if kind not in KINDS:
            raise ValueError(f"unknown service kind {kind!r}")
        label = (name or "").strip() or kind
        cfg = _clean_config(kind, config or {})
        with self._lock:
            cur = self.conn.execute(
                "INSERT INTO stt_services (name, kind, config, created_at) "
                "VALUES (?, ?, ?, ?)",
                (label, kind, json.dumps(cfg), time.time()),
            )
            self.conn.commit()
            new_id = int(cur.lastrowid)
        # The first service anyone creates is the obvious main.
        roles = self.roles()
        if roles.main is None:
            roles.main = new_id
            self.set_roles(roles)
        return self.get(new_id)  # type: ignore[return-value]

    def update(
        self, service_id: int, *, name: Optional[str] = None,
        config: Optional[dict] = None,
    ) -> SttService:
        current = self.get(service_id)
        if current is None:
            raise KeyError(service_id)
        new_name = current.name if name is None else ((name or "").strip() or current.name)
        merged = dict(current.config)
        if config is not None:
            incoming = _clean_config(current.kind, config)
            for key, value in incoming.items():
                # An empty secret means "leave it", so a form that never
                # shows the stored key can still be saved.
                if key in SECRET_KEYS and not value:
                    continue
                merged[key] = value
            for key in KIND_FIELDS[current.kind]:
                if key == "timeout_sec" and key in config and config[key] in (None, ""):
                    merged.pop(key, None)
        with self._lock:
            self.conn.execute(
                "UPDATE stt_services SET name = ?, config = ? WHERE id = ?",
                (new_name, json.dumps(merged), int(service_id)),
            )
            self.conn.commit()
        return self.get(service_id)  # type: ignore[return-value]

    def delete(self, service_id: int) -> None:
        roles = self.roles()
        if roles.main == int(service_id):
            raise ValueError("cannot delete the main service — pick another main first")
        with self._lock:
            self.conn.execute("DELETE FROM stt_services WHERE id = ?", (int(service_id),))
            self.conn.commit()
        roles.fallbacks = [i for i in roles.fallbacks if i != int(service_id)]
        roles.shadows = [i for i in roles.shadows if i != int(service_id)]
        self.set_roles(roles)

    # -- roles -----------------------------------------------------------

    @staticmethod
    def _ids(raw: Optional[str]) -> List[int]:
        if not raw:
            return []
        try:
            data = json.loads(raw)
        except (TypeError, ValueError):
            return []
        out: List[int] = []
        for item in data if isinstance(data, list) else []:
            try:
                value = int(item)
            except (TypeError, ValueError):
                continue
            if value not in out:
                out.append(value)
        return out

    def roles(self) -> SttRoles:
        main_raw = get_setting(self.conn, _KEY_MAIN)
        try:
            main = int(main_raw) if main_raw else None
        except ValueError:
            main = None
        on_empty = get_setting(self.conn, _KEY_FALLBACK_ON_EMPTY)
        return SttRoles(
            main=main,
            fallbacks=self._ids(get_setting(self.conn, _KEY_FALLBACKS)),
            shadows=self._ids(get_setting(self.conn, _KEY_SHADOWS)),
            fallback_on_empty=(
                True if on_empty is None
                else on_empty.lower() in ("1", "true", "yes", "on")
            ),
        )

    def set_roles(self, roles: SttRoles) -> SttRoles:
        """Persist roles, dropping ids that do not exist.

        The main is never its own fallback or shadow: a fallback that is
        the service that just failed cannot help, and a shadow of the
        main measures the main against itself.
        """
        existing = {s.id for s in self.list()}
        main = roles.main if roles.main in existing else None
        fallbacks = [i for i in roles.fallbacks if i in existing and i != main]
        shadows = [i for i in roles.shadows if i in existing and i != main]
        # De-duplicate while keeping the chain's order.
        fallbacks = list(dict.fromkeys(fallbacks))
        shadows = list(dict.fromkeys(shadows))
        set_setting(self.conn, _KEY_MAIN, "" if main is None else str(main))
        set_setting(self.conn, _KEY_FALLBACKS, json.dumps(fallbacks))
        set_setting(self.conn, _KEY_SHADOWS, json.dumps(shadows))
        set_setting(
            self.conn, _KEY_FALLBACK_ON_EMPTY,
            "true" if roles.fallback_on_empty else "false",
        )
        return SttRoles(main, fallbacks, shadows, roles.fallback_on_empty)

    # -- backup ------------------------------------------------------------

    def dump(self) -> List[dict]:
        """Every service with its full config, credentials included.

        Only for the backup archive, which already carries the HA token
        and MQTT password for the same reason: a restore must work
        without re-entering anything.
        """
        return [
            {"id": s.id, "name": s.name, "kind": s.kind,
             "config": s.config, "created_at": s.created_at}
            for s in self.list()
        ]

    def replace_all(self, rows: List[dict]) -> int:
        """Swap the whole table for a dump, keeping ids so roles still fit."""
        clean = []
        for row in rows or []:
            if not isinstance(row, dict) or row.get("kind") not in KINDS:
                continue
            try:
                sid = int(row["id"])
            except (KeyError, TypeError, ValueError):
                continue
            cfg = row.get("config") if isinstance(row.get("config"), dict) else {}
            clean.append((
                sid, str(row.get("name") or row["kind"]), row["kind"],
                json.dumps(_clean_config(row["kind"], cfg)),
                float(row.get("created_at") or time.time()),
            ))
        with self._lock:
            self.conn.execute("DELETE FROM stt_services")
            self.conn.executemany(
                "INSERT INTO stt_services (id, name, kind, config, created_at) "
                "VALUES (?, ?, ?, ?, ?)", clean,
            )
            self.conn.commit()
        return len(clean)

    def is_configured(self) -> bool:
        """Whether services exist or roles were ever written."""
        return bool(self.list()) or get_setting(self.conn, _KEY_MAIN) is not None


def migrate_legacy_settings(store: SttServiceStore, conn, settings) -> bool:
    """Turn the flat pre-0.11 STT settings into services and roles.

    Runs once: only when no service exists and no role was ever stored,
    so it can never overwrite what someone set up by hand. The old
    settings rows are left in place — a downgrade still finds them.

    Reads the legacy values the way the removed getters did: a stored
    override first, the environment default otherwise.
    """
    if store.is_configured():
        return False

    def legacy(key: str, default=None, *, keep_empty: bool = False):
        # The old getters mostly read an empty override as "use the
        # environment default" — clearing the upstream field in the UI
        # stored "" and fell back to UPSTREAM_URI. Only keys and the HA
        # entity treated "" as a deliberate value.
        value = get_setting(conn, key)
        if value is not None and (value or keep_empty):
            return value
        return getattr(settings, key, default)

    def truthy(key: str, default=False) -> bool:
        value = get_setting(conn, key)
        if value is None:
            return bool(getattr(settings, key, default))
        return value.lower() in ("1", "true", "yes", "on")

    upstream_uri = _normalize_wyoming_uri(legacy("upstream_uri", "") or "")
    backend = legacy("stt_backend", "upstream") or "upstream"

    created: Dict[str, int] = {}

    def wyoming(uri: str, name: str) -> Optional[int]:
        if not uri:
            return None
        key = f"wyoming:{uri}"
        if key not in created:
            created[key] = store.create(name, KIND_WYOMING, {"uri": uri}).id
        return created[key]

    main_id: Optional[int] = None
    if backend == "voxtral":
        api_key = legacy("mistral_api_key", "", keep_empty=True) or ""
        main_id = store.create(
            "Voxtral", KIND_VOXTRAL,
            {"api_key": api_key, "model": legacy("mistral_model", "voxtral-mini-latest")},
        ).id
    elif backend == "openai":
        main_id = store.create(
            "OpenAI-compatible", KIND_OPENAI,
            {
                "base_url": legacy("openai_base_url", "") or "",
                "api_key": legacy("openai_api_key", "", keep_empty=True) or "",
                "model": legacy("openai_model", "") or "",
            },
        ).id
    elif backend == "ha":
        main_id = store.create(
            "Home Assistant", KIND_HA,
            {"entity_id": (legacy("ha_stt_entity", "", keep_empty=True) or "").strip()},
        ).id
    else:
        main_id = wyoming(upstream_uri, "Wyoming")

    fallbacks: List[int] = []
    if backend != "upstream" and truthy("stt_local_fallback"):
        fid = wyoming(upstream_uri, "Wyoming")
        if fid is not None:
            fallbacks.append(fid)

    shadows: List[int] = []
    shadow_kind = legacy("shadow_stt_backend", "none") or "none"
    shadow_id: Optional[int] = None
    if shadow_kind == "upstream":
        shadow_id = wyoming(
            _normalize_wyoming_uri(legacy("shadow_upstream_uri", "") or ""),
            "Wyoming (shadow)",
        )
    elif shadow_kind == "voxtral":
        key = legacy("shadow_mistral_api_key", "") or legacy("mistral_api_key", "", keep_empty=True) or ""
        shadow_id = store.create(
            "Voxtral (shadow)", KIND_VOXTRAL,
            {"api_key": key, "model": legacy("shadow_mistral_model", "voxtral-small-latest")},
        ).id
    elif shadow_kind == "openai":
        key = legacy("shadow_openai_api_key", "") or legacy("openai_api_key", "", keep_empty=True) or ""
        shadow_id = store.create(
            "OpenAI-compatible (shadow)", KIND_OPENAI,
            {
                "base_url": legacy("shadow_openai_base_url", "") or legacy("openai_base_url", "") or "",
                "api_key": key,
                "model": legacy("shadow_openai_model", "") or "",
            },
        ).id
    if shadow_id is not None:
        shadows.append(shadow_id)
        # "Let the shadow answer when the primary heard nothing" is what
        # a fallback on empty does now; keep that behaviour for whoever
        # had it switched on.
        if truthy("shadow_rescues_empty", True) and shadow_id not in fallbacks:
            fallbacks.append(shadow_id)

    store.set_roles(SttRoles(
        main=main_id, fallbacks=fallbacks, shadows=shadows,
        fallback_on_empty=truthy("shadow_rescues_empty", True),
    ))
    _LOGGER.info(
        "Migrated STT settings into %d service(s): main=%s fallbacks=%s shadows=%s",
        len(store.list()), main_id, fallbacks, shadows,
    )
    return True
