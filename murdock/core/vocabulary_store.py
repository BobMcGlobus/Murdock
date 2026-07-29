"""Versioned vocabulary snapshots pushed by the HA integration (plan §9).

Source, not dependency: the integration mirrors the exposed entity/area/
floor registry into Murdock; Murdock persists each push as a snapshot and
keeps working from the latest one when HA is unreachable.

The snapshot feeds two consumers:
- the STT vocabulary prompt (tier 1 bias) via :meth:`terms`
- the fuzzy correction index (post-dictionary, phase 2) via
  :meth:`normalized_terms`
"""

from __future__ import annotations

import json
import logging
import re
import sqlite3
import time
import unicodedata
from threading import RLock
from typing import Any, Dict, List, Optional

_LOGGER = logging.getLogger("murdock.vocabulary")

# Rolling history: enough to diff/debug pushes without growing the DB.
_MAX_SNAPSHOTS = 10

_SCHEMA = """
CREATE TABLE IF NOT EXISTS vocabulary_snapshots (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    version TEXT,
    generated_at TEXT,
    created_at REAL NOT NULL,
    payload TEXT NOT NULL
);
"""


def _names_of(items: Any) -> List[str]:
    """Extract display names from a list of strings or dicts."""
    out: List[str] = []
    if not isinstance(items, list):
        return out
    for item in items:
        if isinstance(item, str):
            name = item
        elif isinstance(item, dict):
            name = item.get("name") or ""
        else:
            continue
        name = str(name).strip()
        if name:
            out.append(name)
    return out


def normalize_term(term: str) -> str:
    """Lowercased, accent-stripped, single-spaced form for fuzzy lookup."""
    term = unicodedata.normalize("NFKD", term)
    term = "".join(c for c in term if not unicodedata.combining(c))
    term = term.lower().replace("ß", "ss")
    return re.sub(r"\s+", " ", term).strip()


class VocabularyStore:
    """Thread-safe store for registry vocabulary snapshots."""

    def __init__(self, conn: sqlite3.Connection) -> None:
        self.conn = conn
        self._lock = RLock()
        with self._lock:
            conn.executescript(_SCHEMA)
            conn.commit()

    def save_snapshot(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        """Persist one push and trim history. Returns snapshot metadata."""
        version = payload.get("version")
        generated_at = payload.get("generated_at")
        now = time.time()
        with self._lock:
            cur = self.conn.execute(
                "INSERT INTO vocabulary_snapshots("
                "version, generated_at, created_at, payload) "
                "VALUES(?, ?, ?, ?)",
                (
                    str(version) if version is not None else None,
                    str(generated_at) if generated_at is not None else None,
                    now,
                    json.dumps(payload, ensure_ascii=False),
                ),
            )
            self.conn.execute(
                "DELETE FROM vocabulary_snapshots WHERE id IN ("
                "SELECT id FROM vocabulary_snapshots "
                "ORDER BY id DESC LIMIT -1 OFFSET ?)",
                (_MAX_SNAPSHOTS,),
            )
            self.conn.commit()
            snapshot_id = int(cur.lastrowid)
        terms = self.terms()
        _LOGGER.info(
            "Vocabulary snapshot #%d stored (version=%s, %d terms)",
            snapshot_id, version, len(terms),
        )
        return {
            "snapshot_id": snapshot_id,
            "version": version,
            "generated_at": generated_at,
            "created_at": now,
            "term_count": len(terms),
        }

    def latest(self) -> Optional[Dict[str, Any]]:
        """Latest snapshot incl. parsed payload, or None."""
        with self._lock:
            row = self.conn.execute(
                "SELECT id, version, generated_at, created_at, payload "
                "FROM vocabulary_snapshots ORDER BY id DESC LIMIT 1"
            ).fetchone()
        if row is None:
            return None
        try:
            payload = json.loads(row["payload"])
        except (TypeError, ValueError):
            payload = {}
        return {
            "snapshot_id": row["id"],
            "version": row["version"],
            "generated_at": row["generated_at"],
            "created_at": row["created_at"],
            "payload": payload,
        }

    def terms(self, limit: Optional[int] = None) -> List[str]:
        """Spoken-about terms from the latest snapshot, deduped.

        Priority order matters when a cap applies: entity names first
        (the things actually addressed by voice), then aliases, then
        area and floor names.
        """
        snap = self.latest()
        if snap is None:
            return []
        payload = snap["payload"]
        names: List[str] = []
        aliases: List[str] = []
        for ent in payload.get("entities") or []:
            if not isinstance(ent, dict):
                continue
            name = str(ent.get("name") or "").strip()
            if name:
                names.append(name)
            for alias in ent.get("aliases") or []:
                alias = str(alias).strip()
                if alias:
                    aliases.append(alias)
        ordered = (
            names
            + aliases
            + _names_of(payload.get("areas"))
            + _names_of(payload.get("floors"))
        )
        seen: set = set()
        out: List[str] = []
        for term in ordered:
            key = normalize_term(term)
            if not key or key in seen:
                continue
            seen.add(key)
            out.append(term)
            if limit is not None and len(out) >= limit:
                break
        return out

    def normalized_terms(self) -> Dict[str, str]:
        """``{normalized: original}`` map for the fuzzy correction index."""
        return {normalize_term(t): t for t in self.terms()}
