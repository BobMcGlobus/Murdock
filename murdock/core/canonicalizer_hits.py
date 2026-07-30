"""Counts of applied canonicalizations, so rules can be learned.

A correction that fires again and again is not a fuzzy guess any more —
it is a fact about your household's vocabulary and your STT engine. Once
seen a few times it belongs in the explicit correction dictionary, where
it runs as a deterministic replacement with no fuzzy risk at all.

This store keeps the tally; promoting an entry is a deliberate click, not
an automatism (plan §11: "Vorschlag, nicht Automatik").
"""

from __future__ import annotations

import logging
import sqlite3
import time
from dataclasses import dataclass
from threading import RLock
from typing import Iterable, List

_LOGGER = logging.getLogger("murdock.canonicalizer_hits")

_SCHEMA = """
CREATE TABLE IF NOT EXISTS canonicalizer_hits (
    original TEXT NOT NULL,
    replacement TEXT NOT NULL,
    count INTEGER NOT NULL DEFAULT 0,
    first_seen REAL NOT NULL,
    last_seen REAL NOT NULL,
    PRIMARY KEY (original, replacement)
);
"""


@dataclass
class Hit:
    original: str
    replacement: str
    count: int
    first_seen: float
    last_seen: float


class CanonicalizerHits:
    """Thread-safe tally of applied corrections."""

    def __init__(self, conn: sqlite3.Connection) -> None:
        self.conn = conn
        self._lock = RLock()
        with self._lock:
            conn.executescript(_SCHEMA)
            conn.commit()

    def record(self, replacements: Iterable) -> None:
        """Increment the counter for each applied replacement."""
        now = time.time()
        rows = [
            (r.original, r.replacement)
            for r in replacements
            if getattr(r, "original", "") and getattr(r, "replacement", "")
        ]
        if not rows:
            return
        with self._lock:
            for original, replacement in rows:
                self.conn.execute(
                    "INSERT INTO canonicalizer_hits("
                    "original, replacement, count, first_seen, last_seen) "
                    "VALUES(?, ?, 1, ?, ?) "
                    "ON CONFLICT(original, replacement) DO UPDATE SET "
                    "count = count + 1, last_seen = excluded.last_seen",
                    (original, replacement, now, now),
                )
            self.conn.commit()

    def top(self, *, limit: int = 20) -> List[Hit]:
        with self._lock:
            rows = self.conn.execute(
                "SELECT original, replacement, count, first_seen, last_seen "
                "FROM canonicalizer_hits "
                "ORDER BY count DESC, last_seen DESC LIMIT ?",
                (max(1, min(int(limit), 200)),),
            ).fetchall()
        return [
            Hit(
                original=r["original"],
                replacement=r["replacement"],
                count=r["count"],
                first_seen=r["first_seen"],
                last_seen=r["last_seen"],
            )
            for r in rows
        ]

    def forget(self, original: str, replacement: str) -> bool:
        """Drop one tally — used after promoting it to an explicit rule."""
        with self._lock:
            cur = self.conn.execute(
                "DELETE FROM canonicalizer_hits "
                "WHERE original = ? AND replacement = ?",
                (original, replacement),
            )
            self.conn.commit()
            return cur.rowcount > 0

    def clear(self) -> int:
        with self._lock:
            cur = self.conn.execute("DELETE FROM canonicalizer_hits")
            self.conn.commit()
            return cur.rowcount
