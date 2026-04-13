"""Audit log of every Wyoming session decision.

One row per completed utterance: who was recognised (or not), how long
the audio was, which gate path the session took, and the final
transcript that was handed back to the satellite. Kept short by a
rolling cap so the table never becomes a liability.
"""

from __future__ import annotations

import logging
import sqlite3
import time
from dataclasses import dataclass
from threading import RLock
from typing import List, Optional

_LOGGER = logging.getLogger("voiceid.recognition_log")

# Upper bound on rows kept in the table. Anything older than this is
# trimmed on insert so the audit log stays cheap to query and the DB
# file doesn't grow unbounded on a busy satellite.
_MAX_ROWS_DEFAULT = 2000


# Canonical outcome labels used across the handler + UI.
OUTCOME_MATCH = "match"
OUTCOME_UNKNOWN_FORWARDED = "unknown-forwarded"
OUTCOME_BLOCKED_NO_MATCH = "blocked-no-match"
OUTCOME_BLOCKED_NO_SPEAKERS = "blocked-no-speakers"
OUTCOME_BLOCKED_EMBED_FAILED = "blocked-embed-failed"
OUTCOME_PASSTHROUGH_SHORT = "passthrough-short"
OUTCOME_PASSTHROUGH_NO_SPEAKERS = "passthrough-no-speakers"
OUTCOME_BLOCKED_TV_NOISE = "blocked-tv-noise"
OUTCOME_EMPTY = "empty"


@dataclass
class RecognitionEvent:
    id: int
    created_at: float
    session_id: str
    satellite_id: Optional[str]
    duration_sec: float
    outcome: str
    matched_speaker: Optional[str]
    distance: Optional[float]
    threshold: Optional[float]
    verify_ms: Optional[float]
    transcript: Optional[str]


class RecognitionLog:
    """Thread-safe store for recognition events.

    Writes are cheap (single INSERT + opportunistic trim) and do NOT
    block on heavy queries; the handler can call :meth:`record` from a
    background thread via ``asyncio.to_thread``.
    """

    def __init__(
        self,
        conn: sqlite3.Connection,
        max_rows: int = _MAX_ROWS_DEFAULT,
    ) -> None:
        self.conn = conn
        self.max_rows = max_rows
        self._lock = RLock()

    def record(
        self,
        *,
        session_id: str,
        satellite_id: Optional[str],
        duration_sec: float,
        outcome: str,
        matched_speaker: Optional[str] = None,
        distance: Optional[float] = None,
        threshold: Optional[float] = None,
        verify_ms: Optional[float] = None,
        transcript: Optional[str] = None,
    ) -> int:
        """Insert one event row and return its id."""
        now = time.time()
        # Clamp transcript length so a runaway STT output never bloats
        # the audit table. 1 kB is plenty for voice commands.
        safe_transcript = None
        if transcript:
            safe_transcript = transcript[:1024]
        with self._lock:
            try:
                cur = self.conn.execute(
                    "INSERT INTO recognition_events("
                    "created_at, session_id, satellite_id, duration_sec, "
                    "outcome, matched_speaker, distance, threshold, "
                    "verify_ms, transcript) "
                    "VALUES(?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                    (
                        now,
                        session_id,
                        satellite_id,
                        duration_sec,
                        outcome,
                        matched_speaker,
                        distance,
                        threshold,
                        verify_ms,
                        safe_transcript,
                    ),
                )
                row_id = int(cur.lastrowid)
                self._trim_locked()
                self.conn.commit()
                _LOGGER.info(
                    "Recorded event #%d [%s] outcome=%s speaker=%s "
                    "duration=%.2fs transcript=%r",
                    row_id, session_id, outcome, matched_speaker or "-",
                    duration_sec, (safe_transcript or "")[:60],
                )
                return row_id
            except Exception:
                _LOGGER.exception("Failed to record recognition event")
                return 0

    def _trim_locked(self) -> None:
        """Keep only the latest ``max_rows`` entries. Caller holds the lock."""
        if self.max_rows <= 0:
            return
        self.conn.execute(
            "DELETE FROM recognition_events WHERE id IN ("
            "SELECT id FROM recognition_events "
            "ORDER BY created_at DESC LIMIT -1 OFFSET ?"
            ")",
            (self.max_rows,),
        )

    def list_events(
        self,
        *,
        limit: int = 100,
        outcome: Optional[str] = None,
        speaker: Optional[str] = None,
    ) -> List[RecognitionEvent]:
        clauses: List[str] = []
        params: List = []
        if outcome:
            clauses.append("outcome = ?")
            params.append(outcome)
        if speaker:
            clauses.append("matched_speaker = ?")
            params.append(speaker)
        where = ""
        if clauses:
            where = "WHERE " + " AND ".join(clauses)
        params.append(max(1, min(int(limit), 1000)))
        with self._lock:
            rows = self.conn.execute(
                "SELECT id, created_at, session_id, satellite_id, duration_sec, "
                "outcome, matched_speaker, distance, threshold, verify_ms, transcript "
                f"FROM recognition_events {where} "
                "ORDER BY created_at DESC LIMIT ?",
                params,
            ).fetchall()
        return [
            RecognitionEvent(
                id=row["id"],
                created_at=row["created_at"],
                session_id=row["session_id"],
                satellite_id=row["satellite_id"],
                duration_sec=row["duration_sec"],
                outcome=row["outcome"],
                matched_speaker=row["matched_speaker"],
                distance=row["distance"],
                threshold=row["threshold"],
                verify_ms=row["verify_ms"],
                transcript=row["transcript"],
            )
            for row in rows
        ]

    def stats(self, *, since_seconds: float = 24 * 3600) -> dict:
        """Return counts per outcome and per speaker for the last window."""
        cutoff = time.time() - since_seconds
        with self._lock:
            per_outcome = self.conn.execute(
                "SELECT outcome, COUNT(*) AS n FROM recognition_events "
                "WHERE created_at >= ? GROUP BY outcome",
                (cutoff,),
            ).fetchall()
            per_speaker = self.conn.execute(
                "SELECT matched_speaker AS speaker, COUNT(*) AS n "
                "FROM recognition_events "
                "WHERE created_at >= ? AND matched_speaker IS NOT NULL "
                "GROUP BY matched_speaker ORDER BY n DESC",
                (cutoff,),
            ).fetchall()
        return {
            "window_seconds": since_seconds,
            "per_outcome": {row["outcome"]: row["n"] for row in per_outcome},
            "per_speaker": {row["speaker"]: row["n"] for row in per_speaker},
        }

    def clear(self) -> int:
        with self._lock:
            cur = self.conn.execute("DELETE FROM recognition_events")
            self.conn.commit()
            return cur.rowcount
