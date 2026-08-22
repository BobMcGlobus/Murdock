"""Audit log of every Wyoming session decision.

One row per completed utterance: who was recognised (or not), how long
the audio was, which gate path the session took, and the final
transcript that was handed back to the satellite. Kept short by a
rolling cap so the table never becomes a liability.
"""

from __future__ import annotations

import json
import logging
import sqlite3
import time
from dataclasses import dataclass, field
from threading import RLock
from typing import List, Optional

_LOGGER = logging.getLogger("murdock.recognition_log")

# Upper bound on rows kept in the table. Anything older than this is
# trimmed on insert so the audit log stays cheap to query and the DB
# file doesn't grow unbounded on a busy satellite.
_MAX_ROWS_DEFAULT = 2000


# Canonical outcome labels used across the handler + UI.
OUTCOME_MATCH = "match"
OUTCOME_UNKNOWN_FORWARDED = "unknown-forwarded"
# Margin gate: best speaker matched the absolute threshold, but the
# second-best speaker was too close behind to trust the identity.
OUTCOME_UNCERTAIN_FORWARDED = "uncertain-forwarded"
OUTCOME_BLOCKED_UNCERTAIN = "blocked-uncertain"
OUTCOME_BLOCKED_NO_MATCH = "blocked-no-match"
OUTCOME_BLOCKED_NO_SPEAKERS = "blocked-no-speakers"
OUTCOME_BLOCKED_EMBED_FAILED = "blocked-embed-failed"
OUTCOME_PASSTHROUGH_SHORT = "passthrough-short"
OUTCOME_PASSTHROUGH_NO_SPEAKERS = "passthrough-no-speakers"
OUTCOME_BLOCKED_TV_NOISE = "blocked-tv-noise"
OUTCOME_BLOCKED_EARLY_REJECT = "blocked-early-reject"
OUTCOME_EMPTY = "empty"
# The speaker said the wake word was a mistake.
OUTCOME_CANCELLED = "cancelled"


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
    # A/B shadow: a second STT engine's transcript of the same audio,
    # filled in asynchronously after the event was recorded.
    shadow_transcript: Optional[str] = None
    shadow_engine: Optional[str] = None
    # Speaker weight (plan §11) and best-vs-second-best margin used by
    # the margin gate — logged so "why didn't the rule fire" stays
    # answerable later.
    weight: Optional[float] = None
    margin: Optional[float] = None
    # Wall-clock time each STT engine needed for this utterance, so the
    # A/B comparison covers speed and not just wording.
    transcript_ms: Optional[float] = None
    shadow_ms: Optional[float] = None
    # Breakdown of the primary engine's request: TTFB vs body download,
    # plus what was actually uploaded. One number can't distinguish
    # "the model thought hard" from "the upload crawled".
    transcript_timing: Optional[dict] = None
    # Whether the utterance was whispered, and how strongly — the score
    # is what makes the threshold tunable against real recordings.
    whisper: bool = False
    whisper_score: Optional[float] = None
    # Every enrolled voice heard in the clip, JSON-decoded.
    speakers: List[dict] = field(default_factory=list)


def _decode_timing(raw) -> Optional[dict]:
    """Decode the stored request breakdown, tolerating anything malformed."""
    if not raw:
        return None
    try:
        data = json.loads(raw)
    except (TypeError, ValueError):
        return None
    return data if isinstance(data, dict) else None


def _decode_speakers(raw) -> List[dict]:
    """Decode the stored roster, tolerating anything malformed."""
    if not raw:
        return []
    try:
        data = json.loads(raw)
    except (TypeError, ValueError):
        return []
    return data if isinstance(data, list) else []


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
        weight: Optional[float] = None,
        margin: Optional[float] = None,
        transcript_ms: Optional[float] = None,
        transcript_timing: Optional[dict] = None,
        whisper: bool = False,
        whisper_score: Optional[float] = None,
        speakers: Optional[List[dict]] = None,
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
                    "verify_ms, transcript, "
                    "weight, margin, transcript_ms, whisper, "
                    "whisper_score, speakers, transcript_timing) "
                    "VALUES(?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
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
                        weight,
                        margin,
                        transcript_ms,
                        1 if whisper else 0,
                        whisper_score,
                        json.dumps(speakers) if speakers else None,
                        json.dumps(transcript_timing) if transcript_timing else None,
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
                "outcome, matched_speaker, distance, threshold, verify_ms, "
                "transcript, shadow_transcript, shadow_engine, weight, margin, "
                "transcript_ms, shadow_ms, whisper, whisper_score, speakers, "
                "transcript_timing "
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
                shadow_transcript=row["shadow_transcript"] if "shadow_transcript" in row.keys() else None,
                shadow_engine=row["shadow_engine"] if "shadow_engine" in row.keys() else None,
                weight=row["weight"] if "weight" in row.keys() else None,
                margin=row["margin"] if "margin" in row.keys() else None,
                transcript_ms=row["transcript_ms"] if "transcript_ms" in row.keys() else None,
                shadow_ms=row["shadow_ms"] if "shadow_ms" in row.keys() else None,
                transcript_timing=_decode_timing(
                    row["transcript_timing"]
                    if "transcript_timing" in row.keys() else None
                ),
                whisper=bool(row["whisper"]) if "whisper" in row.keys() and row["whisper"] else False,
                whisper_score=row["whisper_score"] if "whisper_score" in row.keys() else None,
                speakers=_decode_speakers(row["speakers"] if "speakers" in row.keys() else None),
            )
            for row in rows
        ]

    def set_shadow(
        self,
        event_id: int,
        transcript: str,
        engine: str,
        shadow_ms: Optional[float] = None,
    ) -> bool:
        """Attach the A/B shadow transcript to an already-recorded event.

        The shadow engine runs after the main response went out, so this
        is always a late UPDATE by id. Returns False when the row has
        already been trimmed away.
        """
        if not event_id:
            return False
        with self._lock:
            try:
                cur = self.conn.execute(
                    "UPDATE recognition_events SET shadow_transcript = ?, "
                    "shadow_engine = ?, shadow_ms = ? WHERE id = ?",
                    ((transcript or "")[:1024], engine, shadow_ms, event_id),
                )
                self.conn.commit()
                return cur.rowcount > 0
            except Exception:
                _LOGGER.exception("Failed to store shadow transcript")
                return False

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
