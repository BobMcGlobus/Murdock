"""Unknown voice logging with TTL-based auto-cleanup."""

from __future__ import annotations

import asyncio
import logging
import sqlite3
import time
from dataclasses import dataclass
from threading import RLock
from typing import Dict, List, Optional

import numpy as np

from .audio import encode_wav
from .speaker_store import _embedding_to_blob

_LOGGER = logging.getLogger("voiceid.unknown_store")


@dataclass
class UnknownSample:
    id: int
    session_id: str
    satellite_id: Optional[str]
    duration_sec: float
    best_distance: float
    best_speaker: Optional[str]
    liveness_score: Optional[float]
    tag: Optional[str]
    created_at: float


class UnknownStore:
    """CRUD + TTL cleanup for rejected-audio samples."""

    def __init__(
        self,
        conn: sqlite3.Connection,
        ttl_hours: int = 48,
        cleanup_interval_minutes: int = 30,
    ) -> None:
        self.conn = conn
        self.ttl_hours = ttl_hours
        self.cleanup_interval = cleanup_interval_minutes * 60
        self._lock = RLock()
        self._cleanup_task: Optional[asyncio.Task] = None

    def record(
        self,
        session_id: str,
        pcm_bytes: bytes,
        embedding: np.ndarray,
        duration_sec: float,
        best_distance: float,
        best_speaker: Optional[str] = None,
        satellite_id: Optional[str] = None,
        liveness_score: Optional[float] = None,
    ) -> int:
        """Save an unknown audio sample and return its database id."""
        wav_bytes = encode_wav(pcm_bytes)
        emb_blob = _embedding_to_blob(embedding)
        now = time.time()
        with self._lock:
            cur = self.conn.execute(
                "INSERT INTO unknown_samples("
                "session_id, satellite_id, audio, embedding, duration_sec, "
                "best_distance, best_speaker, liveness_score, tag, created_at) "
                "VALUES(?, ?, ?, ?, ?, ?, ?, ?, NULL, ?)",
                (
                    session_id,
                    satellite_id,
                    wav_bytes,
                    emb_blob,
                    duration_sec,
                    best_distance,
                    best_speaker,
                    liveness_score,
                    now,
                ),
            )
            self.conn.commit()
            return int(cur.lastrowid)

    def list_samples(
        self, include_tagged: bool = True, limit: int = 200
    ) -> List[UnknownSample]:
        where = ""
        if not include_tagged:
            where = "WHERE tag IS NULL"
        with self._lock:
            rows = self.conn.execute(
                "SELECT id, session_id, satellite_id, duration_sec, best_distance, "
                "best_speaker, liveness_score, tag, created_at "
                f"FROM unknown_samples {where} ORDER BY created_at DESC LIMIT ?",
                (limit,),
            ).fetchall()
        return [
            UnknownSample(
                id=row["id"],
                session_id=row["session_id"],
                satellite_id=row["satellite_id"],
                duration_sec=row["duration_sec"],
                best_distance=row["best_distance"],
                best_speaker=row["best_speaker"],
                liveness_score=row["liveness_score"],
                tag=row["tag"],
                created_at=row["created_at"],
            )
            for row in rows
        ]

    def get_audio(self, sample_id: int) -> Optional[bytes]:
        with self._lock:
            row = self.conn.execute(
                "SELECT audio FROM unknown_samples WHERE id = ?", (sample_id,)
            ).fetchone()
        return bytes(row["audio"]) if row else None

    def get_sample(self, sample_id: int) -> Optional[Dict]:
        with self._lock:
            row = self.conn.execute(
                "SELECT id, session_id, satellite_id, duration_sec, best_distance, "
                "best_speaker, liveness_score, tag, created_at "
                "FROM unknown_samples WHERE id = ?",
                (sample_id,),
            ).fetchone()
        return dict(row) if row else None

    def tag(self, sample_id: int, tag: str) -> bool:
        with self._lock:
            cur = self.conn.execute(
                "UPDATE unknown_samples SET tag = ? WHERE id = ?",
                (tag, sample_id),
            )
            self.conn.commit()
            return cur.rowcount > 0

    def delete(self, sample_id: int) -> bool:
        with self._lock:
            cur = self.conn.execute(
                "DELETE FROM unknown_samples WHERE id = ?", (sample_id,)
            )
            self.conn.commit()
            return cur.rowcount > 0

    def cleanup_expired(self) -> int:
        """Delete untagged samples older than ``ttl_hours``."""
        cutoff = time.time() - (self.ttl_hours * 3600)
        with self._lock:
            cur = self.conn.execute(
                "DELETE FROM unknown_samples "
                "WHERE tag IS NULL AND created_at < ?",
                (cutoff,),
            )
            self.conn.commit()
        deleted = cur.rowcount
        if deleted:
            _LOGGER.info(
                "Cleaned up %d expired unknown samples (ttl=%dh)",
                deleted, self.ttl_hours,
            )
        return deleted

    def pop_embedding(self, sample_id: int) -> Optional[np.ndarray]:
        """Return the stored embedding for a sample (used when promoting to speaker)."""
        with self._lock:
            row = self.conn.execute(
                "SELECT embedding FROM unknown_samples WHERE id = ?", (sample_id,)
            ).fetchone()
        if row is None:
            return None
        return np.frombuffer(bytes(row["embedding"]), dtype=np.float32).copy()

    async def run_cleanup_loop(self) -> None:
        """Background task: periodically call :meth:`cleanup_expired`."""
        _LOGGER.info(
            "Starting unknown-sample cleanup loop (ttl=%dh, interval=%ds)",
            self.ttl_hours, self.cleanup_interval,
        )
        while True:
            try:
                await asyncio.to_thread(self.cleanup_expired)
            except Exception:
                _LOGGER.exception("Unknown-sample cleanup failed")
            await asyncio.sleep(self.cleanup_interval)
