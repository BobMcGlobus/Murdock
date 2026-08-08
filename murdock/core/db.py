"""SQLite + sqlite-vec setup and schema management."""

from __future__ import annotations

import logging
import sqlite3
from pathlib import Path
from typing import Optional

import sqlite_vec

_LOGGER = logging.getLogger("murdock.db")

_SCHEMA = """
CREATE TABLE IF NOT EXISTS speakers (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    name TEXT NOT NULL UNIQUE,
    ha_user_id TEXT,
    role TEXT,
    enrollment_count INTEGER NOT NULL DEFAULT 0,
    created_at REAL NOT NULL,
    updated_at REAL NOT NULL
);

CREATE TABLE IF NOT EXISTS speaker_samples (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    speaker_id INTEGER NOT NULL REFERENCES speakers(id) ON DELETE CASCADE,
    audio BLOB NOT NULL,
    duration_sec REAL NOT NULL,
    source TEXT,
    filename TEXT,
    created_at REAL NOT NULL
);

CREATE TABLE IF NOT EXISTS unknown_samples (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    session_id TEXT NOT NULL,
    satellite_id TEXT,
    audio BLOB NOT NULL,
    embedding BLOB NOT NULL,
    duration_sec REAL NOT NULL,
    best_distance REAL NOT NULL,
    best_speaker TEXT,
    liveness_score REAL,
    tag TEXT,
    created_at REAL NOT NULL
);

CREATE TABLE IF NOT EXISTS settings (
    key TEXT PRIMARY KEY,
    value TEXT NOT NULL
);

-- Per-satellite sub-centroids: one voiceprint per (speaker, microphone).
-- Different satellites color the embedding differently (mic count,
-- beamforming, room); matching against a same-mic centroid removes that
-- bias. Few rows (speakers × satellites), so plain BLOBs + numpy cosine
-- are plenty — no vec0 needed.
CREATE TABLE IF NOT EXISTS speaker_satellite_centroids (
    speaker_id INTEGER NOT NULL REFERENCES speakers(id) ON DELETE CASCADE,
    satellite_id TEXT NOT NULL,
    embedding BLOB NOT NULL,
    sample_count INTEGER NOT NULL,
    updated_at REAL NOT NULL,
    PRIMARY KEY (speaker_id, satellite_id)
);

CREATE TABLE IF NOT EXISTS recognition_events (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    created_at REAL NOT NULL,
    session_id TEXT NOT NULL,
    satellite_id TEXT,
    duration_sec REAL NOT NULL,
    outcome TEXT NOT NULL,
    matched_speaker TEXT,
    distance REAL,
    threshold REAL,
    verify_ms REAL,
    transcript TEXT,
    emotion TEXT,
    emotion_confidence REAL,
    shadow_transcript TEXT,
    shadow_engine TEXT,
    weight REAL,
    margin REAL,
    transcript_ms REAL,
    shadow_ms REAL,
    whisper INTEGER
);

CREATE INDEX IF NOT EXISTS idx_unknown_created ON unknown_samples(created_at);
CREATE INDEX IF NOT EXISTS idx_speaker_samples ON speaker_samples(speaker_id);
CREATE INDEX IF NOT EXISTS idx_recognition_created ON recognition_events(created_at);
"""

_VEC_SCHEMA = """
CREATE VIRTUAL TABLE IF NOT EXISTS speaker_embeddings USING vec0(
    speaker_id INTEGER PRIMARY KEY,
    embedding FLOAT[192]
);
"""


def _table_columns(conn: sqlite3.Connection, table: str) -> set[str]:
    return {row["name"] for row in conn.execute(f"PRAGMA table_info({table})").fetchall()}


def _migrate(conn: sqlite3.Connection) -> None:
    """Forward-only schema migrations for installs from earlier MVPs."""
    speaker_cols = _table_columns(conn, "speakers")
    if "role" not in speaker_cols:
        conn.execute("ALTER TABLE speakers ADD COLUMN role TEXT")
        _LOGGER.info("Migration: speakers.role added")

    sample_cols = _table_columns(conn, "speaker_samples")
    if "source" not in sample_cols:
        conn.execute("ALTER TABLE speaker_samples ADD COLUMN source TEXT")
        _LOGGER.info("Migration: speaker_samples.source added")
    if "filename" not in sample_cols:
        conn.execute("ALTER TABLE speaker_samples ADD COLUMN filename TEXT")
        _LOGGER.info("Migration: speaker_samples.filename added")
    if "quality_score" not in sample_cols:
        conn.execute("ALTER TABLE speaker_samples ADD COLUMN quality_score REAL")
        _LOGGER.info("Migration: speaker_samples.quality_score added")
    if "satellite_id" not in sample_cols:
        conn.execute("ALTER TABLE speaker_samples ADD COLUMN satellite_id TEXT")
        _LOGGER.info("Migration: speaker_samples.satellite_id added")

    event_cols = _table_columns(conn, "recognition_events")
    if "emotion" not in event_cols:
        conn.execute("ALTER TABLE recognition_events ADD COLUMN emotion TEXT")
        _LOGGER.info("Migration: recognition_events.emotion added")
    if "emotion_confidence" not in event_cols:
        conn.execute("ALTER TABLE recognition_events ADD COLUMN emotion_confidence REAL")
        _LOGGER.info("Migration: recognition_events.emotion_confidence added")
    if "shadow_transcript" not in event_cols:
        conn.execute("ALTER TABLE recognition_events ADD COLUMN shadow_transcript TEXT")
        conn.execute("ALTER TABLE recognition_events ADD COLUMN shadow_engine TEXT")
        _LOGGER.info("Migration: recognition_events shadow columns added")
    if "weight" not in event_cols:
        conn.execute("ALTER TABLE recognition_events ADD COLUMN weight REAL")
        conn.execute("ALTER TABLE recognition_events ADD COLUMN margin REAL")
        _LOGGER.info("Migration: recognition_events weight/margin added")
    if "transcript_ms" not in event_cols:
        conn.execute("ALTER TABLE recognition_events ADD COLUMN transcript_ms REAL")
        conn.execute("ALTER TABLE recognition_events ADD COLUMN shadow_ms REAL")
        _LOGGER.info("Migration: recognition_events STT timings added")
    if "whisper" not in event_cols:
        conn.execute("ALTER TABLE recognition_events ADD COLUMN whisper INTEGER")
        _LOGGER.info("Migration: recognition_events.whisper added")

    conn.commit()


def open_db(db_path: Path) -> sqlite3.Connection:
    """Open a SQLite connection with sqlite-vec loaded and the schema applied."""
    db_path = Path(db_path)
    db_path.parent.mkdir(parents=True, exist_ok=True)

    conn = sqlite3.connect(str(db_path), check_same_thread=False)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA foreign_keys = ON;")
    conn.execute("PRAGMA journal_mode = WAL;")

    conn.enable_load_extension(True)
    sqlite_vec.load(conn)
    conn.enable_load_extension(False)

    conn.executescript(_SCHEMA)
    conn.executescript(_VEC_SCHEMA)
    _migrate(conn)
    conn.commit()

    _LOGGER.info("Opened murdock database at %s", db_path)
    return conn


def get_setting(conn: sqlite3.Connection, key: str, default: Optional[str] = None) -> Optional[str]:
    row = conn.execute("SELECT value FROM settings WHERE key = ?", (key,)).fetchone()
    return row["value"] if row else default


def set_setting(conn: sqlite3.Connection, key: str, value: str) -> None:
    conn.execute(
        "INSERT INTO settings(key, value) VALUES(?, ?) "
        "ON CONFLICT(key) DO UPDATE SET value = excluded.value",
        (key, value),
    )
    conn.commit()
