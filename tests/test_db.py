"""Tests for the SQLite schema, settings store, and migrations.

These guard the silent failure modes: a missing column after an upgrade,
a settings round-trip that doesn't persist, or sqlite-vec not loading.
"""

from __future__ import annotations

import sqlite3

import pytest

from murdock.core.db import (
    _migrate,
    _table_columns,
    get_setting,
    open_db,
    set_setting,
)


def test_open_db_creates_all_tables(tmp_path):
    conn = open_db(tmp_path / "murdock.db")
    tables = {
        row["name"]
        for row in conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table'"
        ).fetchall()
    }
    assert {"speakers", "speaker_samples", "unknown_samples",
            "settings", "recognition_events"} <= tables
    conn.close()


def test_open_db_loads_vec_table(tmp_path):
    """The vec0 virtual table must exist (proves sqlite-vec loaded)."""
    conn = open_db(tmp_path / "murdock.db")
    # Inserting a 192-dim vector should not raise.
    vec = "[" + ",".join("0.1" for _ in range(192)) + "]"
    conn.execute(
        "INSERT INTO speaker_embeddings(speaker_id, embedding) VALUES (?, ?)",
        (1, vec),
    )
    conn.commit()
    row = conn.execute(
        "SELECT speaker_id FROM speaker_embeddings WHERE speaker_id = 1"
    ).fetchone()
    assert row["speaker_id"] == 1
    conn.close()


def test_open_db_is_idempotent(tmp_path):
    """Opening an existing DB again must not error or wipe data."""
    path = tmp_path / "murdock.db"
    conn = open_db(path)
    set_setting(conn, "verify_threshold", "0.42")
    conn.close()

    conn2 = open_db(path)
    assert get_setting(conn2, "verify_threshold") == "0.42"
    conn2.close()


def test_settings_round_trip(tmp_path):
    conn = open_db(tmp_path / "murdock.db")
    assert get_setting(conn, "missing") is None
    assert get_setting(conn, "missing", "fallback") == "fallback"

    set_setting(conn, "mqtt_host", "192.168.1.5")
    assert get_setting(conn, "mqtt_host") == "192.168.1.5"

    # Upsert overwrites.
    set_setting(conn, "mqtt_host", "core-mosquitto")
    assert get_setting(conn, "mqtt_host") == "core-mosquitto"
    conn.close()


def _make_legacy_db(path) -> sqlite3.Connection:
    """Create a DB with the oldest known schema (pre-migration columns)."""
    conn = sqlite3.connect(str(path))
    conn.row_factory = sqlite3.Row
    conn.executescript(
        """
        CREATE TABLE speakers (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            name TEXT NOT NULL UNIQUE,
            ha_user_id TEXT,
            enrollment_count INTEGER NOT NULL DEFAULT 0,
            created_at REAL NOT NULL,
            updated_at REAL NOT NULL
        );
        CREATE TABLE speaker_samples (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            speaker_id INTEGER NOT NULL,
            audio BLOB NOT NULL,
            duration_sec REAL NOT NULL,
            created_at REAL NOT NULL
        );
        CREATE TABLE recognition_events (
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
            transcript TEXT
        );
        """
    )
    conn.commit()
    return conn


def test_migration_adds_missing_columns(tmp_path):
    """_migrate must add every column introduced after the first MVP."""
    conn = _make_legacy_db(tmp_path / "legacy.db")
    _migrate(conn)

    assert "role" in _table_columns(conn, "speakers")
    sample_cols = _table_columns(conn, "speaker_samples")
    assert {"source", "filename", "quality_score", "satellite_id"} <= sample_cols
    event_cols = _table_columns(conn, "recognition_events")
    assert {"emotion", "emotion_confidence"} <= event_cols
    conn.close()


def test_migration_is_idempotent(tmp_path):
    """Running _migrate twice must not raise (no duplicate-column errors)."""
    conn = _make_legacy_db(tmp_path / "legacy.db")
    _migrate(conn)
    _migrate(conn)  # second pass should be a no-op
    assert "role" in _table_columns(conn, "speakers")
    conn.close()


def test_migration_preserves_existing_rows(tmp_path):
    conn = _make_legacy_db(tmp_path / "legacy.db")
    conn.execute(
        "INSERT INTO speakers(name, enrollment_count, created_at, updated_at) "
        "VALUES ('jonas', 3, 1.0, 2.0)"
    )
    conn.commit()
    _migrate(conn)
    row = conn.execute("SELECT name, role FROM speakers WHERE name='jonas'").fetchone()
    assert row["name"] == "jonas"
    assert row["role"] is None  # newly added column defaults to NULL
    conn.close()
