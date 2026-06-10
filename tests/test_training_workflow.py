"""Tests for the voice-pipeline training helpers.

Covers empty-speaker creation and the recognition-event → captured-sample
linkage that powers "assign to speaker" on blocked log entries.
"""

from __future__ import annotations

import struct
import time

import numpy as np
import pytest

from murdock.core.db import open_db
from murdock.core.speaker_store import SpeakerStore
from murdock.core.unknown_store import UnknownStore


def _store(tmp_path) -> SpeakerStore:
    conn = open_db(tmp_path / "murdock.db")
    # embedder/vad aren't touched by create_speaker, so None is fine.
    return SpeakerStore(conn=conn, embedder=None, vad=None)


def test_create_empty_speaker(tmp_path):
    store = _store(tmp_path)
    spk = store.create_speaker("jonas")
    assert spk.id > 0
    assert spk.name == "jonas"
    assert spk.enrollment_count == 0
    # Shows up in the list with no samples.
    assert [s.name for s in store.list_speakers()] == ["jonas"]
    assert store.list_samples(spk.id) == []


def test_create_speaker_with_role(tmp_path):
    store = _store(tmp_path)
    spk = store.create_speaker("anna", role="Familie")
    assert spk.role == "Familie"


def test_create_speaker_rejects_blank(tmp_path):
    store = _store(tmp_path)
    with pytest.raises(ValueError):
        store.create_speaker("   ")


def test_create_speaker_rejects_duplicate(tmp_path):
    store = _store(tmp_path)
    store.create_speaker("jonas")
    with pytest.raises(ValueError):
        store.create_speaker("jonas")


def test_create_speaker_rejects_bad_role(tmp_path):
    store = _store(tmp_path)
    with pytest.raises(ValueError):
        store.create_speaker("jonas", role="Overlord")


# --- Unknown session linkage ---------------------------------------------


def _insert_unknown(conn, session_id, tag=None):
    """Insert an unknown_samples row directly (no audio/embedding needed)."""
    emb = struct.pack("<192f", *([0.0] * 192))
    cur = conn.execute(
        "INSERT INTO unknown_samples(session_id, satellite_id, audio, embedding, "
        "duration_sec, best_distance, best_speaker, liveness_score, tag, created_at) "
        "VALUES(?, NULL, ?, ?, 2.0, 2.0, NULL, NULL, ?, ?)",
        (session_id, b"", emb, tag, time.time()),
    )
    conn.commit()
    return int(cur.lastrowid)


def test_map_sessions_to_samples(tmp_path):
    conn = open_db(tmp_path / "murdock.db")
    store = UnknownStore(conn)
    s1 = _insert_unknown(conn, "sess-a")
    _insert_unknown(conn, "sess-b", tag="assigned:anna")  # tagged → excluded
    mapping = store.map_sessions_to_samples(["sess-a", "sess-b", "sess-c"])
    assert mapping == {"sess-a": s1}


def test_map_sessions_returns_latest_untagged(tmp_path):
    conn = open_db(tmp_path / "murdock.db")
    store = UnknownStore(conn)
    _insert_unknown(conn, "sess-a")
    s2 = _insert_unknown(conn, "sess-a")  # newer for same session
    mapping = store.map_sessions_to_samples(["sess-a"])
    assert mapping == {"sess-a": s2}  # MAX(id) wins


def test_map_sessions_empty(tmp_path):
    conn = open_db(tmp_path / "murdock.db")
    store = UnknownStore(conn)
    assert store.map_sessions_to_samples([]) == {}
    assert store.map_sessions_to_samples(["nope"]) == {}
