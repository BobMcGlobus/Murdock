"""The per-chunk path must not touch the database.

Home Assistant hands Murdock audio in ~10 ms chunks, a hundred a second.
0.11 read the main service from SQLite on every one of them, which on a
small VM was enough to fall below real time: the satellite dropped the
audio it could not hand over and transcripts came back with holes and
missing endings ("Schalte alle Lich-").
"""

from __future__ import annotations

import asyncio
from types import SimpleNamespace

from wyoming.audio import AudioChunk, AudioStart

from murdock.config import Settings
from murdock.core.context import AppContext
from murdock.core.db import open_db
from wyoming_murdock.handler import MurdockHandler


def _ctx(tmp_path, kind="voxtral", config=None):
    ctx = AppContext(
        settings=Settings(), db=open_db(tmp_path / "m.db"), embedder=None,
        vad=None, speakers=SimpleNamespace(list_speakers=lambda: []),
        unknown=None, ha=None, mqtt=None, recognition=None,
    )
    ctx.stt_services.create("Main", kind, config or {"api_key": "k"})
    return ctx


def _handler(ctx):
    return MurdockHandler(
        SimpleNamespace(upstream_uri="tcp://unused:1"), ctx,
        SimpleNamespace(), SimpleNamespace(),
    )


def test_a_hundred_chunks_a_second_cause_no_queries(tmp_path):
    ctx = _ctx(tmp_path)
    handler = _handler(ctx)
    queries = []

    async def scenario():
        await handler.handle_event(AudioStart(rate=16000, width=2, channels=1).event())
        ctx.db.set_trace_callback(queries.append)
        for _ in range(300):  # three seconds of 10 ms chunks
            await handler.handle_event(
                AudioChunk(rate=16000, width=2, channels=1, audio=b"\x00" * 320).event()
            )
        ctx.db.set_trace_callback(None)

    asyncio.run(scenario())
    assert queries == [], f"{len(queries)} queries on the audio path, e.g. {queries[:3]}"
    assert len(handler._audio_buffer) == 300 * 320


def test_the_mode_is_fixed_for_the_whole_session(tmp_path):
    """Switching the main mid-utterance must not flip one half of a session."""
    ctx = _ctx(tmp_path)
    handler = _handler(ctx)

    async def scenario():
        await handler.handle_event(AudioStart(rate=16000, width=2, channels=1).event())
        assert handler._is_voxtral is True
        wyoming = ctx.stt_services.create("Kroko", "wyoming", {"uri": "kroko:10300"})
        roles = ctx.stt_services.roles()
        roles.main = wyoming.id
        ctx.stt_services.set_roles(roles)
        assert ctx.get_stt_backend() == "upstream"
        assert handler._is_voxtral is True
        # The next session picks the change up.
        await handler.handle_event(AudioStart(rate=16000, width=2, channels=1).event())
        assert handler._is_voxtral is False

    asyncio.run(scenario())


def test_the_store_reads_from_memory_and_sees_its_own_writes(tmp_path):
    ctx = _ctx(tmp_path)
    store = ctx.stt_services
    store.list()
    store.roles()
    queries = []
    ctx.db.set_trace_callback(queries.append)
    for _ in range(50):
        ctx.get_main_service()
        ctx.get_stt_backend()
    ctx.db.set_trace_callback(None)
    assert queries == []

    [svc] = store.list()
    store.update(svc.id, name="Renamed")
    assert ctx.get_main_service().name == "Renamed"
    # What a caller changes on its copy never reaches the cache.
    got = store.get(svc.id)
    got.config["api_key"] = "tampered"
    assert store.get(svc.id).config["api_key"] == "k"
