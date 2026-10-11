"""Telling a slow sender from a blocked event loop.

A stream that arrives at a third of real time has two possible causes
that look identical from the outside: Home Assistant (or the satellite's
link) delivering late, or Murdock not reading because its event loop is
busy. The AudioStop line and the log record now carry enough to decide.
"""

from __future__ import annotations

import asyncio
import logging
import time
from types import SimpleNamespace

from wyoming.audio import AudioChunk, AudioStart

from murdock.config import Settings
from murdock.core import loop_monitor
from murdock.core.context import AppContext
from murdock.core.db import open_db
from wyoming_murdock.handler import MurdockHandler


def _ctx(tmp_path):
    ctx = AppContext(
        settings=Settings(), db=open_db(tmp_path / "m.db"), embedder=None,
        vad=None, speakers=SimpleNamespace(list_speakers=lambda: []),
        unknown=None, ha=None, mqtt=None, recognition=None,
    )
    ctx.stt_services.create("Main", "voxtral", {"api_key": "k"})
    return ctx


def _handler(ctx):
    return MurdockHandler(
        SimpleNamespace(upstream_uri="tcp://unused:1"), ctx,
        SimpleNamespace(), SimpleNamespace(),
    )


def _chunk(ms=10):
    return AudioChunk(rate=16000, width=2, channels=1, audio=b"\x00" * (32 * ms)).event()


def test_a_blocking_callback_is_named_and_measured(caplog):
    async def scenario():
        loop_monitor.start()
        await asyncio.sleep(0.12)
        since = time.monotonic()

        async def culprit():
            time.sleep(0.3)  # the kind of call that must not run on the loop

        await asyncio.create_task(culprit())
        await asyncio.sleep(0.12)
        return loop_monitor.max_lag_since(since)

    with caplog.at_level(logging.WARNING, logger="murdock.loop"):
        lag = asyncio.run(scenario())
    assert lag >= 0.2
    assert any("culprit" in r.getMessage() for r in caplog.records), \
        [r.getMessage() for r in caplog.records]


def test_a_slow_sender_is_recorded_as_such(tmp_path, caplog):
    """Murdock reads every chunk at once; the gaps are the sender's."""
    handler = _handler(_ctx(tmp_path))

    async def scenario():
        loop_monitor.start()
        await handler.handle_event(AudioStart(rate=16000, width=2, channels=1).event())
        for _ in range(10):  # 100 ms of audio over ~1.6 s
            await handler.handle_event(_chunk())
            await asyncio.sleep(0.16)
        handler._responded = True  # stop before any transcription
        # Same bookkeeping as the AudioStop path, without a transcription.
        return handler._delivery_ratio()

    ratio = asyncio.run(scenario())
    assert ratio < 0.2
    assert handler._chunk_count == 10
    assert handler._max_gap >= 0.14
    assert handler._max_handle < 0.05, "reading a chunk must be instant"


def test_the_silence_abort_waits_out_a_degraded_stream(tmp_path):
    """No irreversible early decision on audio arriving this far behind."""
    ctx = _ctx(tmp_path)
    probed = []

    class _VAD:
        def analyze_pcm(self, pcm):
            probed.append(len(pcm))
            return SimpleNamespace(speech_seconds=0.0)

    ctx.vad = _VAD()
    handler = _handler(ctx)

    async def scenario():
        await handler.handle_event(AudioStart(rate=16000, width=2, channels=1).event())
        handler._abort_after_sec = 0.1
        # Pretend the stream has been open for two seconds already.
        handler._stream_start -= 2.0
        for _ in range(20):  # 0.2 s of audio "in" 2 s: 10 % delivery
            await handler.handle_event(_chunk())

    asyncio.run(scenario())
    assert probed == [], "dropped a turn on a stream that was arriving late"
    assert handler._no_speech is False


def test_commits_no_longer_wait_for_the_disk(tmp_path):
    conn = open_db(tmp_path / "s.db")
    assert conn.execute("PRAGMA journal_mode").fetchone()[0] == "wal"
    assert conn.execute("PRAGMA synchronous").fetchone()[0] == 1  # NORMAL


def test_a_client_that_hung_up_is_a_warning_not_a_stray_traceback(tmp_path, caplog):
    handler = _handler(_ctx(tmp_path))

    async def _boom(self):
        raise BrokenPipeError(32, "Broken pipe")

    handler._stop_at = time.monotonic() - 5.1
    original = MurdockHandler.__mro__[1].run
    try:
        MurdockHandler.__mro__[1].run = _boom
        with caplog.at_level(logging.WARNING):
            asyncio.run(handler.run())  # must not raise
    finally:
        MurdockHandler.__mro__[1].run = original
    msg = " ".join(r.getMessage() for r in caplog.records)
    assert "closed the connection" in msg and "after AudioStop" in msg
