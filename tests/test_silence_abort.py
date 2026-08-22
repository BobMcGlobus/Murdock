"""Ending a turn nobody started."""

from __future__ import annotations

import asyncio
from types import SimpleNamespace

from wyoming_murdock.handler import MurdockHandler


def _handler(speech_seconds, *, floor=0.2):
    class _VAD:
        def analyze_pcm(self, pcm):
            return SimpleNamespace(speech_seconds=speech_seconds)

    closed = []

    fake = SimpleNamespace(
        _session_id="s1",
        _no_speech=False,
        context=SimpleNamespace(
            vad=_VAD(),
            get_silence_abort_floor=lambda: floor,
        ),
    )

    async def _close(*, send_stop):
        closed.append(send_stop)

    fake._close_upstream = _close
    return fake, closed


def _probe(fake, seconds=3.0):
    audio = b"\x00" * int(seconds * 16000 * 2)
    asyncio.run(MurdockHandler._probe_for_silence(fake, audio))


def test_a_turn_with_no_speech_is_dropped():
    fake, closed = _handler(0.0)
    _probe(fake)
    assert fake._no_speech is True
    # The upstream is dropped, so no transcript is even requested.
    assert closed == [False]


def test_drawing_breath_never_costs_the_turn():
    """The floor is the deciding condition; the timer only says when to look."""
    fake, _ = _handler(0.5)
    _probe(fake)
    assert fake._no_speech is False


def test_the_floor_is_configurable():
    fake, _ = _handler(0.5, floor=1.0)
    _probe(fake)
    assert fake._no_speech is True


def test_a_broken_vad_never_drops_a_turn():
    """Failing open matters more here than anywhere: silence is cheap."""

    class _Boom:
        def analyze_pcm(self, pcm):
            raise RuntimeError("onnx exploded")

    fake = SimpleNamespace(
        _session_id="s1",
        _no_speech=False,
        context=SimpleNamespace(vad=_Boom(), get_silence_abort_floor=lambda: 0.2),
    )
    fake._close_upstream = None  # would explode if reached
    _probe(fake)
    assert fake._no_speech is False


def test_the_setting_clamps_and_defaults(tmp_path):
    from murdock.config import Settings
    from murdock.core.context import AppContext
    from murdock.core.db import open_db

    ctx = AppContext(
        settings=Settings(), db=open_db(tmp_path / "m.db"), embedder=None,
        vad=None, speakers=None, unknown=None, ha=None, mqtt=None,
        recognition=None,
    )
    assert ctx.get_silence_abort_sec() == 3.0
    ctx.set_silence_abort_sec(2.5)
    assert ctx.get_silence_abort_sec() == 2.5
    ctx.set_silence_abort_sec(0)          # off
    assert ctx.get_silence_abort_sec() == 0.0
    ctx.set_silence_abort_sec(9999)
    assert ctx.get_silence_abort_sec() == 30.0
