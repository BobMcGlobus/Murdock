"""Never waiting long on a slow main.

The main used to get its full timeout (8 s) before the fallback chain
even started, so a slow cloud request cost timeout plus fallback — long
enough for Home Assistant to give up on the turn. Now the first fallback
stands by: a local one is asked straight away, a remote one once the
patience has run out, and a slow main is overtaken.
"""

from __future__ import annotations

import asyncio
import time
from types import SimpleNamespace

from murdock.config import Settings
from murdock.core.context import AppContext
from murdock.core.db import open_db
from murdock.core.recognition_log import RecognitionLog
from murdock.core.stt_services import SttRoles
from wyoming_murdock.handler import MurdockHandler


class _Backend:
    def __init__(self, text, delay, calls):
        self.text, self.delay, self.calls = text, delay, calls
        self.last_timing = {}

    async def transcribe(self, audio, **kw):
        self.calls.append(self)
        await asyncio.sleep(self.delay)
        if isinstance(self.text, Exception):
            raise self.text
        return self.text


def _setup(tmp_path, *, main=("Licht an", 0.01), fallback=("Licht an!", 0.01),
           fallback_kind="wyoming", patience=0.2, shadow_fallback=False):
    db = open_db(tmp_path / "m.db")
    ctx = AppContext(
        settings=Settings(), db=db, embedder=None, vad=None,
        speakers=SimpleNamespace(list_speakers=lambda: []), unknown=None,
        ha=None, mqtt=None, recognition=RecognitionLog(db),
    )
    store = ctx.stt_services
    m = store.create("MAI", "openai", {"model": "m", "base_url": "https://openrouter.ai"})
    cfg = {"uri": "192.168.2.25:10303"} if fallback_kind == "wyoming" else {"api_key": "k"}
    f = store.create("Kroko" if fallback_kind == "wyoming" else "Voxtral", fallback_kind, cfg)
    store.set_roles(SttRoles(
        main=m.id, fallbacks=[f.id], shadows=[f.id] if shadow_fallback else [],
    ))
    ctx.set_stt_patience(patience)
    ctx.set_enable_stt_prep(False)
    calls = []
    backends = {
        m.id: _Backend(main[0], main[1], calls),
        f.id: _Backend(fallback[0], fallback[1], calls),
    }
    ctx.build_stt_backend = lambda svc: backends.get(svc.id) if svc else None
    handler = MurdockHandler(
        SimpleNamespace(upstream_uri="tcp://unused:1"), ctx,
        SimpleNamespace(), SimpleNamespace(),
    )
    return ctx, handler, backends[m.id], backends[f.id], calls


def _answer(handler):
    async def go():
        handler._reset_session_state()
        started = time.monotonic()
        text = await handler._main_with_patience(b"\x00" * 3200)
        return text, time.monotonic() - started
    return asyncio.run(go())


def test_a_fast_main_answers_as_before(tmp_path):
    ctx, handler, main, fallback, calls = _setup(tmp_path)
    text, took = _answer(handler)
    assert text == "Licht an"
    assert handler._rescued_by is None
    assert handler._main_result["main_engine"] == "MAI"


def test_a_slow_main_is_overtaken_by_the_local_fallback(tmp_path):
    ctx, handler, main, fallback, calls = _setup(
        tmp_path, main=("Licht an", 3.0), fallback=("Licht an!", 0.05), patience=0.2,
    )
    text, took = _answer(handler)
    assert text == "Licht an!"
    assert handler._rescued_by == "Kroko"
    assert took < 1.0, f"waited {took:.2f}s on a slow main"
    assert handler._late_main is not None
    assert handler._main_result["overtaken"] is True


def test_a_local_fallback_is_asked_at_once_a_remote_one_only_when_needed(tmp_path):
    ctx, handler, main, fallback, calls = _setup(tmp_path, fallback_kind="wyoming")
    _answer(handler)
    assert fallback in calls, "the free local stand-in was not started in parallel"

    ctx, handler, main, fallback, calls = _setup(tmp_path / "r", fallback_kind="voxtral")
    _answer(handler)
    assert fallback not in calls, "a paid remote fallback was asked although the main was fast"


def test_a_stand_in_that_heard_nothing_leaves_the_answer_to_the_main(tmp_path):
    ctx, handler, main, fallback, calls = _setup(
        tmp_path, main=("Licht an", 0.5), fallback=("", 0.01), patience=0.1,
    )
    text, took = _answer(handler)
    assert text == "Licht an"
    assert handler._rescued_by is None


def test_a_late_main_never_writes_into_the_next_session(tmp_path):
    """The overtaken request finishes after the session moved on."""
    from murdock.core.stt_backend import STTBackendError

    ctx, handler, main, fallback, calls = _setup(
        tmp_path, main=(STTBackendError("HTTP 500"), 0.4),
        fallback=("Licht an!", 0.02), patience=0.1,
    )

    async def go():
        handler._reset_session_state()
        await handler._main_with_patience(b"\x00" * 3200)
        late = handler._late_main
        handler._reset_session_state()          # next utterance starts
        await late                              # the old main fails now
        return handler._main_failed, handler._remote_down

    failed, down = asyncio.run(go())
    assert failed is False and down is False


def test_the_overtaken_main_is_logged_once_it_answers(tmp_path):
    """Otherwise its slow tail would vanish from its own statistics."""
    ctx, handler, main, fallback, calls = _setup(
        tmp_path, main=("Licht an", 0.3), fallback=("Licht an!", 0.02), patience=0.05,
    )
    log = ctx.recognition

    async def go():
        handler._reset_session_state()
        await handler._main_with_patience(b"\x00" * 3200)
        event_id = log.record(session_id="s", satellite_id=None,
                              duration_sec=1.0, outcome="match")
        handler._spawn_shadow(event_id, b"")
        await asyncio.sleep(0.6)
        return event_id

    event_id = asyncio.run(go())
    [row] = log.shadows_for([event_id])[event_id]
    assert row["engine"] == "MAI" and row["transcript"] == "Licht an"
    assert row["ms"] >= 250


def test_a_fallback_that_is_also_a_shadow_is_not_asked_twice(tmp_path):
    ctx, handler, main, fallback, calls = _setup(tmp_path, shadow_fallback=True)
    log = ctx.recognition

    async def go():
        handler._reset_session_state()
        await handler._main_with_patience(b"\x00" * 3200)
        event_id = log.record(session_id="s", satellite_id=None,
                              duration_sec=1.0, outcome="match")
        handler._spawn_shadow(event_id, b"\x00" * 3200)
        await asyncio.sleep(0.3)
        return event_id

    event_id = asyncio.run(go())
    assert calls.count(fallback) == 1
    [row] = log.shadows_for([event_id])[event_id]
    assert row["engine"] == "Kroko" and row["transcript"] == "Licht an!"


def test_patience_zero_restores_the_old_behaviour(tmp_path):
    ctx, handler, main, fallback, calls = _setup(tmp_path, patience=0.0)
    _answer(handler)
    assert fallback not in calls
