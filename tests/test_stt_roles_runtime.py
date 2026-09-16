"""Main, fallback chain and shadows at runtime."""

from __future__ import annotations

import asyncio
from types import SimpleNamespace

from murdock.core.stt_backend import STTBackendError, STTNetworkError
from murdock.core.stt_services import KIND_OPENAI, KIND_WYOMING, SttService

import wyoming_murdock.handler as handler_mod
from wyoming_murdock.handler import MurdockHandler


class _Backend:
    """Scripted behaviour: a string answers, an exception is raised."""

    def __init__(self, name, outcome, log=None, delay=0.0, tracker=None):
        self.label = name
        self._outcome = outcome
        self._log = log if log is not None else []
        self._delay = delay
        self._tracker = tracker

    async def transcribe(self, audio, **kw):
        if self._tracker is not None:
            self._tracker["now"] += 1
            self._tracker["peak"] = max(self._tracker["peak"], self._tracker["now"])
        try:
            self._log.append(self.label)
            if self._delay:
                await asyncio.sleep(self._delay)
            if isinstance(self._outcome, Exception):
                raise self._outcome
            return self._outcome
        finally:
            if self._tracker is not None:
                self._tracker["now"] -= 1


def _service(sid, name, remote=False, kind=KIND_OPENAI):
    svc = SttService(sid, name, kind, {})
    svc._remote = remote
    return svc


def _chain_handler(services, outcomes, *, remote_down=False, log=None):
    log = log if log is not None else []
    backends = {
        s.id: (None if outcomes[s.id] is None else _Backend(s.name, outcomes[s.id], log))
        for s in services
    }
    fake = SimpleNamespace(
        _session_id="s1", _remote_down=remote_down, _rescued_by=None,
        _language="de",
        context=SimpleNamespace(
            get_fallback_services=lambda: services,
            is_service_remote=lambda s: s._remote,
            build_stt_backend=lambda s: backends[s.id],
            service_timeout=lambda s: 5.0,
        ),
    )

    async def _prepared(audio):
        return audio

    fake._prepared_upload = _prepared
    return fake, log


def _chain(fake, reason):
    return asyncio.run(
        MurdockHandler._run_fallback_chain(fake, b"\x00" * 3200, reason=reason)
    )


# --- fallback chain -------------------------------------------------------------


def test_the_first_fallback_that_answers_wins():
    a, b = _service(1, "Voxtral"), _service(2, "Kroko")
    fake, log = _chain_handler([a, b], {1: "Licht an", 2: "never"})
    assert _chain(fake, "error") == "Licht an"
    assert fake._rescued_by == "Voxtral"
    assert log == ["Voxtral"]


def test_a_failing_fallback_hands_on_to_the_next():
    a, b = _service(1, "Voxtral"), _service(2, "Kroko")
    fake, log = _chain_handler([a, b], {1: STTBackendError("HTTP 500"), 2: "Licht an"})
    assert _chain(fake, "error") == "Licht an"
    assert fake._rescued_by == "Kroko"


def test_no_internet_skips_the_other_remote_services():
    """The local service at the end is the offline fallback — reached
    without every dead cloud service burning its own timeout first."""
    mai = _service(1, "MAI", remote=True)
    vox = _service(2, "Voxtral", remote=True)
    kroko = _service(3, "Kroko", remote=False, kind=KIND_WYOMING)
    fake, log = _chain_handler(
        [mai, vox, kroko],
        {1: STTNetworkError("unreachable"), 2: "never reached", 3: "Licht an"},
    )
    assert _chain(fake, "error") == "Licht an"
    assert log == ["MAI", "Kroko"]
    assert fake._remote_down is True


def test_a_main_that_could_not_connect_skips_remote_fallbacks_from_the_start():
    vox = _service(1, "Voxtral", remote=True)
    kroko = _service(2, "Kroko", remote=False, kind=KIND_WYOMING)
    fake, log = _chain_handler([vox, kroko], {1: "never", 2: "Licht an"}, remote_down=True)
    assert _chain(fake, "error") == "Licht an"
    assert log == ["Kroko"]


def test_a_local_network_error_says_nothing_about_the_internet():
    """A LAN box being off must not make the chain give up on the cloud."""
    kroko = _service(1, "Kroko", remote=False, kind=KIND_WYOMING)
    vox = _service(2, "Voxtral", remote=True)
    fake, log = _chain_handler([kroko, vox], {1: STTNetworkError("refused"), 2: "Licht an"})
    assert _chain(fake, "error") == "Licht an"
    assert fake._remote_down is False


def test_after_an_error_a_working_fallback_that_hears_nothing_is_the_answer():
    """Nobody said anything; asking further would only add latency."""
    a, b = _service(1, "Voxtral"), _service(2, "Kroko")
    fake, log = _chain_handler([a, b], {1: "", 2: "never"})
    assert _chain(fake, "error") == ""
    assert log == ["Voxtral"]


def test_after_an_empty_main_an_empty_fallback_hands_on():
    """The main already heard nothing; a second empty is no new evidence."""
    a, b = _service(1, "Voxtral"), _service(2, "Kroko")
    fake, log = _chain_handler([a, b], {1: "", 2: "Licht an"})
    assert _chain(fake, "empty") == "Licht an"
    assert log == ["Voxtral", "Kroko"]


def test_an_incomplete_service_is_skipped():
    a, b = _service(1, "half-configured"), _service(2, "Kroko")
    fake, log = _chain_handler([a, b], {1: None, 2: "Licht an"})
    assert _chain(fake, "error") == "Licht an"


def test_nothing_answering_returns_empty():
    a = _service(1, "Voxtral")
    fake, _ = _chain_handler([a], {1: STTBackendError("down")})
    assert _chain(fake, "error") == ""
    assert fake._rescued_by is None


# --- when the chain runs ----------------------------------------------------------


def _assembly(*, main_text, main_failed, on_empty, chain_text="from fallback"):
    calls = []

    async def _primary(audio):
        fake._main_failed = main_failed
        return main_text

    async def _chain_fn(audio, *, reason):
        calls.append(reason)
        return chain_text

    fake = SimpleNamespace(
        _session_id="s1", _main_failed=False, _cancelled=False,
        context=SimpleNamespace(
            get_dictionary_entries=lambda: [],
            get_fallback_on_empty=lambda: on_empty,
            is_cancel_phrase=lambda t: False,
        ),
    )
    fake._primary_transcript = _primary
    fake._run_fallback_chain = _chain_fn
    out = asyncio.run(MurdockHandler._transcribe_and_correct(fake, b"\x00" * 3200))
    return out, calls


def test_a_failed_main_goes_to_the_chain():
    out, calls = _assembly(main_text="", main_failed=True, on_empty=False)
    assert calls == ["error"] and out == "from fallback"


def test_an_empty_main_goes_to_the_chain_when_enabled():
    out, calls = _assembly(main_text="", main_failed=False, on_empty=True)
    assert calls == ["empty"] and out == "from fallback"


def test_an_empty_main_stays_empty_when_disabled():
    out, calls = _assembly(main_text="", main_failed=False, on_empty=False)
    assert calls == [] and out == ""


def test_a_main_that_answered_never_touches_the_chain():
    out, calls = _assembly(main_text="Licht an", main_failed=False, on_empty=True)
    assert calls == [] and out == "Licht an"


# --- shadows ------------------------------------------------------------------------


def _shadow_setup(outcomes, delay=0.02):
    tracker = {"now": 0, "peak": 0}
    log, results = [], []
    services = [_service(i, f"S{i}", kind=KIND_WYOMING) for i in outcomes]
    backends = {s.id: _Backend(s.name, outcomes[s.id], log, delay, tracker) for s in services}
    ctx = SimpleNamespace(
        build_stt_backend=lambda s: backends[s.id],
        service_timeout=lambda s: 5.0,
        get_enable_stt_prep=lambda: False,
        recognition=SimpleNamespace(
            add_shadow_result=lambda event_id, **kw: results.append((event_id, kw)),
        ),
    )
    # Deliberately no per-session attributes: a shadow must not read them.
    fake = SimpleNamespace(context=ctx)
    return fake, services, log, results, tracker


def test_shadows_run_one_at_a_time_and_every_result_is_logged():
    fake, services, log, results, tracker = _shadow_setup(
        {1: "Licht an", 2: STTBackendError("HTTP 401"), 3: "Licht aus"}
    )
    asyncio.run(MurdockHandler._run_shadows(
        fake, 42, b"\x00" * 3200, None, "de", "s1", services,
    ))
    assert log == ["S1", "S2", "S3"]
    assert tracker["peak"] == 1, "two shadows ran at the same time"
    assert [(e, kw["engine"], kw["transcript"], kw["error"]) for e, kw in results] == [
        (42, "S1", "Licht an", None),
        (42, "S2", "", "HTTP 401"),
        (42, "S3", "Licht aus", None),
    ]
    assert all(kw["ms"] is not None for _, kw in results)


def test_shadows_wait_while_a_main_request_is_in_flight():
    """Nothing starts until the answer somebody is waiting for is out."""
    fake, services, log, results, _ = _shadow_setup({1: "Licht an"}, delay=0)

    async def scenario():
        handler_mod._SHADOW_GATE.acquire_main()
        task = asyncio.create_task(MurdockHandler._run_shadows(
            fake, 7, b"\x00" * 3200, None, "de", "s1", services,
        ))
        await asyncio.sleep(0.05)
        started_while_busy = list(log)
        handler_mod._SHADOW_GATE.release_main()
        await asyncio.wait_for(task, timeout=2)
        return started_while_busy

    assert asyncio.run(scenario()) == []
    assert log == ["S1"]


def test_a_shadows_time_excludes_its_wait_at_the_gate():
    """The number is the engine's, not the queue's."""
    fake, services, log, results, _ = _shadow_setup({1: "Licht an"}, delay=0)

    async def scenario():
        handler_mod._SHADOW_GATE.acquire_main()
        task = asyncio.create_task(MurdockHandler._run_shadows(
            fake, 7, b"\x00" * 3200, None, "de", "s1", services,
        ))
        await asyncio.sleep(0.3)
        handler_mod._SHADOW_GATE.release_main()
        await asyncio.wait_for(task, timeout=2)

    asyncio.run(scenario())
    [(_, kw)] = results
    assert kw["ms"] < 250
