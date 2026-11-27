"""The wake word proxy: relay, wake word voiceprints and the rescue."""

from __future__ import annotations

import asyncio
import json
from types import SimpleNamespace

import numpy as np
import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from murdock.api.routes_wake import router as wake_router
from wyoming_murdock.wake_proxy import (
    MODE_RESCUE,
    SpeakerTools,
    SpeakerView,
    Upstream,
    WakePrints,
    WakeProxy,
    would_decide,
)

_BYTES_PER_MS = 32
JONAS, SARAH = 3, 4


class _FakeApp(Upstream):
    """The Voice Support app's side of one run, scripted.

    With a wake word it answers once it has the wake word audio; without
    one (a button press, or Murdock's endpointing run) it only endpoints.
    """

    def __init__(self, verdict: str = "accepted", reason: str | None = None):
        self.verdict = verdict
        self.reason = reason
        self.received_text: list[dict] = []
        self.received_audio = bytearray()
        self.closed = False
        self._out: asyncio.Queue = asyncio.Queue()
        self._wake_ms: int | None = None
        self._answered = False

    async def send(self, message):
        if isinstance(message, str):
            event = json.loads(message)
            self.received_text.append(event)
            if event["type"] == "run-start":
                self._wake_ms = event.get("wake_word_audio_ms") or None
            elif event["type"] == "audio-end":
                if self.verdict == "accepted" or not self._wake_ms:
                    await self._out.put(json.dumps({"type": "stt-vad-end", "timestamp": 2500}))
                await self._out.put(None)
            return
        self.received_audio += message
        if (
            not self._answered and self._wake_ms
            and len(self.received_audio) >= self._wake_ms * _BYTES_PER_MS
        ):
            self._answered = True
            if self.verdict == "accepted":
                await self._out.put(json.dumps({
                    "type": "wake-word-end",
                    "wake_word_output": {"wake_word_phrase": "okay nabu",
                                         "wake_word_audio_ms": self._wake_ms},
                }))
                await self._out.put(json.dumps({"type": "stt-vad-start", "timestamp": 1600}))
            else:
                await self._out.put(json.dumps({
                    "type": "wake-word-rejected",
                    "wake_word_output": {"wake_word_phrase": "okay nabu",
                                         "reason": self.reason},
                }))
                await self._out.put(None)

    async def __aiter__(self):
        while True:
            message = await self._out.get()
            if message is None:
                return
            yield message

    async def close(self):
        self.closed = True


class _Tools(SpeakerTools):
    def __init__(self, wake_vec=(1.0, 0.0, 0.0), command: SpeakerView | None = None):
        self.wake_vec = np.asarray(wake_vec, dtype=np.float32)
        self.command = command or SpeakerView(error="nobody enrolled")
        self.wake_clips: list[bytes] = []
        self.commands: list[bytes] = []

    async def embed_wake(self, clip):
        self.wake_clips.append(clip)
        return self.wake_vec

    async def verify_command(self, audio):
        self.commands.append(audio)
        return self.command

    def speakers(self):
        return {JONAS: "Jonas", SARAH: "Sarah"}


def _proxy(tmp_path, apps, tools=None, *, jonas_prints=0, mode=None):
    queue = list(apps)

    async def connect(url):
        if not queue or queue[0] is None:
            raise OSError("refused")
        return queue.pop(0)

    async def info(url):
        return {"name": "Voice Support", "version": "0.2.1", "api_version": 1,
                "wake_words": [{"id": "okay_nabu", "phrase": "okay nabu"}]}

    proxy = WakeProxy("http://voice:10900", data_dir=tmp_path, tools=tools,
                      connect=connect, fetch_info=info)
    for _ in range(jonas_prints):
        proxy.prints.add(JONAS, "Jonas", np.array([1.0, 0.0, 0.0]), source="enroll")
    if mode:
        proxy.prints.update_settings(mode=mode)
    return proxy


def _run_start(wake_ms: int | None = 1500) -> dict:
    start = {"type": "run-start", "satellite_id": "assist_satellite.wohnzimmer",
             "silence_seconds": 0.7}
    if wake_ms:
        start.update(wake_word_phrase="okay nabu", wake_word_audio_ms=wake_ms)
    return start


def _chunk(value: int, ms: int = 100) -> bytes:
    return bytes([value & 0xFF, 0]) * (ms * 16)


def _audio(n: int = 30) -> bytes:
    return b"".join(_chunk(i) for i in range(n))


def _drive(client, *, wake_ms=1500, chunks=30):
    """Act as Home Assistant: start the run, stream audio, read the answers.

    A rejection ends the run; after an acceptance Home Assistant streams to
    the end and sends audio-end, and the command's end comes back.
    """
    events = []
    with client.websocket_connect("/run") as ws:
        ws.send_text(json.dumps(_run_start(wake_ms)))
        for i in range(chunks):
            ws.send_bytes(_chunk(i))
        events.append(json.loads(ws.receive_text()))
        if events[-1]["type"] != "wake-word-rejected":
            if events[-1]["type"] == "wake-word-end" and len(events) == 1:
                ws.send_text(json.dumps({"type": "audio-end"}))
            while events[-1]["type"] != "stt-vad-end":
                events.append(json.loads(ws.receive_text()))
    return events


def _settle(client, proxy):
    client.portal.call(proxy.wait_idle)


# --- /info -------------------------------------------------------------------


def test_info_is_passed_through_under_murdocks_name(tmp_path):
    proxy = _proxy(tmp_path, [])
    with TestClient(proxy.build_app()) as client:
        info = client.get("/info").json()
    assert info["api_version"] == 1
    assert info["wake_words"] == [{"id": "okay_nabu", "phrase": "okay nabu"}]
    assert info["name"] == "Murdock → Voice Support"


def test_info_without_the_app_is_a_503(tmp_path):
    async def down(url):
        raise OSError("refused")

    proxy = WakeProxy("http://voice:10900", data_dir=tmp_path, fetch_info=down)
    with TestClient(proxy.build_app()) as client:
        assert client.get("/info").status_code == 503


def test_a_recent_info_stands_in_while_the_app_is_away(tmp_path, monkeypatch):
    """Reinstalling or restarting the app must not drop Home Assistant's
    entry; an old answer must not keep satellites sending forever."""
    import wyoming_murdock.wake_proxy as mod

    state = {"up": True}

    async def info(url):
        if not state["up"]:
            raise OSError("refused")
        return {"name": "Voice Support", "version": "0.2.1", "api_version": 1,
                "wake_words": [{"id": "okay_nabu"}]}

    proxy = WakeProxy("http://voice:10900", data_dir=tmp_path, fetch_info=info)
    with TestClient(proxy.build_app()) as client:
        assert client.get("/info").status_code == 200
        state["up"] = False
        assert client.get("/info").json()["wake_words"] == [{"id": "okay_nabu"}]
    again = WakeProxy("http://voice:10900", data_dir=tmp_path, fetch_info=info)
    with TestClient(again.build_app()) as client:
        assert client.get("/info").status_code == 200
        monkeypatch.setattr(mod, "_INFO_STALE_SECONDS", 0)
        assert client.get("/info").status_code == 503


# --- relay ---------------------------------------------------------------------


def test_an_accepted_run_is_relayed_unchanged(tmp_path):
    app = _FakeApp("accepted")
    tools = _Tools()
    proxy = _proxy(tmp_path, [app], tools)
    with TestClient(proxy.build_app()) as client:
        events = _drive(client)
        _settle(client, proxy)
    assert [e["type"] for e in events] == ["wake-word-end", "stt-vad-start", "stt-vad-end"]
    assert app.received_text[0] == _run_start()
    assert app.received_text[-1] == {"type": "audio-end"}
    assert bytes(app.received_audio) == _audio()
    assert app.closed
    # The voice check got the wake word: the 1.5 s before the detection.
    [clip] = tools.wake_clips
    assert len(clip) == 1500 * _BYTES_PER_MS
    [row] = proxy.records
    assert row["verdict"] == "accepted" and row["vad_end_ms"] == 2500


def test_without_the_app_the_run_is_closed_so_home_assistant_falls_back(tmp_path):
    tools = _Tools()
    proxy = _proxy(tmp_path, [None], tools)
    with TestClient(proxy.build_app()) as client:
        with client.websocket_connect("/run") as ws:
            with pytest.raises(Exception):
                ws.receive_text()
        _settle(client, proxy)
    [row] = proxy.records
    assert row["upstream_failed"] is True
    assert tools.wake_clips == []


def test_a_run_without_a_wake_word_is_relayed_but_not_checked(tmp_path):
    """A button press: the app only endpoints, there is nothing to verify."""
    tools = _Tools()
    proxy = _proxy(tmp_path, [_FakeApp("accepted")], tools)
    with TestClient(proxy.build_app()) as client:
        with client.websocket_connect("/run") as ws:
            ws.send_text(json.dumps(_run_start(wake_ms=None)))
            for i in range(5):
                ws.send_bytes(_chunk(i))
            ws.send_text(json.dumps({"type": "audio-end"}))
            assert json.loads(ws.receive_text())["type"] == "stt-vad-end"
        _settle(client, proxy)
    assert tools.wake_clips == []
    [row] = proxy.records
    assert row["wake_word"] is None and row["clip"] is None and row["would"] is None


# --- observe and rescue ------------------------------------------------------------


def test_observe_mode_passes_the_refusal_on_and_says_what_it_would_do(tmp_path):
    app = _FakeApp("rejected", reason="confusable: 'namu'")
    proxy = _proxy(tmp_path, [app], _Tools(), jonas_prints=5)
    with TestClient(proxy.build_app()) as client:
        events = _drive(client)
        _settle(client, proxy)
    assert events[-1]["type"] == "wake-word-rejected"
    [row] = proxy.records
    assert row["reason"] == "confusable: 'namu'"
    assert row["wake_speaker"] == "Jonas"
    assert row["would"] == "would rescue (Jonas)"
    assert row["clip"] and (tmp_path / "wake_runs" / row["clip"]).exists()


def test_rescue_answers_with_an_acceptance_and_a_second_run_endpoints(tmp_path):
    """The app refused Jonas's "Okay Nabu"; his wake voiceprint says it was
    him. Home Assistant hears an acceptance, and the command still ends where
    the app (in an endpoint-only run) says, on Home Assistant's clock."""
    first = _FakeApp("rejected", reason="confusable: 'now'")
    second = _FakeApp("accepted")
    proxy = _proxy(tmp_path, [first, second], _Tools(wake_vec=(1.0, 0.05, 0.0)),
                   jonas_prints=5, mode=MODE_RESCUE)
    with TestClient(proxy.build_app()) as client:
        events = _drive(client)
        _settle(client, proxy)
    assert events[0] == {
        "type": "wake-word-end",
        "wake_word_output": {"wake_word_phrase": "okay nabu", "wake_word_audio_ms": 1500},
    }
    assert events[-1] == {"type": "stt-vad-end", "timestamp": 2500 + 1500}
    # The second run is an ordinary run without a wake word, fed from the
    # end of the wake word on — every byte once, in order.
    assert second.received_text[0]["wake_word_phrase"] is None
    assert second.received_text[0]["wake_word_audio_ms"] == 0
    assert second.received_text[-1] == {"type": "audio-end"}
    assert bytes(second.received_audio) == _audio()[1500 * _BYTES_PER_MS:]
    [row] = proxy.records
    assert row["rescued"] is True and row["would"] == "rescued (Jonas)"
    assert row["vad_end_ms"] == 4000


def test_rescue_mode_leaves_a_stranger_refused(tmp_path):
    first = _FakeApp("rejected", reason="confusable: 'now'")
    proxy = _proxy(tmp_path, [first, _FakeApp()], _Tools(wake_vec=(0.0, 1.0, 0.0)),
                   jonas_prints=5, mode=MODE_RESCUE)
    with TestClient(proxy.build_app()) as client:
        events = _drive(client)
        _settle(client, proxy)
    assert events[-1]["type"] == "wake-word-rejected"
    [row] = proxy.records
    assert row["rescued"] is False and row["wake_speaker"] is None


def test_rescue_needs_enough_voiceprint_samples(tmp_path):
    first = _FakeApp("rejected", reason="confusable: 'now'")
    proxy = _proxy(tmp_path, [first, _FakeApp()], _Tools(), jonas_prints=2,
                   mode=MODE_RESCUE)
    with TestClient(proxy.build_app()) as client:
        events = _drive(client)
        _settle(client, proxy)
    assert events[-1]["type"] == "wake-word-rejected"


def test_without_a_run_for_endpointing_the_refusal_stands(tmp_path):
    """Accepting with nobody to end the command would let Home Assistant
    cut it off at once — refusing is the lesser harm."""
    first = _FakeApp("rejected", reason="confusable: 'now'")
    proxy = _proxy(tmp_path, [first, None], _Tools(), jonas_prints=5, mode=MODE_RESCUE)
    with TestClient(proxy.build_app()) as client:
        events = _drive(client)
        _settle(client, proxy)
    assert events[-1]["type"] == "wake-word-rejected"


# --- learning ------------------------------------------------------------------------


def test_an_accepted_wake_word_teaches_the_commands_speaker(tmp_path):
    tools = _Tools(command=SpeakerView(speaker="Jonas", speaker_id=JONAS, distance=0.12,
                                       threshold=0.35, is_match=True))
    proxy = _proxy(tmp_path, [_FakeApp("accepted")], tools)
    with TestClient(proxy.build_app()) as client:
        _drive(client)
        _settle(client, proxy)
    # The command, from the wake word's end to the command's end, was checked.
    [command] = tools.commands
    assert len(command) == (2500 - 1500) * _BYTES_PER_MS
    [row] = proxy.records
    assert row["learnt"] == "Jonas (auto, 1)"
    assert proxy.prints.speakers[str(JONAS)]["samples"][0]["source"] == "auto"


def test_a_borderline_command_teaches_nothing(tmp_path):
    tools = _Tools(command=SpeakerView(speaker="Jonas", speaker_id=JONAS, distance=0.33,
                                       threshold=0.35, is_match=True))
    proxy = _proxy(tmp_path, [_FakeApp("accepted")], tools)
    with TestClient(proxy.build_app()) as client:
        _drive(client)
        _settle(client, proxy)
    assert proxy.records[0]["learnt"] is None
    assert proxy.prints.speakers == {}


def test_an_enrolment_window_takes_even_refused_wake_words(tmp_path):
    proxy = _proxy(tmp_path, [_FakeApp("rejected", reason="confusable: 'namu'")], _Tools())
    proxy.prints.open_window(SARAH, "Sarah", 60)
    with TestClient(proxy.build_app()) as client:
        _drive(client)
        _settle(client, proxy)
    assert proxy.records[0]["learnt"] == "Sarah (enrolment, 1)"
    assert len(proxy.prints.speakers[str(SARAH)]["samples"]) == 1


# --- voiceprints ---------------------------------------------------------------------


def test_voiceprints_need_a_margin_to_the_next_person(tmp_path):
    prints = WakePrints(tmp_path / "prints.json")
    for _ in range(5):
        prints.add(JONAS, "Jonas", np.array([1.0, 0.0]), source="enroll")
        prints.add(SARAH, "Sarah", np.array([0.995, 0.0998]), source="enroll")
    view = prints.match(np.array([1.0, 0.03]))
    assert view.nearest == "Jonas" and not view.is_match   # Sarah is too close
    prints.update_settings(min_margin=0.0)
    assert prints.match(np.array([1.0, 0.03])).is_match
    # Settings and samples survive a restart.
    again = WakePrints(tmp_path / "prints.json")
    assert again.settings["min_margin"] == 0.0
    assert {r["name"]: r["samples"] for r in again.stats()} == {"Jonas": 5, "Sarah": 5}


def test_an_unknown_mode_is_refused(tmp_path):
    with pytest.raises(ValueError):
        WakePrints(None).update_settings(mode="always")


@pytest.mark.parametrize("verdict, view, expected", [
    ("rejected", SpeakerView(speaker="Jonas", distance=0.2, threshold=0.3, is_match=True),
     "would rescue (Jonas)"),
    ("accepted", SpeakerView(distance=0.6, threshold=0.3, is_match=False, nearest="Jonas"),
     "would veto (no wake voiceprint matches)"),
    ("accepted", SpeakerView(speaker="Jonas", distance=0.2, threshold=0.3, is_match=True),
     "agrees"),
    ("rejected", SpeakerView(distance=0.6, threshold=0.3, is_match=False), "agrees"),
    ("rejected", SpeakerView(error="no wake voiceprints yet"), "app stands"),
    ("accepted", None, "app stands"),
])
def test_what_a_speaker_aware_verdict_would_be(verdict, view, expected):
    assert would_decide(verdict, view) == expected


# --- web API ---------------------------------------------------------------------------


def _api(proxy):
    app = FastAPI()
    app.state.context = SimpleNamespace(wake_proxy=proxy)
    app.include_router(wake_router)
    return TestClient(app)


def test_the_api_opens_an_enrolment_window_by_name(tmp_path):
    proxy = _proxy(tmp_path, [], _Tools())
    with _api(proxy) as client:
        opened = client.post("/api/wake/enroll", json={"speaker": "sarah", "seconds": 120})
        assert opened.status_code == 200 and opened.json()["speaker"] == "Sarah"
        assert client.post("/api/wake/enroll", json={"speaker": "Nobody"}).status_code == 404
        assert client.get("/api/wake/status").json()["enrolling"]["speaker"] == "Sarah"
        assert client.delete("/api/wake/enroll").json() == {"enrolling": None}


def test_the_api_switches_the_mode(tmp_path):
    proxy = _proxy(tmp_path, [], _Tools())
    with _api(proxy) as client:
        assert client.patch("/api/wake/settings", json={"mode": "rescue",
                                                         "threshold": 0.25}).json()["mode"] == "rescue"
        assert client.patch("/api/wake/settings", json={"mode": "x"}).status_code == 400
    assert proxy.prints.settings["threshold"] == 0.25


def test_the_api_is_absent_without_the_proxy():
    app = FastAPI()
    app.state.context = SimpleNamespace()
    app.include_router(wake_router)
    with TestClient(app) as client:
        assert client.get("/api/wake/status").status_code == 404
