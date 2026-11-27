"""The wake word proxy: two verifiers, an authority, labels and replay."""

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
    DE,
    EN,
    SpeakerTools,
    SpeakerView,
    Upstream,
    WakePrints,
    WakeProxy,
)

_BYTES_PER_MS = 32
JONAS, SARAH = 3, 4
EN_URL, DE_URL = "http://en:10900", "http://de:10900"


class _FakeApp(Upstream):
    """A verifier's side of one run, scripted.

    With a wake word it answers once it has the wake word audio; a refusal
    ends its run. After an acceptance it says where speech starts, and where
    the command ends once audio-end arrives.
    """

    def __init__(self, verdict: str = "accepted", reason: str | None = None,
                 vad_end: int = 2500):
        self.verdict = verdict
        self.reason = reason
        self.vad_end = vad_end
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
                    await self._out.put(json.dumps({"type": "stt-vad-end",
                                                    "timestamp": self.vad_end}))
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


def _proxy(tmp_path, en=(), de=None, tools=None, *, authority=None, de_verifies=True):
    """A proxy whose verifiers hand out the given scripted runs in order;
    None in a list means that connection is refused."""
    queues = {f"ws://en:10900/run": list(en)}
    if de is not None:
        queues["ws://de:10900/run"] = list(de)

    async def connect(url):
        queue = queues.get(url)
        if not queue or queue[0] is None:
            if queue:
                queue.pop(0)
            raise OSError("refused")
        return queue.pop(0)

    async def info(url):
        return {"name": "Voice Support", "version": "0.2.1", "api_version": 1,
                "wake_words": [{"id": "okay_nabu", "phrase": "okay nabu"}]}

    proxy = WakeProxy(EN_URL, secondary_url=DE_URL if de is not None else None,
                      data_dir=tmp_path, tools=tools, connect=connect, fetch_info=info)
    if de is not None and de_verifies:
        proxy.verifies[DE] = {"okay_nabu"}
    if authority:
        proxy.prints.update_settings(authority=authority)
    return proxy


def _run_start(wake_ms: int | None = 1500) -> dict:
    start = {"type": "run-start", "satellite_id": "assist_satellite.wohnzimmer",
             "silence_seconds": 0.7}
    if wake_ms:
        start.update(wake_word_phrase="Okay Nabu", wake_word_audio_ms=wake_ms)
    return start


def _chunk(value: int, ms: int = 100) -> bytes:
    return bytes([value & 0xFF, 0]) * (ms * 16)


def _audio(n: int = 30) -> bytes:
    return b"".join(_chunk(i) for i in range(n))


def _drive(client, *, wake_ms=1500, chunks=30):
    """Act as Home Assistant: start, stream, read the answers to the end."""
    events = []
    with client.websocket_connect("/run") as ws:
        ws.send_text(json.dumps(_run_start(wake_ms)))
        for i in range(chunks):
            ws.send_bytes(_chunk(i))
        events.append(json.loads(ws.receive_text()))
        if events[-1]["type"] != "wake-word-rejected":
            ws.send_text(json.dumps({"type": "audio-end"}))
            while events[-1]["type"] != "stt-vad-end":
                events.append(json.loads(ws.receive_text()))
    return events


def _settle(client, proxy):
    client.portal.call(proxy.wait_idle)


# --- /info -------------------------------------------------------------------


def test_info_is_passed_through_under_murdocks_name(tmp_path):
    proxy = _proxy(tmp_path)
    with TestClient(proxy.build_app()) as client:
        info = client.get("/info").json()
    assert info["api_version"] == 1 and info["name"] == "Murdock → Voice Support"
    assert proxy.verifies[EN] == {"okay_nabu"}


def test_info_without_the_app_is_a_503(tmp_path):
    async def down(url):
        raise OSError("refused")

    proxy = WakeProxy(EN_URL, data_dir=tmp_path, fetch_info=down)
    with TestClient(proxy.build_app()) as client:
        assert client.get("/info").status_code == 503


def test_a_recent_info_stands_in_while_the_app_is_away(tmp_path, monkeypatch):
    import wyoming_murdock.wake_proxy as mod

    state = {"up": True}

    async def info(url):
        if not state["up"]:
            raise OSError("refused")
        return {"name": "Voice Support", "version": "0.2.1", "api_version": 1,
                "wake_words": [{"id": "okay_nabu"}]}

    proxy = WakeProxy(EN_URL, data_dir=tmp_path, fetch_info=info)
    with TestClient(proxy.build_app()) as client:
        assert client.get("/info").status_code == 200
        state["up"] = False
        assert client.get("/info").json()["wake_words"] == [{"id": "okay_nabu"}]
    again = WakeProxy(EN_URL, data_dir=tmp_path, fetch_info=info)
    with TestClient(again.build_app()) as client:
        assert client.get("/info").status_code == 200
        monkeypatch.setattr(mod, "_INFO_STALE_SECONDS", 0)
        assert client.get("/info").status_code == 503


# --- one verifier --------------------------------------------------------------


def test_an_accepted_run_is_relayed_unchanged(tmp_path):
    app = _FakeApp("accepted")
    tools = _Tools()
    proxy = _proxy(tmp_path, en=[app], tools=tools)
    with TestClient(proxy.build_app()) as client:
        events = _drive(client)
        _settle(client, proxy)
    assert [e["type"] for e in events] == ["wake-word-end", "stt-vad-start", "stt-vad-end"]
    assert app.received_text[0] == _run_start()
    assert app.received_text[-1] == {"type": "audio-end"}
    assert bytes(app.received_audio) == _audio()
    [clip] = tools.wake_clips
    assert len(clip) == 1500 * _BYTES_PER_MS
    [row] = proxy.records
    assert row["verdict"] == "accepted" and row["en"] == "accepted" and row["de"] is None
    assert row["vad_end_ms"] == 2500


def test_without_the_app_the_run_is_closed_so_home_assistant_falls_back(tmp_path):
    proxy = _proxy(tmp_path, en=[None], tools=_Tools())
    with TestClient(proxy.build_app()) as client:
        with client.websocket_connect("/run") as ws:
            with pytest.raises(Exception):
                ws.receive_text()
        _settle(client, proxy)
    assert proxy.records[0]["upstream_failed"] is True


def test_a_run_without_a_wake_word_is_relayed_but_not_checked(tmp_path):
    tools = _Tools()
    proxy = _proxy(tmp_path, en=[_FakeApp()], tools=tools)
    with TestClient(proxy.build_app()) as client:
        with client.websocket_connect("/run") as ws:
            ws.send_text(json.dumps(_run_start(wake_ms=None)))
            for i in range(5):
                ws.send_bytes(_chunk(i))
            ws.send_text(json.dumps({"type": "audio-end"}))
            assert json.loads(ws.receive_text())["type"] == "stt-vad-end"
        _settle(client, proxy)
    assert tools.wake_clips == [] and proxy.records[0]["clip"] is None


# --- two verifiers -----------------------------------------------------------------


def test_the_german_verifier_hears_everything_and_only_the_log_hears_it(tmp_path):
    en = _FakeApp("accepted")
    de = _FakeApp("rejected", reason="confusable: 'na du'")
    proxy = _proxy(tmp_path, en=[en], de=[de], tools=_Tools())
    with TestClient(proxy.build_app()) as client:
        events = _drive(client)
        _settle(client, proxy)
    # English is the authority: Home Assistant hears it alone.
    assert [e["type"] for e in events] == ["wake-word-end", "stt-vad-start", "stt-vad-end"]
    assert de.received_text[0] == _run_start()
    assert bytes(de.received_audio).startswith(_audio()[:1500 * _BYTES_PER_MS])
    [row] = proxy.records
    assert (row["en"], row["de"], row["de_reason"]) == ("accepted", "rejected",
                                                        "confusable: 'na du'")
    assert row["verdict"] == "accepted" and row["authority"] == EN


def test_with_german_authority_home_assistant_hears_the_german_verdict(tmp_path):
    """English refuses the German "Nabu" as "namu"; German accepts it, and
    its command boundaries are the ones Home Assistant gets."""
    en = _FakeApp("rejected", reason="confusable: 'namu'")
    de = _FakeApp("accepted", vad_end=3100)
    proxy = _proxy(tmp_path, en=[en], de=[de], tools=_Tools(), authority=DE)
    with TestClient(proxy.build_app()) as client:
        events = _drive(client)
        _settle(client, proxy)
    assert events[0]["type"] == "wake-word-end"
    assert events[-1] == {"type": "stt-vad-end", "timestamp": 3100}
    # Every byte reached the German verifier even after English let go.
    assert bytes(de.received_audio) == _audio()
    [row] = proxy.records
    assert row["authority"] == DE and row["verdict"] == "accepted"
    assert row["en"] == "rejected" and row["vad_end_ms"] == 3100


def test_german_authority_refuses_what_german_refuses(tmp_path):
    en = _FakeApp("accepted")
    de = _FakeApp("rejected", reason="confusable: 'na du'")
    proxy = _proxy(tmp_path, en=[en], de=[de], tools=_Tools(), authority=DE)
    with TestClient(proxy.build_app()) as client:
        events = _drive(client)
        _settle(client, proxy)
    assert events == [{"type": "wake-word-rejected",
                       "wake_word_output": {"wake_word_phrase": "okay nabu",
                                            "reason": "confusable: 'na du'"}}]
    assert proxy.records[0]["verdict"] == "rejected"


def test_a_long_command_outlives_the_other_verifiers_refusal(tmp_path):
    """The German verifier refuses and hangs up at once; the English
    authority's run must go on to the end of the command."""
    en = _FakeApp("accepted", vad_end=9000)
    de = _FakeApp("rejected", reason="last")
    proxy = _proxy(tmp_path, en=[en], de=[de], tools=_Tools())
    with TestClient(proxy.build_app()) as client:
        events = _drive(client, chunks=90)
        _settle(client, proxy)
    assert events[-1] == {"type": "stt-vad-end", "timestamp": 9000}
    assert bytes(en.received_audio) == _audio(90)


def test_a_wake_word_german_does_not_verify_goes_by_english(tmp_path):
    proxy = _proxy(tmp_path, en=[_FakeApp("accepted")],
                   de=[_FakeApp("rejected", reason="x")], tools=_Tools(),
                   authority=DE, de_verifies=False)
    with TestClient(proxy.build_app()) as client:
        events = _drive(client)
        _settle(client, proxy)
    assert events[0]["type"] == "wake-word-end"
    assert proxy.records[0]["authority"] == EN


def test_without_the_german_verifier_english_decides(tmp_path):
    proxy = _proxy(tmp_path, en=[_FakeApp("accepted")], de=[None], tools=_Tools(),
                   authority=DE)
    with TestClient(proxy.build_app()) as client:
        events = _drive(client)
        _settle(client, proxy)
    assert events[0]["type"] == "wake-word-end"
    [row] = proxy.records
    assert row["authority"] == EN and row["de"] is None


# --- labels, comparison and replay --------------------------------------------------


def test_a_label_window_marks_the_runs_and_the_comparison_counts_them(tmp_path):
    proxy = _proxy(tmp_path, en=[_FakeApp("rejected", reason="confusable: 'namu'"),
                                 _FakeApp("accepted")],
                   de=[_FakeApp("accepted"), _FakeApp("rejected", reason="confusable")],
                   tools=_Tools())
    proxy.open_label_window("wake", 60)
    with TestClient(proxy.build_app()) as client:
        _drive(client)
        proxy.open_label_window("false", 60)
        _drive(client)
        _settle(client, proxy)
    assert [r["label"] for r in proxy.records] == ["wake", "false"]
    comparison = proxy.comparison()
    assert comparison["wake"] == {"runs": 1, "en_accepted": 0, "de_accepted": 1,
                                  "en_judged": 1, "de_judged": 1}
    assert comparison["false"]["en_accepted"] == 1 and comparison["false"]["de_accepted"] == 0


def test_past_runs_can_be_labelled_by_time(tmp_path):
    proxy = _proxy(tmp_path, en=[_FakeApp("accepted")], tools=_Tools())
    with TestClient(proxy.build_app()) as client:
        _drive(client)
        _settle(client, proxy)
    at = proxy.records[0]["at"]
    assert proxy.label_range("wake", at, at) == 1
    assert proxy.history()[0]["label"] == "wake" and proxy.records[0]["label"] == "wake"
    with pytest.raises(ValueError):
        proxy.label_range("maybe", at, at)


def test_stored_clips_replay_through_a_verifier(tmp_path):
    proxy = _proxy(tmp_path, en=[_FakeApp("accepted"), _FakeApp("rejected", reason="r")],
                   de=[_FakeApp("accepted"), _FakeApp("accepted")], tools=_Tools())
    with TestClient(proxy.build_app()) as client:
        _drive(client)
        _drive(client)
        _settle(client, proxy)
        proxy.label_range("wake", "0", "9")
        # Replay against a verifier that now refuses the first and takes the second.
        replays = [_FakeApp("rejected", reason="confusable"), _FakeApp("accepted")]

        async def connect(url):
            return replays.pop(0)

        proxy._connect = connect
        result = client.portal.call(lambda: proxy.replay("de", label="wake"))
    assert result["summary"] == {"wake": {"runs": 2, "accepted": 1, "rejected": 1,
                                          "no_verdict": 0}}
    assert [r["recorded"] for r in result["results"]] == ["accepted", "accepted"]


# --- voiceprints ------------------------------------------------------------------------


def test_an_accepted_wake_word_teaches_the_commands_speaker(tmp_path):
    tools = _Tools(command=SpeakerView(speaker="Jonas", speaker_id=JONAS, distance=0.12,
                                       threshold=0.35, is_match=True))
    proxy = _proxy(tmp_path, en=[_FakeApp("accepted")], tools=tools)
    with TestClient(proxy.build_app()) as client:
        _drive(client)
        _settle(client, proxy)
    [command] = tools.commands
    assert len(command) == (2500 - 1500) * _BYTES_PER_MS
    assert proxy.records[0]["learnt"] == "Jonas (auto, 1)"


def test_a_false_alarm_teaches_nothing_even_if_accepted(tmp_path):
    tools = _Tools(command=SpeakerView(speaker="Jonas", speaker_id=JONAS, distance=0.12,
                                       threshold=0.35, is_match=True))
    proxy = _proxy(tmp_path, en=[_FakeApp("accepted")], tools=tools)
    proxy.open_label_window("false", 60)
    with TestClient(proxy.build_app()) as client:
        _drive(client)
        _settle(client, proxy)
    assert proxy.records[0]["learnt"] is None and proxy.prints.speakers == {}


def test_an_enrolment_window_takes_even_refused_wake_words(tmp_path):
    proxy = _proxy(tmp_path, en=[_FakeApp("rejected", reason="confusable: 'namu'")],
                   tools=_Tools())
    proxy.prints.open_window(SARAH, "Sarah", 60)
    with TestClient(proxy.build_app()) as client:
        _drive(client)
        _settle(client, proxy)
    [row] = proxy.records
    assert row["learnt"] == "Sarah (enrolment, 1)" and row["label"] == "wake"


def test_voiceprints_need_a_margin_to_the_next_person(tmp_path):
    prints = WakePrints(tmp_path / "prints.json")
    for _ in range(5):
        prints.add(JONAS, "Jonas", np.array([1.0, 0.0]), source="enroll")
        prints.add(SARAH, "Sarah", np.array([0.995, 0.0998]), source="enroll")
    view = prints.match(np.array([1.0, 0.03]))
    assert view.nearest == "Jonas" and not view.is_match
    prints.update_settings(min_margin=0.0)
    assert prints.match(np.array([1.0, 0.03])).is_match
    again = WakePrints(tmp_path / "prints.json")
    assert again.settings["min_margin"] == 0.0


def test_an_unknown_authority_is_refused(tmp_path):
    with pytest.raises(ValueError):
        WakePrints(None).update_settings(authority="fr")


# --- web API ---------------------------------------------------------------------------


def _api(proxy):
    app = FastAPI()
    app.state.context = SimpleNamespace(wake_proxy=proxy)
    app.include_router(wake_router)
    return TestClient(app)


def test_the_api_switches_the_authority_only_to_a_configured_verifier(tmp_path):
    with _api(_proxy(tmp_path)) as client:
        assert client.patch("/api/wake/settings", json={"authority": "de"}).status_code == 400
    with _api(_proxy(tmp_path, de=[])) as client:
        assert client.patch("/api/wake/settings",
                            json={"authority": "de"}).json()["authority"] == "de"


def test_the_api_opens_windows_and_labels_by_time(tmp_path):
    proxy = _proxy(tmp_path, tools=_Tools())
    with _api(proxy) as client:
        assert client.post("/api/wake/label", json={"label": "false", "seconds": 60}
                           ).json()["label"] == "false"
        assert client.get("/api/wake/status").json()["labelling"]["label"] == "false"
        assert client.delete("/api/wake/label").json() == {"labelling": None}
        assert client.post("/api/wake/enroll", json={"speaker": "sarah"}
                           ).json()["speaker"] == "Sarah"
        assert client.post("/api/wake/label-range",
                           json={"label": "wake", "start": "0", "end": "9"}
                           ).json() == {"labelled": 0, "label": "wake"}


def test_the_api_is_absent_without_the_proxy():
    app = FastAPI()
    app.state.context = SimpleNamespace()
    app.include_router(wake_router)
    with TestClient(app) as client:
        assert client.get("/api/wake/status").status_code == 404
