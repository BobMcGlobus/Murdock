"""The wake word proxy relays Voice Support runs unchanged and keeps a copy."""

from __future__ import annotations

import asyncio
import json

import pytest
from fastapi.testclient import TestClient

from wyoming_murdock.wake_proxy import SpeakerView, Upstream, WakeProxy, would_decide

_BYTES_PER_MS = 32


class _FakeApp(Upstream):
    """The Voice Support app's side of one run, scripted.

    It answers the way the real app does: a verdict once it has the wake
    word audio, then (if it accepted) where the command starts and ends.
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
                self._wake_ms = event.get("wake_word_audio_ms")
            elif event["type"] == "audio-end":
                if self.verdict == "accepted":
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


def _proxy(tmp_path, app: _FakeApp | None, view: SpeakerView | None = None):
    async def connect(url):
        if app is None:
            raise OSError("refused")
        return app

    async def check(clip: bytes) -> SpeakerView:
        check.clips.append(clip)
        return view or SpeakerView(error="nobody enrolled")

    check.clips = []

    async def info(url):
        return {"name": "Voice Support", "version": "0.2.1", "api_version": 1,
                "wake_words": [{"id": "okay_nabu", "phrase": "okay nabu"}]}

    proxy = WakeProxy("http://voice:10900", data_dir=tmp_path, speaker_check=check,
                      connect=connect, fetch_info=info)
    return proxy, check


def _run_start(wake_ms: int | None = 1500) -> dict:
    start = {"type": "run-start", "satellite_id": "assist_satellite.wohnzimmer",
             "silence_seconds": 0.7}
    if wake_ms:
        start.update(wake_word_phrase="okay nabu", wake_word_audio_ms=wake_ms)
    return start


def _chunk(value: int, ms: int = 100) -> bytes:
    return bytes([value & 0xFF, 0]) * (ms * 16)


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
            events.append(json.loads(ws.receive_text()))       # stt-vad-start
            ws.send_text(json.dumps({"type": "audio-end"}))
            events.append(json.loads(ws.receive_text()))       # stt-vad-end
    return events


def _settle(client, proxy):
    client.portal.call(proxy.wait_idle)


def test_info_is_passed_through_under_murdocks_name(tmp_path):
    proxy, _ = _proxy(tmp_path, _FakeApp())
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


def test_an_accepted_run_is_relayed_unchanged(tmp_path):
    app = _FakeApp("accepted")
    proxy, check = _proxy(tmp_path, app)
    with TestClient(proxy.build_app()) as client:
        events = _drive(client)
        _settle(client, proxy)
    assert [e["type"] for e in events] == ["wake-word-end", "stt-vad-start", "stt-vad-end"]
    # Home Assistant's messages and audio reach the app untouched.
    assert app.received_text[0] == _run_start()
    assert app.received_text[-1] == {"type": "audio-end"}
    assert bytes(app.received_audio) == b"".join(_chunk(i) for i in range(30))
    assert app.closed
    # The speaker check got the wake word: the 1.5 s before the detection.
    [clip] = check.clips
    assert len(clip) == 1500 * _BYTES_PER_MS
    [row] = proxy.records
    assert row["verdict"] == "accepted" and row["accepted_boundary_ms"] == 1500
    assert row["vad_end_ms"] == 2500 and row["satellite_id"] == "assist_satellite.wohnzimmer"


def test_a_rejection_is_relayed_and_recorded_with_its_reason(tmp_path):
    app = _FakeApp("rejected", reason="confusable: 'namu'")
    proxy, _ = _proxy(tmp_path, app, SpeakerView(
        speaker="Jonas", distance=0.21, threshold=0.3, is_match=True, nearest="Jonas",
    ))
    with TestClient(proxy.build_app()) as client:
        events = _drive(client)
        _settle(client, proxy)
    assert events[-1]["type"] == "wake-word-rejected"
    [row] = proxy.records
    assert row["verdict"] == "rejected"
    assert row["reason"] == "confusable: 'namu'"
    assert row["would"] == "would rescue (Jonas)"
    # The clip and the log line are on disk for listening back.
    assert row["clip"] and (tmp_path / "wake_runs" / row["clip"]).exists()
    lines = (tmp_path / "wake_runs" / "runs.jsonl").read_text(encoding="utf-8").splitlines()
    assert json.loads(lines[-1])["reason"] == "confusable: 'namu'"


def test_a_run_without_a_wake_word_is_relayed_but_not_checked(tmp_path):
    """A button press: the app only endpoints, there is nothing to verify."""
    app = _FakeApp("accepted")
    proxy, check = _proxy(tmp_path, app)
    with TestClient(proxy.build_app()) as client:
        with client.websocket_connect("/run") as ws:
            ws.send_text(json.dumps(_run_start(wake_ms=None)))
            for i in range(5):
                ws.send_bytes(_chunk(i))
            ws.send_text(json.dumps({"type": "audio-end"}))
            assert json.loads(ws.receive_text())["type"] == "stt-vad-end"
        _settle(client, proxy)
    assert check.clips == []
    [row] = proxy.records
    assert row["wake_word"] is None and row["clip"] is None and row["would"] is None


def test_without_the_app_the_run_is_closed_so_home_assistant_falls_back(tmp_path):
    proxy, check = _proxy(tmp_path, None)
    with TestClient(proxy.build_app()) as client:
        with client.websocket_connect("/run") as ws:
            with pytest.raises(Exception):
                ws.receive_text()
        _settle(client, proxy)
    [row] = proxy.records
    assert row["upstream_failed"] is True
    assert check.clips == []


@pytest.mark.parametrize("verdict, view, expected", [
    ("rejected", SpeakerView(speaker="Jonas", distance=0.2, threshold=0.3, is_match=True),
     "would rescue (Jonas)"),
    ("accepted", SpeakerView(distance=0.6, threshold=0.3, is_match=False, nearest="Jonas"),
     "would veto (no enrolled voice)"),
    ("accepted", SpeakerView(speaker="Jonas", distance=0.2, threshold=0.3, is_match=True),
     "agrees"),
    ("rejected", SpeakerView(distance=0.6, threshold=0.3, is_match=False), "agrees"),
    ("rejected", SpeakerView(error="nobody enrolled"), "app stands"),
    ("accepted", None, "app stands"),
])
def test_what_a_speaker_aware_verdict_would_be(verdict, view, expected):
    assert would_decide(verdict, view) == expected
