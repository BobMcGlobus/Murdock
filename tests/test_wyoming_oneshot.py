"""The one-shot Wyoming helper must not stop at the first transcript.

Same defect as the streaming reader fixed in 0.9.4, in the other place
it lives: the helper behind the A/B shadow and the local fallback. A
streaming recogniser sends one Transcript per firmed-up word, so
returning the first truncated every utterance — reported as "wann
Avengers Doomsday rauskomm" against the cloud's "rauskommt?".
"""

from __future__ import annotations

import asyncio

import pytest
from wyoming.asr import Transcript, TranscriptChunk, TranscriptStop

import murdock.core.stt_backend as mod


class _FakeClient:
    def __init__(self, events, hang_after=True):
        self._events = list(events)
        self._hang = hang_after
        self.written = []

    async def __aenter__(self):
        return self

    async def __aexit__(self, *a):
        return False

    async def write_event(self, event):
        self.written.append(event)

    async def read_event(self):
        if self._events:
            return self._events.pop(0)
        if self._hang:
            # A server that keeps the socket open after its last word.
            await asyncio.sleep(10)
        return None


def _run(monkeypatch, events, hang_after=True):
    client = _FakeClient(events, hang_after)
    monkeypatch.setattr(
        mod, "_WYOMING_FINAL_GRACE", 0.05, raising=False
    )

    class _AsyncClient:
        @staticmethod
        def from_uri(uri):
            return client

    import wyoming.client

    monkeypatch.setattr(wyoming.client, "AsyncClient", _AsyncClient)
    return asyncio.run(
        mod.transcribe_via_wyoming("tcp://x:1", b"\x00" * 3200, timeout=5)
    )


def _t(text):
    return Transcript(text=text).event()


def test_the_newest_transcript_wins(monkeypatch):
    assert _run(monkeypatch, [
        _t("Kannst du mir sagen"),
        _t("Kannst du mir sagen, wann Avengers Doomsday rauskomm"),
        _t("Kannst du mir sagen, wann Avengers Doomsday rauskommt?"),
    ]) == "Kannst du mir sagen, wann Avengers Doomsday rauskommt?"


def test_a_one_shot_server_is_unaffected(monkeypatch):
    """It sends exactly one and then nothing; the grace costs one wait."""
    assert _run(monkeypatch, [_t("Licht an")]) == "Licht an"


def test_transcript_stop_ends_it_immediately(monkeypatch):
    """An explicit end of stream should not wait out the grace period."""
    assert _run(monkeypatch, [
        _t("Licht im Wohnzimmer an"),
        TranscriptStop().event(),
    ]) == "Licht im Wohnzimmer an"


def test_streaming_chunks_are_not_mistaken_for_a_result(monkeypatch):
    """transcript-chunk is partial by definition in the protocol."""
    assert _run(monkeypatch, [
        TranscriptChunk(text="Licht ").event(),
        TranscriptChunk(text="an").event(),
        _t("Licht an."),
    ]) == "Licht an."


def test_an_empty_final_does_not_erase_a_good_one(monkeypatch):
    assert _run(monkeypatch, [_t("Schalte das Licht"), _t("")]) == \
        "Schalte das Licht"


def test_nothing_at_all_is_still_an_error(monkeypatch):
    """Silence from the server is a failure, not an empty transcript —
    that is what lets the caller fall back."""
    with pytest.raises(mod.STTBackendError):
        _run(monkeypatch, [], hang_after=False)
