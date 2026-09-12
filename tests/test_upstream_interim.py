"""Interim transcripts from a streaming upstream must not be taken as final.

A streaming recogniser emits a Transcript every time it firms up another
word, each one a little longer than the last. Resolving on the first one
truncated every utterance mid-word — "was ist die Hauptstadt von
Niedersach" — while a one-shot engine, whose single Transcript arrives
after AudioStop, looked fine.
"""

from __future__ import annotations

import asyncio
from types import SimpleNamespace

import pytest
from wyoming.asr import Transcript

from wyoming_murdock.handler import MurdockHandler


class _Upstream:
    """Replays a scripted event sequence, then closes."""

    def __init__(self, events, handler=None, stop_after=None, hang=False):
        self._events = list(events)
        self._handler = handler
        self._stop_after = stop_after
        # A real server usually keeps the socket open after its last
        # word, so the reader has to decide by silence rather than by
        # the connection ending.
        self._hang = hang
        self._read = 0

    async def read_event(self):
        if self._read == self._stop_after and self._handler is not None:
            # Home Assistant stopped sending audio at this point.
            self._handler._audio_stop_sent = True
        if not self._events:
            if self._hang:
                await asyncio.sleep(10)
            return None
        self._read += 1
        return self._events.pop(0)


def _run(events, *, stop_after=None, hang=False):
    """Build the stand-in and drive the reader on one event loop.

    The future has to be created inside the loop that later resolves it,
    so the handler is assembled in the coroutine rather than outside it.
    """

    async def _go():
        fake = SimpleNamespace(
            _session_id="s1",
            _upstream_interim="",
            _audio_stop_sent=False,
            _streaming_upstream=False,
            _cancelled=False,
            # Cancel phrases are checked against interims; off by default
            # here so these tests stay about the interim handling itself.
            context=SimpleNamespace(is_cancel_phrase=lambda text: False),
            _upstream_transcript=asyncio.get_running_loop().create_future(),
        )
        fake._upstream_client = _Upstream(events, fake, stop_after, hang)
        await MurdockHandler._read_upstream_transcript(fake)
        return fake._upstream_transcript.result()

    return asyncio.run(_go())


def _t(text):
    return Transcript(text=text).event()


def test_a_streaming_upstream_is_not_cut_off_mid_word():
    """The reported symptom, as a test."""
    assert _run(
        [
            _t("Was ist die"),
            _t("Was ist die Hauptstadt"),
            _t("Was ist die Hauptstadt von Niedersach"),
            _t("Was ist die Hauptstadt von Niedersachsen?"),
        ],
        stop_after=3,  # AudioStop lands after the third interim
    ) == "Was ist die Hauptstadt von Niedersachsen?"


def test_a_one_shot_upstream_still_works():
    """Its single Transcript arrives after AudioStop and is final."""
    assert _run([_t("Licht im Wohnzimmer an")], stop_after=0) ==         "Licht im Wohnzimmer an"


def test_a_session_ending_early_falls_back_to_the_best_interim():
    """Half an answer beats none when the upstream hangs up."""
    assert _run([_t("Wir müssen nochmal einen"),
                 _t("Wir müssen nochmal einen Latenztest")]) ==         "Wir müssen nochmal einen Latenztest"


def test_nothing_at_all_still_resolves_empty():
    assert _run([]) == ""


def test_an_empty_final_does_not_erase_a_good_interim():
    """A final that heard nothing must not throw away what was understood."""
    assert _run([_t("Schalte das Licht"), _t("")], stop_after=1) ==         "Schalte das Licht"


def test_a_cancel_phrase_in_an_interim_marks_the_session():
    """The point of catching it mid-sentence is to stop before the act."""

    async def _go():
        fake = SimpleNamespace(
            _session_id="s1",
            _upstream_interim="",
            _audio_stop_sent=False,
            _streaming_upstream=False,
            _cancelled=False,
            context=SimpleNamespace(
                is_cancel_phrase=lambda text: text.lower().startswith("abbruch")
            ),
            _upstream_transcript=asyncio.get_running_loop().create_future(),
        )
        fake._upstream_client = _Upstream(
            [_t("Abbruch"), _t("Abbruch bitte")], fake, 2
        )
        await MurdockHandler._read_upstream_transcript(fake)
        return fake._cancelled

    assert asyncio.run(_go()) is True


def test_a_streaming_server_is_not_cut_off_after_audiostop():
    """The tail of the utterance is still in the server's buffer.

    "Final = the first transcript after AudioStop" was not enough: a
    streaming recogniser keeps emitting longer ones afterwards, and
    taking the first cut the end off — "Fahre alle Rollladen der gesam".
    """
    assert _run(
        [
            _t("Fahre alle"),                      # interim — marks it streaming
            _t("Fahre alle Rollladen der gesam"),  # after AudioStop, still growing
            _t("Fahre alle Rollladen der gesamten Wohnung runter."),
        ],
        stop_after=1,
    ) == "Fahre alle Rollladen der gesamten Wohnung runter."


def test_a_one_shot_server_never_pays_the_grace():
    """It sends nothing before AudioStop, so its first is final."""
    import time

    t0 = time.monotonic()
    assert _run([_t("Licht an")], stop_after=0) == "Licht an"
    # No waiting: the grace is only spent on a server that showed itself
    # to be streaming.
    assert (time.monotonic() - t0) < 0.2


def test_silence_decides_when_the_server_keeps_the_socket_open():
    """The realistic shape: nothing closes, so the end is inferred."""
    import time

    import wyoming_murdock.handler as h

    original = h._UPSTREAM_FINAL_GRACE
    h._UPSTREAM_FINAL_GRACE = 0.1
    try:
        t0 = time.monotonic()
        result = _run(
            [_t("Fahre alle"), _t("Fahre alle Rollladen der gesamten Wohnung runter.")],
            stop_after=1, hang=True,
        )
        waited = time.monotonic() - t0
    finally:
        h._UPSTREAM_FINAL_GRACE = original

    assert result == "Fahre alle Rollladen der gesamten Wohnung runter."
    # It waited out the grace rather than hanging on the open socket.
    assert 0.05 < waited < 2.0
