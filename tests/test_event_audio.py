"""Keeping the audio behind a transcript, so it can be listened to.

A wrong transcript has two possible causes — the audio was already bad,
or the engine misread good audio — and they are indistinguishable from
the text alone. The log keeps both recordings of the newest utterances:
what the microphone sent, and the copy the service received.
"""

from __future__ import annotations

import asyncio
import io
import wave
from types import SimpleNamespace

import pytest
from fastapi import HTTPException

from murdock.api.routes_recognition import event_audio, list_events
from murdock.core.db import open_db
from murdock.core.recognition_log import RecognitionLog


def _log(tmp_path, name="m.db"):
    return RecognitionLog(open_db(tmp_path / name))


def _event(log, session="s1"):
    return log.record(
        session_id=session, satellite_id="wz", duration_sec=1.0, outcome="match"
    )


def test_both_recordings_come_back_as_playable_wav(tmp_path):
    log = _log(tmp_path)
    event_id = _event(log)
    log.store_audio(event_id, "mic", b"\x01\x02" * 1600)
    log.store_audio(event_id, "upload", b"\x03\x04" * 800)

    ctx = SimpleNamespace(recognition=log)
    resp = asyncio.run(event_audio(event_id, "mic", ctx))
    assert resp.media_type == "audio/wav"
    with wave.open(io.BytesIO(resp.body)) as wf:
        assert wf.getframerate() == 16000
        assert wf.getnchannels() == 1
        assert wf.getnframes() == 1600

    upload = asyncio.run(event_audio(event_id, "upload", ctx))
    assert len(upload.body) < len(resp.body)


def test_an_event_without_audio_is_a_404_not_an_empty_file(tmp_path):
    log = _log(tmp_path)
    ctx = SimpleNamespace(recognition=log)
    with pytest.raises(HTTPException) as err:
        asyncio.run(event_audio(_event(log), "mic", ctx))
    assert err.value.status_code == 404


def test_only_the_two_known_kinds_are_served(tmp_path):
    log = _log(tmp_path)
    with pytest.raises(HTTPException) as err:
        asyncio.run(event_audio(_event(log), "../../etc/passwd", SimpleNamespace(recognition=log)))
    assert err.value.status_code == 400


def test_older_utterances_lose_their_audio(tmp_path):
    """Storage is bounded by count — a debugging aid, not an archive."""
    log = _log(tmp_path)
    ids = [_event(log, f"s{i}") for i in range(5)]
    for event_id in ids:
        log.store_audio(event_id, "mic", b"\x00" * 3200, keep_events=3)
    kept = log.audio_for(ids)
    assert sorted(kept) == sorted(ids[-3:])
    assert log.get_audio(ids[0], "mic") is None


def test_the_log_says_which_recordings_it_still_has(tmp_path):
    log = _log(tmp_path)
    event_id = _event(log)
    log.store_audio(event_id, "mic", b"\x00" * 3200)

    ctx = SimpleNamespace(
        recognition=log,
        unknown=SimpleNamespace(map_sessions_to_samples=lambda ids: {}),
        satellite_label=lambda sid: sid or "",
    )
    out = asyncio.run(list_events(limit=10, outcome=None, speaker=None, ctx=ctx))
    assert out.events[0].audio == ["mic"]


def test_clearing_the_log_takes_the_recordings_with_it(tmp_path):
    log = _log(tmp_path)
    event_id = _event(log)
    log.store_audio(event_id, "mic", b"\x00" * 3200)
    log.clear()
    assert log.get_audio(event_id, "mic") is None


def test_the_handler_stores_the_microphone_copy_and_the_upload_copy(tmp_path):
    """The two copies differ exactly when the upload was conditioned."""
    from murdock.config import Settings
    from murdock.core.context import AppContext
    from wyoming_murdock.handler import MurdockHandler

    db = open_db(tmp_path / "h.db")
    log = RecognitionLog(db)
    ctx = AppContext(
        settings=Settings(), db=db, embedder=None, vad=None,
        speakers=None, unknown=None, ha=None, mqtt=None, recognition=log,
    )
    ctx.stt_services.create("Main", "voxtral", {"api_key": "k"})
    handler = MurdockHandler(
        SimpleNamespace(upstream_uri="tcp://unused:1"), ctx,
        SimpleNamespace(), SimpleNamespace(),
    )
    event_id = _event(log)
    handler._session_audio = b"\x01\x02" * 1600
    handler._prepared_audio = b"\x05\x06" * 1200

    async def scenario():
        handler._store_event_audio(event_id)
        await asyncio.sleep(0.05)

    asyncio.run(scenario())
    assert log.get_audio(event_id, "mic")[0] == b"\x01\x02" * 1600
    assert log.get_audio(event_id, "upload")[0] == b"\x05\x06" * 1200

    # Switched off, nothing is kept.
    other = _event(log, "s2")
    ctx.set_enable_event_audio(False)
    handler._session_audio = b"\x01\x02" * 1600
    asyncio.run(scenario())
    assert log.get_audio(other, "mic") is None
