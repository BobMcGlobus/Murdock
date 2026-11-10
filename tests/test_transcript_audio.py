"""Which audio the gate verifies and which audio buffered engines transcribe.

Drives ``_finish_session`` end to end with fakes, so the gate, extraction
and the buffered engine run for real.
"""

from __future__ import annotations

import asyncio
import time
from types import SimpleNamespace

import numpy as np

from murdock.config import Settings
from murdock.core.context import AppContext
from murdock.core.db import open_db
from murdock.core.recognition_log import RecognitionLog
from murdock.core.speaker_store import VerificationResult
from murdock.core.stt_backend import STTBackendError
from murdock.core.stt_services import SttRoles
from wyoming_murdock.handler import MurdockHandler

_RATE = 16000

# Each sound is a constant sample value.
CHIME = 500
JONAS = 2000      # enrolled
STRANGER = 3000   # not enrolled


def _tone(value: int, seconds: float) -> np.ndarray:
    return np.full(int(seconds * _RATE), value, dtype=np.int16)


def _gap(seconds: float) -> np.ndarray:
    return np.zeros(int(seconds * _RATE), dtype=np.int16)


def _pcm(*parts: np.ndarray) -> bytes:
    return np.concatenate(parts).astype("<i2").tobytes()


def _samples(pcm: bytes) -> np.ndarray:
    return np.frombuffer(pcm, dtype="<i2")


def _seconds_of(value: int, pcm: bytes) -> float:
    return float(np.sum(_samples(pcm) == value)) / _RATE


class _RunVAD:
    """Speech segments are runs of one voice; the chime is not speech."""

    def analyze_pcm(self, pcm: bytes):
        s = _samples(pcm)
        segments, start = [], None
        for i in range(len(s) + 1):
            v = s[i] if i < len(s) else 0
            if start is not None and v != s[start]:
                segments.append((start / _RATE, i / _RATE))
                start = None
            if start is None and v not in (0, CHIME):
                start = i
        return SimpleNamespace(
            segments=segments,
            speech_seconds=sum(e - st for st, e in segments),
        )


class _ShareEmbedder:
    """The embedding is the share of sound samples that are Jonas."""

    def __init__(self):
        self.inputs: list[bytes] = []

    def embed_pcm(self, pcm: bytes) -> np.ndarray:
        self.inputs.append(pcm)
        s = _samples(pcm)
        voiced = s[s != 0]
        if voiced.size == 0:
            raise ValueError("no speech")
        return np.array([float(np.mean(voiced == JONAS))])


class _Speakers:
    def __init__(self):
        self.enrolled: list[bytes] = []

    def list_speakers(self):
        return [SimpleNamespace(id=1, name="Jonas")]

    def get_speaker(self, speaker_id):
        return SimpleNamespace(id=1, name="Jonas", role="admin")

    def verify_embedding(self, embedding, threshold=0.3, **_kw):
        distance = 1.0 - float(embedding[0])
        hit = distance <= threshold
        return VerificationResult(
            is_match=hit,
            matched_speaker="Jonas" if hit else None,
            matched_speaker_id=1 if hit else None,
            distance=distance,
            threshold=threshold,
            all_distances={"Jonas": distance},
        )

    def auto_enroll_embedding(self, speaker_id, embedding, audio, *_args):
        self.enrolled.append(audio)


class _Engine:
    """A buffered engine that records what it was sent."""

    def __init__(self, fail: bool = False):
        self.fail = fail
        self.audio: list[bytes] = []
        self.last_timing = {}

    async def transcribe(self, audio, **_kw):
        self.audio.append(audio)
        if self.fail:
            raise STTBackendError("main down")
        return "Licht an"


def _setup(tmp_path, *, require_match=False, extraction=True, main_fails=False,
           auto_enroll=False):
    db = open_db(tmp_path / "m.db")
    embedder = _ShareEmbedder()
    speakers = _Speakers()
    ctx = AppContext(
        settings=Settings(), db=db, embedder=embedder, vad=_RunVAD(),
        speakers=speakers, unknown=None, ha=None, mqtt=None,
        recognition=RecognitionLog(db),
    )
    main = ctx.stt_services.create("OpenAI", "openai", {"api_key": "k", "model": "m"})
    fallback = ctx.stt_services.create("Groq", "openai", {"api_key": "k", "model": "m"})
    ctx.stt_services.set_roles(SttRoles(main=main.id, fallbacks=[fallback.id]))
    engines = {main.id: _Engine(fail=main_fails), fallback.id: _Engine()}
    ctx.build_stt_backend = lambda svc: engines.get(svc.id) if svc else None
    ctx.set_enable_stt_prep(False)
    ctx.set_min_liveness_score(0.0)
    ctx.set_skip_leading_seconds(1.0)
    ctx.set_require_match(require_match)
    ctx.set_enable_extraction(extraction)
    ctx.set_unknown_logging(False)
    ctx.set_auto_enroll(auto_enroll)
    ctx.detect_whisper = lambda audio: None

    async def _no_media(*_a, **_kw):
        return 0.0

    ctx.compute_media_tightening = _no_media
    published: list[dict] = []
    ctx.publish_recognition = lambda **kw: published.append(kw)

    async def _publish_now(**kw):
        published.append(kw)

    ctx.publish_recognition_now = _publish_now

    handler = MurdockHandler(
        SimpleNamespace(upstream_uri="tcp://unused:1"), ctx,
        SimpleNamespace(), SimpleNamespace(),
    )
    sent: list[str] = []
    shadow_audio: list[bytes] = []

    async def _send(text, label=""):
        sent.append(text)

    handler._send_transcript_to_satellite = _send
    handler._spawn_shadow = lambda event_id, audio: shadow_audio.append(audio)
    return SimpleNamespace(
        handler=handler, engine=engines[main.id], fallback=engines[fallback.id],
        embedder=embedder, speakers=speakers, sent=sent, published=published,
        shadow_audio=shadow_audio,
    )


def _run(handler, pcm: bytes, *, early_match: bool = False) -> None:
    async def go():
        handler._reset_session_state()
        assert handler._is_voxtral
        if early_match:
            handler._early_match, handler._early_match_id = "Jonas", 1
            handler._early_distance = 0.05
        handler._audio_buffer = bytearray(pcm)
        handler._stream_start = time.monotonic() - len(pcm) / (2 * _RATE)
        await handler._finish_session()
        await asyncio.sleep(0.05)  # let fire-and-forget log writes land
    asyncio.run(go())


def _known(published) -> list:
    return [p["speaker"] for p in published if p.get("is_known")]


def test_engine_gets_untrimmed_audio(tmp_path):
    s = _setup(tmp_path, extraction=False)
    pcm = _pcm(_tone(CHIME, 0.3), _gap(0.2), _tone(JONAS, 2.5))
    _run(s.handler, pcm)
    assert s.sent == ["Licht an"]
    assert s.engine.audio == [pcm]


def test_embedder_gets_speech_without_the_chime(tmp_path):
    s = _setup(tmp_path, extraction=False)
    # Jonas starts inside the one-second chime window.
    pcm = _pcm(_tone(CHIME, 0.4), _gap(0.1), _tone(JONAS, 2.5))
    _run(s.handler, pcm)
    [verified] = s.embedder.inputs
    assert _seconds_of(CHIME, verified) == 0
    assert _seconds_of(JONAS, verified) == 2.5


def test_short_regions_dropped_by_extraction_are_transcribed(tmp_path):
    s = _setup(tmp_path)
    # The 0.4 s word is too short to score, so extraction drops it.
    pcm = _pcm(_tone(CHIME, 1.0), _tone(JONAS, 2.0), _gap(0.1), _tone(JONAS, 0.4))
    _run(s.handler, pcm)
    assert s.sent == ["Licht an"]
    assert s.engine.audio == [pcm]


def test_minority_speaker_blocked_by_require_match(tmp_path):
    s = _setup(tmp_path, require_match=True)
    pcm = _pcm(_tone(CHIME, 1.0), _tone(JONAS, 0.7), _gap(0.1), _tone(STRANGER, 2.3))
    _run(s.handler, pcm)
    assert s.sent == [""]
    assert s.engine.audio == []
    assert _known(s.published) == []


def test_minority_speaker_not_published_as_known(tmp_path):
    s = _setup(tmp_path)
    pcm = _pcm(_tone(CHIME, 1.0), _tone(JONAS, 0.7), _gap(0.1), _tone(STRANGER, 2.3))
    _run(s.handler, pcm)
    assert s.sent == ["Licht an"]
    assert s.engine.audio == [pcm]
    assert s.published and _known(s.published) == []


def test_majority_speaker_is_extracted_and_recognised(tmp_path):
    s = _setup(tmp_path, require_match=True)
    pcm = _pcm(_tone(CHIME, 1.0), _tone(JONAS, 2.5), _gap(0.1), _tone(STRANGER, 0.8))
    _run(s.handler, pcm)
    assert s.sent == ["Licht an"]
    assert s.engine.audio == [pcm]
    assert _known(s.published)[-1] == "Jonas"


def test_speech_in_the_chime_window_is_verified(tmp_path):
    s = _setup(tmp_path, require_match=True, extraction=False)
    pcm = _pcm(_tone(STRANGER, 0.9), _gap(0.1), _tone(JONAS, 1.5))
    _run(s.handler, pcm)
    assert s.sent == [""]
    assert s.engine.audio == []


def test_chime_window_speech_counts_toward_majority(tmp_path):
    s = _setup(tmp_path, require_match=True)
    pcm = _pcm(_tone(STRANGER, 1.9), _gap(0.1), _tone(JONAS, 1.5))
    _run(s.handler, pcm)
    assert s.sent == [""]
    assert _known(s.published) == []


def test_early_match_is_reverified(tmp_path):
    s = _setup(tmp_path, require_match=True)
    pcm = _pcm(_tone(CHIME, 1.0), _tone(JONAS, 0.7), _gap(0.1), _tone(STRANGER, 2.5))
    _run(s.handler, pcm, early_match=True)
    assert s.sent == [""]
    assert s.engine.audio == []


def test_fallback_and_shadows_get_the_full_audio(tmp_path):
    s = _setup(tmp_path, main_fails=True)
    pcm = _pcm(_tone(CHIME, 1.0), _tone(JONAS, 2.0), _gap(0.1), _tone(JONAS, 0.4))
    _run(s.handler, pcm)
    assert s.sent == ["Licht an"]
    assert s.fallback.audio == [pcm]
    assert s.shadow_audio == [pcm]


def test_no_auto_enroll_when_extraction_declined(tmp_path):
    s = _setup(tmp_path, require_match=True, auto_enroll=True)
    # Jonas's two 0.5 s words are too short to score, so he falls below
    # the majority; the stranger's 0.7 s is scored. The whole clip still
    # matches.
    pcm = _pcm(
        _tone(CHIME, 1.0),
        _tone(JONAS, 0.7), _gap(0.1),
        _tone(JONAS, 0.5), _gap(0.1),
        _tone(JONAS, 0.5), _gap(0.1),
        _tone(STRANGER, 0.7),
    )
    _run(s.handler, pcm)
    assert _known(s.published) == ["Jonas"]
    assert s.speakers.enrolled == []
