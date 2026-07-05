"""Tests for the OpenAI-compatible backend, fallback config and A/B shadow."""

from __future__ import annotations

import asyncio
from types import SimpleNamespace

import pytest

from murdock.config import Settings
from murdock.core.context import AppContext
from murdock.core.db import open_db
from murdock.core.recognition_log import RecognitionLog
from murdock.core.stt_backend import (
    OpenAICompatibleBackend,
    STTBackendError,
    VoxtralBackend,
)


def _ctx(tmp_path):
    db = open_db(tmp_path / "m.db")
    settings = SimpleNamespace(
        stt_backend="upstream",
        mistral_api_key="mk", mistral_model="voxtral-mini-latest",
        openai_base_url="https://api.openai.com",
        openai_api_key=None, openai_model="gpt-4o-transcribe",
        stt_local_fallback=False,
        shadow_stt_backend="none",
        shadow_upstream_uri=None,
        shadow_mistral_model="voxtral-small-latest",
        shadow_openai_base_url="", shadow_openai_api_key=None,
        shadow_openai_model="",
        upstream_uri="tcp://localhost:10300",
    )
    return AppContext(
        settings=settings, db=db, embedder=None, vad=None, speakers=None,
        unknown=None, ha=None, mqtt=None, recognition=None,
    )


# --- backend classes ----------------------------------------------------------


def test_openai_backend_label_and_base_url():
    b = OpenAICompatibleBackend(api_key="", model="whisper-large-v3-turbo",
                                base_url="https://api.groq.com/openai/")
    assert b.base_url == "https://api.groq.com/openai"
    assert b.label == "openai:whisper-large-v3-turbo"
    assert b.is_openrouter is False


def test_openrouter_detected_and_api_root_normalised():
    # Bare host → /api appended so /v1/audio/transcriptions resolves.
    b = OpenAICompatibleBackend(api_key="k", model="openai/whisper-large-v3-turbo",
                                base_url="https://openrouter.ai")
    assert b.is_openrouter is True
    assert b.base_url == "https://openrouter.ai/api"
    # Already-correct base stays untouched.
    b2 = OpenAICompatibleBackend(api_key="k", model="openai/whisper-large-v3-turbo",
                                 base_url="https://openrouter.ai/api/")
    assert b2.base_url == "https://openrouter.ai/api"


def test_openrouter_request_is_json_base64():
    import base64

    b = OpenAICompatibleBackend(api_key="k", model="openai/whisper-large-v3-turbo",
                                base_url="https://openrouter.ai")
    wav = b"RIFFfakewav"
    kwargs = b._request_kwargs(wav, "de")
    assert "files" not in kwargs
    payload = kwargs["json"]
    assert payload["model"] == "openai/whisper-large-v3-turbo"
    assert payload["input_audio"]["format"] == "wav"
    assert base64.b64decode(payload["input_audio"]["data"]) == wav


def test_standard_request_is_multipart():
    b = OpenAICompatibleBackend(api_key="k", model="gpt-4o-transcribe")
    kwargs = b._request_kwargs(b"RIFFfakewav", "de")
    assert "json" not in kwargs
    assert kwargs["files"]["file"][0] == "audio.wav"
    assert kwargs["data"] == {"model": "gpt-4o-transcribe", "language": "de"}


def test_voxtral_is_openai_compatible():
    b = VoxtralBackend(api_key="k", model="voxtral-small-latest")
    assert isinstance(b, OpenAICompatibleBackend)
    assert b.base_url == "https://api.mistral.ai"
    assert b.label == "voxtral:voxtral-small-latest"


def test_backend_raises_on_connection_failure():
    # Unroutable target → transcribe must raise, not return "".
    b = OpenAICompatibleBackend(
        api_key="", model="m", base_url="http://127.0.0.1:1", timeout=2.0
    )
    with pytest.raises(STTBackendError):
        asyncio.run(b.transcribe(b"\x00\x00" * 16000))


# --- context wiring -------------------------------------------------------------


def test_stt_backend_accepts_openai(tmp_path):
    ctx = _ctx(tmp_path)
    ctx.set_stt_backend("openai")
    assert ctx.get_stt_backend() == "openai"
    with pytest.raises(ValueError):
        ctx.set_stt_backend("bogus")


def test_active_cloud_backend_dispatch(tmp_path):
    ctx = _ctx(tmp_path)
    assert ctx.get_active_cloud_backend() is None  # upstream
    ctx.set_stt_backend("voxtral")
    assert isinstance(ctx.get_active_cloud_backend(), VoxtralBackend)
    ctx.set_stt_backend("openai")
    backend = ctx.get_active_cloud_backend()
    assert isinstance(backend, OpenAICompatibleBackend)
    assert backend.model == "gpt-4o-transcribe"


def test_openai_backend_requires_model(tmp_path):
    ctx = _ctx(tmp_path)
    # No model anywhere (empty settings default, no override) → no backend.
    ctx.settings.openai_model = ""
    assert ctx.get_openai_backend() is None
    ctx.set_openai_model("whisper-large-v3-turbo")
    ctx.set_openai_base_url("https://api.groq.com/openai")
    b = ctx.get_openai_backend()
    assert b.base_url == "https://api.groq.com/openai"


def test_local_fallback_round_trip(tmp_path):
    ctx = _ctx(tmp_path)
    assert ctx.get_stt_local_fallback() is False
    ctx.set_stt_local_fallback(True)
    assert ctx.get_stt_local_fallback() is True


# --- shadow engine ---------------------------------------------------------------


def test_shadow_backend_validation(tmp_path):
    ctx = _ctx(tmp_path)
    assert ctx.get_shadow_stt_backend() == "none"
    ctx.set_shadow_stt_backend("voxtral")
    assert ctx.get_shadow_stt_backend() == "voxtral"
    with pytest.raises(ValueError):
        ctx.set_shadow_stt_backend("bogus")


def test_shadow_voxtral_uses_own_model_and_main_key(tmp_path):
    ctx = _ctx(tmp_path)
    ctx.set_shadow_stt_backend("voxtral")
    b = ctx.get_shadow_backend()
    assert isinstance(b, VoxtralBackend)
    assert b.model == "voxtral-small-latest"
    assert b.api_key == "mk"  # primary Mistral key


def test_shadow_openai_key_falls_back_to_primary(tmp_path):
    ctx = _ctx(tmp_path)
    ctx.set_openai_api_key("primary-key")
    ctx.set_shadow_stt_backend("openai")
    ctx.set_shadow_openai_model("whisper-large-v3-turbo")
    b = ctx.get_shadow_backend()
    assert b.api_key == "primary-key"
    ctx.set_shadow_openai_api_key("own-key")
    assert ctx.get_shadow_backend().api_key == "own-key"


def test_shadow_upstream_uri_normalised(tmp_path):
    ctx = _ctx(tmp_path)
    ctx.set_shadow_upstream_uri("host:10301")
    assert ctx.get_shadow_upstream_uri() == "tcp://host:10301"


# --- recognition log shadow column -----------------------------------------------


def test_set_shadow_round_trip(tmp_path):
    conn = open_db(tmp_path / "m.db")
    log = RecognitionLog(conn)
    event_id = log.record(
        session_id="s1", satellite_id=None, duration_sec=2.0,
        outcome="match", transcript="licht an",
    )
    assert event_id > 0
    assert log.set_shadow(event_id, "licht an bitte", "openai:whisper") is True
    ev = log.list_events(limit=1)[0]
    assert ev.shadow_transcript == "licht an bitte"
    assert ev.shadow_engine == "openai:whisper"
    # Unknown id → False.
    assert log.set_shadow(999999, "x", "y") is False


def test_defaults():
    s = Settings()
    assert s.stt_local_fallback is False
    assert s.shadow_stt_backend == "none"
    assert s.openai_model == "gpt-4o-transcribe"
