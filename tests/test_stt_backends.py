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
        shadow_mistral_api_key=None,
        shadow_openai_base_url="", shadow_openai_api_key=None,
        shadow_openai_model="",
        upstream_uri="tcp://localhost:10300",
        enable_stt_vocabulary=False, stt_vocabulary="",
        enable_stt_dictionary=False, stt_dictionary="",
        enable_dual_transcript=False,
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
    assert kwargs["data"] == {
        "model": "gpt-4o-transcribe", "language": "de", "temperature": "0",
    }


def test_language_reaches_both_request_shapes():
    """HA sends the language; neither shape may drop it.

    Regression: the OpenRouter branch built its JSON body without the
    hint, so the primary engine re-detected the language on every
    utterance while the multipart branch honoured it.
    """
    orb = OpenAICompatibleBackend(api_key="k", model="openai/whisper-large-v3-turbo",
                                  base_url="https://openrouter.ai")
    assert orb._request_kwargs(b"RIFF", "de")["json"]["language"] == "de"
    std = OpenAICompatibleBackend(api_key="k", model="gpt-4o-transcribe")
    assert std._request_kwargs(b"RIFF", "de")["data"]["language"] == "de"
    # No hint from HA — the field stays absent rather than being sent empty.
    assert "language" not in orb._request_kwargs(b"RIFF", None)["json"]
    assert "language" not in std._request_kwargs(b"RIFF", None)["data"]


def test_temperature_is_pinned_to_zero():
    """Whisper's temperature-fallback cascade is the multi-second tail.

    Without an explicit 0 the endpoint re-decodes the whole clip up to
    six times when its own quality thresholds fail.
    """
    orb = OpenAICompatibleBackend(api_key="k", model="openai/whisper-large-v3-turbo",
                                  base_url="https://openrouter.ai")
    assert orb._request_kwargs(b"RIFF", "de")["json"]["temperature"] == 0
    std = OpenAICompatibleBackend(api_key="k", model="gpt-4o-transcribe")
    # Multipart form fields must be strings, not floats.
    assert std._request_kwargs(b"RIFF", "de")["data"]["temperature"] == "0"


def test_vocabulary_prompt_sent_where_supported():
    b = OpenAICompatibleBackend(api_key="k", model="whisper-large-v3-turbo",
                                prompt="Fehenlichter, Bed-Lightstrip")
    kwargs = b._request_kwargs(b"RIFF", None)
    assert kwargs["data"]["prompt"] == "Fehenlichter, Bed-Lightstrip"


def test_vocabulary_prompt_skipped_for_voxtral_and_openrouter():
    v = VoxtralBackend(api_key="k")
    v.prompt = "Fehenlichter"
    assert "prompt" not in v._request_kwargs(b"RIFF", None)["data"]
    orb = OpenAICompatibleBackend(api_key="k", model="openai/whisper-large-v3-turbo",
                                  base_url="https://openrouter.ai",
                                  prompt="Fehenlichter")
    assert "prompt" not in orb._request_kwargs(b"RIFF", None)["json"]


def test_context_injects_vocabulary_into_openai_backend(tmp_path):
    ctx = _ctx(tmp_path)
    ctx.set_openai_model("whisper-large-v3-turbo")
    assert ctx.get_openai_backend().prompt is None
    ctx.set_enable_stt_vocabulary(True)
    ctx.set_stt_vocabulary("Fehenlichter, Sat1")
    assert ctx.get_openai_backend().prompt == "Fehenlichter, Sat1"
    # Toggle off → prompt gone even though the text stays stored.
    ctx.set_enable_stt_vocabulary(False)
    assert ctx.get_openai_backend().prompt is None


def test_dual_transcript_needs_shadow(tmp_path):
    ctx = _ctx(tmp_path)
    ctx.set_enable_dual_transcript(True)
    assert ctx.dual_transcript_active() is False  # no shadow engine
    ctx.set_shadow_stt_backend("voxtral")
    assert ctx.dual_transcript_active() is True
    ctx.set_enable_dual_transcript(False)
    assert ctx.dual_transcript_active() is False


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
    assert b.api_key == "mk"  # primary Mistral key (fallback)


def test_shadow_voxtral_own_key_wins(tmp_path):
    ctx = _ctx(tmp_path)
    ctx.set_shadow_stt_backend("voxtral")
    assert ctx.has_shadow_mistral_api_key() is False
    ctx.set_shadow_mistral_api_key("shadow-mk")
    assert ctx.has_shadow_mistral_api_key() is True
    assert ctx.get_shadow_backend().api_key == "shadow-mk"
    # Clearing falls back to the primary key again.
    ctx.set_shadow_mistral_api_key("")
    assert ctx.get_shadow_backend().api_key == "mk"


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
