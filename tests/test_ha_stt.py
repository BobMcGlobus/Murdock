"""Transcribing through a Home Assistant speech-to-text entity.

Home Assistant Cloud's transcription is not a Wyoming service and has no
API of its own; the only way in is the `/api/stt/{entity}` endpoint Home
Assistant exposes for its own frontend.
"""

from __future__ import annotations

import asyncio
from types import SimpleNamespace

import pytest

from murdock.config import Settings
from murdock.core.context import AppContext
from murdock.core.db import open_db
from murdock.core.stt_backend import HomeAssistantSTTBackend, STTBackendError


def _backend(**kw):
    base = dict(base_url="http://ha:8123", token="tok",
                entity_id="stt.home_assistant_cloud")
    base.update(kw)
    return HomeAssistantSTTBackend(**base)


def test_the_header_carries_every_field_home_assistant_demands():
    """It answers 400 if a single one is missing."""
    h = _backend()._headers("de", 16000, 2, 1)["X-Speech-Content"]
    fields = dict(p.strip().split("=", 1) for p in h.split(";"))
    assert set(fields) == {
        "format", "codec", "sample_rate", "bit_rate", "channel", "language",
    }
    assert fields["format"] == "wav" and fields["codec"] == "pcm"
    assert fields["sample_rate"] == "16000"
    assert fields["bit_rate"] == "16"      # bit depth, from the sample width
    assert fields["channel"] == "1"


def test_a_bare_language_is_grown_into_a_locale():
    """Home Assistant wants de-DE, and Murdock stores plain `de`."""
    def lang(tag):
        h = _backend()._headers(tag, 16000, 2, 1)["X-Speech-Content"]
        return dict(p.strip().split("=", 1) for p in h.split(";"))["language"]

    assert lang("de") == "de-DE"
    assert lang("de-DE") == "de-DE"     # a full tag is left alone
    assert lang(None) == "en-US"        # never sent empty


def test_it_is_not_configured_without_all_three_parts():
    assert _backend().configured
    assert not _backend(token="").configured
    assert not _backend(base_url="").configured
    assert not _backend(entity_id="").configured


def _ctx(tmp_path, entity, ha_ok=True):
    ctx = AppContext(
        settings=Settings(), db=open_db(tmp_path / "m.db"), embedder=None,
        vad=None, speakers=None, unknown=None,
        ha=SimpleNamespace(base_url="http://ha:8123" if ha_ok else "",
                           token="tok" if ha_ok else ""),
        mqtt=None, recognition=None,
    )
    ctx.set_ha_stt_entity(entity)
    return ctx


def test_pointing_it_at_murdock_itself_is_refused(tmp_path):
    """Murdock is an STT provider in HA; this would call itself forever."""
    assert _ctx(tmp_path, "stt.murdock").get_ha_stt_backend() is None
    assert _ctx(tmp_path, "stt.Murdock_Proxy").get_ha_stt_backend() is None
    # A real entity still works.
    assert _ctx(tmp_path, "stt.home_assistant_cloud").get_ha_stt_backend()


def test_no_backend_without_home_assistant_credentials(tmp_path):
    assert _ctx(tmp_path, "stt.home_assistant_cloud", ha_ok=False) \
        .get_ha_stt_backend() is None
    assert _ctx(tmp_path, "").get_ha_stt_backend() is None


def test_it_is_selectable_as_a_backend(tmp_path):
    ctx = _ctx(tmp_path, "stt.home_assistant_cloud")
    assert "ha" in ctx.STT_BACKENDS
    ctx.set_stt_backend("ha")
    assert ctx.get_active_cloud_backend() is not None


def test_a_refusal_is_not_mistaken_for_silence(monkeypatch):
    """HA answers 200 with result="error" — that is a failure, not "".

    Treating it as an empty transcript would hide a broken Cloud
    subscription behind "I didn't catch that".
    """
    import murdock.core.stt_backend as mod

    class _Resp:
        status_code = 200
        text = ""

        async def aread(self):
            pass

        async def aclose(self):
            pass

        def json(self):
            return {"text": "", "result": "error"}

    class _Client:
        def __init__(self, **kw):
            pass

        async def __aenter__(self):
            return self

        async def __aexit__(self, *a):
            return False

        def build_request(self, *a, **kw):
            return object()

        async def send(self, request, stream=False):
            return _Resp()

    monkeypatch.setattr(mod.httpx, "AsyncClient", _Client)
    with pytest.raises(STTBackendError):
        asyncio.run(_backend().transcribe(b"\x00" * 3200))
