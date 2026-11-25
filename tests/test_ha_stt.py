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
from murdock.core.stt_services import SttRoles


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
    # xx-XX is wrong for some: there is no en-EN, and Cloud answers 415.
    assert lang("en") == "en-US"
    assert lang("ja") == "ja-JP"
    assert lang("sv") == "sv-SE"


def test_the_locale_is_chosen_from_what_the_entity_lists():
    from murdock.core.stt_backend import _pick_locale

    cloud = ["en-US", "en-GB", "de-DE", "de-CH"]
    assert _pick_locale("en", cloud) == "en-US"
    assert _pick_locale("en-GB", cloud) == "en-GB"     # asked for, so kept
    assert _pick_locale("en_gb", cloud) == "en-GB"     # spelling is loose
    assert _pick_locale("de", cloud) == "de-DE"
    # No en-US on offer: another English beats a guaranteed 415.
    assert _pick_locale("en", ["en-GB", "en-AU"]) == "en-GB"
    # An entity that lists bare codes gets the bare code.
    assert _pick_locale("de", ["en", "de"]) == "de"
    # Sent exactly as the entity spells it.
    assert _pick_locale("en", ["en_US"]) == "en_US"
    # Nothing to choose from, or nothing that fits: the guess.
    assert _pick_locale("en", []) == "en-US"
    assert _pick_locale("fr", ["en-US"]) == "fr-FR"


class _FakeHA:
    """Home Assistant's /api/stt: GET lists capabilities, POST transcribes."""

    def __init__(self, languages, get_fails=False):
        self.languages = languages
        self.get_fails = get_fails
        self.gets = 0
        self.sent_languages = []

    def client_class(self):
        ha = self

        class _Resp:
            def __init__(self, data):
                self.status_code = 200
                self.text = ""
                self._data = data

            async def aread(self):
                pass

            async def aclose(self):
                pass

            def json(self):
                return self._data

        class _Client:
            def __init__(self, **kw):
                pass

            async def __aenter__(self):
                return self

            async def __aexit__(self, *a):
                return False

            async def get(self, url, headers=None):
                ha.gets += 1
                if ha.get_fails:
                    raise OSError("refused")
                return _Resp({"languages": ha.languages})

            def build_request(self, method, url, headers=None, content=None):
                header = headers["X-Speech-Content"]
                fields = dict(p.strip().split("=", 1) for p in header.split(";"))
                ha.sent_languages.append(fields["language"])
                return object()

            async def send(self, request, stream=False):
                return _Resp({"text": "Licht an", "result": "success"})

        return _Client


def test_cloud_gets_a_locale_it_lists_for_a_bare_code(monkeypatch):
    """Issue #2: pipeline language `en`, Cloud lists en-US/en-GB — Murdock
    sent en-EN and every request came back 415."""
    import murdock.core.stt_backend as mod

    ha = _FakeHA(["en-US", "en-GB"])
    monkeypatch.setattr(mod, "_HA_LANGUAGES", {})
    monkeypatch.setattr(mod.httpx, "AsyncClient", ha.client_class())
    backend = _backend()
    assert asyncio.run(backend.transcribe(b"\x00" * 3200, language="en")) == "Licht an"
    assert asyncio.run(backend.transcribe(b"\x00" * 3200, language="en")) == "Licht an"
    assert ha.sent_languages == ["en-US", "en-US"]
    assert ha.gets == 1     # the list is fetched once, not per utterance


def test_without_the_list_the_guess_is_sent(monkeypatch):
    import murdock.core.stt_backend as mod

    ha = _FakeHA([], get_fails=True)
    monkeypatch.setattr(mod, "_HA_LANGUAGES", {})
    monkeypatch.setattr(mod.httpx, "AsyncClient", ha.client_class())
    assert asyncio.run(_backend().transcribe(b"\x00" * 3200, language="de")) == "Licht an"
    assert ha.sent_languages == ["de-DE"]


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
    ctx.stt_services.create("HA Cloud", "ha", {"entity_id": entity})
    return ctx


def _built(ctx):
    [svc] = ctx.stt_services.list()
    return ctx.build_stt_backend(svc)


def test_pointing_it_at_murdock_itself_is_refused(tmp_path):
    """Murdock is an STT provider in HA; this would call itself forever."""
    assert _built(_ctx(tmp_path / "a", "stt.murdock")) is None
    assert _built(_ctx(tmp_path / "b", "stt.Murdock_Proxy")) is None
    # A real entity still works.
    assert _built(_ctx(tmp_path / "c", "stt.home_assistant_cloud"))


def test_no_backend_without_home_assistant_credentials(tmp_path):
    assert _built(_ctx(tmp_path / "a", "stt.home_assistant_cloud", ha_ok=False)) is None
    assert _built(_ctx(tmp_path / "b", "")) is None


def test_it_is_selectable_as_a_backend(tmp_path):
    ctx = _ctx(tmp_path, "stt.home_assistant_cloud")
    [svc] = ctx.stt_services.list()
    ctx.stt_services.set_roles(SttRoles(main=svc.id, fallbacks=[], shadows=[]))
    assert ctx.get_stt_backend() == "ha"
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


def test_the_probe_says_which_part_is_missing(tmp_path):
    """'It doesn't work' has three causes; the test must name one."""
    import asyncio

    from murdock.api.routes_stt_services import TestIn, test_service

    out = asyncio.run(test_service(TestIn(kind="ha", config={"entity_id": ""}), _ctx(tmp_path, "")))
    assert out.ok is False and "incomplete" in (out.error or "")


def test_a_language_mismatch_is_named_rather_than_a_bare_415(monkeypatch, tmp_path):
    """Home Assistant answers an unsupported language with 415 and no reason."""
    import asyncio

    import murdock.core.stt_backend as mod
    from murdock.api.routes_stt_services import TestIn, test_service

    async def _caps(self):
        return {"languages": ["en-US", "en-GB"], "formats": ["wav"]}

    monkeypatch.setattr(mod.HomeAssistantSTTBackend, "capabilities", _caps)
    ctx = _ctx(tmp_path, "stt.home_assistant_cloud")
    ctx.set_stt_language("de")
    [svc] = ctx.stt_services.list()
    probe = TestIn(id=svc.id)
    out = asyncio.run(test_service(probe, ctx))
    assert out.ok is True
    assert out.language_ok is False
    assert "en-US" in out.languages

    # A supported language compares on the primary subtag: HA lists
    # locales, Murdock stores a bare code.
    async def _caps_de(self):
        return {"languages": ["de-DE", "en-US"], "formats": ["wav"]}

    monkeypatch.setattr(mod.HomeAssistantSTTBackend, "capabilities", _caps_de)
    assert asyncio.run(test_service(probe, ctx)).language_ok is True
