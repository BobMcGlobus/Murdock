"""The /api/stt-services endpoints and the backup of services."""

from __future__ import annotations

import asyncio
import io
import json
import zipfile
from types import SimpleNamespace

import pytest
from fastapi import HTTPException

from murdock.api import routes_stt_services as routes
from murdock.config import Settings
from murdock.core.context import AppContext
from murdock.core.db import open_db


def _ctx(tmp_path, name="m.db"):
    return AppContext(
        settings=Settings(), db=open_db(tmp_path / name), embedder=None,
        vad=None, speakers=None, unknown=None,
        ha=SimpleNamespace(base_url="http://ha:8123", token="tok", configured=True),
        mqtt=None, recognition=None,
    )


def _run(coro):
    return asyncio.run(coro)


def test_a_key_never_comes_back_out(tmp_path):
    ctx = _ctx(tmp_path)
    out = _run(routes.create_service(routes.ServiceIn(
        name="MAI", kind="openai",
        config={"base_url": "https://openrouter.ai", "api_key": "sk-secret",
                "model": "microsoft/mai-transcribe-2"},
    ), ctx))
    assert "sk-secret" not in out.model_dump_json()
    [svc] = out.services
    assert svc.secrets_set == ["api_key"]
    assert svc.remote is True and svc.complete is True
    # The first service becomes the main by itself.
    assert out.roles.main == svc.id


def test_saving_the_form_with_an_empty_key_keeps_the_stored_one(tmp_path):
    ctx = _ctx(tmp_path)
    svc = ctx.stt_services.create("Voxtral", "voxtral", {"api_key": "k1"})
    _run(routes.update_service(
        svc.id, routes.ServicePatch(name="Voxtral small", config={"api_key": "", "model": "voxtral-small-latest"}), ctx,
    ))
    stored = ctx.stt_services.get(svc.id)
    assert stored.config["api_key"] == "k1"
    assert stored.name == "Voxtral small"


def test_the_main_cannot_be_deleted(tmp_path):
    ctx = _ctx(tmp_path)
    main = ctx.stt_services.create("Kroko", "wyoming", {"uri": "kroko:10300"})
    other = ctx.stt_services.create("Voxtral", "voxtral", {"api_key": "k"})
    with pytest.raises(HTTPException) as err:
        _run(routes.delete_service(main.id, ctx))
    assert err.value.status_code == 409
    out = _run(routes.delete_service(other.id, ctx))
    assert [s.id for s in out.services] == [main.id]


def test_roles_are_stored_in_order_and_cleaned(tmp_path):
    ctx = _ctx(tmp_path)
    a = ctx.stt_services.create("A", "wyoming", {"uri": "a:1"})
    b = ctx.stt_services.create("B", "voxtral", {"api_key": "k"})
    c = ctx.stt_services.create("C", "wyoming", {"uri": "c:1"})
    out = _run(routes.set_roles(routes.RolesModel(
        main=b.id, fallbacks=[c.id, b.id, a.id, 999], shadows=[a.id, b.id],
        fallback_on_empty=False,
    ), ctx))
    assert out.roles.main == b.id
    assert out.roles.fallbacks == [c.id, a.id]
    assert out.roles.shadows == [a.id]
    assert out.roles.fallback_on_empty is False


def test_a_wyoming_main_becomes_the_upstream(tmp_path):
    ctx = _ctx(tmp_path)
    a = ctx.stt_services.create("A", "voxtral", {"api_key": "k"})
    b = ctx.stt_services.create("Kroko", "wyoming", {"uri": "192.168.2.25:10303"})
    _run(routes.set_roles(routes.RolesModel(main=b.id), ctx))
    assert ctx.get_stt_backend() == "upstream"
    assert ctx.get_upstream_uri() == "tcp://192.168.2.25:10303"
    _run(routes.set_roles(routes.RolesModel(main=a.id), ctx))
    assert ctx.get_stt_backend() == "voxtral"


def test_an_incomplete_service_is_reported_before_anything_is_sent(tmp_path):
    ctx = _ctx(tmp_path)
    out = _run(routes.test_service(routes.TestIn(kind="openai", config={"model": ""}), ctx))
    assert out.ok is False and "incomplete" in out.error


def test_an_unreachable_wyoming_server_fails_the_test(tmp_path):
    ctx = _ctx(tmp_path)
    out = _run(routes.test_service(
        routes.TestIn(kind="wyoming", config={"uri": "127.0.0.1:1"}), ctx,
    ))
    assert out.ok is False and out.error


def test_services_survive_a_backup_round_trip(tmp_path):
    from murdock.api.routes_backup import dump_settings

    src = _ctx(tmp_path, "a.db")
    kroko = src.stt_services.create("Kroko", "wyoming", {"uri": "kroko:10300"})
    mai = src.stt_services.create("MAI", "openai", {"api_key": "sk", "model": "m"})
    roles = src.stt_services.roles()
    roles.shadows = [mai.id]
    src.stt_services.set_roles(roles)

    dst = _ctx(tmp_path, "b.db")
    dst.stt_services.create("Something else", "voxtral", {"api_key": "x"})
    from murdock.api.routes_backup import apply_settings

    apply_settings(dst.db, dump_settings(src.db))
    dst.stt_services.replace_all(json.loads(json.dumps(src.stt_services.dump())))

    assert [(s.id, s.name) for s in dst.stt_services.list()] == [(kroko.id, "Kroko"), (mai.id, "MAI")]
    assert dst.stt_services.get(mai.id).config["api_key"] == "sk"
    assert dst.get_main_service().name == "Kroko"
    assert [s.name for s in dst.get_shadow_services()] == ["MAI"]


def test_the_backup_archive_carries_the_services(tmp_path):
    from murdock.api.routes_backup import export_backup

    ctx = _ctx(tmp_path)
    ctx.speakers = SimpleNamespace(list_speakers=lambda: [])
    ctx.stt_services.create("Kroko", "wyoming", {"uri": "kroko:10300"})

    async def read():
        resp = await export_backup(ctx)
        chunks = [c async for c in resp.body_iterator]
        return b"".join(c if isinstance(c, bytes) else c.encode() for c in chunks)

    zf = zipfile.ZipFile(io.BytesIO(_run(read())))
    [row] = json.loads(zf.read("stt_services.json"))
    assert row["name"] == "Kroko" and row["config"]["uri"] == "tcp://kroko:10300"
