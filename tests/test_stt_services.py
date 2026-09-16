"""Speech-to-text services: storage, roles, and the one-time migration."""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from murdock.config import Settings
from murdock.core.context import AppContext
from murdock.core.db import open_db, set_setting
from murdock.core.stt_services import (
    KIND_HA,
    KIND_OPENAI,
    KIND_VOXTRAL,
    KIND_WYOMING,
    SttRoles,
    SttService,
    SttServiceStore,
    _host_is_local,
    migrate_legacy_settings,
)


def _store(tmp_path):
    return SttServiceStore(open_db(tmp_path / "m.db"))


def _ctx(tmp_path, **settings):
    return AppContext(
        settings=Settings(**settings), db=open_db(tmp_path / "m.db"),
        embedder=None, vad=None, speakers=None, unknown=None,
        ha=SimpleNamespace(base_url="http://192.168.2.10:8123", token="tok",
                           configured=True),
        mqtt=None, recognition=None,
    )


# --- store --------------------------------------------------------------------


def test_the_first_service_becomes_main(tmp_path):
    store = _store(tmp_path)
    kroko = store.create("Kroko", KIND_WYOMING, {"uri": "192.168.2.25:10303"})
    assert store.roles().main == kroko.id
    # A bare host:port grows its scheme, the same as the old URI field.
    assert kroko.config["uri"] == "tcp://192.168.2.25:10303"
    # A second one does not steal the role.
    store.create("MAI", KIND_OPENAI, {"model": "microsoft/mai-transcribe-2"})
    assert store.roles().main == kroko.id


def test_unknown_config_keys_are_dropped(tmp_path):
    store = _store(tmp_path)
    s = store.create("x", KIND_WYOMING, {"uri": "h:1", "api_key": "leak", "junk": 1})
    assert set(s.config) == {"uri"}


def test_credentials_never_leave_through_public(tmp_path):
    store = _store(tmp_path)
    s = store.create("MAI", KIND_OPENAI, {
        "base_url": "https://openrouter.ai", "api_key": "sk-secret",
        "model": "microsoft/mai-transcribe-2",
    })
    out = s.public()
    assert "api_key" not in out["config"]
    assert "sk-secret" not in repr(out)
    assert out["secrets_set"] == ["api_key"]


def test_an_empty_secret_on_update_keeps_the_stored_one(tmp_path):
    """A form that never shows the key must still be savable."""
    store = _store(tmp_path)
    s = store.create("MAI", KIND_OPENAI, {"api_key": "sk-secret", "model": "m"})
    updated = store.update(s.id, config={"api_key": "", "model": "m2"})
    assert updated.config["api_key"] == "sk-secret"
    assert updated.config["model"] == "m2"


def test_the_main_cannot_be_deleted(tmp_path):
    store = _store(tmp_path)
    main = store.create("Kroko", KIND_WYOMING, {"uri": "h:1"})
    with pytest.raises(ValueError):
        store.delete(main.id)


def test_deleting_a_service_removes_it_from_every_role(tmp_path):
    store = _store(tmp_path)
    main = store.create("Kroko", KIND_WYOMING, {"uri": "h:1"})
    other = store.create("Voxtral", KIND_VOXTRAL, {"api_key": "k"})
    store.set_roles(SttRoles(main=main.id, fallbacks=[other.id], shadows=[other.id]))
    store.delete(other.id)
    roles = store.roles()
    assert roles.fallbacks == [] and roles.shadows == []


# --- roles ----------------------------------------------------------------------


def test_the_main_is_never_its_own_fallback_or_shadow(tmp_path):
    store = _store(tmp_path)
    a = store.create("A", KIND_WYOMING, {"uri": "h:1"})
    b = store.create("B", KIND_WYOMING, {"uri": "h:2"})
    roles = store.set_roles(SttRoles(main=a.id, fallbacks=[a.id, b.id], shadows=[a.id, b.id]))
    assert roles.fallbacks == [b.id]
    assert roles.shadows == [b.id]


def test_the_fallback_order_is_kept_and_deduplicated(tmp_path):
    store = _store(tmp_path)
    a, b, c, d = (store.create(n, KIND_WYOMING, {"uri": f"h:{i}"})
                  for i, n in enumerate("ABCD"))
    roles = store.set_roles(SttRoles(main=a.id, fallbacks=[d.id, b.id, d.id, c.id]))
    assert roles.fallbacks == [d.id, b.id, c.id]


def test_ids_that_do_not_exist_are_dropped(tmp_path):
    store = _store(tmp_path)
    a = store.create("A", KIND_WYOMING, {"uri": "h:1"})
    roles = store.set_roles(SttRoles(main=a.id, fallbacks=[999], shadows=[1000]))
    assert roles.fallbacks == [] and roles.shadows == []


def test_fallback_on_empty_defaults_on(tmp_path):
    assert _store(tmp_path).roles().fallback_on_empty is True


# --- remote or local ------------------------------------------------------------


@pytest.mark.parametrize("host, local", [
    ("192.168.2.25", True),
    ("10.0.0.5", True),
    ("127.0.0.1", True),
    ("localhost", True),
    ("core-whisper", True),             # container / single-label name
    ("homeassistant.local", True),
    ("host.docker.internal", True),
    ("api.mistral.ai", False),
    ("openrouter.ai", False),
    ("8.8.8.8", False),
    ("", True),                          # unknown never gets skipped
])
def test_host_classification(host, local):
    assert _host_is_local(host) is local


def test_service_remoteness_by_kind():
    assert SttService(1, "k", KIND_WYOMING, {"uri": "tcp://192.168.2.25:10303"}).is_remote() is False
    assert SttService(2, "m", KIND_OPENAI, {"base_url": "https://openrouter.ai"}).is_remote() is True
    assert SttService(3, "p", KIND_OPENAI, {"base_url": "http://192.168.2.25:5092"}).is_remote() is False
    assert SttService(4, "v", KIND_VOXTRAL, {}).is_remote() is True
    assert SttService(5, "h", KIND_HA, {}).is_remote(ha_base_url="http://192.168.2.10:8123") is False


# --- migration ------------------------------------------------------------------


def test_an_upstream_install_becomes_one_wyoming_main(tmp_path):
    ctx = _ctx(tmp_path)
    set_setting(ctx.db, "upstream_uri", "tcp://192.168.2.25:10303")
    assert migrate_legacy_settings(ctx.stt_services, ctx.db, ctx.settings)
    main = ctx.get_main_service()
    assert main.kind == KIND_WYOMING
    assert main.config["uri"] == "tcp://192.168.2.25:10303"
    assert ctx.get_stt_backend() == "upstream"


def test_a_cloud_main_keeps_its_credentials(tmp_path):
    ctx = _ctx(tmp_path)
    set_setting(ctx.db, "stt_backend", "openai")
    set_setting(ctx.db, "openai_base_url", "https://openrouter.ai")
    set_setting(ctx.db, "openai_api_key", "sk-secret")
    set_setting(ctx.db, "openai_model", "microsoft/mai-transcribe-2")
    migrate_legacy_settings(ctx.stt_services, ctx.db, ctx.settings)
    main = ctx.get_main_service()
    assert main.kind == KIND_OPENAI
    assert main.config == {
        "base_url": "https://openrouter.ai", "api_key": "sk-secret",
        "model": "microsoft/mai-transcribe-2",
    }
    assert ctx.get_stt_backend() == "openai"


def test_the_local_fallback_becomes_a_wyoming_fallback(tmp_path):
    ctx = _ctx(tmp_path)
    set_setting(ctx.db, "stt_backend", "voxtral")
    set_setting(ctx.db, "mistral_api_key", "k")
    set_setting(ctx.db, "stt_local_fallback", "true")
    set_setting(ctx.db, "upstream_uri", "tcp://192.168.2.25:10303")
    migrate_legacy_settings(ctx.stt_services, ctx.db, ctx.settings)
    [fb] = ctx.get_fallback_services()
    assert fb.kind == KIND_WYOMING and fb.config["uri"] == "tcp://192.168.2.25:10303"


def test_a_shadow_becomes_a_shadow_service(tmp_path):
    """The shape of the reporter's setup: HA main, Wyoming shadow, rescue off."""
    ctx = _ctx(tmp_path)
    set_setting(ctx.db, "stt_backend", "ha")
    set_setting(ctx.db, "ha_stt_entity", "stt.home_assistant_cloud")
    set_setting(ctx.db, "shadow_stt_backend", "upstream")
    set_setting(ctx.db, "shadow_upstream_uri", "tcp://192.168.2.25:10303")
    set_setting(ctx.db, "shadow_rescues_empty", "false")
    migrate_legacy_settings(ctx.stt_services, ctx.db, ctx.settings)

    assert ctx.get_main_service().kind == KIND_HA
    [shadow] = ctx.get_shadow_services()
    assert shadow.config["uri"] == "tcp://192.168.2.25:10303"
    # Rescue was off, so the shadow does not also become a fallback.
    assert ctx.get_fallback_services() == []
    assert ctx.get_fallback_on_empty() is False


def test_rescue_on_makes_the_shadow_a_fallback_too(tmp_path):
    """'Let the shadow answer when the primary heard nothing' is what a
    fallback on empty does now."""
    ctx = _ctx(tmp_path)
    set_setting(ctx.db, "upstream_uri", "tcp://192.168.2.25:10303")
    set_setting(ctx.db, "shadow_stt_backend", "voxtral")
    set_setting(ctx.db, "mistral_api_key", "k")
    set_setting(ctx.db, "shadow_rescues_empty", "true")
    migrate_legacy_settings(ctx.stt_services, ctx.db, ctx.settings)
    [fb] = ctx.get_fallback_services()
    assert fb.kind == KIND_VOXTRAL
    # The shadow key falls back to the main key, as the old getter did.
    assert fb.config["api_key"] == "k"


def test_a_cleared_upstream_field_means_the_environment_default(tmp_path):
    """Clearing the field stored "", which the old getter read as "use
    UPSTREAM_URI". Taking it literally migrated such installs to no main
    service at all."""
    ctx = _ctx(tmp_path)
    ctx.settings.upstream_uri = "tcp://core-whisper:10300"
    set_setting(ctx.db, "upstream_uri", "")
    set_setting(ctx.db, "openai_model", "")
    migrate_legacy_settings(ctx.stt_services, ctx.db, ctx.settings)
    main = ctx.get_main_service()
    assert main is not None and main.config["uri"] == "tcp://core-whisper:10300"


def test_a_cleared_key_stays_cleared(tmp_path):
    """An emptied key was deliberate; the environment's must not return."""
    ctx = _ctx(tmp_path)
    ctx.settings.mistral_api_key = "env-key"
    set_setting(ctx.db, "stt_backend", "voxtral")
    set_setting(ctx.db, "mistral_api_key", "")
    migrate_legacy_settings(ctx.stt_services, ctx.db, ctx.settings)
    assert ctx.get_main_service().config.get("api_key", "") == ""


def test_migration_runs_once_and_never_overwrites(tmp_path):
    ctx = _ctx(tmp_path)
    set_setting(ctx.db, "upstream_uri", "tcp://192.168.2.25:10303")
    assert migrate_legacy_settings(ctx.stt_services, ctx.db, ctx.settings) is True
    ctx.stt_services.create("MAI", KIND_OPENAI, {"model": "m"})
    before = [s.id for s in ctx.stt_services.list()]
    assert migrate_legacy_settings(ctx.stt_services, ctx.db, ctx.settings) is False
    assert [s.id for s in ctx.stt_services.list()] == before


def test_the_same_wyoming_server_is_created_once(tmp_path):
    """Main and fallback pointing at one server are one service, not two."""
    ctx = _ctx(tmp_path)
    set_setting(ctx.db, "upstream_uri", "tcp://192.168.2.25:10303")
    set_setting(ctx.db, "shadow_stt_backend", "upstream")
    set_setting(ctx.db, "shadow_upstream_uri", "tcp://192.168.2.25:10303")
    migrate_legacy_settings(ctx.stt_services, ctx.db, ctx.settings)
    assert len(ctx.stt_services.list()) == 1


# --- backends built from services -------------------------------------------------


def test_each_backend_carries_the_service_name(tmp_path):
    ctx = _ctx(tmp_path)
    store = ctx.stt_services
    for name, kind, cfg in [
        ("Kroko", KIND_WYOMING, {"uri": "h:1"}),
        ("MAI", KIND_OPENAI, {"model": "m", "base_url": "https://openrouter.ai"}),
        ("Voxtral", KIND_VOXTRAL, {"api_key": "k"}),
        ("Cloud", KIND_HA, {"entity_id": "stt.home_assistant_cloud"}),
    ]:
        backend = ctx.build_stt_backend(store.create(name, kind, cfg))
        assert backend is not None, name
        assert name in backend.label


def test_incomplete_services_build_nothing(tmp_path):
    ctx = _ctx(tmp_path)
    store = ctx.stt_services
    assert ctx.build_stt_backend(store.create("a", KIND_OPENAI, {})) is None
    assert ctx.build_stt_backend(store.create("b", KIND_VOXTRAL, {})) is None
    assert ctx.build_stt_backend(store.create("c", KIND_WYOMING, {})) is None
    # Murdock's own entity would call itself.
    assert ctx.build_stt_backend(store.create("d", KIND_HA, {"entity_id": "stt.murdock"})) is None


def test_per_service_timeout_overrides_the_global_one(tmp_path):
    ctx = _ctx(tmp_path)
    store = ctx.stt_services
    local = store.create("Kroko", KIND_WYOMING, {"uri": "h:1", "timeout_sec": 3})
    cloud = store.create("MAI", KIND_OPENAI, {"model": "m"})
    assert ctx.service_timeout(local) == 3.0
    assert ctx.service_timeout(cloud) == ctx.get_stt_timeout()
