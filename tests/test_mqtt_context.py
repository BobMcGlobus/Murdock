"""Tests for the MQTT context cache (subscribe side).

This is the "token raus" path: HA pushes TV/presence onto retained
context topics and Murdock folds them into a cache that the verify gate
reads. These tests cover topic parsing, payload coercion, room/global
fallback, and staleness — all without needing a live broker (aiomqtt is
imported lazily, so the client constructs fine offline).
"""

from __future__ import annotations

import time
from types import SimpleNamespace

import pytest

from murdock.core.mqtt_integration import (
    _ACTIVE_SAT_TTL_SECONDS,
    _CONTEXT_TTL_SECONDS,
    MQTTClient,
)


def _msg(topic: str, payload):
    """Fake aiomqtt message: only .topic and .payload are read."""
    if isinstance(payload, str):
        payload = payload.encode("utf-8")
    return SimpleNamespace(topic=topic, payload=payload)


def _client():
    return MQTTClient(host="x", topic_prefix="murdock")


def test_configured_requires_host():
    assert MQTTClient(host="").configured is False
    assert MQTTClient(host="broker").configured is True


def test_context_json_payload_parsed():
    c = _client()
    c._handle_context_message(_msg("murdock/context/wohnzimmer/tv", '{"playing": true}'))
    assert c.is_tv_playing(room="wohnzimmer") is True


def test_context_room_then_global_fallback():
    c = _client()
    c._handle_context_message(_msg("murdock/context/global/tv", '{"playing": false}'))
    # No room-specific entry → falls back to global.
    assert c.is_tv_playing(room="kueche") is False
    # Room-specific overrides global.
    c._handle_context_message(_msg("murdock/context/kueche/tv", '{"playing": true}'))
    assert c.is_tv_playing(room="kueche") is True
    # Other room still sees global.
    assert c.is_tv_playing(room="bad") is False


def test_context_unknown_returns_none():
    c = _client()
    assert c.is_tv_playing(room="anywhere") is None
    assert c.is_present(room="anywhere") is None


def test_context_empty_payload_clears_entry():
    c = _client()
    c._handle_context_message(_msg("murdock/context/wohnzimmer/tv", '{"playing": true}'))
    assert c.is_tv_playing(room="wohnzimmer") is True
    c._handle_context_message(_msg("murdock/context/wohnzimmer/tv", ""))
    assert c.is_tv_playing(room="wohnzimmer") is None


def test_context_bare_scalar_true():
    c = _client()
    c._handle_context_message(_msg("murdock/context/wohnzimmer/tv", "true"))
    assert c.is_tv_playing(room="wohnzimmer") is True


def test_context_bad_topic_shape_ignored():
    c = _client()
    # Too few / too many path segments → ignored, no crash.
    c._handle_context_message(_msg("murdock/context/onlyroom", '{"playing": true}'))
    c._handle_context_message(_msg("murdock/context/a/b/c", '{"playing": true}'))
    assert c.context_snapshot() == {}


def test_context_wrong_prefix_ignored():
    c = _client()
    c._handle_context_message(_msg("other/context/room/tv", '{"playing": true}'))
    assert c.context_snapshot() == {}


@pytest.mark.parametrize("payload,expected", [
    ('{"playing": true}', True),
    ('{"playing": false}', False),
    ('{"value": "on"}', True),
    ('{"value": "off"}', False),
    ('{"state": "playing"}', True),
    ('{"value": 1}', True),
    ('{"value": 0}', False),
])
def test_coerce_bool_variants(payload, expected):
    c = _client()
    c._handle_context_message(_msg("murdock/context/r/tv", payload))
    assert c.is_tv_playing(room="r") is expected


def test_presence_lookup():
    c = _client()
    c._handle_context_message(_msg("murdock/context/buero/presence", '{"present": true}'))
    assert c.is_present(room="buero") is True
    c._handle_context_message(_msg("murdock/context/buero/presence", '{"occupied": false}'))
    assert c.is_present(room="buero") is False


def test_context_staleness():
    c = _client()
    c._handle_context_message(_msg("murdock/context/wohnzimmer/tv", '{"playing": true}'))
    # Backdate the arrival timestamp beyond the TTL.
    value, _ = c._context[("wohnzimmer", "tv")]
    c._context[("wohnzimmer", "tv")] = (value, time.monotonic() - _CONTEXT_TTL_SECONDS - 10)
    assert c.is_tv_playing(room="wohnzimmer") is None
    assert c.context_snapshot() == {}


def test_context_snapshot_shape():
    c = _client()
    c._handle_context_message(_msg("murdock/context/wohnzimmer/tv", '{"playing": true}'))
    snap = c.context_snapshot()
    assert "wohnzimmer" in snap
    assert snap["wohnzimmer"]["tv"]["value"] == {"playing": True}
    assert "age_seconds" in snap["wohnzimmer"]["tv"]


def test_active_satellite_roundtrip():
    c = _client()
    assert c.get_active_satellite() is None
    c._handle_active_satellite(_msg("murdock/active_satellite", "Arbeitszimmer"))
    assert c.get_active_satellite() == "Arbeitszimmer"


def test_active_satellite_empty_clears():
    c = _client()
    c._handle_active_satellite(_msg("murdock/active_satellite", "Kueche"))
    assert c.get_active_satellite() == "Kueche"
    c._handle_active_satellite(_msg("murdock/active_satellite", ""))
    assert c.get_active_satellite() is None


def test_active_satellite_staleness():
    c = _client()
    c._handle_active_satellite(_msg("murdock/active_satellite", "Wohnzimmer"))
    sat_id, area, _ = c._active_satellite
    c._active_satellite = (sat_id, area, time.monotonic() - _ACTIVE_SAT_TTL_SECONDS - 5)
    assert c.get_active_satellite() is None
    assert c.get_active_satellite_area() is None


def test_active_satellite_json_with_area():
    c = _client()
    c._handle_active_satellite(_msg(
        "murdock/active_satellite",
        '{"id": "assist_satellite.sat1", "area": "Arbeitszimmer"}',
    ))
    assert c.get_active_satellite() == "assist_satellite.sat1"
    assert c.get_active_satellite_area() == "Arbeitszimmer"


def test_active_satellite_bare_string_no_area():
    c = _client()
    c._handle_active_satellite(_msg("murdock/active_satellite", "Kueche"))
    assert c.get_active_satellite() == "Kueche"
    assert c.get_active_satellite_area() is None


def test_media_in_area_tightens_is_tv_playing():
    c = _client()
    # A media player in Wohnzimmer is playing.
    c._handle_context_message(_msg(
        "murdock/context/media/media_player.wohnzimmer_tv",
        '{"playing": true, "area": "Wohnzimmer"}',
    ))
    assert c.is_tv_playing(room="Wohnzimmer") is True
    # A different room sees no media context → None (no signal).
    assert c.is_tv_playing(room="Arbeitszimmer") is None


def test_media_in_area_not_playing_is_false():
    c = _client()
    c._handle_context_message(_msg(
        "murdock/context/media/media_player.radio",
        '{"playing": false, "area": "Kueche"}',
    ))
    assert c.is_tv_playing(room="Kueche") is False


def test_media_and_legacy_tv_combine():
    c = _client()
    # Legacy single-TV topic says not playing…
    c._handle_context_message(_msg("murdock/context/Wohnzimmer/tv", '{"playing": false}'))
    # …but a media player in the room is playing → overall True.
    c._handle_context_message(_msg(
        "murdock/context/media/media_player.soundbar",
        '{"playing": true, "area": "Wohnzimmer"}',
    ))
    assert c.is_tv_playing(room="Wohnzimmer") is True


def test_discovery_config_count_and_topics():
    c = _client()
    configs = c._discovery_configs()
    # 7 sensors + 1 binary_sensor.
    assert len(configs) == 8
    object_ids = {object_id for _, object_id, _ in configs}
    assert "current_speaker" in object_ids
    assert "speaker_recognized" in object_ids
    # Every config must carry availability + device grouping.
    for _, _, cfg in configs:
        assert cfg["availability_topic"] == "murdock/status"
        assert cfg["device"]["identifiers"] == ["murdock"]
        assert cfg["state_topic"].startswith("murdock/")
