"""Tests for the per-satellite × per-source media-restriction matrix.

Builds a minimal AppContext with light fakes — compute_media_tightening
only touches settings, the sqlite db, the MQTT client and the HA client.
"""

from __future__ import annotations

import asyncio
import time
from types import SimpleNamespace

import pytest

from murdock.core.context import AppContext
from murdock.core.db import open_db
from murdock.core.mqtt_integration import MQTTClient


def _ctx(tmp_path, tv_boost=0.05):
    db = open_db(tmp_path / "m.db")
    settings = SimpleNamespace(tv_threshold_boost=tv_boost)
    mqtt = MQTTClient(host="broker")

    async def _ha_tv():
        return False

    ha = SimpleNamespace(is_tv_playing=_ha_tv)
    ctx = AppContext(
        settings=settings, db=db, embedder=None, vad=None,
        speakers=None, unknown=None, ha=ha, mqtt=mqtt, recognition=None,
    )
    return ctx, mqtt


def _play(mqtt, entity, area, playing=True):
    mqtt._context[("media", entity)] = (
        {"playing": playing, "area": area}, time.monotonic(),
    )


def _tighten(ctx, sat, area):
    return asyncio.run(ctx.compute_media_tightening(sat, area))


def test_restriction_round_trip(tmp_path):
    ctx, _ = _ctx(tmp_path)
    assert ctx.get_media_restrictions() == {}
    ctx.set_media_restriction("sat1", "media_player.tv", 0.12)
    assert ctx.get_media_restrictions() == {"sat1": {"media_player.tv": 0.12}}
    # Clear collapses the empty satellite entry.
    ctx.set_media_restriction("sat1", "media_player.tv", None)
    assert ctx.get_media_restrictions() == {}


def test_set_requires_ids(tmp_path):
    ctx, _ = _ctx(tmp_path)
    with pytest.raises(ValueError):
        ctx.set_media_restriction("", "media_player.tv", 0.1)
    with pytest.raises(ValueError):
        ctx.set_media_restriction("sat1", "", 0.1)


def test_no_media_no_tightening(tmp_path):
    ctx, mqtt = _ctx(tmp_path)
    mqtt._connected = True
    assert _tighten(ctx, "sat1", "Wohnzimmer") == 0.0


def test_same_room_default_boost(tmp_path):
    ctx, mqtt = _ctx(tmp_path, tv_boost=0.05)
    mqtt._connected = True
    _play(mqtt, "media_player.tv", "Wohnzimmer")
    # No matrix entry → default boost because it's in the satellite's room.
    assert _tighten(ctx, "sat1", "Wohnzimmer") == pytest.approx(0.05)


def test_other_room_no_effect(tmp_path):
    ctx, mqtt = _ctx(tmp_path)
    mqtt._connected = True
    _play(mqtt, "media_player.tv", "Kueche")
    assert _tighten(ctx, "sat1", "Wohnzimmer") == 0.0


def test_matrix_override_wins(tmp_path):
    ctx, mqtt = _ctx(tmp_path, tv_boost=0.05)
    mqtt._connected = True
    _play(mqtt, "media_player.tv", "Wohnzimmer")
    ctx.set_media_restriction("sat1", "media_player.tv", 0.20)
    # Explicit delta replaces the default boost.
    assert _tighten(ctx, "sat1", "Wohnzimmer") == pytest.approx(0.20)


def test_matrix_zero_disables_source(tmp_path):
    ctx, mqtt = _ctx(tmp_path, tv_boost=0.05)
    mqtt._connected = True
    _play(mqtt, "media_player.radio", "Wohnzimmer")
    # Explicit 0 means this source does not restrict this satellite,
    # even though it plays in the same room.
    ctx.set_media_restriction("sat1", "media_player.radio", 0.0)
    assert _tighten(ctx, "sat1", "Wohnzimmer") == 0.0


def test_strongest_source_wins(tmp_path):
    ctx, mqtt = _ctx(tmp_path, tv_boost=0.05)
    mqtt._connected = True
    _play(mqtt, "media_player.tv", "Wohnzimmer")
    _play(mqtt, "media_player.radio", "Wohnzimmer")
    ctx.set_media_restriction("sat1", "media_player.tv", 0.30)
    ctx.set_media_restriction("sat1", "media_player.radio", 0.10)
    # max, not sum.
    assert _tighten(ctx, "sat1", "Wohnzimmer") == pytest.approx(0.30)


def test_matrix_is_per_satellite(tmp_path):
    ctx, mqtt = _ctx(tmp_path, tv_boost=0.05)
    mqtt._connected = True
    _play(mqtt, "media_player.tv", "Wohnzimmer")
    ctx.set_media_restriction("sat1", "media_player.tv", 0.30)
    # sat2 has no rule → default boost (same room); sat1 → its override.
    assert _tighten(ctx, "sat1", "Wohnzimmer") == pytest.approx(0.30)
    assert _tighten(ctx, "sat2", "Wohnzimmer") == pytest.approx(0.05)
