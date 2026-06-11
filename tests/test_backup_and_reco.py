"""Tests for the settings-backup helpers and the threshold recommendation."""

from __future__ import annotations

import numpy as np
import pytest

from murdock.api.routes_backup import apply_settings, dump_settings
from murdock.api.routes_settings import recommend_threshold
from murdock.core.db import open_db, set_setting


# --- settings dump / apply ----------------------------------------------------


def test_settings_dump_apply_round_trip(tmp_path):
    src = open_db(tmp_path / "src.db")
    set_setting(src, "verify_threshold", "0.42")
    set_setting(src, "mqtt_host", "core-mosquitto")
    set_setting(src, "media_restrictions", '{"sat1": {"media_player.tv": 0.2}}')

    dumped = dump_settings(src)
    assert dumped["verify_threshold"] == "0.42"
    assert dumped["mqtt_host"] == "core-mosquitto"

    dst = open_db(tmp_path / "dst.db")
    n = apply_settings(dst, dumped)
    assert n == len(dumped)
    assert dump_settings(dst) == dumped


def test_apply_settings_skips_non_strings(tmp_path):
    dst = open_db(tmp_path / "dst.db")
    n = apply_settings(dst, {"good": "1", "bad": 5, 7: "x"})
    assert n == 1
    assert dump_settings(dst) == {"good": "1"}


# --- threshold recommendation --------------------------------------------------


def _dists(mean, std, n, seed):
    rng = np.random.default_rng(seed)
    return list(np.clip(rng.normal(mean, std, n), 0.0, 2.0))


def test_reco_clean_separation():
    genuine = _dists(0.15, 0.03, 50, 1)
    impostor = _dists(0.60, 0.05, 30, 2)
    r = recommend_threshold(genuine, impostor, current=0.30)
    assert r["status"] == "ok"
    assert r["separation"] > 0
    # Recommendation lies in the gap between the two distributions.
    assert r["genuine_p95"] < r["recommended"] < r["impostor_p05"]


def test_reco_overlap_flagged():
    genuine = _dists(0.30, 0.08, 50, 3)
    impostor = _dists(0.35, 0.08, 50, 4)
    r = recommend_threshold(genuine, impostor, current=0.30)
    assert r["status"] == "overlap"
    assert r["separation"] < 0
    assert r["recommended"] is not None


def test_reco_insufficient_data():
    r = recommend_threshold([0.1] * 5, [0.6] * 2, current=0.30)
    assert r["status"] == "insufficient_data"
    assert "recommended" not in r or r.get("recommended") is None
    assert r["genuine_count"] == 5
    assert r["impostor_count"] == 2


def test_reco_clamped():
    # Pathologically low distances should still yield a sane floor.
    r = recommend_threshold([0.001] * 20, [0.02] * 10, current=0.30)
    assert r["recommended"] >= 0.05
