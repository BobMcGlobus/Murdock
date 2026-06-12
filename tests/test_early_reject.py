"""Tests for the early-reject decision logic and the new defaults."""

from __future__ import annotations

import pytest

from murdock.config import Settings
from wyoming_murdock.handler import (
    _EARLY_REJECT_MEDIA_FACTOR,
    early_reject_decision,
)


def test_reject_only_far_beyond_threshold():
    # Borderline distances stay in the "let full verification decide" gap.
    assert early_reject_decision(0.40, 0.30, 0.25, media_playing=False) is False
    assert early_reject_decision(0.54, 0.30, 0.25, media_playing=False) is False
    # Catastrophically off → reject.
    assert early_reject_decision(0.55, 0.30, 0.25, media_playing=False) is True
    assert early_reject_decision(0.80, 0.30, 0.25, media_playing=False) is True


def test_media_halves_the_margin():
    # 0.45 is below 0.30+0.25 without media, but ≥ 0.30+0.125 with it.
    assert early_reject_decision(0.45, 0.30, 0.25, media_playing=False) is False
    assert early_reject_decision(0.45, 0.30, 0.25, media_playing=True) is True
    assert _EARLY_REJECT_MEDIA_FACTOR == pytest.approx(0.5)


def test_reject_respects_effective_threshold():
    # A satellite/speaker-resolved threshold shifts the bar with it.
    assert early_reject_decision(0.55, 0.40, 0.25, media_playing=False) is False
    assert early_reject_decision(0.66, 0.40, 0.25, media_playing=False) is True


def test_defaults_are_safe():
    s = Settings()
    # Hard gate off by default — a fresh install must not make the
    # satellite unusable before any speakers are enrolled.
    assert s.require_speaker_match is False
    # Early reject is opt-in.
    assert s.enable_early_reject is False
    assert s.early_reject_margin == pytest.approx(0.25)
