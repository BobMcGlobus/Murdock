"""Tests for Platt-scaling confidence calibration.

Covers the fit on separable and degenerate data, probability monotonicity
and bounds, and the Calibrator persistence round-trip.
"""

from __future__ import annotations

import numpy as np
import pytest

from murdock.core.calibration import (
    Calibrator,
    calibrator_from_pairs,
    fit_platt,
)


def _separable(n=200, seed=0):
    """Genuine distances low (~0.12), impostor high (~0.6)."""
    rng = np.random.default_rng(seed)
    genuine = np.clip(rng.normal(0.12, 0.04, n), 0.0, 2.0)
    impostor = np.clip(rng.normal(0.60, 0.08, n), 0.0, 2.0)
    distances = np.concatenate([genuine, impostor])
    labels = np.concatenate([np.ones(n), np.zeros(n)])
    return distances, labels


def test_fit_returns_params_on_separable_data():
    distances, labels = _separable()
    params = fit_platt(distances, labels)
    assert params is not None
    a, b = params
    # Lower distance ⇒ same speaker ⇒ A must be positive.
    assert a > 0


def test_fit_none_when_single_class():
    # Only genuine, no impostor.
    assert fit_platt([0.1, 0.12, 0.09], [1, 1, 1]) is None
    # Only impostor.
    assert fit_platt([0.6, 0.7], [0, 0]) is None
    # Empty.
    assert fit_platt([], []) is None


def test_probability_monotonic_and_bounded():
    distances, labels = _separable()
    cal = calibrator_from_pairs(distances, labels)
    assert cal.fitted
    p_close = cal.probability(0.10)
    p_mid = cal.probability(0.35)
    p_far = cal.probability(0.65)
    # Monotonically decreasing in distance.
    assert p_close > p_mid > p_far
    # A clearly-genuine distance is confident, a clearly-impostor one isn't.
    assert p_close > 0.8
    assert p_far < 0.2
    # Bounds.
    for d in (-1.0, 0.0, 0.5, 1.0, 5.0):
        p = cal.probability(d)
        assert 0.0 <= p <= 1.0


def test_calibrator_from_pairs_counts():
    distances, labels = _separable(n=50)
    cal = calibrator_from_pairs(distances, labels)
    assert cal.n_genuine == 50
    assert cal.n_impostor == 50
    assert cal.fitted_at > 0


def test_calibrator_unfitted_returns_none():
    cal = Calibrator()
    assert cal.fitted is False
    assert cal.probability(0.2) is None


def test_calibrator_from_pairs_degenerate_is_unfitted():
    cal = calibrator_from_pairs([0.1, 0.2], [1, 1])
    assert cal.fitted is False
    assert cal.n_genuine == 2
    assert cal.n_impostor == 0


def test_calibrator_dict_round_trip():
    distances, labels = _separable(n=30)
    cal = calibrator_from_pairs(distances, labels)
    restored = Calibrator.from_dict(cal.to_dict())
    assert restored.fitted == cal.fitted
    assert restored.a == pytest.approx(cal.a)
    assert restored.b == pytest.approx(cal.b)
    assert restored.n_genuine == cal.n_genuine
    # Same probability for the same distance after a round-trip.
    assert restored.probability(0.2) == pytest.approx(cal.probability(0.2))


def test_from_dict_handles_none_and_empty():
    assert Calibrator.from_dict(None).fitted is False
    assert Calibrator.from_dict({}).fitted is False


def test_fit_stable_with_overlapping_classes():
    # Heavily overlapping classes shouldn't blow up; just less confident.
    rng = np.random.default_rng(1)
    genuine = np.clip(rng.normal(0.30, 0.10, 100), 0, 2)
    impostor = np.clip(rng.normal(0.35, 0.10, 100), 0, 2)
    distances = np.concatenate([genuine, impostor])
    labels = np.concatenate([np.ones(100), np.zeros(100)])
    cal = calibrator_from_pairs(distances, labels)
    assert cal.fitted
    # Still monotone, just shallow.
    assert cal.probability(0.1) >= cal.probability(0.5)
