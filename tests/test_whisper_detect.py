"""Tests for whisper detection.

Whispering is defined by the *absence* of vocal-fold vibration, so the
fixtures are built around that: a harmonic stack stands in for voiced
speech, shaped noise for a whisper. Loudness is varied independently
because "quiet" must never be mistaken for "whispered" — a distant
speaker is quiet too.
"""

from __future__ import annotations

import numpy as np
import pytest

from murdock.core.audio import float32_to_pcm16_bytes
from murdock.core.whisper_detect import (
    DEFAULT_THRESHOLD,
    analyze,
    analyze_pcm,
)

_SR = 16000


def _t(seconds: float = 2.0) -> np.ndarray:
    return np.arange(int(seconds * _SR)) / _SR


def voiced(f0: float = 130.0, amp: float = 0.25, seconds: float = 2.0):
    t = _t(seconds)
    s = sum((1.0 / k) * np.sin(2 * np.pi * f0 * k * t) for k in range(1, 25))
    s *= 0.6 + 0.4 * np.sin(2 * np.pi * 3 * t)
    return (amp * s / np.abs(s).max()).astype(np.float32)


def whispered(amp: float = 0.05, seconds: float = 2.0, seed: int = 7):
    rng = np.random.default_rng(seed)
    t = _t(seconds)
    n = rng.normal(0, 1, t.size)
    spec = np.fft.rfft(n)
    freqs = np.fft.rfftfreq(t.size, 1.0 / _SR)
    spec *= np.clip((freqs - 200) / 1500, 0, 1) * np.exp(-freqs / 6000)
    s = np.fft.irfft(spec, t.size)
    s *= 0.6 + 0.4 * np.sin(2 * np.pi * 3 * t)
    return (amp * s / np.abs(s).max()).astype(np.float32)


def test_voiced_speech_is_not_whisper():
    f = analyze(voiced())
    assert f.is_whisper is False
    assert f.voiced_ratio > 0.8
    assert f.score < 0.2


def test_whisper_is_detected():
    f = analyze(whispered())
    assert f.is_whisper is True
    assert f.voiced_ratio < 0.2
    assert f.score > DEFAULT_THRESHOLD


@pytest.mark.parametrize("amp", [0.02, 0.05, 0.12, 0.3])
def test_loudness_does_not_decide(amp):
    """A quiet voice is not a whisper, and a loud whisper is still one."""
    assert analyze(voiced(amp=amp)).is_whisper is False
    assert analyze(whispered(amp=amp)).is_whisper is True


@pytest.mark.parametrize("f0", [85, 110, 130, 200, 260])
def test_pitch_range_stays_voiced(f0):
    """Deep and high voices alike must read as voiced."""
    assert analyze(voiced(f0=f0)).is_whisper is False


def test_silence_yields_no_opinion():
    rng = np.random.default_rng(1)
    quiet = (0.0005 * rng.normal(0, 1, int(2 * _SR))).astype(np.float32)
    f = analyze(quiet)
    # Room tone must not be reported as whispering.
    assert f.is_whisper is False
    assert f.voiced_ratio == 0.0


def test_threshold_is_adjustable():
    # The synthetic whisper saturates at 1.0, so raising the bar can't
    # flip it — verify the plumbing and the comparison instead.
    from murdock.core.whisper_detect import WhisperFeatures

    assert analyze(whispered(), threshold=0.42).threshold == 0.42
    # Lowering the bar to zero makes even voiced speech qualify.
    assert analyze(voiced(), threshold=0.0).is_whisper is True

    borderline = WhisperFeatures(0.5, 0.3, 0.3, 0.3, 0.05, 0.4, threshold=0.6)
    assert borderline.is_whisper is False
    assert WhisperFeatures(0.6, 0.3, 0.3, 0.3, 0.05, 0.4, 0.6).is_whisper is True


def test_empty_and_tiny_inputs():
    assert analyze(np.zeros(0, dtype=np.float32)).is_whisper is False
    assert analyze(np.zeros(100, dtype=np.float32)).score == 0.0
    assert analyze_pcm(b"").is_whisper is False


def test_pcm_entry_point_matches_waveform():
    sig = whispered()
    from_wave = analyze(sig)
    from_pcm = analyze_pcm(float32_to_pcm16_bytes(sig))
    assert from_pcm.is_whisper == from_wave.is_whisper
    assert from_pcm.score == pytest.approx(from_wave.score, abs=0.1)


def test_features_are_reported_for_tuning():
    """The log needs the raw signals, not just the verdict."""
    f = analyze(whispered())
    assert 0.0 <= f.harmonicity <= 1.0
    assert 0.0 <= f.zero_crossing_rate <= 1.0
    assert 0.0 <= f.spectral_tilt <= 1.0
    assert f.rms > 0.0
    assert f.threshold == DEFAULT_THRESHOLD


def _room_tone(level: float, seconds: float, seed: int = 2) -> np.ndarray:
    """Steady broadband noise — an air conditioner, a fan, a fridge."""
    rng = np.random.default_rng(seed)
    return rng.normal(0, level, int(seconds * _SR)).astype(np.float32)


def _utterance(signal: np.ndarray, noise_level: float) -> np.ndarray:
    """Speech with room tone under it, plus the usual lead-in and tail."""
    return np.concatenate([
        _room_tone(noise_level, 1.7),
        signal + _room_tone(noise_level, len(signal) / _SR),
        _room_tone(noise_level, 1.8, seed=3),
    ])


@pytest.mark.parametrize("noise", [0.001, 0.006, 0.015, 0.03])
def test_room_noise_does_not_turn_speech_into_a_whisper(noise):
    """Reported from a real install: an air conditioner scored 0.78.

    An absolute silence floor let the noise frames count as active, and
    having no periodicity they dragged the voiced ratio down until
    ordinary speech cleared the whisper bar. The activity bar now rises
    with the room, so the same voice reads the same in any room.
    """
    f = analyze(_utterance(voiced(), noise))
    assert not f.is_whisper, f"score {f.score:.2f} at noise {noise}"
    assert f.voiced_ratio > 0.8


@pytest.mark.parametrize("noise", [0.001, 0.006, 0.015, 0.03])
def test_whispering_is_still_caught_through_the_same_noise(noise):
    """The noise fix must not cost the feature its actual job."""
    f = analyze(_utterance(whispered(), noise))
    assert f.is_whisper, f"score {f.score:.2f} at noise {noise}"


def test_a_quiet_whisper_in_a_loud_room_is_still_judged():
    """Never return 'not whispering' when the truth is 'could not tell'.

    Capping the activity bar at half the loudest frame keeps the
    utterance in the running however loud the room is.
    """
    f = analyze(_utterance(whispered(amp=0.03), 0.02))
    assert f.voiced_ratio == 0.0
    assert f.score > 0.0
