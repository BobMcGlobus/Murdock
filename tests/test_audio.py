"""Tests for audio decode / resample / conversion paths.

These guard the quiet corruption bugs: a resample that returns near-empty
audio, a mono mixdown that doubles length, or a PCM round-trip that
drifts. Bad audio here produces embeddings that match nothing, so a
silent regression would degrade recognition without any error.
"""

from __future__ import annotations

import numpy as np
import pytest

from murdock.core.audio import (
    decode_wav,
    encode_wav,
    float32_to_pcm16_bytes,
    pcm16_bytes_to_float32,
    rms_dbfs,
    to_mono_16k_pcm,
)


def _sine_pcm16(freq=440, rate=16000, secs=0.1, channels=1):
    t = np.linspace(0, secs, int(rate * secs), endpoint=False)
    wave = np.sin(2 * np.pi * freq * t) * 0.5
    if channels > 1:
        wave = np.repeat(wave[:, None], channels, axis=1).reshape(-1)
    return (wave * 32767).astype(np.int16).tobytes()


def test_pcm_float_round_trip():
    pcm = _sine_pcm16()
    floats = pcm16_bytes_to_float32(pcm)
    assert floats.dtype == np.float32
    assert floats.max() <= 1.0 and floats.min() >= -1.0
    back = float32_to_pcm16_bytes(floats)
    # Round-trip should be near-identical (within quantisation).
    a = np.frombuffer(pcm, dtype=np.int16).astype(np.int32)
    b = np.frombuffer(back, dtype=np.int16).astype(np.int32)
    assert np.max(np.abs(a - b)) <= 2


def test_pcm_float_empty():
    assert pcm16_bytes_to_float32(b"").shape == (0,)


def test_wav_encode_decode_round_trip():
    pcm = _sine_pcm16(rate=16000, secs=0.05)
    wav = encode_wav(pcm, sample_rate=16000, sample_width=2, channels=1)
    assert wav[:4] == b"RIFF" and wav[8:12] == b"WAVE"
    frames, rate, width, channels = decode_wav(wav)
    assert rate == 16000 and width == 2 and channels == 1
    assert frames == pcm


def test_to_mono_16k_noop_when_already_correct():
    pcm = _sine_pcm16(rate=16000, secs=0.1, channels=1)
    out = to_mono_16k_pcm(pcm, 16000, 2, 1)
    assert out == pcm


def test_to_mono_16k_downmixes_stereo():
    pcm = _sine_pcm16(rate=16000, secs=0.1, channels=2)
    out = to_mono_16k_pcm(pcm, 16000, 2, 2)
    # Mono output should be half the sample count of stereo input.
    assert len(out) == len(pcm) // 2


def test_to_mono_16k_resamples_down():
    pcm = _sine_pcm16(rate=48000, secs=0.1, channels=1)
    out = to_mono_16k_pcm(pcm, 48000, 2, 1)
    n_out = len(out) // 2
    expected = int(round(0.1 * 16000))
    # Allow a couple of samples of rounding slack.
    assert abs(n_out - expected) <= 2


def test_to_mono_16k_resamples_up():
    pcm = _sine_pcm16(rate=8000, secs=0.1, channels=1)
    out = to_mono_16k_pcm(pcm, 8000, 2, 1)
    n_out = len(out) // 2
    assert abs(n_out - int(round(0.1 * 16000))) <= 2


def test_to_mono_16k_rejects_non_16bit():
    with pytest.raises(ValueError):
        to_mono_16k_pcm(b"\x00\x00\x00", rate=16000, width=1, channels=1)


def test_rms_dbfs_silence_is_floor():
    # Empty input hits the explicit -120 sentinel.
    assert rms_dbfs(b"") == -120.0
    # Digital silence (all zeros) isn't the sentinel but must read as
    # extremely quiet thanks to the epsilon floor.
    silent = np.zeros(1600, dtype=np.int16).tobytes()
    assert rms_dbfs(silent) < -100.0


def test_rms_dbfs_full_scale_near_zero():
    full = (np.ones(1600, dtype=np.int16) * 32767).tobytes()
    db = rms_dbfs(full)
    assert -1.0 < db <= 0.0


def test_rms_dbfs_quieter_is_more_negative():
    loud = _sine_pcm16(secs=0.1)
    quiet = float32_to_pcm16_bytes(pcm16_bytes_to_float32(loud) * 0.1)
    assert rms_dbfs(quiet) < rms_dbfs(loud)
