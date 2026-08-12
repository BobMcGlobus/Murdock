"""Conditioning of the audio that goes to a cloud STT endpoint."""

from __future__ import annotations

import numpy as np

from murdock.core.audio import float32_to_pcm16_bytes, pcm16_bytes_to_float32
from murdock.core.stt_prep import (
    normalize_level,
    prepare_for_upload,
    trim_head_and_tail,
)

RATE = 16000


class _FakeVAD:
    """Reports fixed speech segments, in seconds."""

    def __init__(self, segments):
        self._segments = segments

    def analyze_waveform(self, audio):
        class _R:
            segments = self._segments

        return _R()


def _clip(total_sec: float, speech_from: float, speech_to: float) -> bytes:
    """Quiet room tone with a loud region in the middle."""
    n = int(total_sec * RATE)
    audio = np.random.default_rng(3).normal(scale=0.001, size=n).astype(np.float32)
    a, b = int(speech_from * RATE), int(speech_to * RATE)
    t = np.arange(b - a, dtype=np.float32) / RATE
    audio[a:b] += 0.3 * np.sin(2 * np.pi * 180.0 * t)
    return float32_to_pcm16_bytes(audio)


def test_trims_the_ends_but_keeps_the_middle():
    """Internal pauses must survive — splicing them risks clipped onsets."""
    pcm = _clip(4.0, 1.0, 3.0)
    vad = _FakeVAD([(1.0, 1.4), (2.6, 3.0)])  # a gap in the middle
    out = trim_head_and_tail(pcm, vad)
    kept_sec = len(out) / (RATE * 2)
    # 1.0→3.0 plus 0.2s padding either side = 2.4s. If the gap had been
    # spliced out this would be closer to 1.2s.
    assert 2.3 < kept_sec < 2.5


def test_a_barely_detecting_vad_is_distrusted():
    """Silero under-detects whispering — the audio users most need kept.

    A sliver of detected speech in a long clip is far more likely to be
    a missed detection than a genuinely short command, so nothing is
    trimmed rather than confidently throwing the rest away.
    """
    pcm = _clip(3.0, 1.0, 1.05)
    vad = _FakeVAD([(1.0, 1.02)])  # 20 ms in 3 s — well under the floor
    assert trim_head_and_tail(pcm, vad) == pcm


def test_trim_is_a_no_op_without_speech():
    pcm = _clip(2.0, 0.5, 1.0)
    assert trim_head_and_tail(pcm, _FakeVAD([])) == pcm
    assert trim_head_and_tail(pcm, None) == pcm


def test_trim_survives_a_broken_vad():
    """A VAD failure must not cost the utterance."""

    class _Boom:
        def analyze_waveform(self, audio):
            raise RuntimeError("onnx exploded")

    pcm = _clip(2.0, 0.5, 1.0)
    assert trim_head_and_tail(pcm, _Boom()) == pcm


def test_quiet_speech_is_brought_up():
    quiet = float32_to_pcm16_bytes(
        (0.01 * np.sin(2 * np.pi * 180.0 * np.arange(RATE) / RATE)).astype(np.float32)
    )
    louder = normalize_level(quiet)
    before = np.abs(pcm16_bytes_to_float32(quiet)).max()
    after = np.abs(pcm16_bytes_to_float32(louder)).max()
    assert after > before


def test_normalisation_never_clips():
    hot = float32_to_pcm16_bytes(
        (0.98 * np.sin(2 * np.pi * 180.0 * np.arange(RATE) / RATE)).astype(np.float32)
    )
    out = pcm16_bytes_to_float32(normalize_level(hot))
    assert np.abs(out).max() <= 0.995


def test_pure_silence_is_left_alone():
    """Amplifying room tone just invites the decoder to invent words."""
    silence = float32_to_pcm16_bytes(np.zeros(RATE, dtype=np.float32))
    assert normalize_level(silence) == silence
    faint = float32_to_pcm16_bytes(
        np.random.default_rng(1).normal(scale=0.0005, size=RATE).astype(np.float32)
    )
    assert normalize_level(faint) == faint


def test_prepare_reports_what_it_did():
    pcm = _clip(4.0, 1.0, 3.0)
    out, info = prepare_for_upload(pcm, _FakeVAD([(1.0, 3.0)]))
    assert len(out) < len(pcm)
    assert info["trimmed_ms"] > 0
    # Normalisation moves the level toward the target from either side —
    # this clip starts hot, so it is attenuated.
    assert abs(info["level_after_dbfs"] - (-20.0)) < 1.0


def test_prepare_reports_none_when_nothing_changed():
    pcm = _clip(2.4, 0.2, 2.2)
    out, info = prepare_for_upload(pcm, _FakeVAD([]), normalize=False)
    assert out == pcm
    assert info is None
