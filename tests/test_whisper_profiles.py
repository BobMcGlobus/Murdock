"""Tests for whisper voice profiles.

Two properties matter more than the feature itself:

* a whispered sample must **never** enter the normal voiceprint — it
  would drag everyone's ordinary recognition toward a voice with no
  pitch;
* the whisper centroid must only be consulted when asked, so "whisper
  detected" can never become a way past the gate for someone who never
  enrolled a whisper.
"""

from __future__ import annotations

import numpy as np
import pytest

from murdock.core.db import open_db
from murdock.core.speaker_store import (
    STYLE_NORMAL,
    STYLE_WHISPER,
    WHISPER_PROFILE_MIN_SAMPLES,
    SpeakerStore,
)


class _StyleEmbedder:
    """Embeds by style: whispers land far from normal speech, as in life."""

    EMBEDDING_DIM = 192

    def __init__(self):
        self.calls = 0

    def embed_pcm(self, pcm: bytes) -> np.ndarray:
        self.calls += 1
        vec = np.zeros(self.EMBEDDING_DIM, dtype=np.float32)
        # The style marker is the first PCM sample. Enrollment hands over
        # raw PCM here, and the centroid rebuild decodes the stored WAV
        # back to raw PCM, so byte 0 is the marker in both paths.
        marker = pcm[0] if pcm else 0
        if marker == 1:          # whispered
            vec[1] = 1.0
        else:                    # normal
            vec[0] = 1.0
        return vec

    @staticmethod
    def average(embs):
        return np.mean(np.stack(embs), axis=0).astype(np.float32)

    @staticmethod
    def cosine_distance(a, b):
        na, nb = np.linalg.norm(a), np.linalg.norm(b)
        if na < 1e-9 or nb < 1e-9:
            return 2.0
        return float(1.0 - np.dot(a, b) / (na * nb))


def _pcm(marker: int, seconds: float = 2.0) -> bytes:
    """Mono 16 kHz PCM whose first sample encodes the style."""
    n = int(seconds * 16000)
    buf = bytearray(n * 2)
    buf[0] = marker
    return bytes(buf)


def _store(tmp_path):
    db = open_db(tmp_path / "m.db")
    return SpeakerStore(db, _StyleEmbedder(), vad=None, threshold=0.38)


def _enroll(store, name, marker, style, n=1):
    for _ in range(n):
        store.enroll(
            speaker_name=name,
            pcm_bytes=_pcm(marker),
            duration_sec=2.0,
            source="upload",
            style=style,
        )


def test_whisper_samples_stay_out_of_the_normal_voiceprint(tmp_path):
    store = _store(tmp_path)
    _enroll(store, "Jonas", marker=0, style=STYLE_NORMAL, n=2)
    _enroll(store, "Jonas", marker=1, style=STYLE_WHISPER, n=2)

    # A normal utterance must still match cleanly: if the whisper samples
    # had been averaged in, this distance would be roughly 0.3, not ~0.
    normal = np.zeros(192, dtype=np.float32)
    normal[0] = 1.0
    result = store.verify_embedding(normal, 0.38)
    assert result.is_match is True
    assert result.distance < 0.05


def test_whisper_matches_only_when_asked(tmp_path):
    store = _store(tmp_path)
    _enroll(store, "Jonas", marker=0, style=STYLE_NORMAL, n=2)
    _enroll(store, "Jonas", marker=1, style=STYLE_WHISPER,
            n=WHISPER_PROFILE_MIN_SAMPLES)

    whispered = np.zeros(192, dtype=np.float32)
    whispered[1] = 1.0

    # Without the flag the whisper is a stranger — this is the property
    # that keeps the feature from being a bypass.
    plain = store.verify_embedding(whispered, 0.38)
    assert plain.is_match is False

    # With it, the same-style centroid closes the gap.
    with_flag = store.verify_embedding(whispered, 0.38, whisper=True)
    assert with_flag.is_match is True
    assert with_flag.matched_speaker == "Jonas"


def test_no_whisper_profile_means_no_match(tmp_path):
    """Someone who never enrolled a whisper stays unknown when whispering."""
    store = _store(tmp_path)
    _enroll(store, "Jonas", marker=0, style=STYLE_NORMAL, n=2)

    whispered = np.zeros(192, dtype=np.float32)
    whispered[1] = 1.0
    result = store.verify_embedding(whispered, 0.38, whisper=True)
    assert result.is_match is False


def test_one_whisper_sample_is_not_enough(tmp_path):
    store = _store(tmp_path)
    _enroll(store, "Jonas", marker=0, style=STYLE_NORMAL, n=2)
    _enroll(store, "Jonas", marker=1, style=STYLE_WHISPER, n=1)

    whispered = np.zeros(192, dtype=np.float32)
    whispered[1] = 1.0
    assert store.verify_embedding(whispered, 0.38, whisper=True).is_match is False


def test_whisper_counts_are_reported(tmp_path):
    store = _store(tmp_path)
    _enroll(store, "Jonas", marker=0, style=STYLE_NORMAL, n=2)
    _enroll(store, "Jonas", marker=1, style=STYLE_WHISPER, n=3)
    speaker = store.get_speaker_by_name("Jonas")
    assert store.whisper_profile_counts()[speaker.id] == 3


def test_samples_report_their_style(tmp_path):
    store = _store(tmp_path)
    _enroll(store, "Jonas", marker=0, style=STYLE_NORMAL)
    _enroll(store, "Jonas", marker=1, style=STYLE_WHISPER)
    speaker = store.get_speaker_by_name("Jonas")
    styles = sorted(
        (s.get("style") or STYLE_NORMAL) for s in store.list_samples(speaker.id)
    )
    assert styles == [STYLE_NORMAL, STYLE_WHISPER]


def test_enrollment_count_excludes_whispers(tmp_path):
    """The headline count is about the normal voiceprint."""
    store = _store(tmp_path)
    _enroll(store, "Jonas", marker=0, style=STYLE_NORMAL, n=2)
    _enroll(store, "Jonas", marker=1, style=STYLE_WHISPER, n=2)
    speaker = store.get_speaker_by_name("Jonas")
    assert speaker.enrollment_count == 2


def test_deleting_whisper_samples_drops_the_profile(tmp_path):
    store = _store(tmp_path)
    _enroll(store, "Jonas", marker=0, style=STYLE_NORMAL, n=2)
    _enroll(store, "Jonas", marker=1, style=STYLE_WHISPER, n=2)
    speaker = store.get_speaker_by_name("Jonas")

    for s in store.list_samples(speaker.id):
        if (s.get("style") or STYLE_NORMAL) == STYLE_WHISPER:
            store.delete_sample(s["id"])

    whispered = np.zeros(192, dtype=np.float32)
    whispered[1] = 1.0
    assert store.verify_embedding(whispered, 0.38, whisper=True).is_match is False
    assert store.whisper_profile_counts().get(speaker.id, 0) == 0
