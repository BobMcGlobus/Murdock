"""The shadow engine answering when the primary heard nothing.

A transducer that is unsure returns nothing rather than guessing. That
is honest and useless to the person waiting — where a second engine is
already configured for the A/B comparison, it can answer instead.
"""

from __future__ import annotations

import asyncio
from types import SimpleNamespace

import pytest

from wyoming_murdock.handler import MurdockHandler


def _handler(*, enabled=True, shadow="voxtral", shadow_text="Licht an",
             fails=False):
    async def _shadow_transcribe(audio):
        if fails:
            raise RuntimeError("provider down")
        return shadow_text, "voxtral:voxtral-mini-latest"

    fake = SimpleNamespace(
        _session_id="s1",
        _rescued_by=None,
        _transcript_timing={"ttfb_ms": 400.0},
        context=SimpleNamespace(
            get_shadow_rescues_empty=lambda: enabled,
            get_shadow_stt_backend=lambda: shadow,
            get_stt_timeout=lambda: 8.0,
        ),
    )
    fake._shadow_transcribe = _shadow_transcribe
    return fake


def _rescue(fake):
    return asyncio.run(MurdockHandler._rescue_empty_transcript(fake, b"\x00" * 3200))


def test_shadow_answers_when_the_primary_heard_nothing():
    fake = _handler()
    assert _rescue(fake) == "Licht an"
    assert fake._rescued_by == "voxtral:voxtral-mini-latest"


def test_rescue_is_skipped_when_disabled():
    fake = _handler(enabled=False)
    assert _rescue(fake) == ""
    assert fake._rescued_by is None


def test_rescue_is_skipped_without_a_shadow_engine():
    fake = _handler(shadow="none")
    assert _rescue(fake) == ""


def test_a_shadow_that_also_heard_nothing_changes_nothing():
    """Two engines agreeing on silence is a real answer, not a failure."""
    fake = _handler(shadow_text="   ")
    assert _rescue(fake) == ""
    assert fake._rescued_by is None


def test_a_failing_shadow_never_costs_the_utterance():
    """The empty transcript stands; the rescue must not raise."""
    fake = _handler(fails=True)
    assert _rescue(fake) == ""
    assert fake._rescued_by is None


def test_the_log_records_which_engine_answered():
    fake = _handler()
    _rescue(fake)
    timing = MurdockHandler._timing_with_rescue(fake)
    assert timing["rescued_by"] == "voxtral:voxtral-mini-latest"
    # The primary's own breakdown is preserved alongside it.
    assert timing["ttfb_ms"] == 400.0


def test_timing_is_untouched_when_no_rescue_happened():
    fake = _handler(enabled=False)
    _rescue(fake)
    assert MurdockHandler._timing_with_rescue(fake) == {"ttfb_ms": 400.0}
