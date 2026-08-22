"""How something was said, from measurements already taken.

The whisper detector computes a level and a voicing strength for every
utterance anyway. Turning those into a coarse label costs nothing and
gives automations something to react to — a quiet request deserves a
quiet answer.

The hard part is that "loud" has no absolute meaning: it depends on the
microphone, the distance, the room. So the level is judged against a
**running median of that satellite's own recent utterances** rather than
a fixed number. A satellite calibrates itself within a handful of turns
and needs no configuration.

Deliberately coarse. Three bands and a whisper, not a mood: the
measurements support "quieter and flatter than usual for this room", and
they do not support anything more confident than that.
"""

from __future__ import annotations

import logging
from collections import defaultdict, deque
from typing import Deque, Dict, Optional

_LOGGER = logging.getLogger("murdock.voice_style")

STYLE_WHISPERED = "whispered"
STYLE_QUIET = "quiet"
STYLE_NORMAL = "normal"
STYLE_ANIMATED = "animated"

#: Utterances kept per satellite for the running baseline.
_WINDOW = 25

#: Below this many samples the baseline is not trustworthy and everything
#: reads as normal — a confident label from two data points is a guess.
_MIN_SAMPLES = 5

#: How far from the median counts as quiet / animated, as a ratio.
_QUIET_RATIO = 0.6
_ANIMATED_RATIO = 1.7


class VoiceStyleTracker:
    """Per-satellite running baseline of speech level."""

    def __init__(self, window: int = _WINDOW) -> None:
        self._window = window
        self._levels: Dict[str, Deque[float]] = defaultdict(
            lambda: deque(maxlen=window)
        )

    def _median(self, key: str) -> Optional[float]:
        vals = sorted(self._levels[key])
        if len(vals) < _MIN_SAMPLES:
            return None
        mid = len(vals) // 2
        if len(vals) % 2:
            return vals[mid]
        return (vals[mid - 1] + vals[mid]) / 2.0

    def classify(
        self,
        satellite_id: Optional[str],
        rms: Optional[float],
        *,
        whispered: bool = False,
    ) -> str:
        """Label this utterance and fold it into the baseline.

        Whispering wins outright — it is a measured acoustic property,
        not a loudness judgement, and the whisper detector is better at
        it than a level comparison could be.
        """
        key = satellite_id or "_"
        level = float(rms) if rms else 0.0
        baseline = self._median(key)
        # A whisper still updates the baseline; excluding it would make
        # the median drift up in a room where someone often whispers.
        if level > 0:
            self._levels[key].append(level)

        if whispered:
            return STYLE_WHISPERED
        if baseline is None or baseline <= 0 or level <= 0:
            return STYLE_NORMAL
        ratio = level / baseline
        if ratio <= _QUIET_RATIO:
            return STYLE_QUIET
        if ratio >= _ANIMATED_RATIO:
            return STYLE_ANIMATED
        return STYLE_NORMAL

    def baseline_for(self, satellite_id: Optional[str]) -> Optional[float]:
        """The current reference level, or None while still learning."""
        return self._median(satellite_id or "_")
