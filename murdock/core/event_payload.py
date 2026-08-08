"""Canonical recognition-event payload.

Single source of truth for the JSON shape of ``speaker_recognition_detected``
— fired to HA via REST and mirrored on the MQTT event topic. The HA custom
integration consumes exactly this shape, so both sinks must never drift
apart.

Semantics (integration plan §21):
- ``speaker`` is the speaker name for a verified match, ``None`` otherwise
  — the event fires on EVERY completed utterance, including non-recognition,
  so a stale speaker never lingers in downstream state.
- ``reason`` explains a null speaker: "unknown", "uncertain", "tv-noise",
  "early-reject", "short", "no-speakers".
- ``nearest_speaker``/``nearest_distance`` name the closest enrolled
  profile (for a match: the runner-up, which feeds the margin gate).
"""

from __future__ import annotations

import time
from typing import Any, Dict, List, Optional


def build_recognition_payload(
    *,
    speaker: str,
    is_known: bool,
    confidence: float,
    satellite_id: Optional[str],
    distance: Optional[float] = None,
    threshold: Optional[float] = None,
    nearest_speaker: Optional[str] = None,
    nearest_distance: Optional[float] = None,
    weight: Optional[float] = None,
    margin: Optional[float] = None,
    uncertain: bool = False,
    reason: Optional[str] = None,
    role: Optional[str] = None,
    emotion: Optional[str] = None,
    emotion_confidence: Optional[float] = None,
    ambiguities: Optional[List[Dict[str, str]]] = None,
    whisper: bool = False,
    speakers: Optional[List[Dict[str, Any]]] = None,
) -> Dict[str, Any]:
    """Build the event payload dict.

    ``speaker`` still arrives as the legacy display string ("unknown",
    "tv-noise", …) from the handler; it is nulled here for non-matches and
    the display string moves into ``reason`` unless an explicit reason was
    given.
    """
    payload: Dict[str, Any] = {
        "speaker": speaker if is_known else None,
        "confidence": round(confidence, 4),
        "satellite_id": satellite_id,
        "is_known": is_known,
        "timestamp": time.time(),
    }
    if not is_known:
        payload["reason"] = reason or (speaker if speaker else "unknown")
    if distance is not None:
        payload["distance"] = round(distance, 4)
    if threshold is not None:
        payload["threshold"] = round(threshold, 4)
    if nearest_speaker is not None:
        payload["nearest_speaker"] = nearest_speaker
    if nearest_distance is not None:
        payload["nearest_distance"] = round(nearest_distance, 4)
    if weight is not None:
        payload["weight"] = round(weight, 4)
    if margin is not None:
        payload["margin"] = round(margin, 4)
    if uncertain:
        payload["uncertain"] = True
    if role is not None:
        payload["role"] = role
    if emotion is not None:
        payload["emotion"] = emotion
    if emotion_confidence is not None:
        payload["emotion_confidence"] = round(emotion_confidence, 4)
    if ambiguities:
        payload["ambiguities"] = ambiguities
    if whisper:
        payload["whisper"] = True
    # More than one enrolled voice in the clip — the gate followed the
    # dominant one, but "who else was there" is useful on its own.
    if speakers and len(speakers) > 1:
        payload["speakers"] = speakers
    return payload
