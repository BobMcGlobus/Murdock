"""Turn the enrollment data Murdock already has into concrete advice.

Sample quality scores, per-satellite tags and the recognition log all
exist, but they only answer questions you already thought to ask. This
module inverts that: it looks for the *specific* reasons a profile
underperforms and names the fix — "Anna has no samples from the kitchen
satellite, add three there" beats a quality number every time.

Every finding carries the numbers behind it, so the advice can be
checked rather than believed. Deliberately conservative: silence is the
right answer for a healthy profile, and a wall of nitpicks would train
users to ignore the panel.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence

_LOGGER = logging.getLogger("murdock.enrollment_coach")

# A profile below this many samples is thin no matter how good they are:
# a single recording session can't cover how a voice actually varies.
MIN_HEALTHY_SAMPLES = 3

# Mean composite quality below this is worth mentioning.
LOW_QUALITY_MEAN = 0.45

# Distance headroom: matches landing this close to the threshold are
# only just passing, and small changes (a cold, a moved satellite) will
# start pushing them over.
TIGHT_HEADROOM = 0.06

# A satellite needs this many same-mic samples before a sub-profile is
# built. Mirrors SATELLITE_PROFILE_MIN_SAMPLES in speaker_store.
SATELLITE_PROFILE_MIN_SAMPLES = 3

# Only advise on satellites the speaker actually uses.
MIN_EVENTS_FOR_SATELLITE_ADVICE = 5

# Drift: compare the newest matches against the older ones.
DRIFT_MIN_EVENTS = 12
DRIFT_DELTA = 0.05

SEVERITY_INFO = "info"
SEVERITY_WARN = "warn"


@dataclass
class Finding:
    """One actionable observation about one speaker."""

    speaker: str
    code: str
    severity: str
    message: str
    detail: Dict = field(default_factory=dict)

    def as_dict(self) -> Dict:
        return {
            "speaker": self.speaker,
            "code": self.code,
            "severity": self.severity,
            "message": self.message,
            "detail": self.detail,
        }


def _mean(values: Sequence[float]) -> Optional[float]:
    vals = [v for v in values if v is not None]
    return sum(vals) / len(vals) if vals else None


def analyse_speaker(
    *,
    name: str,
    samples: Sequence[Dict],
    events: Sequence[Dict],
    threshold: float,
) -> List[Finding]:
    """Findings for a single speaker.

    ``samples`` are rows from ``speaker_samples`` (``quality_score``,
    ``satellite_id``); ``events`` are that speaker's matches from the
    recognition log (``distance``, ``satellite_id``, ``created_at``,
    newest first).
    """
    findings: List[Finding] = []

    # 1. Too few samples — the most common real problem.
    if len(samples) < MIN_HEALTHY_SAMPLES:
        findings.append(
            Finding(
                speaker=name,
                code="few_samples",
                severity=SEVERITY_WARN,
                message=(
                    f"Only {len(samples)} sample(s) enrolled. Add at least "
                    f"{MIN_HEALTHY_SAMPLES - len(samples)} more, ideally "
                    f"recorded on different days."
                ),
                detail={"samples": len(samples),
                        "recommended": MIN_HEALTHY_SAMPLES},
            )
        )

    # 2. Poor material. Quality scores are only present for samples that
    #    went through scoring, so absence is not a finding.
    quality = _mean([s.get("quality_score") for s in samples])
    if quality is not None and quality < LOW_QUALITY_MEAN:
        findings.append(
            Finding(
                speaker=name,
                code="low_quality",
                severity=SEVERITY_INFO,
                message=(
                    f"Average sample quality is {quality:.2f}. Re-record in a "
                    f"quieter moment, closer to the microphone."
                ),
                detail={"quality_mean": round(quality, 3),
                        "bar": LOW_QUALITY_MEAN},
            )
        )

    # 3. Satellites this speaker uses but has no samples from. This is
    #    the concrete one: without same-mic samples no sub-profile is
    #    built, so microphone colouring is never compensated.
    events_by_sat: Dict[str, int] = {}
    for e in events:
        sat = e.get("satellite_id")
        if sat:
            events_by_sat[sat] = events_by_sat.get(sat, 0) + 1
    samples_by_sat: Dict[str, int] = {}
    for s in samples:
        sat = s.get("satellite_id")
        if sat:
            samples_by_sat[sat] = samples_by_sat.get(sat, 0) + 1

    for sat, count in sorted(
        events_by_sat.items(), key=lambda kv: kv[1], reverse=True
    ):
        if count < MIN_EVENTS_FOR_SATELLITE_ADVICE:
            continue
        have = samples_by_sat.get(sat, 0)
        if have >= SATELLITE_PROFILE_MIN_SAMPLES:
            continue
        missing = SATELLITE_PROFILE_MIN_SAMPLES - have
        findings.append(
            Finding(
                speaker=name,
                code="satellite_coverage",
                severity=SEVERITY_WARN,
                message=(
                    f"Speaks via “{sat}” often ({count} recognitions) but has "
                    f"{have} sample(s) from it — {missing} more would enable a "
                    f"per-satellite voice profile and cancel that "
                    f"microphone's colouring."
                ),
                detail={
                    "satellite_id": sat,
                    "events": count,
                    "samples": have,
                    "missing": missing,
                },
            )
        )

    # 4. Passing, but only just.
    distances = [e["distance"] for e in events if e.get("distance") is not None]
    mean_distance = _mean(distances)
    if mean_distance is not None and distances:
        headroom = threshold - mean_distance
        if 0 <= headroom < TIGHT_HEADROOM:
            findings.append(
                Finding(
                    speaker=name,
                    code="tight_headroom",
                    severity=SEVERITY_WARN,
                    message=(
                        f"Matches average {mean_distance:.3f} against a "
                        f"threshold of {threshold:.3f} — only {headroom:.3f} "
                        f"of headroom. More samples would push this down."
                    ),
                    detail={
                        "mean_distance": round(mean_distance, 4),
                        "threshold": round(threshold, 4),
                        "headroom": round(headroom, 4),
                    },
                )
            )

    # 5. Drift: newest matches consistently worse than older ones. A
    #    voice changes, microphones move, rooms get rearranged.
    if len(distances) >= DRIFT_MIN_EVENTS:
        half = len(distances) // 2
        recent = _mean(distances[:half])      # events are newest-first
        older = _mean(distances[half:])
        if recent is not None and older is not None:
            delta = recent - older
            if delta >= DRIFT_DELTA:
                findings.append(
                    Finding(
                        speaker=name,
                        code="drift",
                        severity=SEVERITY_INFO,
                        message=(
                            f"Recent matches are {delta:.3f} worse than older "
                            f"ones ({older:.3f} → {recent:.3f}). The profile "
                            f"may be going stale — add a fresh sample."
                        ),
                        detail={
                            "recent_mean": round(recent, 4),
                            "older_mean": round(older, 4),
                            "delta": round(delta, 4),
                        },
                    )
                )

    return findings


def analyse(
    *,
    speakers: Sequence[Dict],
    samples_by_speaker: Dict[int, Sequence[Dict]],
    events_by_speaker: Dict[str, Sequence[Dict]],
    threshold: float,
) -> List[Finding]:
    """Findings across every enrolled speaker, most severe first."""
    out: List[Finding] = []
    for sp in speakers:
        out.extend(
            analyse_speaker(
                name=sp["name"],
                samples=samples_by_speaker.get(sp["id"], []),
                events=events_by_speaker.get(sp["name"], []),
                threshold=threshold,
            )
        )
    order = {SEVERITY_WARN: 0, SEVERITY_INFO: 1}
    out.sort(key=lambda f: (order.get(f.severity, 2), f.speaker))
    return out
