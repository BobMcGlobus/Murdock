"""Transcript post-processing: the correction dictionary and its hints.

One of the weapons against systematic STT mistakes (vocabulary biasing
via the backend ``prompt`` lives in stt_backend, fuzzy name correction
in canonicalizer):

* **Correction dictionary** — user-maintained list of known
  misrecognitions. Each entry either *replaces* the wrong phrase with
  the right one (deterministic, keeps HA's local intent matching
  working — the entity is called "Fehenlichter", not "fehlende
  Lichter") or *annotates* it with the likely correction as an
  alternative reading for an LLM agent.

These are pure text functions, deliberately free of any Murdock state so
they are trivially testable.
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field
from typing import Iterable, List, Optional, Sequence, Tuple

_LOGGER = logging.getLogger("murdock.transcript_tools")

MODE_REPLACE = "replace"
MODE_ANNOTATE = "annotate"

# Hint kinds (plan §13). "alternative" = a different reading of the same
# span. "additional" and "reading" came from the removed dual-transcript
# merge; they stay defined because the HA integration still understands
# them and events from older versions may carry them.
HINT_ALTERNATIVE = "alternative"
HINT_ADDITIONAL = "additional"
HINT_READING = "reading"


@dataclass
class TranscriptHint:
    """One ambiguity: what was heard, and what it might be instead."""

    original: str
    alternative: str
    kind: str = HINT_ALTERNATIVE

    def as_dict(self) -> dict:
        return {
            "original": self.original,
            "alternative": self.alternative,
            "kind": self.kind,
        }


@dataclass
class TranscriptResult:
    """Both renderings of a processed transcript plus its ambiguities.

    ``clean`` keeps HA's local intent matching working; ``annotated``
    carries the ``[oder: …]`` markers inline (legacy behaviour). Which
    one goes to the satellite is the caller's decision — see the
    transcript hint mode.
    """

    clean: str
    annotated: str
    hints: List[TranscriptHint] = field(default_factory=list)

# Entry separators in the dictionary text: "wrong -> right" replaces,
# "wrong ~> right" annotates.
_SEP_RE = re.compile(r"\s*(->|~>)\s*")

# Below this SequenceMatcher ratio the transcripts are considered too
# different for a word merge — show both readings whole instead.



def parse_correction_dictionary(text: str) -> List[Tuple[str, str, str]]:
    """Parse dictionary text into ``(wrong, right, mode)`` entries.

    One entry per line; ``#`` starts a comment; blank/invalid lines are
    skipped. Entries are returned longest-wrong-phrase-first so a longer
    phrase wins over a substring of itself when applied in order.
    """
    entries: List[Tuple[str, str, str]] = []
    for line in (text or "").splitlines():
        line = line.split("#", 1)[0].strip()
        if not line:
            continue
        parts = _SEP_RE.split(line, maxsplit=1)
        if len(parts) != 3:
            continue
        wrong, sep, right = parts[0].strip(), parts[1], parts[2].strip()
        if not wrong or not right:
            continue
        mode = MODE_REPLACE if sep == "->" else MODE_ANNOTATE
        entries.append((wrong, right, mode))
    entries.sort(key=lambda e: len(e[0]), reverse=True)
    return entries


def apply_correction_dictionary_ex(
    transcript: str, entries: List[Tuple[str, str, str]]
) -> TranscriptResult:
    """Apply dictionary entries, keeping clean and annotated renderings.

    Matching is case-insensitive on word boundaries (multi-word phrases
    allowed). Replace mode substitutes the correction in *both*
    renderings — it is deterministic and keeps HA's local intent matching
    working. Annotate mode leaves the clean rendering untouched and
    records the correction as a hint, which the annotated rendering
    additionally spells out inline as ``[oder: correction]``.
    """
    if not transcript or not entries:
        return TranscriptResult(clean=transcript, annotated=transcript)

    clean = transcript
    hints: List[TranscriptHint] = []
    for wrong, right, mode in entries:
        pattern = re.compile(r"\b" + re.escape(wrong) + r"\b", re.IGNORECASE)
        if mode == MODE_REPLACE:
            clean = pattern.sub(right, clean)
            continue
        # Annotate: collect one hint per distinct matched spelling.
        seen: set = set()
        for match in pattern.finditer(clean):
            original = match.group(0)
            if original.casefold() in seen:
                continue
            seen.add(original.casefold())
            hints.append(TranscriptHint(original=original, alternative=right))

    annotated = clean
    for hint in hints:
        pattern = re.compile(
            r"\b" + re.escape(hint.original) + r"\b", re.IGNORECASE
        )
        annotated = pattern.sub(
            lambda m: f"{m.group(0)} [oder: {hint.alternative}]", annotated
        )
    return TranscriptResult(clean=clean, annotated=annotated, hints=hints)


def apply_correction_dictionary(
    transcript: str, entries: List[Tuple[str, str, str]]
) -> str:
    """Annotated rendering only — the inline-marker behaviour."""
    return apply_correction_dictionary_ex(transcript, entries).annotated


def resolve_hints(
    result: TranscriptResult, known_terms: Optional[Iterable[str]]
) -> TranscriptResult:
    """Decide before marking (plan §13).

    An ambiguity is only worth passing on when it is genuinely
    ambiguous. If exactly one of the two readings matches a known entity
    name, that reading wins outright: the clean transcript is rewritten
    to it and the hint is dropped. Marking it anyway would ask the agent
    to re-decide something we already know.

    ``known_terms`` are the vocabulary terms (entity/area/floor names and
    aliases); matching is on the normalized form.
    """
    if not result.hints or not known_terms:
        return result
    from murdock.core.vocabulary_store import normalize_term

    known = {normalize_term(t) for t in known_terms if t}
    if not known:
        return result

    clean = result.clean
    remaining: List[TranscriptHint] = []
    for hint in result.hints:
        if hint.kind != HINT_ALTERNATIVE or not hint.original:
            remaining.append(hint)
            continue
        orig_known = normalize_term(hint.original) in known
        alt_known = normalize_term(hint.alternative) in known
        if orig_known == alt_known:
            # Both or neither are entities — genuinely ambiguous.
            remaining.append(hint)
            continue
        if alt_known:
            pattern = re.compile(
                r"\b" + re.escape(hint.original) + r"\b", re.IGNORECASE
            )
            clean, n = pattern.subn(hint.alternative, clean, count=1)
            if not n:
                remaining.append(hint)
                continue
            _LOGGER.info(
                "Hint resolved to known entity: %r → %r",
                hint.original, hint.alternative,
            )
        else:
            _LOGGER.debug(
                "Hint dropped, heard reading is the known entity: %r",
                hint.original,
            )
    if clean == result.clean and len(remaining) == len(result.hints):
        return result
    return TranscriptResult(
        clean=clean, annotated=result.annotated, hints=remaining
    )


def render_hints(hints: Sequence[TranscriptHint]) -> List[dict]:
    """Serialise hints for the recognition event payload."""
    return [h.as_dict() for h in hints]
