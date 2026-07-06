"""Transcript post-processing: correction dictionary and dual-engine merge.

Two of the three weapons against systematic STT mistakes (the third —
vocabulary biasing via the backend ``prompt`` — lives in stt_backend):

* **Correction dictionary** — user-maintained list of known
  misrecognitions. Each entry either *replaces* the wrong phrase with
  the right one (deterministic, keeps HA's local intent matching
  working — the entity is called "Fehenlichter", not "fehlende
  Lichter") or *annotates* it with the likely correction as an
  alternative reading for an LLM agent.

* **Dual-transcript merge** — align the primary and shadow engines'
  transcripts word-by-word and mark disagreements inline as
  ``primary [oder: shadow]``. Engines fail on *different* words, so
  the union carries strictly more information than either alone; the
  conversation agent picks the reading that makes sense.

Both are pure text functions, deliberately free of any Murdock state so
they are trivially testable.
"""

from __future__ import annotations

import difflib
import logging
import re
import string
from typing import List, Tuple

_LOGGER = logging.getLogger("murdock.transcript_tools")

MODE_REPLACE = "replace"
MODE_ANNOTATE = "annotate"

# Entry separators in the dictionary text: "wrong -> right" replaces,
# "wrong ~> right" annotates.
_SEP_RE = re.compile(r"\s*(->|~>)\s*")

# Below this SequenceMatcher ratio the transcripts are considered too
# different for a word merge — show both readings whole instead.
_MERGE_MIN_SIMILARITY = 0.3

_PUNCT_TABLE = str.maketrans("", "", string.punctuation + "„“”‚‘’«»…")


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


def apply_correction_dictionary(
    transcript: str, entries: List[Tuple[str, str, str]]
) -> str:
    """Apply parsed dictionary entries to a transcript.

    Matching is case-insensitive on word boundaries (multi-word phrases
    allowed). Replace mode substitutes the correction; annotate mode
    keeps the recognised text and appends ``[oder: correction]`` so an
    LLM can choose.
    """
    if not transcript or not entries:
        return transcript
    out = transcript
    for wrong, right, mode in entries:
        pattern = re.compile(
            r"\b" + re.escape(wrong) + r"\b", re.IGNORECASE
        )
        if mode == MODE_REPLACE:
            out = pattern.sub(right, out)
        else:
            out = pattern.sub(lambda m: f"{m.group(0)} [oder: {right}]", out)
    return out


def _norm_token(token: str) -> str:
    """Casefold and strip punctuation for diff matching."""
    return token.translate(_PUNCT_TABLE).casefold()


def merge_transcripts(primary: str, shadow: str) -> str:
    """Merge two engines' transcripts, marking disagreements inline.

    The primary transcript is the base; wherever the shadow reads
    differently, its reading is appended as ``[oder: …]``. Tokens are
    compared casefolded and punctuation-stripped so "Lichter," vs
    "lichter" never triggers a false disagreement. When the transcripts
    barely overlap, a word merge would be noise — the shadow reading is
    then appended whole as ``[alternative Lesart: …]``.
    """
    p, s = (primary or "").strip(), (shadow or "").strip()
    if not p:
        return s
    if not s:
        return p

    p_tokens = p.split()
    s_tokens = s.split()
    p_norm = [_norm_token(t) for t in p_tokens]
    s_norm = [_norm_token(t) for t in s_tokens]
    if p_norm == s_norm:
        return p

    matcher = difflib.SequenceMatcher(a=p_norm, b=s_norm, autojunk=False)
    if matcher.ratio() < _MERGE_MIN_SIMILARITY:
        return f"{p} [alternative Lesart: {s}]"

    out: List[str] = []
    for op, i1, i2, j1, j2 in matcher.get_opcodes():
        if op == "equal":
            out.extend(p_tokens[i1:i2])
        elif op == "replace":
            out.extend(p_tokens[i1:i2])
            out.append(f"[oder: {' '.join(s_tokens[j1:j2])}]")
        elif op == "delete":
            # Primary-only words: keep them, the base is the primary.
            out.extend(p_tokens[i1:i2])
        elif op == "insert":
            # Shadow heard extra words the primary missed — they can
            # flip the meaning ("nicht"!), so surface them.
            out.append(f"[oder zusätzlich: {' '.join(s_tokens[j1:j2])}]")
    return " ".join(out)
