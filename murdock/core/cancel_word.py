"""Recognising an explicit "stop, that wasn't for you".

A wake word fires on a phone call, on the television, on a sentence that
merely rhymed. The utterance is already being processed by then, and the
only thing that reliably stops it is the person saying so.

Matching is deliberately narrow. A cancel word has to *lead* the
utterance or be the whole of it — anywhere-in-the-string matching would
let "kein Abbruch nötig" kill a legitimate request, which is a far worse
failure than missing a cancel. Phonetic comparison covers the ways a
recogniser mangles a short word ("Abruch", "ab Bruch") without opening
that door.
"""

from __future__ import annotations

import logging
from typing import Iterable, List

from .canonicalize import koelner_phonetik

_LOGGER = logging.getLogger("murdock.cancel")

#: Words beyond the cancel phrase that still count as leading it. "Abbruch
#: bitte" and "stop stop stop" are cancels; a whole sentence is not.
_MAX_TRAILING_WORDS = 2


def parse_cancel_words(raw: str) -> List[str]:
    """Split the configured list into individual phrases."""
    if not raw:
        return []
    parts = [p.strip() for p in raw.replace(";", ",").split(",")]
    return [p for p in parts if p]


def _words(text: str) -> List[str]:
    cleaned = "".join(c if c.isalnum() or c.isspace() else " " for c in text)
    return [w for w in cleaned.lower().split() if w]


def _phonetic(word: str) -> str:
    try:
        return koelner_phonetik(word)
    except Exception:  # pragma: no cover - defensive
        return ""


def is_cancel(text: str, cancel_words: Iterable[str]) -> bool:
    """True when ``text`` is a cancellation rather than a request.

    The phrase must start the utterance, and at most a couple of words
    may follow it — enough for "Abbruch bitte", not enough for a command
    that happens to begin with the word.
    """
    words = _words(text or "")
    if not words:
        return False

    for phrase in cancel_words:
        target = _words(phrase)
        if not target or len(target) > len(words):
            continue
        head = words[: len(target)]
        if head == target:
            trailing = len(words) - len(target)
            if trailing <= _MAX_TRAILING_WORDS:
                return True
            continue
        # Phonetic fallback: short words come back mangled often enough
        # that an exact match alone would miss real cancellations.
        head_key = " ".join(_phonetic(w) for w in head)
        target_key = " ".join(_phonetic(w) for w in target)
        if head_key and head_key == target_key:
            trailing = len(words) - len(target)
            if trailing <= _MAX_TRAILING_WORDS:
                return True
    return False
