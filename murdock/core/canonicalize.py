"""Map a transcript onto the entity names that actually exist.

The STT engine guesses at names it has never seen; Murdock knows the
exact list of valid ones (mirrored from Home Assistant's registry). So
instead of trying to *persuade* the engine up front — a bias prompt only
some backends even accept — the transcript is corrected afterwards:
spans that are unmistakably a near-miss of a known name are replaced by
that name.

This works for every backend, including the default local Wyoming
upstream, and it produces plain deterministic text, so Home Assistant's
local intent matching gets *better* rather than worse — "Bad-Lightstrip"
matches no entity, "Bett-Lightstrip" does.

Three stages, deliberately conservative:

1. **Candidates** — Kölner Phonetik indexes the vocabulary. German STT
   confusions are phonetic ("Bad" and "Bett" both encode to ``12``), so
   this finds the plausible pairs that pure edit distance would rank far
   apart.
2. **Scoring** — sequence ratio, character-trigram overlap and a
   phonetic-equality bonus. Phonetics may *propose*, never decide: "Bad"
   and "Bett" sound alike, and only the vocabulary knows which one is a
   real entity.
3. **Gating** — a replacement needs both an absolute score and a clear
   lead over the runner-up candidate. Same reasoning as the speaker
   margin gate: a narrow win is a guess, and a wrong replacement is
   worse than no replacement.
"""

from __future__ import annotations

import difflib
import logging
import re
import unicodedata
from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional, Sequence, Set, Tuple

_LOGGER = logging.getLogger("murdock.canonicalize")

# Conservative defaults. The score is a weighted blend (see _score), so
# 0.82 means "clearly the same word, mangled" rather than "vaguely
# similar". The margin keeps ambiguous pairs — two entities that are both
# plausible — out of the deterministic path; those belong in a hint.
DEFAULT_MIN_SCORE = 0.82
DEFAULT_MIN_MARGIN = 0.10

# Identical Kölner Phonetik codes are the signature of a mishearing
# rather than a coincidence, so that case is allowed a lower lexical bar
# — the pattern the reference implementation calls a "strong-evidence
# policy". It still has to clear the margin gate, so an ambiguous pair is
# never decided this way.
_PHONETIC_RELIEF = 0.10

# Cheap blocking bar: below this trigram overlap and without a phonetic
# match, a term is not a contender and is skipped before the (much more
# expensive) sequence-ratio computation.
_BLOCKING_TRIGRAM = 0.2

# Signal weights: lexical similarity carries the decision, phonetics only
# tips an already-close call.
_W_RATIO = 0.5
_W_TRIGRAM = 0.3
_W_PHONETIC = 0.2

# Spans shorter than this are never replaced — too little signal, and
# short German words collide with everything.
_MIN_SPAN_CHARS = 4

# Common German words that must never be rewritten into an entity name,
# however close the match. Without this, "Licht an" happily becomes
# "Licht Anna". Deliberately short: the score and margin gates do the
# heavy lifting, this only covers the highest-traffic collisions.
_STOPWORDS = frozenset("""
aber alle als am an auch auf aus bei bin bis bist da damit dann das dass
dein deine dem den denn der des dessen die dies diese dieser dieses doch
dort du ein eine einem einen einer eines er es etwas euer für gegen
gewesen habe haben hat hatte hier hin ich ihr im in ist ja jetzt kann
kein keine machen mach mal man mehr mein meine mich mir mit nach nicht
nichts noch nun nur ob oder ohne schon sehr sein seine seid sind so
soll sowie über um und uns unser vom von vor war waren was weg weil
weiter wenn wer werde werden wie wieder will wir wird wo zu zum zur
""".split())


@dataclass
class Replacement:
    """One applied correction, for the log and the learning loop."""

    original: str
    replacement: str
    score: float
    margin: float

    def as_dict(self) -> dict:
        return {
            "original": self.original,
            "replacement": self.replacement,
            "score": round(self.score, 4),
            "margin": round(self.margin, 4),
        }


def _fold(text: str) -> str:
    """Lowercase, strip accents, unify ß — the comparison form."""
    text = unicodedata.normalize("NFKD", text.replace("ß", "ss"))
    text = "".join(c for c in text if not unicodedata.combining(c))
    return text.lower()


# Kölner Phonetik ------------------------------------------------------
#
# Chosen over Soundex/Metaphone because those are tuned for English.
# Codes: 0 vowels, 1 b/p, 2 d/t, 3 f/v/w, 4 g/k/q, 5 l, 6 m/n, 7 r,
# 8 s/z/c(soft), 48 x.

_KP_BEFORE_C_HARD = set("AHKLOQRUX")
_KP_C_HARD_INNER = set("AHKOQUX")


def koelner_phonetik(word: str) -> str:
    """Return the Kölner Phonetik code of a single word ("" if empty)."""
    w = _fold(word)
    w = re.sub(r"[^a-z]", "", w).upper()
    if not w:
        return ""

    codes: List[str] = []
    for i, ch in enumerate(w):
        # Explicit sets, never `x in "STR"`: at word edges the neighbour is
        # the empty string, and "" is a substring of everything — which
        # silently sent every final D/T down the soft branch.
        prev = w[i - 1] if i else None
        nxt = w[i + 1] if i + 1 < len(w) else None
        if ch in "AEIJOUY":
            codes.append("0")
        elif ch == "H":
            continue
        elif ch == "B":
            codes.append("1")
        elif ch == "P":
            codes.append("3" if nxt == "H" else "1")
        elif ch in "DT":
            codes.append("8" if nxt in ("C", "S", "Z") else "2")
        elif ch in "FVW":
            codes.append("3")
        elif ch in "GKQ":
            codes.append("4")
        elif ch == "X":
            # After a k-sound the "k" part is already encoded.
            codes.append("8" if prev in ("C", "K", "Q") else "48")
        elif ch == "L":
            codes.append("5")
        elif ch in "MN":
            codes.append("6")
        elif ch == "R":
            codes.append("7")
        elif ch in "SZ":
            codes.append("8")
        elif ch == "C":
            if i == 0:
                codes.append("4" if nxt in _KP_BEFORE_C_HARD else "8")
            elif prev in ("S", "Z"):
                codes.append("8")
            else:
                codes.append("4" if nxt in _KP_C_HARD_INNER else "8")
        # Anything else (digits were stripped) contributes nothing.

    flat = "".join(codes)
    # Collapse repeats, then drop zeros except a leading one.
    collapsed: List[str] = []
    for c in flat:
        if not collapsed or collapsed[-1] != c:
            collapsed.append(c)
    head, tail = collapsed[:1], collapsed[1:]
    return "".join(head + [c for c in tail if c != "0"])


def phonetic_key(text: str) -> str:
    """Phonetic code of a whole span (words joined, order preserved)."""
    parts = [koelner_phonetik(w) for w in re.split(r"\s+", text.strip()) if w]
    return "-".join(p for p in parts if p)


_WS_RE = re.compile(r"\s+")


def _trigrams(text: str) -> Set[str]:
    inner = _WS_RE.sub(" ", text.strip())
    s = "  " + inner + "  "
    return {s[i:i + 3] for i in range(len(s) - 2)}


def _jaccard(a: Set[str], b: Set[str]) -> float:
    if not a or not b:
        return 0.0
    return len(a & b) / len(a | b)


def _score(
    folded_span: str,
    span_trigrams: Set[str],
    span_phonetic: str,
    cand: "_Candidate",
) -> Tuple[float, bool]:
    """Blended similarity plus whether the phonetic codes are identical."""
    trig = _jaccard(span_trigrams, cand.trigrams)
    phonetic_hit = bool(span_phonetic) and span_phonetic == cand.phonetic
    ratio = difflib.SequenceMatcher(None, folded_span, cand.folded).ratio()
    score = _W_RATIO * ratio + _W_TRIGRAM * trig
    if phonetic_hit:
        score += _W_PHONETIC
    return score, phonetic_hit


@dataclass
class _Candidate:
    term: str
    folded: str
    phonetic: str
    trigrams: Set[str]
    token_count: int


class Canonicalizer:
    """Rewrites near-misses of known vocabulary terms in a transcript."""

    def __init__(
        self,
        terms: Iterable[str],
        *,
        min_score: float = DEFAULT_MIN_SCORE,
        min_margin: float = DEFAULT_MIN_MARGIN,
    ) -> None:
        self.min_score = min_score
        self.min_margin = min_margin
        self._candidates: List[_Candidate] = []
        self._known_folded: Set[str] = set()
        seen: Set[str] = set()
        for term in terms:
            term = (term or "").strip()
            if not term:
                continue
            folded = _fold(term)
            self._known_folded.add(folded)
            if folded in seen:
                continue
            seen.add(folded)
            self._candidates.append(
                _Candidate(
                    term=term,
                    folded=folded,
                    phonetic=phonetic_key(term),
                    trigrams=_trigrams(folded),
                    token_count=len(term.split()),
                )
            )
        self._max_tokens = max(
            (c.token_count for c in self._candidates), default=0
        )

    @property
    def term_count(self) -> int:
        return len(self._candidates)

    def _best_two(self, folded_span: str) -> Tuple[
        Optional[_Candidate], float, float, bool
    ]:
        """Best candidate for a span, its lead over the runner-up, and
        whether that best match is phonetically identical.

        Trigram overlap acts as a cheap blocking step: computing the
        sequence ratio against every term for every window would cost
        hundreds of milliseconds in the response path, and a term sharing
        almost no character trigrams *and* no phonetic code is never a
        serious contender. The margin is therefore measured among the
        candidates that survive blocking.
        """
        span_trigrams = _trigrams(folded_span)
        span_phonetic = phonetic_key(folded_span)
        best: Optional[_Candidate] = None
        best_score = 0.0
        best_phonetic = False
        second_score = 0.0
        for cand in self._candidates:
            if (
                _jaccard(span_trigrams, cand.trigrams) < _BLOCKING_TRIGRAM
                and span_phonetic != cand.phonetic
            ):
                continue
            s, phonetic_hit = _score(
                folded_span, span_trigrams, span_phonetic, cand
            )
            if s > best_score:
                best, best_score, second_score = cand, s, best_score
                best_phonetic = phonetic_hit
            elif s > second_score:
                second_score = s
        return best, best_score, best_score - second_score, best_phonetic

    def canonicalize(self, transcript: str) -> Tuple[str, List[Replacement]]:
        """Return the corrected transcript and what was changed.

        Longer spans are considered first, so a two-word entity name wins
        over one of its words matching something else. Text already
        matching a known term is left alone — there is nothing to fix.
        """
        if not transcript or not self._candidates:
            return transcript, []

        tokens = transcript.split()
        if not tokens:
            return transcript, []

        replacements: List[Replacement] = []
        consumed: Set[int] = set()
        out: List[Optional[str]] = list(tokens)
        max_n = min(self._max_tokens, len(tokens)) or 1

        for n in range(max_n, 0, -1):
            for start in range(0, len(tokens) - n + 1):
                idx = range(start, start + n)
                if any(i in consumed for i in idx):
                    continue
                span = " ".join(tokens[start:start + n])
                stripped = span.strip(".,;:!?\"'“”„»«")
                if len(stripped) < _MIN_SPAN_CHARS:
                    continue
                folded = _fold(stripped)
                if folded in self._known_folded:
                    continue  # already a valid name
                if n == 1 and folded in _STOPWORDS:
                    continue
                cand, score, margin, phonetic_hit = self._best_two(folded)
                if cand is None:
                    continue
                floor = self.min_score
                if phonetic_hit:
                    floor -= _PHONETIC_RELIEF
                if score < floor:
                    continue
                if margin < self.min_margin:
                    _LOGGER.debug(
                        "Ambiguous span %r: %r scored %.3f but only leads by "
                        "%.3f — left alone", stripped, cand.term, score, margin,
                    )
                    continue
                # Keep whatever punctuation surrounded the span.
                lead = span[:len(span) - len(span.lstrip(".,;:!?\"'“”„»«"))]
                trail = span[len(span.rstrip(".,;:!?\"'“”„»«")):]
                out[start] = f"{lead}{cand.term}{trail}"
                for i in list(idx)[1:]:
                    out[i] = None
                consumed.update(idx)
                replacements.append(
                    Replacement(
                        original=stripped,
                        replacement=cand.term,
                        score=score,
                        margin=margin,
                    )
                )

        if not replacements:
            return transcript, []
        corrected = " ".join(tok for tok in out if tok is not None)
        for r in replacements:
            _LOGGER.info(
                "Canonicalised %r → %r (score=%.3f, margin=%.3f)",
                r.original, r.replacement, r.score, r.margin,
            )
        return corrected, replacements


def canonicalize(
    transcript: str,
    terms: Sequence[str],
    *,
    min_score: float = DEFAULT_MIN_SCORE,
    min_margin: float = DEFAULT_MIN_MARGIN,
) -> Tuple[str, List[Replacement]]:
    """One-shot convenience wrapper around :class:`Canonicalizer`."""
    return Canonicalizer(
        terms, min_score=min_score, min_margin=min_margin
    ).canonicalize(transcript)
