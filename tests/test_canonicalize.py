"""Tests for entity-name canonicalization (Kölner Phonetik + lexical)."""

from __future__ import annotations

import pytest

from murdock.core.canonicalize import (
    DEFAULT_MIN_MARGIN,
    DEFAULT_MIN_SCORE,
    Canonicalizer,
    canonicalize,
    koelner_phonetik,
    phonetic_key,
)

TERMS = [
    "Bett-Lightstrip", "Deckenlampe", "Wohnzimmer", "Schlafzimmer",
    "Küche", "Fehenlichter", "Stehlampe", "Obergeschoss",
    "Kaffeemaschine", "Fernseher",
]


# ----------------------------------------------------------------------
# Kölner Phonetik
# ----------------------------------------------------------------------


@pytest.mark.parametrize(
    "word,code",
    [
        ("Müller", "657"),
        ("Mueller", "657"),   # umlaut spelling variant collapses to the same
        ("Wikipedia", "3412"),
        ("Fax", "348"),
        ("Tanz", "268"),
        ("Bad", "12"),
        ("Bett", "12"),       # the pair that motivated all of this
        ("Baden", "126"),
    ],
)
def test_koelner_reference_values(word, code):
    assert koelner_phonetik(word) == code


def test_final_consonant_is_not_treated_as_pre_sibilant():
    """Regression: `nxt in "CSZ"` is True for the empty string, which sent
    every word-final D/T down the soft branch (Bad → 18 instead of 12)."""
    assert koelner_phonetik("Bad") == "12"
    assert koelner_phonetik("Rat") == "72"
    # A genuine pre-sibilant D/T still encodes as 8.
    assert koelner_phonetik("Tanz") == "268"


def test_koelner_edge_cases():
    assert koelner_phonetik("") == ""
    assert koelner_phonetik("123") == ""
    assert phonetic_key("Bad-Lightstrip") == phonetic_key("Bett-Lightstrip")
    assert phonetic_key("") == ""


# ----------------------------------------------------------------------
# Corrections that should happen
# ----------------------------------------------------------------------


@pytest.mark.parametrize(
    "text,expected",
    [
        # Phonetically identical, lexically close — the headline case.
        ("schalte das Bad-Lightstrip ein", "schalte das Bett-Lightstrip ein"),
        # Plain misspellings from a mishearing.
        ("mach die Dekenlampe an", "mach die Deckenlampe an"),
        ("spiel Musik im Wohnzimer", "spiel Musik im Wohnzimmer"),
        ("mach die Kaffemaschine an", "mach die Kaffeemaschine an"),
        ("mach den Fernsehr an", "mach den Fernseher an"),
    ],
)
def test_canonicalizes_near_misses(text, expected):
    out, reps = canonicalize(text, TERMS)
    assert out == expected
    assert len(reps) == 1
    assert reps[0].score >= DEFAULT_MIN_SCORE - 0.11


# ----------------------------------------------------------------------
# Corrections that must NOT happen
# ----------------------------------------------------------------------


@pytest.mark.parametrize(
    "text",
    [
        "mach die Deckenlampe im Wohnzimmer an",  # already valid
        "mach das Licht an",                      # common words only
        "wie ist das Wetter heute",
        "erzähl mir einen Witz",
        "wer ist im Garten",                      # no such entity
        "stelle einen Timer",
        "und dann noch das",                       # pure stopwords
    ],
)
def test_leaves_ordinary_speech_alone(text):
    out, reps = canonicalize(text, TERMS)
    assert out == text
    assert reps == []


def test_ambiguous_pair_is_never_decided():
    """Two plausible entities → the margin gate must refuse.

    This is the whole point of the margin: replacing "Bad-Licht" with one
    of two equally close names would be a coin flip, and a wrong
    replacement is worse than none.
    """
    terms = ["Bett-Licht", "Bad-Licht"]
    out, reps = canonicalize("schalte Batt-Licht ein", terms)
    assert out == "schalte Batt-Licht ein"
    assert reps == []


def test_margin_gate_can_be_tightened():
    # Loose margin: the correction goes through.
    out, reps = canonicalize(
        "mach die Dekenlampe an", TERMS, min_margin=0.0
    )
    assert out == "mach die Deckenlampe an"
    # An impossible margin blocks everything.
    out, reps = canonicalize(
        "mach die Dekenlampe an", TERMS, min_margin=1.5
    )
    assert out == "mach die Dekenlampe an"
    assert reps == []


def test_score_floor_can_be_tightened():
    out, reps = canonicalize(
        "schalte das Bad-Lightstrip ein", TERMS, min_score=0.99
    )
    assert reps == []


def test_short_spans_are_ignored():
    # "Bad" alone is below the minimum span length, so it is never touched
    # even though it is phonetically identical to a known term.
    out, reps = canonicalize("Bad", ["Bett"])
    assert out == "Bad"
    assert reps == []


# ----------------------------------------------------------------------
# Mechanics
# ----------------------------------------------------------------------


def test_multi_token_entity_name():
    terms = ["Lampe im Flur"]
    out, reps = canonicalize("schalte Lampe im Flurr aus", terms)
    assert out == "schalte Lampe im Flur aus"
    assert reps[0].original == "Lampe im Flurr"


def test_punctuation_is_preserved():
    out, _ = canonicalize("mach die Dekenlampe an.", TERMS)
    assert out.endswith("an.")
    out, reps = canonicalize("Dekenlampe, bitte", TERMS)
    assert out.startswith("Deckenlampe,")


def test_idempotent():
    once, _ = canonicalize("mach die Dekenlampe an", TERMS)
    twice, reps = canonicalize(once, TERMS)
    assert twice == once
    assert reps == []


def test_empty_inputs():
    assert canonicalize("", TERMS) == ("", [])
    assert canonicalize("mach das Licht an", []) == ("mach das Licht an", [])
    assert canonicalize("text", ["", "   "]) == ("text", [])


def test_replacement_serialises_for_the_event_payload():
    _, reps = canonicalize("mach die Dekenlampe an", TERMS)
    d = reps[0].as_dict()
    assert d["original"] == "Dekenlampe"
    assert d["replacement"] == "Deckenlampe"
    assert 0.0 < d["score"] <= 1.0
    assert d["margin"] >= DEFAULT_MIN_MARGIN


def test_term_count_deduplicates():
    c = Canonicalizer(["Lampe", "lampe", "LAMPE", "Küche"])
    assert c.term_count == 2


def test_stays_fast_on_a_large_vocabulary():
    """The response path can't afford hundreds of milliseconds here."""
    import time

    terms = [f"Gerät {i:03d} Wohnzimmer" for i in range(400)]
    c = Canonicalizer(terms)
    text = "schalte bitte das Licht im Wohnzimmer und in der Küche ein"
    start = time.monotonic()
    for _ in range(10):
        c.canonicalize(text)
    per_call = (time.monotonic() - start) / 10
    assert per_call < 0.05, f"{per_call * 1000:.1f} ms per call"
