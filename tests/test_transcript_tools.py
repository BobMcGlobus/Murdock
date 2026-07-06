"""Tests for the correction dictionary and the dual-transcript merge."""

from __future__ import annotations

import pytest

from murdock.core.transcript_tools import (
    apply_correction_dictionary,
    merge_transcripts,
    parse_correction_dictionary,
)


# --- dictionary parsing --------------------------------------------------------


def test_parse_modes_comments_and_invalid_lines():
    text = """
    # bekannte Fehler
    fehlende Lichter -> Fehenlichter
    Bad-Lightstrip ~> Bed-Lightstrip   # Voxtral-Klassiker
    kaputte zeile ohne separator
    -> nur rechts
    links ->
    """
    entries = parse_correction_dictionary(text)
    assert ("fehlende Lichter", "Fehenlichter", "replace") in entries
    assert ("Bad-Lightstrip", "Bed-Lightstrip", "annotate") in entries
    assert len(entries) == 2


def test_parse_sorts_longest_first():
    entries = parse_correction_dictionary(
        "licht -> Licht\nfehlende lichter -> Fehenlichter"
    )
    assert entries[0][0] == "fehlende lichter"


# --- dictionary application ----------------------------------------------------


def test_replace_is_case_insensitive_and_word_bounded():
    entries = parse_correction_dictionary("fehlende Lichter -> Fehenlichter")
    out = apply_correction_dictionary(
        "Schalte Fehlende lichter aus", entries
    )
    assert out == "Schalte Fehenlichter aus"
    # "lichterkette" must not match "lichter" as a substring.
    entries2 = parse_correction_dictionary("lichter -> Lichter")
    assert apply_correction_dictionary("Die Lichterkette an", entries2) == \
        "Die Lichterkette an"


def test_annotate_keeps_original_and_appends_alternative():
    entries = parse_correction_dictionary("Bad-Lightstrip ~> Bed-Lightstrip")
    out = apply_correction_dictionary("Mach den Bad-Lightstrip an", entries)
    assert out == "Mach den Bad-Lightstrip [oder: Bed-Lightstrip] an"


def test_longer_phrase_wins_over_substring():
    entries = parse_correction_dictionary(
        "lichter -> LEDs\nfehlende lichter -> Fehenlichter"
    )
    out = apply_correction_dictionary("schalte fehlende lichter aus", entries)
    assert "Fehenlichter" in out
    assert "LEDs" not in out


def test_apply_noop_without_entries():
    assert apply_correction_dictionary("hallo", []) == "hallo"
    assert apply_correction_dictionary("", [("a", "b", "replace")]) == ""


# --- dual merge -----------------------------------------------------------------


def test_merge_identical_returns_primary():
    assert merge_transcripts("Licht an", "Licht an") == "Licht an"


def test_merge_ignores_case_and_punctuation_noise():
    # Different casing/punctuation must not create a fake disagreement.
    assert merge_transcripts(
        "Schalte das Licht an.", "schalte das licht an"
    ) == "Schalte das Licht an."


def test_merge_marks_single_disagreement():
    out = merge_transcripts(
        "Schalte fehlende Lichter aus",
        "Schalte Fehenlichter aus",
    )
    assert out == "Schalte fehlende Lichter [oder: Fehenlichter] aus"


def test_merge_marks_shadow_only_words():
    out = merge_transcripts("mach das Licht an", "mach das Licht nicht an")
    assert "[oder zusätzlich: nicht]" in out


def test_merge_empty_sides():
    assert merge_transcripts("", "nur schatten") == "nur schatten"
    assert merge_transcripts("nur primär", "") == "nur primär"


def test_merge_low_similarity_appends_whole_alternative():
    out = merge_transcripts(
        "Wetterbericht für morgen bitte",
        "Der Drucker im Keller brennt",
    )
    assert out.startswith("Wetterbericht für morgen bitte")
    assert "[alternative Lesart: Der Drucker im Keller brennt]" in out
