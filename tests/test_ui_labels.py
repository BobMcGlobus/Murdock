"""Every outcome the handler can produce must be readable in the UI.

`uncertain-forwarded` shipped for months rendering as its own raw key in
the recognition log, because adding an outcome in Python and adding its
label in JavaScript are two separate acts and nothing connected them.
"""

from __future__ import annotations

import pathlib
import re

import murdock.core.recognition_log as rl

_UI = pathlib.Path(__file__).resolve().parent.parent / "murdock" / "ui" / "static"


def _declared_outcomes() -> set[str]:
    return {
        getattr(rl, name)
        for name in dir(rl)
        if name.startswith("OUTCOME_") and isinstance(getattr(rl, name), str)
    }


def test_every_outcome_has_a_badge_and_a_translation():
    i18n = (_UI / "i18n.js").read_text(encoding="utf-8")
    mapped = set(re.findall(r'"([a-z-]+)":\s*\{\s*label:\s*t\(', i18n))

    missing_badge = _declared_outcomes() - mapped
    assert not missing_badge, f"no badge for: {sorted(missing_badge)}"

    for outcome in _declared_outcomes():
        key = f'"outcome.{outcome}"'
        # Once per locale, plus the map entry itself.
        assert i18n.count(key) >= 3, f"{key} is missing from a locale"


def test_both_locales_define_the_same_keys():
    """A key present in one language only renders as its own name."""
    i18n = (_UI / "i18n.js").read_text(encoding="utf-8")
    blocks = re.findall(r"\n        (?:en|de):\s*\{(.*?)\n        \},", i18n, re.S)
    assert len(blocks) == 2, "expected exactly an EN and a DE block"
    en, de = ({k for k in re.findall(r'\n\s+"([^"]+)":', b)} for b in blocks)
    assert not (en - de), f"only in EN: {sorted(en - de)[:10]}"
    assert not (de - en), f"only in DE: {sorted(de - en)[:10]}"


def _tag_stream(html: str):
    """Positions of every form/details boundary, in document order."""
    import re

    for m in re.finditer(
        r'<details[^>]*>|</details>|<form[^>]*>|</form>', html
    ):
        yield m.start(), m.group(0)


def test_no_details_block_spans_a_form_boundary():
    """A `</details>` written at the wrong nesting level balances the
    tag count while swallowing everything after it.

    One did exactly that: the A/B shadow block ran past the end of the
    transcription form, through an entire other form, and into the MQTT
    card — taking the Save button with it. Counting tags found nothing
    wrong, because the count was right.
    """
    html = (_UI / "index.html").read_text(encoding="utf-8")
    depth = 0
    offenders = []
    for pos, tag in _tag_stream(html):
        if tag.startswith("<details"):
            depth += 1
        elif tag == "</details>":
            depth -= 1
            assert depth >= 0, f"stray </details> at {pos}"
        elif tag.startswith("</form") and depth:
            offenders.append(pos)
    assert not offenders, (
        f"a <details> is still open where a form ends, at {offenders}"
    )
    assert depth == 0, "a <details> is never closed"


def test_no_form_hides_its_save_button_in_a_collapsed_block():
    """Folding the A/B shadow section away took the only Save button
    with it, leaving the whole transcription form unsavable.

    A `<details>` is closed by default, so a submit button inside one is
    invisible until the user expands a section they have no reason to
    open.
    """
    html = (_UI / "index.html").read_text(encoding="utf-8")
    problems = []
    pos = 0
    while True:
        start = html.find("<form", pos)
        if start == -1:
            break
        end = html.find("</form>", start)
        if end == -1:
            break
        form = html[start:end]
        pos = end

        # Every <details> range inside this form.
        ranges = []
        d = form.find("<details")
        while d != -1:
            close = form.find("</details>", d)
            if close == -1:
                break
            ranges.append((d, close))
            d = form.find("<details", close)

        s = form.find('type="submit"')
        while s != -1:
            if any(lo < s < hi for lo, hi in ranges):
                form_id = form[:60].split('id="')[-1].split('"')[0]
                problems.append(form_id or "(unnamed form)")
            s = form.find('type="submit"', s + 1)

    assert not problems, (
        f"submit button hidden inside a collapsed <details>: {problems}"
    )
