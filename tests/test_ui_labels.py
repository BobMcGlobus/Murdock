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
