"""Dispatcher handlers must stay on the event loop.

Home Assistant classifies a dispatcher target through `HassJob`: a
coroutine runs on the loop, a function marked `@callback` runs on the
loop, and **anything else is treated as an executor job and run in a
worker thread**. A plain `def` handler that calls `async_write_ha_state`
therefore writes state from the wrong thread — which Home Assistant logs
as a warning about crashing or corrupting data, and which is how this
was reported.

Parsed rather than imported: the integration needs Home Assistant to
import, which the test image deliberately does not carry.
"""

from __future__ import annotations

import ast
import pathlib

_COMPONENT = (
    pathlib.Path(__file__).resolve().parent.parent
    / "custom_components" / "murdock"
)


def _decorated_with_callback(node: ast.FunctionDef) -> bool:
    for dec in node.decorator_list:
        if isinstance(dec, ast.Name) and dec.id == "callback":
            return True
        if isinstance(dec, ast.Attribute) and dec.attr == "callback":
            return True
    return False


def _handlers():
    for path in sorted(_COMPONENT.glob("*.py")):
        tree = ast.parse(path.read_text(encoding="utf-8"))
        for node in ast.walk(tree):
            if isinstance(node, ast.FunctionDef) and node.name == "_handle_update":
                yield path.name, node


def test_every_dispatch_handler_is_a_callback():
    plain = [
        f"{fname}:{node.lineno}"
        for fname, node in _handlers()
        if not _decorated_with_callback(node)
    ]
    assert not plain, (
        "these run in a worker thread and must not write state: "
        f"{plain}"
    )


def test_handlers_exist_at_all():
    """Guards the guard: a renamed handler must not silently pass."""
    assert list(_handlers()), "no _handle_update found — did they move?"


def test_state_writes_are_guarded_against_a_removed_entity():
    """A dispatch can land while an entity is being torn down."""
    for fname, node in _handlers():
        src = ast.unparse(node)
        if "async_write_ha_state" not in src:
            continue
        assert "self.hass is None" in src, (
            f"{fname}:{node.lineno} writes state without checking that the "
            "entity is still attached to hass"
        )
