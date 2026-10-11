"""Harness for running the integration against a real Home Assistant.

Two adjustments, both about the test environment rather than the code:

* Home Assistant's test config dir carries its own ``custom_components``
  package and is put first on ``sys.path``; importing ours before that
  happens keeps it the one Home Assistant finds.
* On Windows (local runs only) the event loop needs a socket pair for
  itself, which the plugin's socket blocker refuses. HTTP is mocked by
  ``aioclient_mock`` regardless, so nothing reaches a network.
"""

import pathlib
import sys

ROOT = pathlib.Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import custom_components.murdock  # noqa: E402,F401

if sys.platform == "win32":
    import pytest_socket

    pytest_socket.disable_socket = lambda *args, **kwargs: None
    pytest_socket.socket_allow_hosts = lambda *args, **kwargs: None
