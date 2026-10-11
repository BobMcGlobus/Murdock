"""Watching the event loop for stalls.

Murdock serves Wyoming, the web API and MQTT from one asyncio loop, and
the Wyoming handler reads the next audio chunk only once the previous
event has been handled. Anything that holds the loop therefore stops
Murdock reading, Home Assistant's writes back up behind it, and so does
the end-of-speech detection that runs on the same stream. A satellite
then appears to deliver audio slower than real time.

Two instruments tell that apart from a slow sender:

* a **lag probe** — a task that sleeps a fixed interval and measures by
  how much it oversleeps. Overshoot is time the loop was busy elsewhere.
* a **slow-callback log** — every callback the loop runs is timed, and
  one that holds it longer than a threshold is logged by name. This is
  what asyncio's debug mode does, without the rest of debug mode's cost.
"""

from __future__ import annotations

import asyncio
import logging
import time
from collections import deque
from typing import Deque, Optional, Tuple

_LOGGER = logging.getLogger("murdock.loop")

#: How often the lag probe wakes up.
_PROBE_INTERVAL = 0.05

#: A callback holding the loop longer than this is logged by name.
_SLOW_CALLBACK_SEC = 0.15

#: Recent probe overshoots, (monotonic time, lag seconds).
_samples: Deque[Tuple[float, float]] = deque(maxlen=2400)  # ~2 min at 50 ms

_installed = False
_probe_task: Optional[asyncio.Task] = None


def max_lag_since(since: float) -> float:
    """Worst loop stall, in seconds, observed after ``since`` (monotonic)."""
    worst = 0.0
    for at, lag in reversed(_samples):
        if at < since:
            break
        if lag > worst:
            worst = lag
    return worst


def _install_slow_callback_log() -> None:
    """Time every callback the loop runs; name the ones that hold it."""
    global _installed
    if _installed:
        return
    original = asyncio.events.Handle._run

    def _timed_run(self):  # type: ignore[no-untyped-def]
        start = time.perf_counter()
        try:
            return original(self)
        finally:
            held = time.perf_counter() - start
            if held >= _SLOW_CALLBACK_SEC:
                _LOGGER.warning(
                    "Event loop held %.0f ms by %s", held * 1000,
                    _describe(self),
                )

    asyncio.events.Handle._run = _timed_run  # type: ignore[method-assign]
    _installed = True


def _describe(handle) -> str:
    """A readable name for what a loop callback was running."""
    callback = getattr(handle, "_callback", None)
    owner = getattr(callback, "__self__", None)
    if isinstance(owner, asyncio.Task):
        coro = owner.get_coro()
        frame = getattr(coro, "cr_frame", None)
        where = ""
        if frame is not None:
            where = f" at {frame.f_code.co_filename.rsplit('/', 1)[-1]}:{frame.f_lineno}"
        return f"task {getattr(coro, '__qualname__', coro)!s}{where}"
    return repr(callback)[:160]


async def _probe() -> None:
    while True:
        start = time.monotonic()
        await asyncio.sleep(_PROBE_INTERVAL)
        lag = max(0.0, time.monotonic() - start - _PROBE_INTERVAL)
        _samples.append((time.monotonic(), lag))


def start() -> None:
    """Begin watching the running loop. Safe to call more than once."""
    global _probe_task
    _install_slow_callback_log()
    if _probe_task is None or _probe_task.done():
        _probe_task = asyncio.get_running_loop().create_task(_probe())
