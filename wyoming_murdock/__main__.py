"""Entry point: starts both the Wyoming proxy and the FastAPI web UI."""

from __future__ import annotations

import asyncio
import logging
import signal
from functools import partial

import uvicorn
from wyoming.server import AsyncServer

from murdock import __version__
from murdock.api.app import create_app
from murdock.config import get_settings
from murdock.core.context import build_context
from murdock.core.info_cache import UpstreamInfoCache, parse_languages

from .handler import MurdockHandler

_LOGGER = logging.getLogger("murdock.main")


async def _run_web_ui(app, host: str, port: int) -> None:
    config = uvicorn.Config(
        app, host=host, port=port, log_level="info", access_log=False
    )
    server = uvicorn.Server(config)
    await server.serve()


async def main() -> None:
    settings = get_settings()
    logging.basicConfig(
        level=getattr(logging, settings.log_level.upper(), logging.INFO),
        format="%(asctime)s %(levelname)s [%(name)s] %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    _LOGGER.info("Murdock v%s starting…", __version__)
    _LOGGER.info("Listen: %s  Upstream: %s", settings.listen_uri, settings.upstream_uri)
    _LOGGER.info("Data dir: %s   Model dir: %s", settings.data_dir, settings.model_dir)

    context = build_context(settings)
    speakers = context.speakers.list_speakers()
    _LOGGER.info(
        "Speaker store ready — %d speaker(s) enrolled, threshold=%.3f",
        len(speakers), context.get_verify_threshold(),
    )

    # Start the MQTT bridge if enabled. Non-blocking — the connection
    # loop runs in the background and reconnects on its own, so a missing
    # broker never delays startup.
    if context.get_mqtt_enabled():
        await context.apply_mqtt_settings()
        _LOGGER.info(
            "MQTT enabled → %s:%d (connecting in background)",
            context.get_mqtt_host() or "(none)", context.get_mqtt_port(),
        )

    # Runtime-evaluated language override: first the SQLite setting, then
    # the env var fallback. Evaluated lazily inside the cache so UI edits
    # take effect immediately without restart.
    env_override = parse_languages(settings.advertised_languages)

    def _override_provider():
        runtime = context.get_advertised_languages()
        if runtime:
            return runtime
        return env_override

    # Use whatever the DB says (override > env default) so a previous
    # run's UI override survives container restarts.
    initial_upstream = context.get_upstream_uri()
    info_cache = UpstreamInfoCache(
        initial_upstream,
        ttl=settings.info_cache_ttl_seconds,
        override_provider=_override_provider,
    )
    context.info_cache = info_cache  # allow REST routes to poke it

    server = AsyncServer.from_uri(settings.listen_uri)
    handler_factory = partial(MurdockHandler, info_cache, context)

    app = create_app(context)

    # Fit confidence calibration if enabled and not yet fitted. Runs in the
    # background so a slow re-embed of all samples never delays startup;
    # until it lands, confidence falls back to 1 - distance.
    if context.get_enable_calibration() and not context.get_calibrator().fitted:
        if len(speakers) >= 2:
            _LOGGER.info("Calibration not fitted yet — scheduling background fit")
            context.schedule_recalibration()

    cleanup_task = asyncio.create_task(context.unknown.run_cleanup_loop())
    web_task = asyncio.create_task(
        _run_web_ui(app, settings.web_host, settings.web_port)
    )
    wyoming_task = asyncio.create_task(server.run(handler_factory))
    # Kick off the language prime AFTER the listener is up so that slow
    # or unreachable upstreams never delay the Wyoming bind — HA can
    # connect immediately and Describe requests will refresh on demand.
    prime_task = asyncio.create_task(info_cache.prime())

    stop_event = asyncio.Event()

    def _handle_signal(*_: object) -> None:
        _LOGGER.info("Received shutdown signal")
        stop_event.set()

    loop = asyncio.get_running_loop()
    for sig_name in ("SIGINT", "SIGTERM"):
        if hasattr(signal, sig_name):
            try:
                loop.add_signal_handler(getattr(signal, sig_name), _handle_signal)
            except NotImplementedError:
                # Windows doesn't support signal handlers in asyncio.
                pass

    try:
        done, pending = await asyncio.wait(
            {wyoming_task, web_task, asyncio.create_task(stop_event.wait())},
            return_when=asyncio.FIRST_COMPLETED,
        )
    finally:
        cleanup_task.cancel()
        web_task.cancel()
        wyoming_task.cancel()
        for task in (cleanup_task, web_task, wyoming_task):
            try:
                await task
            except (asyncio.CancelledError, Exception):
                pass
        await context.mqtt.stop()
        await context.ha.close()
        context.db.close()
        _LOGGER.info("Murdock stopped")


def run() -> None:
    asyncio.run(main())


if __name__ == "__main__":
    run()
