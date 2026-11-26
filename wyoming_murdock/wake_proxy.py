"""A listening post between Home Assistant and its Voice Support app.

Home Assistant 2026.12 (Labs: "Next-generation voice") sends every voice
run to the Voice Support app: the wake word audio a satellite buffered
before its detection, then the command, so the app can second-guess the
wake word and decide where the command ends. The wake word audio never
reaches a speech-to-text provider — the pipeline cuts it off first — so
this is the only place Murdock can hear it.

This module puts Murdock in that path without taking anything over. It
speaks the app's protocol (``GET /info`` and a websocket at ``/run``),
forwards every message in both directions unchanged, and keeps a copy:

* the run's metadata and the app's verdict, with its latency,
* the audio around the wake word, written to ``<data_dir>/wake_runs``,
* what Murdock's own speaker check makes of the wake word.

Nothing it learns changes the verdict yet. The log line says what a
speaker-aware verdict *would* have been, so a week of real wake words can
show whether overriding the app is worth doing before anything does it.

Point Home Assistant's Voice support integration at Murdock (on a
Supervisor install: remove the entry, add the integration again and
untick "use the app", then enter Murdock's host and port 10900) and set
``WAKE_PROXY_UPSTREAM`` to where the real app listens.
"""

from __future__ import annotations

import asyncio
import json
import logging
import time
import wave
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Awaitable, Callable, Optional

import httpx
from fastapi import FastAPI, WebSocket
from fastapi.responses import JSONResponse

_LOGGER = logging.getLogger("murdock.wake_proxy")

_RATE = 16000
_BYTES_PER_MS = 32  # 16 kHz, 16-bit, mono — fixed by the protocol
#: What the app holds at most; a run longer than this is not buffered further.
_MAX_RUN_MS = 30_000
#: Audio kept after the wake word in the stored clip, for listening back.
_CLIP_TAIL_MS = 1500
#: Clips older than the newest this many are deleted.
_KEEP_RUNS = 300
_INFO_TIMEOUT = 5.0
_CONNECT_TIMEOUT = 1.0
#: How long the app's last /info answer stands in while it is down. Long
#: enough to reinstall or restart the app without Home Assistant dropping
#: the entry; short, because a stale answer keeps satellites sending wake
#: word audio (and detecting at their lower threshold) with nobody checking.
_INFO_STALE_SECONDS = 30 * 60
_INFO_REFRESH_SECONDS = 300


class Upstream:
    """The app's side of one run. A thin seam so tests need no socket."""

    async def send(self, message: str | bytes) -> None:  # pragma: no cover
        raise NotImplementedError

    def __aiter__(self):  # pragma: no cover
        raise NotImplementedError

    async def close(self) -> None:  # pragma: no cover
        raise NotImplementedError


class _WebsocketUpstream(Upstream):
    def __init__(self, connection) -> None:
        self._connection = connection

    async def send(self, message: str | bytes) -> None:
        await self._connection.send(message)

    async def __aiter__(self):
        from websockets.exceptions import ConnectionClosed

        try:
            async for message in self._connection:
                yield message
        except ConnectionClosed:
            return

    async def close(self) -> None:
        await self._connection.close()


async def _connect_websocket(url: str) -> Upstream:
    from websockets.asyncio.client import connect

    connection = await asyncio.wait_for(
        connect(url, max_size=None, open_timeout=_CONNECT_TIMEOUT, close_timeout=1.0),
        timeout=_CONNECT_TIMEOUT + 0.5,
    )
    return _WebsocketUpstream(connection)


@dataclass
class RunRecord:
    """What one run looked like from the middle."""

    started: float = field(default_factory=time.monotonic)
    started_at: str = field(
        default_factory=lambda: datetime.now().isoformat(timespec="seconds")
    )
    satellite_id: Optional[str] = None
    wake_word_phrase: Optional[str] = None
    wake_word_audio_ms: Optional[int] = None
    silence_seconds: Optional[float] = None
    audio: bytearray = field(default_factory=bytearray)
    audio_bytes_seen: int = 0
    verdict: Optional[str] = None          # "accepted" | "rejected"
    reason: Optional[str] = None
    verdict_ms: Optional[float] = None     # from run-start
    accepted_boundary_ms: Optional[int] = None
    vad_start_ms: Optional[int] = None
    vad_end_ms: Optional[int] = None
    error: Optional[str] = None
    upstream_failed: bool = False

    def on_ha_text(self, text: str) -> None:
        try:
            message = json.loads(text)
        except ValueError:
            return
        if message.get("type") != "run-start":
            return
        self.satellite_id = message.get("satellite_id")
        self.wake_word_phrase = message.get("wake_word_phrase")
        ms = message.get("wake_word_audio_ms")
        self.wake_word_audio_ms = int(ms) if isinstance(ms, (int, float)) and ms else None
        self.silence_seconds = message.get("silence_seconds")

    def on_audio(self, chunk: bytes) -> None:
        self.audio_bytes_seen += len(chunk)
        room = _MAX_RUN_MS * _BYTES_PER_MS - len(self.audio)
        if room > 0:
            self.audio += chunk[:room]

    def on_app_text(self, text: str) -> None:
        try:
            event = json.loads(text)
        except ValueError:
            return
        kind = event.get("type")
        output = event.get("wake_word_output") or {}
        if kind == "wake-word-end":
            self.verdict = "accepted"
            self.verdict_ms = (time.monotonic() - self.started) * 1000
            boundary = output.get("wake_word_audio_ms")
            self.accepted_boundary_ms = int(boundary) if isinstance(boundary, int) else None
        elif kind == "wake-word-rejected":
            self.verdict = "rejected"
            self.reason = output.get("reason")
            self.verdict_ms = (time.monotonic() - self.started) * 1000
        elif kind == "stt-vad-start":
            self.vad_start_ms = event.get("timestamp")
        elif kind == "stt-vad-end":
            self.vad_end_ms = event.get("timestamp")
        elif kind == "error":
            self.error = f"{event.get('code')}: {event.get('message')}"

    @property
    def has_wake_word(self) -> bool:
        return bool(self.wake_word_phrase and self.wake_word_audio_ms)

    def wake_clip(self) -> bytes:
        """The wake word: the satellite's pre-roll, up to where the app says
        it ends when it accepted."""
        end_ms = self.accepted_boundary_ms or self.wake_word_audio_ms or 0
        end = min(len(self.audio), end_ms * _BYTES_PER_MS)
        return bytes(self.audio[: end - end % 2])

    def stored_clip(self) -> bytes:
        end_ms = (self.wake_word_audio_ms or 0) + _CLIP_TAIL_MS
        end = min(len(self.audio), end_ms * _BYTES_PER_MS)
        return bytes(self.audio[: end - end % 2])


@dataclass
class SpeakerView:
    """Murdock's own reading of the wake word."""

    speaker: Optional[str] = None
    distance: Optional[float] = None
    threshold: Optional[float] = None
    is_match: bool = False
    nearest: Optional[str] = None
    speech_ms: Optional[float] = None
    error: Optional[str] = None


def would_decide(app_verdict: Optional[str], view: Optional[SpeakerView]) -> str:
    """What a speaker-aware verdict would be. Logged only, never sent.

    An enrolled voice keeps a wake word the app refused (the German "Okay
    Nabu" that Kaldi hears as "namu"); a voice nobody enrolled loses one
    the app accepted (the television). Without a reading, the app stands.
    """
    if view is None or view.error or view.distance is None:
        return "app stands"
    if app_verdict == "rejected" and view.is_match:
        return f"would rescue ({view.speaker})"
    if app_verdict == "accepted" and not view.is_match:
        return "would veto (no enrolled voice)"
    return "agrees"


class WakeProxy:
    """The ``/info`` + ``/run`` relay with a tap on every run."""

    def __init__(
        self,
        upstream_url: str,
        *,
        data_dir: Optional[Path] = None,
        speaker_check: Optional[Callable[[bytes], Awaitable[SpeakerView]]] = None,
        connect: Callable[[str], Awaitable[Upstream]] = _connect_websocket,
        fetch_info: Optional[Callable[[str], Awaitable[dict]]] = None,
    ) -> None:
        self.upstream_url = upstream_url.rstrip("/")
        self.runs_dir = Path(data_dir) / "wake_runs" if data_dir else None
        self._speaker_check = speaker_check
        self._connect = connect
        self._fetch_info = fetch_info or self._fetch_info_http
        self.records: list[dict] = []          # newest last, for tests and the API
        self._background: set[asyncio.Task] = set()
        self._active = 0
        self._info: Optional[dict] = None
        self._info_at: Optional[float] = None   # wall clock, survives restarts
        self._load_info()

    @property
    def _info_path(self) -> Optional[Path]:
        return self.runs_dir / "info.json" if self.runs_dir else None

    def _load_info(self) -> None:
        path = self._info_path
        if path is None or not path.exists():
            return
        try:
            stored = json.loads(path.read_text(encoding="utf-8"))
            self._info, self._info_at = stored["info"], float(stored["at"])
        except (OSError, ValueError, KeyError, TypeError):
            pass

    async def current_info(self) -> dict:
        """The app's /info, or its last answer while it is briefly away."""
        try:
            data = await self._fetch_info(self.upstream_url)
        except Exception as exc:
            age = time.time() - self._info_at if self._info_at else None
            if self._info is not None and age is not None and age < _INFO_STALE_SECONDS:
                _LOGGER.warning(
                    "Voice Support app at %s unreachable (%s) — answering with "
                    "its info from %.0f s ago", self.upstream_url, exc, age,
                )
                return self._info
            raise
        self._info, self._info_at = data, time.time()
        path = self._info_path
        if path is not None:
            try:
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_text(json.dumps({"info": data, "at": self._info_at}),
                                encoding="utf-8")
            except OSError:
                pass
        return data

    async def keep_info_fresh(self) -> None:
        """Ask the app now and then, so a short outage has a recent answer."""
        while True:
            try:
                await self.current_info()
            except Exception as exc:
                _LOGGER.debug("Voice Support app info refresh failed: %s", exc)
            await asyncio.sleep(_INFO_REFRESH_SECONDS)

    @property
    def run_url(self) -> str:
        base = self.upstream_url
        if base.startswith("https://"):
            base = "wss://" + base[len("https://"):]
        elif base.startswith("http://"):
            base = "ws://" + base[len("http://"):]
        return f"{base}/run"

    async def _fetch_info_http(self, url: str) -> dict:
        async with httpx.AsyncClient(timeout=_INFO_TIMEOUT) as client:
            resp = await client.get(f"{url}/info")
            resp.raise_for_status()
            return resp.json()

    def build_app(self) -> FastAPI:
        app = FastAPI(title="Murdock wake word proxy", docs_url=None, redoc_url=None,
                      openapi_url=None)

        @app.get("/info")
        async def info():
            try:
                data = await self.current_info()
            except Exception as exc:
                _LOGGER.warning("Voice Support app at %s unreachable: %s",
                                self.upstream_url, exc)
                # Home Assistant retries the entry and runs without the app
                # in the meantime — the same as if the app itself were down.
                return JSONResponse({"error": "upstream unreachable"}, status_code=503)
            if isinstance(data, dict) and data.get("name"):
                data = {**data, "name": f"Murdock → {data['name']}"}
            return data

        @app.get("/health")
        async def health():
            return {"ok": True, "upstream": self.upstream_url}

        @app.websocket("/run")
        async def run(websocket: WebSocket):
            await self.relay(websocket)

        return app

    async def relay(self, websocket: WebSocket) -> None:
        self._active += 1
        try:
            await self._relay(websocket)
        finally:
            self._active -= 1

    async def _relay(self, websocket: WebSocket) -> None:
        await websocket.accept()
        record = RunRecord()
        try:
            upstream = await self._connect(self.run_url)
        except Exception as exc:
            _LOGGER.warning("Voice Support app not reachable for a run: %s", exc)
            record.upstream_failed = True
            # Closing tells Home Assistant there is no app for this run; it
            # falls back to the satellite's detection and its own segmenter.
            await websocket.close()
            self._finish(record)
            return

        upstream_open = True

        async def ha_to_app() -> None:
            nonlocal upstream_open
            while True:
                message = await websocket.receive()
                if message.get("type") == "websocket.disconnect":
                    return
                text, data = message.get("text"), message.get("bytes")
                if text is not None:
                    record.on_ha_text(text)
                    payload: str | bytes = text
                elif data is not None:
                    record.on_audio(data)
                    payload = data
                else:
                    continue
                if upstream_open:
                    try:
                        await upstream.send(payload)
                    except Exception:
                        # Keep draining Home Assistant; it closes when it is done.
                        upstream_open = False

        async def app_to_ha() -> None:
            async for message in upstream:
                if isinstance(message, str):
                    record.on_app_text(message)
                    await websocket.send_text(message)

        to_app = asyncio.create_task(ha_to_app())
        to_ha = asyncio.create_task(app_to_ha())
        try:
            done, _ = await asyncio.wait(
                {to_app, to_ha}, return_when=asyncio.FIRST_COMPLETED
            )
            if to_ha in done and not to_app.done():
                # The app is finished with this run (it closes after a
                # rejection, or after the last event). Let Home Assistant
                # finish sending; it closes the socket itself.
                _LOGGER.debug("[wake] app ended the run first; draining HA")
                upstream_open = False
                try:
                    await asyncio.wait_for(to_app, timeout=5.0)
                except asyncio.TimeoutError:
                    _LOGGER.debug("[wake] HA kept the run open after the app ended it")
            _LOGGER.debug("[wake] run over (%s first)", "HA" if to_app in done else "app")
        except Exception:
            _LOGGER.debug("Wake proxy run failed", exc_info=True)
        finally:
            for task in (to_app, to_ha):
                if not task.done():
                    task.cancel()
            for task in (to_app, to_ha):
                try:
                    await task
                except (asyncio.CancelledError, Exception):
                    pass
            # The run is over for Home Assistant; nothing below may hold it.
            self._finish(record)
            try:
                await asyncio.wait_for(upstream.close(), timeout=1.0)
            except (asyncio.TimeoutError, Exception):
                pass
            try:
                await websocket.close()
            except Exception:
                pass

    def _finish(self, record: RunRecord) -> None:
        """Analyse and store off the relay path; the run is already over."""
        task = asyncio.create_task(self._analyse(record))
        self._background.add(task)
        task.add_done_callback(self._background.discard)

    async def wait_idle(self, timeout: float = 5.0) -> None:
        """For tests: let open runs end and their analyses finish."""
        deadline = time.monotonic() + timeout
        while (self._active or self._background) and time.monotonic() < deadline:
            if self._background:
                await asyncio.gather(*list(self._background), return_exceptions=True)
            else:
                await asyncio.sleep(0.01)

    async def _analyse(self, record: RunRecord) -> None:
        view: Optional[SpeakerView] = None
        if record.has_wake_word and self._speaker_check is not None:
            clip = record.wake_clip()
            if clip:
                try:
                    view = await self._speaker_check(clip)
                except Exception as exc:  # never let analysis take anything down
                    view = SpeakerView(error=str(exc) or type(exc).__name__)
        decision = would_decide(record.verdict, view) if record.has_wake_word else None
        clip_name = self._store_clip(record)
        row = {
            "at": record.started_at,
            "satellite_id": record.satellite_id,
            "wake_word": record.wake_word_phrase,
            "wake_word_audio_ms": record.wake_word_audio_ms,
            "silence_seconds": record.silence_seconds,
            "verdict": record.verdict,
            "reason": record.reason,
            "verdict_ms": round(record.verdict_ms, 1) if record.verdict_ms else None,
            "accepted_boundary_ms": record.accepted_boundary_ms,
            "vad_start_ms": record.vad_start_ms,
            "vad_end_ms": record.vad_end_ms,
            "audio_ms": round(record.audio_bytes_seen / _BYTES_PER_MS),
            "error": record.error,
            "upstream_failed": record.upstream_failed,
            "speaker": view.speaker if view else None,
            "distance": round(view.distance, 4) if view and view.distance is not None else None,
            "threshold": view.threshold if view else None,
            "nearest": view.nearest if view else None,
            "speech_ms": view.speech_ms if view else None,
            "speaker_error": view.error if view else None,
            "would": decision,
            "clip": clip_name,
        }
        self.records.append(row)
        del self.records[:-_KEEP_RUNS]
        self._append_jsonl(row)
        if record.has_wake_word:
            _LOGGER.info(
                "[wake] %s: %s — app %s%s%s · voice %s · %s",
                record.satellite_id or "?", record.wake_word_phrase,
                record.verdict or "no verdict",
                f" ({record.reason})" if record.reason else "",
                f" in {record.verdict_ms:.0f} ms" if record.verdict_ms else "",
                _describe_view(view), decision,
            )
        elif record.upstream_failed:
            _LOGGER.info("[wake] %s: run without the app (unreachable)",
                         record.satellite_id or "?")
        else:
            _LOGGER.debug(
                "[wake] %s: run without a wake word, %d ms, command end %s",
                record.satellite_id or "?", row["audio_ms"], record.vad_end_ms,
            )

    def _store_clip(self, record: RunRecord) -> Optional[str]:
        if self.runs_dir is None or not record.has_wake_word or not record.audio:
            return None
        try:
            self.runs_dir.mkdir(parents=True, exist_ok=True)
            stamp = datetime.now().strftime("%Y%m%d-%H%M%S-%f")[:-3]
            satellite = (record.satellite_id or "unknown").split(".")[-1][:40]
            name = f"{stamp}-{satellite}-{record.verdict or 'none'}.wav"
            with wave.open(str(self.runs_dir / name), "wb") as w:
                w.setnchannels(1)
                w.setsampwidth(2)
                w.setframerate(_RATE)
                w.writeframes(record.stored_clip())
            clips = sorted(self.runs_dir.glob("*.wav"))
            for old in clips[:-_KEEP_RUNS]:
                old.unlink(missing_ok=True)
            return name
        except OSError as exc:
            _LOGGER.debug("Could not store wake clip: %s", exc)
            return None

    def _append_jsonl(self, row: dict) -> None:
        if self.runs_dir is None:
            return
        try:
            self.runs_dir.mkdir(parents=True, exist_ok=True)
            with open(self.runs_dir / "runs.jsonl", "a", encoding="utf-8") as fh:
                fh.write(json.dumps(row, ensure_ascii=False) + "\n")
        except OSError as exc:
            _LOGGER.debug("Could not append wake run: %s", exc)


def _describe_view(view: Optional[SpeakerView]) -> str:
    if view is None:
        return "not checked"
    if view.error:
        return f"check failed ({view.error})"
    if view.is_match:
        return f"{view.speaker} d={view.distance:.3f} (th {view.threshold:.3f})"
    if view.nearest is not None and view.distance is not None:
        return f"unknown, nearest {view.nearest} d={view.distance:.3f} (th {view.threshold:.3f})"
    return "unknown"


def speaker_check_for(context) -> Callable[[bytes], Awaitable[SpeakerView]]:
    """Murdock's ordinary speaker check, applied to a wake word clip.

    The voiceprints were enrolled from commands, not from the wake word,
    so distances here run higher than in the gate. That is part of what
    the log is for: seeing how far apart they are before trusting them.
    """
    from .handler import _MODEL_LOCK

    async def check(clip: bytes) -> SpeakerView:
        if context.embedder is None or context.speakers is None:
            return SpeakerView(error="no speaker model")
        if not context.speakers.list_speakers():
            return SpeakerView(error="nobody enrolled")
        audio = clip
        speech_ms = None
        if context.vad is not None:
            try:
                result = await asyncio.to_thread(context.vad.analyze_pcm, clip)
                speech_ms = round(float(result.speech_seconds) * 1000)
                if result.segments:
                    # Embed speech only: the pre-roll starts with whatever
                    # the room was doing before anyone spoke.
                    pieces = [
                        clip[int(s * _RATE) * 2:int(e * _RATE) * 2]
                        for s, e in result.segments
                    ]
                    voiced = b"".join(pieces)
                    if len(voiced) >= int(0.4 * _RATE) * 2:
                        audio = voiced
            except Exception:
                _LOGGER.debug("VAD on wake clip failed", exc_info=True)
        threshold = context.get_verify_threshold()
        async with _MODEL_LOCK:
            loop = asyncio.get_running_loop()
            embedding = await loop.run_in_executor(None, context.embedder.embed_pcm, audio)
            result = await loop.run_in_executor(
                None, lambda: context.speakers.verify_embedding(embedding, threshold),
            )
        if result is None:
            return SpeakerView(error="no result", speech_ms=speech_ms)
        nearest = None
        nearest_distance = result.distance
        distances = getattr(result, "all_distances", None) or {}
        if distances:
            nearest, nearest_distance = min(distances.items(), key=lambda kv: kv[1])
        return SpeakerView(
            speaker=result.matched_speaker if result.is_match else None,
            distance=float(nearest_distance if nearest_distance is not None else result.distance),
            threshold=float(result.threshold),
            is_match=bool(result.is_match),
            nearest=nearest,
            speech_ms=speech_ms,
        )

    return check


async def run_wake_proxy(proxy: WakeProxy, host: str, port: int) -> None:
    """Serve the proxy next to the web UI, on its own port and without the
    web API's authentication: Home Assistant's client sends no token, and
    the real app has none either."""
    import uvicorn

    config = uvicorn.Config(
        proxy.build_app(), host=host, port=port, log_level="warning",
        access_log=False, ws_max_size=16 * 1024 * 1024, proxy_headers=False,
    )
    server = uvicorn.Server(config)
    _LOGGER.info("Wake word proxy on %s:%d → %s", host, port, proxy.upstream_url)
    refresher = asyncio.create_task(proxy.keep_info_fresh())
    try:
        await server.serve()
    finally:
        refresher.cancel()


__all__ = [
    "RunRecord",
    "SpeakerView",
    "Upstream",
    "WakeProxy",
    "run_wake_proxy",
    "speaker_check_for",
    "would_decide",
]
