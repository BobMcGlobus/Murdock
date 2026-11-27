"""Murdock between Home Assistant and its Voice Support app.

Home Assistant 2026.12 (Labs: "Next-generation voice") sends every voice
run to the Voice Support app: the wake word audio a satellite buffered
before its detection, then the command, so the app can second-guess the
wake word and decide where the command ends. The wake word audio never
reaches a speech-to-text provider — the pipeline cuts it off first — so
this is the only place Murdock can hear it.

Murdock speaks the app's protocol (``GET /info`` and a websocket at
``/run``) and relays both directions. On top of that:

* **Wake word voiceprints.** Each enrolled person gets a voiceprint of the
  wake word itself — the same short phrase every time, so it separates
  people far better than the voiceprints built from varied commands. They
  are learnt from accepted wake words whose command the ordinary speaker
  check attributes to that person, and from an enrolment window opened
  through the web API ("the next wake words are Sarah's").
* **Rescue** (mode ``rescue``): the app's verifier is English-only, and a
  German "Okay Nabu" often comes back as "namu" or "now" — refused. When
  the wake word's voice matches someone's wake word voiceprint closely
  enough, Murdock answers Home Assistant with an acceptance instead. The
  app stops endpointing after a refusal, and Home Assistant ends the
  command the moment its run goes quiet, so Murdock opens a second run at
  the app without a wake word — the app then only endpoints — feeds it the
  audio from the wake word on, and passes its boundaries back shifted onto
  Home Assistant's clock.
* **Observe** (mode ``observe``, the default): nothing is changed; the log
  says what a rescue would have done.

Every run's metadata, the app's verdict and the audio around the wake word
are kept in ``<data_dir>/wake_runs``.

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
import numpy as np
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
#: Home Assistant waits 2 s for a verdict, the satellite about as long. A
#: rescue has to be decided well inside that, counted from the run's start.
_RESCUE_DEADLINE_SECONDS = 1.4
#: Wake word voiceprint samples kept per person, newest win.
_MAX_PRINTS = 40
#: Speech shorter than this after trimming is not trusted on its own; the
#: whole clip is embedded instead (CAM++ needs 0.5 s at the very least).
_MIN_VOICED_SECONDS = 0.6

MODE_OBSERVE = "observe"
MODE_RESCUE = "rescue"
MODES = (MODE_OBSERVE, MODE_RESCUE)

DEFAULT_SETTINGS = {
    "mode": MODE_OBSERVE,
    # Cosine distance to a person's wake word centroid. Text-dependent
    # prints sit much closer than the command ones; calibrate from the
    # leave-one-out figures the status endpoint reports.
    "threshold": 0.30,
    "min_samples": 5,
    # Distance gap to the next person, so two similar voices never let
    # the closer one through by a hair.
    "min_margin": 0.05,
}


# ---------------------------------------------------------------------------
# Upstream seam
# ---------------------------------------------------------------------------


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


# ---------------------------------------------------------------------------
# What a run looked like
# ---------------------------------------------------------------------------


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
    verdict: Optional[str] = None          # the app's: "accepted" | "rejected"
    reason: Optional[str] = None
    verdict_ms: Optional[float] = None     # from run-start
    accepted_boundary_ms: Optional[int] = None
    vad_start_ms: Optional[int] = None
    vad_end_ms: Optional[int] = None
    error: Optional[str] = None
    upstream_failed: bool = False
    rescued: bool = False

    def on_ha_text(self, text: str) -> Optional[dict]:
        try:
            message = json.loads(text)
        except ValueError:
            return None
        if message.get("type") == "run-start":
            self.satellite_id = message.get("satellite_id")
            self.wake_word_phrase = message.get("wake_word_phrase")
            ms = message.get("wake_word_audio_ms")
            self.wake_word_audio_ms = int(ms) if isinstance(ms, (int, float)) and ms else None
            self.silence_seconds = message.get("silence_seconds")
        return message

    def on_audio(self, chunk: bytes) -> None:
        self.audio_bytes_seen += len(chunk)
        room = _MAX_RUN_MS * _BYTES_PER_MS - len(self.audio)
        if room > 0:
            self.audio += chunk[:room]

    def on_app_event(self, event: dict, offset_ms: int = 0) -> None:
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
        elif kind == "stt-vad-start" and isinstance(event.get("timestamp"), int):
            self.vad_start_ms = event["timestamp"] + offset_ms
        elif kind == "stt-vad-end" and isinstance(event.get("timestamp"), int):
            self.vad_end_ms = event["timestamp"] + offset_ms
        elif kind == "error":
            self.error = f"{event.get('code')}: {event.get('message')}"

    @property
    def has_wake_word(self) -> bool:
        return bool(self.wake_word_phrase and self.wake_word_audio_ms)

    @property
    def pre_roll_bytes(self) -> int:
        return (self.wake_word_audio_ms or 0) * _BYTES_PER_MS

    def pre_roll(self) -> bytes:
        """The satellite's buffered audio before its detection: the wake word."""
        end = min(len(self.audio), self.pre_roll_bytes)
        return bytes(self.audio[: end - end % 2])

    def command(self) -> bytes:
        """The command after the wake word, up to where the run said it ended."""
        start = (self.accepted_boundary_ms or self.wake_word_audio_ms or 0) * _BYTES_PER_MS
        end = len(self.audio)
        if self.vad_end_ms:
            end = min(end, self.vad_end_ms * _BYTES_PER_MS)
        start -= start % 2
        end -= end % 2
        return bytes(self.audio[start:end]) if end > start else b""

    def stored_clip(self) -> bytes:
        end_ms = (self.wake_word_audio_ms or 0) + _CLIP_TAIL_MS
        end = min(len(self.audio), end_ms * _BYTES_PER_MS)
        return bytes(self.audio[: end - end % 2])


@dataclass
class SpeakerView:
    """A reading of who spoke: from wake word voiceprints or the command."""

    speaker: Optional[str] = None
    speaker_id: Optional[int] = None
    distance: Optional[float] = None
    threshold: Optional[float] = None
    is_match: bool = False
    nearest: Optional[str] = None
    margin: Optional[float] = None
    samples: Optional[int] = None
    error: Optional[str] = None


def would_decide(app_verdict: Optional[str], view: Optional[SpeakerView]) -> str:
    """What the voice says about the app's verdict.

    An enrolled voice keeps a wake word the app refused (the German "Okay
    Nabu" that Kaldi hears as "namu"); a voice nobody enrolled would lose
    one the app accepted (the television). Without a reading, the app
    stands.
    """
    if view is None or view.error or view.distance is None:
        return "app stands"
    if app_verdict == "rejected" and view.is_match:
        return f"would rescue ({view.speaker})"
    if app_verdict == "accepted" and not view.is_match:
        return "would veto (no wake voiceprint matches)"
    return "agrees"


# ---------------------------------------------------------------------------
# Wake word voiceprints
# ---------------------------------------------------------------------------


def _cosine_distance(a: np.ndarray, b: np.ndarray) -> float:
    na, nb = float(np.linalg.norm(a)), float(np.linalg.norm(b))
    if na < 1e-9 or nb < 1e-9:
        return 2.0
    return 1.0 - float(np.dot(a, b) / (na * nb))


class WakePrints:
    """Per-person voiceprints of the wake word, kept as a small JSON file.

    Text-dependent: every sample is the same phrase, so the centroid is
    tight and a short clip is enough. The file holds the mode and the
    thresholds too, so the experiment is configured in one place.
    """

    def __init__(self, path: Optional[Path]) -> None:
        self.path = path
        self.speakers: dict[str, dict] = {}
        self.settings: dict = dict(DEFAULT_SETTINGS)
        #: (speaker_id, name, until — wall clock) while an enrolment window is open.
        self.enrolling: Optional[tuple[int, str, float]] = None
        self._load()

    # -- persistence ----------------------------------------------------
    def _load(self) -> None:
        if self.path is None or not self.path.exists():
            return
        try:
            data = json.loads(self.path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            return
        self.speakers = data.get("speakers") or {}
        self.settings.update({k: v for k, v in (data.get("settings") or {}).items()
                              if k in DEFAULT_SETTINGS})

    def save(self) -> None:
        if self.path is None:
            return
        try:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            tmp = self.path.with_suffix(".tmp")
            tmp.write_text(json.dumps({"speakers": self.speakers,
                                       "settings": self.settings}), encoding="utf-8")
            tmp.replace(self.path)
        except OSError as exc:
            _LOGGER.warning("Could not save wake voiceprints: %s", exc)

    # -- settings --------------------------------------------------------
    def update_settings(self, **changes) -> dict:
        for key, value in changes.items():
            if value is None or key not in DEFAULT_SETTINGS:
                continue
            if key == "mode" and value not in MODES:
                raise ValueError(f"mode must be one of {', '.join(MODES)}")
            self.settings[key] = value
        self.save()
        return dict(self.settings)

    # -- samples ---------------------------------------------------------
    def add(self, speaker_id: int, name: str, embedding: np.ndarray, *,
            source: str, clip: Optional[str] = None) -> int:
        entry = self.speakers.setdefault(str(speaker_id), {"name": name, "samples": []})
        entry["name"] = name
        entry["samples"].append({
            "emb": [round(float(x), 6) for x in embedding],
            "at": datetime.now().isoformat(timespec="seconds"),
            "source": source,
            "clip": clip,
        })
        del entry["samples"][:-_MAX_PRINTS]
        self.save()
        return len(entry["samples"])

    def remove(self, speaker_id: int) -> bool:
        removed = self.speakers.pop(str(speaker_id), None) is not None
        if removed:
            self.save()
        return removed

    def _centroids(self) -> list[tuple[int, str, np.ndarray, int]]:
        out = []
        for sid, entry in self.speakers.items():
            samples = [np.asarray(s["emb"], dtype=np.float32) for s in entry.get("samples", [])]
            if not samples:
                continue
            centroid = np.mean(samples, axis=0)
            out.append((int(sid), entry.get("name", sid), centroid, len(samples)))
        return out

    def match(self, embedding: np.ndarray) -> SpeakerView:
        threshold = float(self.settings["threshold"])
        usable = [c for c in self._centroids() if c[3] >= int(self.settings["min_samples"])]
        if not usable:
            return SpeakerView(error="no wake voiceprints yet", threshold=threshold)
        scored = sorted(
            ((_cosine_distance(embedding, centroid), sid, name, n)
             for sid, name, centroid, n in usable),
            key=lambda t: t[0],
        )
        distance, sid, name, n = scored[0]
        margin = (scored[1][0] - distance) if len(scored) > 1 else None
        ok = distance <= threshold and (
            margin is None or margin >= float(self.settings["min_margin"])
        )
        return SpeakerView(
            speaker=name if ok else None, speaker_id=sid if ok else None,
            distance=distance, threshold=threshold, is_match=ok, nearest=name,
            margin=margin, samples=n,
        )

    def stats(self) -> list[dict]:
        """Per person: samples, and how far each sample sits from the
        centroid of the others — the figure to calibrate the threshold on."""
        rows = []
        others_all = self._centroids()
        for sid, name, _centroid, n in others_all:
            samples = [np.asarray(s["emb"], dtype=np.float32)
                       for s in self.speakers[str(sid)]["samples"]]
            loo = []
            if n >= 2:
                total = np.sum(samples, axis=0)
                for s in samples:
                    loo.append(_cosine_distance(s, (total - s) / (n - 1)))
            # Distance of this person's samples to everybody else's centroid.
            cross = []
            for osid, _oname, ocentroid, _on in others_all:
                if osid != sid:
                    cross.extend(_cosine_distance(s, ocentroid) for s in samples)
            sources: dict[str, int] = {}
            for s in self.speakers[str(sid)]["samples"]:
                sources[s.get("source", "?")] = sources.get(s.get("source", "?"), 0) + 1
            rows.append({
                "speaker_id": sid, "name": name, "samples": n, "sources": sources,
                "loo_mean": round(float(np.mean(loo)), 4) if loo else None,
                "loo_max": round(float(np.max(loo)), 4) if loo else None,
                "to_others_min": round(float(np.min(cross)), 4) if cross else None,
            })
        return rows

    # -- enrolment window ----------------------------------------------------
    def open_window(self, speaker_id: int, name: str, seconds: float) -> None:
        self.enrolling = (speaker_id, name, time.time() + seconds)

    def close_window(self) -> None:
        self.enrolling = None

    def window_speaker(self) -> Optional[tuple[int, str]]:
        if self.enrolling and self.enrolling[2] > time.time():
            return self.enrolling[0], self.enrolling[1]
        self.enrolling = None
        return None


# ---------------------------------------------------------------------------
# Speaker tools: Murdock's embedder and gate, applied to wake runs
# ---------------------------------------------------------------------------


class SpeakerTools:
    """What the proxy needs from the rest of Murdock. Tests substitute it."""

    async def embed_wake(self, clip: bytes) -> np.ndarray:  # pragma: no cover
        raise NotImplementedError

    async def verify_command(self, audio: bytes) -> SpeakerView:  # pragma: no cover
        raise NotImplementedError

    def speakers(self) -> dict[int, str]:  # pragma: no cover
        raise NotImplementedError


class ContextSpeakerTools(SpeakerTools):
    def __init__(self, context) -> None:
        self.context = context

    async def _speech_only(self, audio: bytes) -> bytes:
        if self.context.vad is None:
            return audio
        try:
            result = await asyncio.to_thread(self.context.vad.analyze_pcm, audio)
        except Exception:
            _LOGGER.debug("VAD on wake audio failed", exc_info=True)
            return audio
        if not result.segments:
            return audio
        voiced = b"".join(
            audio[int(s * _RATE) * 2:int(e * _RATE) * 2] for s, e in result.segments
        )
        # A clipped "Okay Nabu" is too little to trust on its own.
        if len(voiced) < int(_MIN_VOICED_SECONDS * _RATE) * 2:
            return audio
        return voiced

    async def _embed(self, audio: bytes) -> np.ndarray:
        from .handler import _MODEL_LOCK

        if self.context.embedder is None:
            raise RuntimeError("no speaker model")
        async with _MODEL_LOCK:
            loop = asyncio.get_running_loop()
            return await loop.run_in_executor(None, self.context.embedder.embed_pcm, audio)

    async def embed_wake(self, clip: bytes) -> np.ndarray:
        return await self._embed(await self._speech_only(clip))

    async def verify_command(self, audio: bytes) -> SpeakerView:
        if self.context.speakers is None or not self.context.speakers.list_speakers():
            return SpeakerView(error="nobody enrolled")
        if len(audio) < _RATE * 2:  # under a second: not worth a verdict
            return SpeakerView(error="command too short")
        embedding = await self._embed(await self._speech_only(audio))
        threshold = self.context.get_verify_threshold()
        result = await asyncio.get_running_loop().run_in_executor(
            None, lambda: self.context.speakers.verify_embedding(embedding, threshold),
        )
        if result is None:
            return SpeakerView(error="no result")
        return SpeakerView(
            speaker=result.matched_speaker if result.is_match else None,
            speaker_id=result.matched_speaker_id if result.is_match else None,
            distance=float(result.distance), threshold=float(result.threshold),
            is_match=bool(result.is_match),
        )

    def speakers(self) -> dict[int, str]:
        if self.context.speakers is None:
            return {}
        return {int(s.id): s.name for s in self.context.speakers.list_speakers()}


def speaker_tools_for(context) -> SpeakerTools:
    return ContextSpeakerTools(context)


# ---------------------------------------------------------------------------
# The proxy
# ---------------------------------------------------------------------------


class _Run:
    """Live state of one relayed run."""

    def __init__(self, record: RunRecord, upstream: Upstream) -> None:
        self.record = record
        self.target: Optional[Upstream] = upstream
        self.outbox: asyncio.Queue = asyncio.Queue()
        self.consumed = 0                       # HA audio bytes the pump has handled
        self.wake_task: Optional[asyncio.Task] = None
        self.wake_embedding: Optional[np.ndarray] = None
        self.wake_view: Optional[SpeakerView] = None
        self.second: Optional[Upstream] = None


class WakeProxy:
    """The ``/info`` + ``/run`` relay, with wake word voiceprints."""

    def __init__(
        self,
        upstream_url: str,
        *,
        data_dir: Optional[Path] = None,
        tools: Optional[SpeakerTools] = None,
        connect: Callable[[str], Awaitable[Upstream]] = _connect_websocket,
        fetch_info: Optional[Callable[[str], Awaitable[dict]]] = None,
    ) -> None:
        self.upstream_url = upstream_url.rstrip("/")
        self.runs_dir = Path(data_dir) / "wake_runs" if data_dir else None
        self.tools = tools
        self._connect = connect
        self._fetch_info = fetch_info or self._fetch_info_http
        self.prints = WakePrints(self.runs_dir / "prints.json" if self.runs_dir else None)
        self.records: list[dict] = []          # newest last, for tests and the API
        self._background: set[asyncio.Task] = set()
        self._active = 0
        self._info: Optional[dict] = None
        self._info_at: Optional[float] = None   # wall clock, survives restarts
        self._load_info()

    # -- /info -------------------------------------------------------------
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

    # -- one run -----------------------------------------------------------
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
            self._finish(record, None)
            return

        run = _Run(record, upstream)

        async def from_ha() -> None:
            """Read Home Assistant; the pump forwards, in order."""
            try:
                while True:
                    message = await websocket.receive()
                    if message.get("type") == "websocket.disconnect":
                        return
                    text, data = message.get("text"), message.get("bytes")
                    if text is not None:
                        record.on_ha_text(text)
                        run.outbox.put_nowait(("text", text))
                    elif data is not None:
                        record.on_audio(data)
                        run.outbox.put_nowait(("audio", data))
                        self._maybe_start_wake_check(run)
            finally:
                run.outbox.put_nowait(None)

        async def pump() -> None:
            """The one sender to the app, so a switch cannot reorder audio."""
            while True:
                item = await run.outbox.get()
                if item is None:
                    return
                kind = item[0]
                if kind == "audio":
                    run.consumed += len(item[1])
                    await self._send(run, item[1])
                elif kind == "text":
                    await self._send(run, item[1])
                elif kind == "switch":
                    second, boundary = item[1], item[2]
                    run.target = second
                    # Shaped like Home Assistant's own run without a wake
                    # word (a button press): the app only endpoints it.
                    await self._send(run, json.dumps({
                        "type": "run-start",
                        "satellite_id": record.satellite_id,
                        "wake_word_phrase": None,
                        "wake_word_audio_ms": 0,
                        "silence_seconds": record.silence_seconds,
                    }))
                    backlog = bytes(record.audio[boundary:run.consumed])
                    if backlog:
                        await self._send(run, backlog)

        async def to_ha() -> None:
            await self._pass_events(run, websocket, upstream, offset_ms=0, first=True)

        tasks = [asyncio.create_task(from_ha()), asyncio.create_task(pump()),
                 asyncio.create_task(to_ha())]
        reader, pumper, answerer = tasks
        try:
            done, _ = await asyncio.wait({reader, answerer},
                                         return_when=asyncio.FIRST_COMPLETED)
            if answerer in done and not reader.done():
                # The app is finished with this run (it closes after a
                # rejection, or after the last event). Home Assistant
                # closes the socket itself once it is done.
                try:
                    await asyncio.wait_for(asyncio.shield(reader), timeout=5.0)
                except asyncio.TimeoutError:
                    _LOGGER.debug("[wake] HA kept the run open after the app ended it")
            elif reader in done and not answerer.done():
                # Home Assistant hung up; give the app a moment for last words.
                try:
                    await asyncio.wait_for(asyncio.shield(answerer), timeout=0.5)
                except asyncio.TimeoutError:
                    pass
        except Exception:
            _LOGGER.debug("Wake proxy run failed", exc_info=True)
        finally:
            for task in tasks:
                if not task.done():
                    task.cancel()
            for task in tasks:
                try:
                    await task
                except (asyncio.CancelledError, Exception):
                    pass
            # The run is over for Home Assistant; nothing below may hold it.
            self._finish(record, run)
            for up in (upstream, run.second):
                if up is None:
                    continue
                try:
                    await asyncio.wait_for(up.close(), timeout=1.0)
                except (asyncio.TimeoutError, Exception):
                    pass
            try:
                await websocket.close()
            except Exception:
                pass

    async def _send(self, run: _Run, payload: str | bytes) -> None:
        target = run.target
        if target is None:
            return
        try:
            await target.send(payload)
        except Exception:
            # The app has let the run go; keep draining Home Assistant.
            if run.target is target:
                run.target = None

    def _maybe_start_wake_check(self, run: _Run) -> None:
        record = run.record
        if (
            run.wake_task is None and self.tools is not None and record.has_wake_word
            and len(record.audio) >= record.pre_roll_bytes
        ):
            run.wake_task = asyncio.create_task(self._wake_check(run, record.pre_roll()))

    async def _wake_check(self, run: _Run, clip: bytes) -> None:
        try:
            run.wake_embedding = await self.tools.embed_wake(clip)
            run.wake_view = self.prints.match(run.wake_embedding)
        except Exception as exc:
            run.wake_view = SpeakerView(error=str(exc) or type(exc).__name__)

    async def _pass_events(self, run: _Run, websocket: WebSocket, upstream: Upstream,
                           *, offset_ms: int, first: bool) -> None:
        """The app's events to Home Assistant; a refusal may be rescued."""
        record = run.record
        async for message in upstream:
            if not isinstance(message, str):
                continue
            try:
                event = json.loads(message)
            except ValueError:
                continue
            if first:
                record.on_app_event(event)
                if event.get("type") == "wake-word-rejected":
                    second = await self._try_rescue(run)
                    if second is not None:
                        await websocket.send_text(json.dumps({
                            "type": "wake-word-end",
                            "wake_word_output": {
                                "wake_word_phrase": record.wake_word_phrase,
                                "wake_word_audio_ms": record.wake_word_audio_ms,
                            },
                        }))
                        # From here the second run endpoints the command.
                        await self._pass_events(
                            run, websocket, second,
                            offset_ms=record.wake_word_audio_ms or 0, first=False,
                        )
                        return
            else:
                if event.get("type") in ("wake-word-end", "wake-word-rejected"):
                    continue  # nothing to verify in the second run
                record.on_app_event(event, offset_ms=offset_ms)
                if event.get("type") in ("stt-vad-start", "stt-vad-end") and \
                        isinstance(event.get("timestamp"), int):
                    event = {**event, "timestamp": event["timestamp"] + offset_ms}
                message = json.dumps(event)
            await websocket.send_text(message)

    async def _try_rescue(self, run: _Run) -> Optional[Upstream]:
        """A second run for an enrolled voice the app refused, or None."""
        if self.prints.settings.get("mode") != MODE_RESCUE or run.wake_task is None:
            return None
        remaining = _RESCUE_DEADLINE_SECONDS - (time.monotonic() - run.record.started)
        if remaining <= 0:
            return None
        try:
            await asyncio.wait_for(asyncio.shield(run.wake_task), timeout=remaining)
        except asyncio.TimeoutError:
            _LOGGER.info("[wake] no voice verdict in time — the refusal stands")
            return None
        view = run.wake_view
        if view is None or not view.is_match:
            return None
        try:
            second = await asyncio.wait_for(self._connect(self.run_url), timeout=0.5)
        except Exception as exc:
            _LOGGER.warning("[wake] cannot open a run for endpointing (%s) — "
                            "the refusal stands", exc)
            return None
        run.second = second
        run.record.rescued = True
        run.outbox.put_nowait(("switch", second, run.record.pre_roll_bytes))
        return second

    # -- after the run -----------------------------------------------------
    def _finish(self, record: RunRecord, run: Optional[_Run]) -> None:
        """Analyse, learn and store off the relay path; the run is over."""
        task = asyncio.create_task(self._analyse(record, run))
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

    async def _analyse(self, record: RunRecord, run: Optional[_Run]) -> None:
        wake_view: Optional[SpeakerView] = None
        embedding: Optional[np.ndarray] = None
        command_view: Optional[SpeakerView] = None
        learnt: Optional[str] = None
        clip_name = self._store_clip(record)

        if record.has_wake_word and self.tools is not None:
            if run is not None and run.wake_task is not None:
                try:
                    await run.wake_task
                except Exception:
                    pass
                embedding, wake_view = run.wake_embedding, run.wake_view
            if embedding is None:
                clip = record.pre_roll()
                if clip:
                    try:
                        embedding = await self.tools.embed_wake(clip)
                        wake_view = self.prints.match(embedding)
                    except Exception as exc:
                        wake_view = SpeakerView(error=str(exc) or type(exc).__name__)
            if embedding is not None:
                learnt, command_view = await self._learn(record, embedding, clip_name)

        decision = would_decide(record.verdict, wake_view) if record.has_wake_word else None
        if record.rescued:
            decision = f"rescued ({wake_view.speaker if wake_view else '?'})"
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
            "rescued": record.rescued,
            "wake_speaker": wake_view.speaker if wake_view else None,
            "wake_nearest": wake_view.nearest if wake_view else None,
            "wake_distance": round(wake_view.distance, 4)
            if wake_view and wake_view.distance is not None else None,
            "wake_margin": round(wake_view.margin, 4)
            if wake_view and wake_view.margin is not None else None,
            "wake_error": wake_view.error if wake_view else None,
            "command_speaker": command_view.speaker if command_view else None,
            "command_distance": round(command_view.distance, 4)
            if command_view and command_view.distance is not None else None,
            "learnt": learnt,
            "would": decision,
            "clip": clip_name,
        }
        self.records.append(row)
        del self.records[:-_KEEP_RUNS]
        self._append_jsonl(row)
        if record.has_wake_word:
            _LOGGER.info(
                "[wake] %s: %s — app %s%s%s · wake voice %s · %s%s",
                record.satellite_id or "?", record.wake_word_phrase,
                record.verdict or "no verdict",
                f" ({record.reason})" if record.reason else "",
                f" in {record.verdict_ms:.0f} ms" if record.verdict_ms else "",
                _describe_view(wake_view), decision,
                f" · learnt for {learnt}" if learnt else "",
            )
        elif record.upstream_failed:
            _LOGGER.info("[wake] %s: run without the app (unreachable)",
                         record.satellite_id or "?")
        else:
            _LOGGER.debug(
                "[wake] %s: run without a wake word, %d ms, command end %s",
                record.satellite_id or "?", row["audio_ms"], record.vad_end_ms,
            )

    async def _learn(self, record: RunRecord, embedding: np.ndarray,
                     clip_name: Optional[str]) -> tuple[Optional[str], Optional[SpeakerView]]:
        """Add this wake word to someone's voiceprint, if it is known whose."""
        window = self.prints.window_speaker()
        if window is not None:
            sid, name = window
            n = self.prints.add(sid, name, embedding, source="enroll", clip=clip_name)
            return f"{name} (enrolment, {n})", None
        if record.verdict != "accepted" and not record.rescued:
            return None, None
        command = record.command()
        try:
            view = await self.tools.verify_command(command)
        except Exception as exc:
            return None, SpeakerView(error=str(exc) or type(exc).__name__)
        # Only an unambiguous command teaches: a wrong print is worse than none.
        if view.is_match and view.speaker_id is not None and view.distance is not None \
                and view.threshold is not None and view.distance <= view.threshold - 0.05:
            n = self.prints.add(view.speaker_id, view.speaker or str(view.speaker_id),
                                embedding, source="auto", clip=clip_name)
            return f"{view.speaker} (auto, {n})", view
        return None, view

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

    def status(self, recent: int = 20) -> dict[str, Any]:
        window = self.prints.window_speaker()
        return {
            "upstream": self.upstream_url,
            "settings": dict(self.prints.settings),
            "enrolling": {"speaker": window[1], "seconds_left": round(
                self.prints.enrolling[2] - time.time())} if window else None,
            "voiceprints": self.prints.stats(),
            "recent": self.records[-recent:],
        }


def _describe_view(view: Optional[SpeakerView]) -> str:
    if view is None:
        return "not checked"
    if view.error:
        return f"– ({view.error})"
    margin = f", margin {view.margin:.3f}" if view.margin is not None else ""
    if view.is_match:
        return f"{view.speaker} d={view.distance:.3f} (th {view.threshold:.3f}{margin})"
    return (f"no match, nearest {view.nearest} d={view.distance:.3f} "
            f"(th {view.threshold:.3f}{margin})")


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
    _LOGGER.info("Wake word proxy on %s:%d → %s (mode %s)", host, port,
                 proxy.upstream_url, proxy.prints.settings.get("mode"))
    refresher = asyncio.create_task(proxy.keep_info_fresh())
    try:
        await server.serve()
    finally:
        refresher.cancel()


__all__ = [
    "MODES",
    "RunRecord",
    "SpeakerTools",
    "SpeakerView",
    "Upstream",
    "WakePrints",
    "WakeProxy",
    "run_wake_proxy",
    "speaker_tools_for",
    "would_decide",
]
