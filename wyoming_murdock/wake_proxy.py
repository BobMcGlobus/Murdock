"""Murdock between Home Assistant and its wake word verifiers.

Home Assistant 2026.12 (Labs: "Next-generation voice") sends every voice
run to the Voice Support app: the wake word audio a satellite buffered
before its detection, then the command, so the app can second-guess the
wake word and decide where the command ends. The wake word audio never
reaches a speech-to-text provider — the pipeline cuts it off first — so
this is the only place Murdock can hear it.

Murdock speaks the app's protocol (``GET /info`` and a websocket at
``/run``) and sends every run to up to two verifiers at once:

* **en** — the official app. Its Kaldi model is English, so a German
  "Okay Nabu" is often refused as "namu" or "now".
* **de** — the same app built with a German model ("Voice Support
  (Deutsch)", ``voice_support_de/``), which checks the phrase the way a
  German says it, against German rivals such as "okay, na du".

One of them is the **authority**: its verdict and its command boundaries go
to Home Assistant, unchanged. The other only answers into the log, so the
two can be compared on every real wake word before the German one is
trusted. A wake word the German verifier does not know (Hey Jarvis) always
goes by the English one.

Alongside, as an extra indicator rather than a decision:

* **Wake word voiceprints** — the voice of the wake word itself, per
  enrolled person, learnt from accepted wake words whose command the
  ordinary speaker check attributes to them, or from an enrolment window.
* **Labels** — a window during which every wake word is marked as meant
  ("wake") or as a deliberate false alarm ("false"), and labelling of past
  runs by time. Labelled clips can be **replayed** against either verifier
  to calibrate its thresholds.

Every run's metadata, both verdicts and the audio around the wake word are
kept in ``<data_dir>/wake_runs``.
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
_CHUNK_BYTES = 100 * _BYTES_PER_MS
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
#: Wake word voiceprint samples kept per person, newest win.
_MAX_PRINTS = 40
#: Speech shorter than this after trimming is not trusted on its own; the
#: whole clip is embedded instead (CAM++ needs 0.5 s at the very least).
_MIN_VOICED_SECONDS = 0.6
#: A replayed clip that has not been judged by then counts as no verdict.
_REPLAY_VERDICT_SECONDS = 3.0

EN, DE = "en", "de"
SOURCES = (EN, DE)
LABEL_WAKE, LABEL_FALSE = "wake", "false"
LABELS = (LABEL_WAKE, LABEL_FALSE)

DEFAULT_SETTINGS = {
    # Whose verdict and boundaries reach Home Assistant.
    "authority": EN,
    # Wake word voiceprints, an indicator only: cosine distance to a
    # person's centroid, samples needed, and the gap to the next person.
    "threshold": 0.40,
    "min_samples": 5,
    "min_margin": 0.05,
}


def _normalize(name: Optional[str]) -> str:
    """"Okay Nabu", "okay-nabu" and "okay_nabu" are one wake word."""
    lowered = (name or "").lower().replace("-", "_").replace(" ", "_")
    return "".join(c for c in lowered if c.isalnum() or c == "_")


# ---------------------------------------------------------------------------
# Upstream seam
# ---------------------------------------------------------------------------


class Upstream:
    """A verifier's side of one run. A thin seam so tests need no socket."""

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


def _ws_url(base: str) -> str:
    if base.startswith("https://"):
        base = "wss://" + base[len("https://"):]
    elif base.startswith("http://"):
        base = "ws://" + base[len("http://"):]
    return f"{base}/run"


# ---------------------------------------------------------------------------
# What a run looked like
# ---------------------------------------------------------------------------


@dataclass
class Verdict:
    verdict: Optional[str] = None          # "accepted" | "rejected"
    reason: Optional[str] = None
    ms: Optional[float] = None             # from run-start
    boundary_ms: Optional[int] = None


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
    verdicts: dict[str, Verdict] = field(default_factory=dict)
    authority: str = EN
    vad_start_ms: Optional[int] = None
    vad_end_ms: Optional[int] = None
    error: Optional[str] = None
    upstream_failed: bool = False
    label: Optional[str] = None

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

    def on_app_event(self, source: str, event: dict) -> None:
        kind = event.get("type")
        output = event.get("wake_word_output") or {}
        verdict = self.verdicts.setdefault(source, Verdict())
        if kind == "wake-word-end":
            verdict.verdict = "accepted"
            verdict.ms = (time.monotonic() - self.started) * 1000
            boundary = output.get("wake_word_audio_ms")
            verdict.boundary_ms = int(boundary) if isinstance(boundary, int) else None
        elif kind == "wake-word-rejected":
            verdict.verdict = "rejected"
            verdict.reason = output.get("reason")
            verdict.ms = (time.monotonic() - self.started) * 1000
        elif source == self.authority:
            if kind == "stt-vad-start" and isinstance(event.get("timestamp"), int):
                self.vad_start_ms = event["timestamp"]
            elif kind == "stt-vad-end" and isinstance(event.get("timestamp"), int):
                self.vad_end_ms = event["timestamp"]
            elif kind == "error":
                self.error = f"{event.get('code')}: {event.get('message')}"

    def verdict_of(self, source: str) -> Optional[str]:
        v = self.verdicts.get(source)
        return v.verdict if v else None

    @property
    def verdict(self) -> Optional[str]:
        """The verdict Home Assistant acted on."""
        return self.verdict_of(self.authority)

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
        authority = self.verdicts.get(self.authority)
        boundary = (authority.boundary_ms if authority else None) or self.wake_word_audio_ms or 0
        start = boundary * _BYTES_PER_MS
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


# ---------------------------------------------------------------------------
# Wake word voiceprints
# ---------------------------------------------------------------------------


def _cosine_distance(a: np.ndarray, b: np.ndarray) -> float:
    na, nb = float(np.linalg.norm(a)), float(np.linalg.norm(b))
    if na < 1e-9 or nb < 1e-9:
        return 2.0
    return 1.0 - float(np.dot(a, b) / (na * nb))


class WakePrints:
    """Per-person voiceprints of the wake word, and the proxy's settings,
    kept as one small JSON file."""

    def __init__(self, path: Optional[Path]) -> None:
        self.path = path
        self.speakers: dict[str, dict] = {}
        self.settings: dict = dict(DEFAULT_SETTINGS)
        #: (speaker_id, name, until — wall clock) while an enrolment window is open.
        self.enrolling: Optional[tuple[int, str, float]] = None
        self._load()

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

    def update_settings(self, **changes) -> dict:
        for key, value in changes.items():
            if value is None or key not in DEFAULT_SETTINGS:
                continue
            if key == "authority" and value not in SOURCES:
                raise ValueError(f"authority must be one of {', '.join(SOURCES)}")
            self.settings[key] = value
        self.save()
        return dict(self.settings)

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
            if samples:
                out.append((int(sid), entry.get("name", sid), np.mean(samples, axis=0),
                            len(samples)))
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
        """Per person: samples, and how far each sits from the centroid of
        the others — the figure a threshold is chosen on."""
        rows = []
        centroids = self._centroids()
        for sid, name, _centroid, n in centroids:
            samples = [np.asarray(s["emb"], dtype=np.float32)
                       for s in self.speakers[str(sid)]["samples"]]
            loo = []
            if n >= 2:
                total = np.sum(samples, axis=0)
                loo = [_cosine_distance(s, (total - s) / (n - 1)) for s in samples]
            cross = [_cosine_distance(s, oc) for osid, _o, oc, _on in centroids
                     if osid != sid for s in samples]
            rows.append({
                "speaker_id": sid, "name": name, "samples": n,
                "loo_mean": round(float(np.mean(loo)), 4) if loo else None,
                "loo_max": round(float(np.max(loo)), 4) if loo else None,
                "to_others_min": round(float(np.min(cross)), 4) if cross else None,
            })
        return rows

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

    def __init__(self, record: RunRecord, targets: dict[str, Upstream]) -> None:
        self.record = record
        self.targets: dict[str, Optional[Upstream]] = dict(targets)
        self.outbox: asyncio.Queue = asyncio.Queue()
        self.wake_task: Optional[asyncio.Task] = None
        self.wake_embedding: Optional[np.ndarray] = None
        self.wake_view: Optional[SpeakerView] = None


class WakeProxy:
    """The ``/info`` + ``/run`` relay to one or two verifiers."""

    def __init__(
        self,
        upstream_url: str,
        *,
        secondary_url: Optional[str] = None,
        data_dir: Optional[Path] = None,
        tools: Optional[SpeakerTools] = None,
        connect: Callable[[str], Awaitable[Upstream]] = _connect_websocket,
        fetch_info: Optional[Callable[[str], Awaitable[dict]]] = None,
    ) -> None:
        self.urls: dict[str, str] = {EN: upstream_url.rstrip("/")}
        if secondary_url:
            self.urls[DE] = secondary_url.rstrip("/")
        self.runs_dir = Path(data_dir) / "wake_runs" if data_dir else None
        self.tools = tools
        self._connect = connect
        self._fetch_info = fetch_info or self._fetch_info_http
        self.prints = WakePrints(self.runs_dir / "prints.json" if self.runs_dir else None)
        self.records: list[dict] = []          # newest last, since start
        self._background: set[asyncio.Task] = set()
        self._active = 0
        self._info: Optional[dict] = None
        self._info_at: Optional[float] = None   # wall clock, survives restarts
        #: Wake word ids each verifier says it can verify.
        self.verifies: dict[str, set[str]] = {}
        #: (label, until — wall clock) while a labelling window is open.
        self.labelling: Optional[tuple[str, float]] = None
        self._load_info()

    @property
    def upstream_url(self) -> str:
        return self.urls[EN]

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
            self._note_wake_words(EN, self._info)
        except (OSError, ValueError, KeyError, TypeError):
            pass

    def _note_wake_words(self, source: str, info: dict) -> None:
        try:
            self.verifies[source] = {
                _normalize(w.get("id")) for w in info.get("wake_words") or []
            }
        except AttributeError:
            pass

    async def current_info(self) -> dict:
        """The official app's /info, or its last answer while it is away."""
        try:
            data = await self._fetch_info(self.urls[EN])
        except Exception as exc:
            age = time.time() - self._info_at if self._info_at else None
            if self._info is not None and age is not None and age < _INFO_STALE_SECONDS:
                _LOGGER.warning(
                    "Voice Support app at %s unreachable (%s) — answering with "
                    "its info from %.0f s ago", self.urls[EN], exc, age,
                )
                return self._info
            raise
        self._info, self._info_at = data, time.time()
        self._note_wake_words(EN, data)
        path = self._info_path
        if path is not None:
            try:
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_text(json.dumps({"info": data, "at": self._info_at}),
                                encoding="utf-8")
            except OSError:
                pass
        return data

    async def refresh_secondary_info(self) -> None:
        if DE not in self.urls:
            return
        try:
            self._note_wake_words(DE, await self._fetch_info(self.urls[DE]))
        except Exception as exc:
            _LOGGER.debug("German verifier info refresh failed: %s", exc)

    async def keep_info_fresh(self) -> None:
        """Ask the verifiers now and then, so a short outage has a recent answer."""
        while True:
            try:
                await self.current_info()
            except Exception as exc:
                _LOGGER.debug("Voice Support app info refresh failed: %s", exc)
            await self.refresh_secondary_info()
            await asyncio.sleep(_INFO_REFRESH_SECONDS)

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
                                self.urls[EN], exc)
                return JSONResponse({"error": "upstream unreachable"}, status_code=503)
            if isinstance(data, dict) and data.get("name"):
                data = {**data, "name": f"Murdock → {data['name']}"}
            return data

        @app.get("/health")
        async def health():
            return {"ok": True, "upstreams": self.urls}

        @app.websocket("/run")
        async def run(websocket: WebSocket):
            await self.relay(websocket)

        return app

    # -- one run -----------------------------------------------------------
    def _authority_for(self, record: RunRecord, run: _Run) -> str:
        """The verifier Home Assistant hears for this run."""
        if (
            self.prints.settings.get("authority") == DE
            and run.targets.get(DE) is not None
            and _normalize(record.wake_word_phrase) in self.verifies.get(DE, set())
        ):
            return DE
        return EN

    async def relay(self, websocket: WebSocket) -> None:
        self._active += 1
        try:
            await self._relay(websocket)
        finally:
            self._active -= 1

    async def _relay(self, websocket: WebSocket) -> None:
        await websocket.accept()
        record = RunRecord()
        targets: dict[str, Upstream] = {}
        for source, url in self.urls.items():
            try:
                targets[source] = await self._connect(_ws_url(url))
            except Exception as exc:
                level = logging.WARNING if source == EN else logging.INFO
                _LOGGER.log(level, "[wake] %s verifier not reachable for a run: %s",
                            source, exc)
        if EN not in targets:
            record.upstream_failed = True
            for up in targets.values():
                await self._close(up)
            # Closing tells Home Assistant there is no app for this run; it
            # falls back to the satellite's detection and its own segmenter.
            await websocket.close()
            self._finish(record, None)
            return

        run = _Run(record, targets)
        authority_known = asyncio.Event()

        async def from_ha() -> None:
            try:
                while True:
                    message = await websocket.receive()
                    if message.get("type") == "websocket.disconnect":
                        return
                    text, data = message.get("text"), message.get("bytes")
                    if text is not None:
                        parsed = record.on_ha_text(text)
                        if parsed and parsed.get("type") == "run-start":
                            record.authority = self._authority_for(record, run)
                            authority_known.set()
                        run.outbox.put_nowait(text)
                    elif data is not None:
                        record.on_audio(data)
                        run.outbox.put_nowait(data)
                        self._maybe_start_wake_check(run)
            finally:
                run.outbox.put_nowait(None)

        async def pump() -> None:
            """The one sender to the verifiers: every message to each, in order."""
            while True:
                item = await run.outbox.get()
                if item is None:
                    return
                for source in list(run.targets):
                    target = run.targets[source]
                    if target is None:
                        continue
                    try:
                        await target.send(item)
                    except Exception:
                        # That verifier has let the run go; keep draining HA.
                        run.targets[source] = None

        async def answers(source: str) -> None:
            await authority_known.wait()
            upstream = targets[source]
            async for message in upstream:
                if not isinstance(message, str):
                    continue
                try:
                    event = json.loads(message)
                except ValueError:
                    continue
                record.on_app_event(source, event)
                if source == record.authority:
                    await websocket.send_text(message)

        reader = asyncio.create_task(from_ha())
        pumper = asyncio.create_task(pump())
        listeners = {source: asyncio.create_task(answers(source)) for source in targets}
        known = asyncio.create_task(authority_known.wait())
        tasks = [reader, pumper, known, *listeners.values()]
        try:
            await asyncio.wait({reader, known}, return_when=asyncio.FIRST_COMPLETED)
            authority = listeners.get(record.authority)
            # The run lasts as long as Home Assistant streams and the
            # authority answers; the other verifier never ends it.
            while True:
                if reader.done():
                    # Home Assistant hung up; a moment for the authority's last words.
                    if authority is not None and not authority.done():
                        await asyncio.wait({authority}, timeout=0.5)
                    break
                if authority is None or authority.done():
                    # The authority let the run go (after a refusal, or its
                    # last event); Home Assistant closes the socket itself.
                    await asyncio.wait({reader}, timeout=5.0)
                    if not reader.done():
                        _LOGGER.debug("[wake] HA kept the run open after the app ended it")
                    break
                await asyncio.wait({reader, authority}, return_when=asyncio.FIRST_COMPLETED)
            # The other verifier's verdict is wanted for the log, not waited on long.
            other = [t for s, t in listeners.items() if s != record.authority and not t.done()]
            if other:
                await asyncio.wait(other, timeout=0.5)
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
            self._finish(record, run)
            for up in targets.values():
                await self._close(up)
            try:
                await websocket.close()
            except Exception:
                pass

    @staticmethod
    async def _close(up: Optional[Upstream]) -> None:
        if up is None:
            return
        try:
            await asyncio.wait_for(up.close(), timeout=1.0)
        except (asyncio.TimeoutError, Exception):
            pass

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

    # -- after the run -----------------------------------------------------
    def _finish(self, record: RunRecord, run: Optional[_Run]) -> None:
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

    def _window_label(self) -> Optional[str]:
        if self.labelling and self.labelling[1] > time.time():
            return self.labelling[0]
        self.labelling = None
        return None

    async def _analyse(self, record: RunRecord, run: Optional[_Run]) -> None:
        wake_view: Optional[SpeakerView] = None
        embedding: Optional[np.ndarray] = None
        command_view: Optional[SpeakerView] = None
        learnt: Optional[str] = None
        if record.has_wake_word:
            record.label = self._window_label()
            if self.prints.window_speaker() is not None and record.label is None:
                record.label = LABEL_WAKE   # an enrolment is meant wake words
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

        en, de = record.verdicts.get(EN, Verdict()), record.verdicts.get(DE, Verdict())
        row = {
            "at": record.started_at,
            "satellite_id": record.satellite_id,
            "wake_word": record.wake_word_phrase,
            "wake_word_audio_ms": record.wake_word_audio_ms,
            "silence_seconds": record.silence_seconds,
            "authority": record.authority,
            "verdict": record.verdict,
            "en": en.verdict, "en_reason": en.reason,
            "en_ms": round(en.ms, 1) if en.ms is not None else None,
            "de": de.verdict, "de_reason": de.reason,
            "de_ms": round(de.ms, 1) if de.ms is not None else None,
            "vad_start_ms": record.vad_start_ms,
            "vad_end_ms": record.vad_end_ms,
            "audio_ms": round(record.audio_bytes_seen / _BYTES_PER_MS),
            "error": record.error,
            "upstream_failed": record.upstream_failed,
            "label": record.label,
            "wake_speaker": wake_view.speaker if wake_view else None,
            "wake_nearest": wake_view.nearest if wake_view else None,
            "wake_distance": round(wake_view.distance, 4)
            if wake_view and wake_view.distance is not None else None,
            "wake_error": wake_view.error if wake_view else None,
            "command_speaker": command_view.speaker if command_view else None,
            "command_distance": round(command_view.distance, 4)
            if command_view and command_view.distance is not None else None,
            "learnt": learnt,
            "clip": clip_name,
        }
        self.records.append(row)
        del self.records[:-_KEEP_RUNS]
        self._append_jsonl(row)
        if record.has_wake_word:
            _LOGGER.info(
                "[wake] %s: %s — en %s · de %s → %s (%s) · voice %s%s%s",
                record.satellite_id or "?", record.wake_word_phrase,
                _describe_verdict(en), _describe_verdict(de) if DE in self.urls else "–",
                record.verdict or "no verdict", record.authority,
                _describe_view(wake_view),
                f" · label {record.label}" if record.label else "",
                f" · learnt for {learnt}" if learnt else "",
            )
        elif record.upstream_failed:
            _LOGGER.info("[wake] %s: run without the app (unreachable)",
                         record.satellite_id or "?")
        else:
            _LOGGER.debug("[wake] %s: run without a wake word, %d ms, command end %s",
                          record.satellite_id or "?", row["audio_ms"], record.vad_end_ms)

    async def _learn(self, record: RunRecord, embedding: np.ndarray,
                     clip_name: Optional[str]) -> tuple[Optional[str], Optional[SpeakerView]]:
        """Add this wake word to someone's voiceprint, if it is known whose."""
        window = self.prints.window_speaker()
        if window is not None:
            sid, name = window
            n = self.prints.add(sid, name, embedding, source="enroll", clip=clip_name)
            return f"{name} (enrolment, {n})", None
        if record.verdict != "accepted" or record.label == LABEL_FALSE:
            return None, None
        try:
            view = await self.tools.verify_command(record.command())
        except Exception as exc:
            return None, SpeakerView(error=str(exc) or type(exc).__name__)
        # Only an unambiguous command teaches: a wrong print is worse than none.
        if view.is_match and view.speaker_id is not None and view.distance is not None \
                and view.threshold is not None and view.distance <= view.threshold - 0.05:
            n = self.prints.add(view.speaker_id, view.speaker or str(view.speaker_id),
                                embedding, source="auto", clip=clip_name)
            return f"{view.speaker} (auto, {n})", view
        return None, view

    # -- storage -------------------------------------------------------------
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

    @property
    def _jsonl(self) -> Optional[Path]:
        return self.runs_dir / "runs.jsonl" if self.runs_dir else None

    def _append_jsonl(self, row: dict) -> None:
        path = self._jsonl
        if path is None:
            return
        try:
            path.parent.mkdir(parents=True, exist_ok=True)
            with open(path, "a", encoding="utf-8") as fh:
                fh.write(json.dumps(row, ensure_ascii=False) + "\n")
        except OSError as exc:
            _LOGGER.debug("Could not append wake run: %s", exc)

    def history(self) -> list[dict]:
        """Every recorded run, oldest first (memory holds only since start)."""
        path = self._jsonl
        if path is None or not path.exists():
            return list(self.records)
        rows = []
        for line in path.read_text(encoding="utf-8").splitlines():
            try:
                row = json.loads(line)
            except ValueError:
                continue
            # Rows from before the second verifier had one verdict: the app's.
            if "en" not in row and "verdict" in row:
                row["en"], row["en_reason"] = row.get("verdict"), row.get("reason")
            rows.append(row)
        return rows

    def label_range(self, label: Optional[str], start: str, end: str) -> int:
        """Label (or, with None, unlabel) the recorded wake words in [start, end]."""
        if label is not None and label not in LABELS:
            raise ValueError(f"label must be one of {', '.join(LABELS)}")
        path = self._jsonl
        rows = self.history()
        changed = 0
        for row in rows:
            if row.get("wake_word") and start <= (row.get("at") or "") <= end:
                row["label"] = label
                changed += 1
        if path is not None and changed:
            tmp = path.with_suffix(".tmp")
            tmp.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows),
                           encoding="utf-8")
            tmp.replace(path)
        by_at = {(r.get("at"), r.get("clip")): r.get("label") for r in rows}
        for row in self.records:
            if (row.get("at"), row.get("clip")) in by_at:
                row["label"] = by_at[(row.get("at"), row.get("clip"))]
        return changed

    def open_label_window(self, label: str, seconds: float) -> None:
        if label not in LABELS:
            raise ValueError(f"label must be one of {', '.join(LABELS)}")
        self.labelling = (label, time.time() + seconds)

    def close_label_window(self) -> None:
        self.labelling = None

    # -- calibration -----------------------------------------------------------
    def comparison(self) -> dict:
        """How each verifier did on the labelled wake words, and how often
        the two agreed on the rest."""
        rows = [r for r in self.history() if r.get("wake_word")]
        out: dict[str, Any] = {}
        for label in LABELS:
            labelled = [r for r in rows if r.get("label") == label]
            out[label] = {
                "runs": len(labelled),
                **{f"{s}_accepted": sum(1 for r in labelled if r.get(s) == "accepted")
                   for s in SOURCES},
                **{f"{s}_judged": sum(1 for r in labelled if r.get(s)) for s in SOURCES},
            }
        both = [r for r in rows if r.get("en") and r.get("de")]
        out["unlabelled_agreement"] = {
            "runs": len(both),
            "agree": sum(1 for r in both if r["en"] == r["de"]),
            "de_only_accepted": sum(1 for r in both
                                    if r["de"] == "accepted" and r["en"] != "accepted"),
            "en_only_accepted": sum(1 for r in both
                                    if r["en"] == "accepted" and r["de"] != "accepted"),
        }
        return out

    async def replay(self, source: str, *, label: Optional[str] = None,
                     limit: int = 100) -> dict:
        """Send stored clips through a verifier again: after changing its
        thresholds, this says what it now makes of the same wake words."""
        if source not in self.urls:
            raise ValueError(f"no {source} verifier configured")
        if self.runs_dir is None:
            raise ValueError("no data directory")
        rows = [r for r in self.history()
                if r.get("clip") and r.get("wake_word")
                and (label is None or r.get("label") == label)][-limit:]
        results = []
        for row in rows:
            path = self.runs_dir / row["clip"]
            if not path.exists():
                continue
            with wave.open(str(path), "rb") as w:
                audio = w.readframes(w.getnframes())
            verdict, reason = await self._replay_one(source, row, audio)
            results.append({"at": row.get("at"), "clip": row["clip"],
                            "label": row.get("label"), "verdict": verdict,
                            "reason": reason, "recorded": row.get(source)})
        summary: dict[str, dict] = {}
        for result in results:
            key = result["label"] or "unlabelled"
            entry = summary.setdefault(key, {"runs": 0, "accepted": 0, "rejected": 0,
                                             "no_verdict": 0})
            entry["runs"] += 1
            entry[result["verdict"] or "no_verdict"] += 1
        return {"source": source, "summary": summary, "results": results}

    async def _replay_one(self, source: str, row: dict,
                          audio: bytes) -> tuple[Optional[str], Optional[str]]:
        try:
            upstream = await self._connect(_ws_url(self.urls[source]))
        except Exception as exc:
            return None, f"unreachable: {exc}"
        try:
            await upstream.send(json.dumps({
                "type": "run-start",
                "satellite_id": row.get("satellite_id"),
                "wake_word_phrase": row.get("wake_word"),
                "wake_word_audio_ms": row.get("wake_word_audio_ms") or 1500,
                "silence_seconds": None,
            }))
            for i in range(0, len(audio), _CHUNK_BYTES):
                await upstream.send(audio[i:i + _CHUNK_BYTES])

            async def verdict() -> tuple[Optional[str], Optional[str]]:
                async for message in upstream:
                    if not isinstance(message, str):
                        continue
                    event = json.loads(message)
                    output = event.get("wake_word_output") or {}
                    if event.get("type") == "wake-word-end":
                        return "accepted", None
                    if event.get("type") == "wake-word-rejected":
                        return "rejected", output.get("reason")
                return None, "closed without a verdict"

            try:
                return await asyncio.wait_for(verdict(), timeout=_REPLAY_VERDICT_SECONDS)
            except asyncio.TimeoutError:
                return None, "no verdict in time"
        finally:
            try:
                await upstream.send(json.dumps({"type": "audio-end"}))
            except Exception:
                pass
            await self._close(upstream)

    def status(self, recent: int = 20) -> dict[str, Any]:
        window = self.prints.window_speaker()
        label = self._window_label()
        return {
            "upstreams": dict(self.urls),
            "verifies": {k: sorted(v) for k, v in self.verifies.items()},
            "settings": dict(self.prints.settings),
            "enrolling": {"speaker": window[1], "seconds_left": round(
                self.prints.enrolling[2] - time.time())} if window else None,
            "labelling": {"label": label, "seconds_left": round(
                self.labelling[1] - time.time())} if label else None,
            "voiceprints": self.prints.stats(),
            "comparison": self.comparison(),
            "recent": self.records[-recent:] if recent else [],
        }


def _describe_verdict(v: Verdict) -> str:
    if v.verdict is None:
        return "no verdict"
    text = v.verdict
    if v.reason:
        text += f" ({v.reason})"
    if v.ms is not None:
        text += f" {v.ms:.0f} ms"
    return text


def _describe_view(view: Optional[SpeakerView]) -> str:
    if view is None:
        return "not checked"
    if view.error:
        return f"– ({view.error})"
    if view.is_match:
        return f"{view.speaker} d={view.distance:.3f}"
    return f"no match, nearest {view.nearest} d={view.distance:.3f}"


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
    _LOGGER.info("Wake word proxy on %s:%d → %s (authority %s)", host, port,
                 ", ".join(f"{k} {v}" for k, v in proxy.urls.items()),
                 proxy.prints.settings.get("authority"))
    refresher = asyncio.create_task(proxy.keep_info_fresh())
    try:
        await server.serve()
    finally:
        refresher.cancel()


__all__ = [
    "DE",
    "EN",
    "LABELS",
    "RunRecord",
    "SpeakerTools",
    "SpeakerView",
    "Upstream",
    "WakePrints",
    "WakeProxy",
    "run_wake_proxy",
    "speaker_tools_for",
]
