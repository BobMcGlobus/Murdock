"""Silero VAD wrapper for speech-presence detection.

Used for:
    * Enrollment quality control — reject samples with too little speech.
    * Optional pre-verification trimming in the Wyoming proxy.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from threading import Lock
from typing import List, Optional, Tuple

import numpy as np

from .audio import pcm16_bytes_to_float32

_LOGGER = logging.getLogger("murdock.vad")

_WINDOW_SAMPLES = 512  # Silero VAD v4/v5 at 16 kHz: 512 new samples per step.
_CONTEXT_SAMPLES = 64  # Silero v5 prepends the previous step's tail samples.
_SAMPLE_RATE = 16000


@dataclass
class VADResult:
    """Summary of a VAD pass over an audio clip."""

    total_seconds: float
    speech_seconds: float
    speech_ratio: float
    segments: List[Tuple[float, float]]
    peak_probability: float

    @property
    def has_enough_speech(self) -> bool:
        return self.speech_seconds >= 1.0 and self.speech_ratio >= 0.5


class SileroVAD:
    """Thin onnxruntime wrapper around Silero VAD.

    Lazy-loads the model on first use.
    """

    def __init__(
        self,
        model_path: Path,
        speech_threshold: float = 0.5,
        min_silence_ms: int = 200,
    ) -> None:
        self.model_path = Path(model_path)
        self.speech_threshold = speech_threshold
        self.min_silence_samples = int(min_silence_ms * _SAMPLE_RATE / 1000)
        self._session = None  # type: ignore[assignment]
        self._state: Optional[np.ndarray] = None
        self._context: Optional[np.ndarray] = None
        self._input_names: List[str] = []
        self._output_names: List[str] = []
        self._is_v5: bool = True
        self._lock = Lock()

    def _ensure_session(self) -> None:
        if self._session is not None:
            return
        if not self.model_path.exists():
            raise FileNotFoundError(
                f"Silero VAD ONNX model not found at {self.model_path}."
            )
        import onnxruntime as ort

        sess_opts = ort.SessionOptions()
        sess_opts.intra_op_num_threads = 1
        self._session = ort.InferenceSession(
            str(self.model_path),
            sess_options=sess_opts,
            providers=["CPUExecutionProvider"],
        )
        self._input_names = [inp.name for inp in self._session.get_inputs()]
        self._output_names = [out.name for out in self._session.get_outputs()]
        # Silero v5 uses a single 'state' input; v4 uses separate 'h' and 'c'.
        self._is_v5 = "state" in self._input_names
        _LOGGER.info(
            "Loaded Silero VAD from %s (inputs=%s, outputs=%s, version=%s)",
            self.model_path, self._input_names, self._output_names,
            "v5" if self._is_v5 else "v4",
        )

    def _reset_state(self) -> None:
        if self._is_v5:
            self._state = np.zeros((2, 1, 128), dtype=np.float32)
            # v5 prepends the previous step's last 64 samples to the input.
            self._context = np.zeros((1, _CONTEXT_SAMPLES), dtype=np.float32)
        else:
            # v4 keeps two independent LSTM states; no audio context.
            self._state = {
                "h": np.zeros((2, 1, 64), dtype=np.float32),
                "c": np.zeros((2, 1, 64), dtype=np.float32),
            }  # type: ignore[assignment]
            self._context = None

    def _run_chunk(self, chunk: np.ndarray) -> float:
        assert self._session is not None
        assert self._state is not None

        sr_tensor = np.array(_SAMPLE_RATE, dtype=np.int64)
        # Always (1, samples) float32.
        chunk_tensor = chunk[np.newaxis, :].astype(np.float32)

        if self._is_v5:
            assert self._context is not None
            # Concatenate the previous tail context (64 samples) with the new
            # 512-sample chunk → (1, 576). Without this prefix the model gets
            # cold inputs and returns probabilities near zero.
            input_tensor = np.concatenate([self._context, chunk_tensor], axis=1)
            inputs = {
                "input": input_tensor,
                "state": self._state,
                "sr": sr_tensor,
            }
        else:
            assert isinstance(self._state, dict)
            input_tensor = chunk_tensor
            inputs = {
                "input": input_tensor,
                "h": self._state["h"],
                "c": self._state["c"],
                "sr": sr_tensor,
            }
        # Drop inputs the model doesn't declare (safety net).
        inputs = {k: v for k, v in inputs.items() if k in self._input_names}

        outputs = self._session.run(self._output_names, inputs)
        named = dict(zip(self._output_names, outputs))

        # Pick the probability output by name when possible, otherwise by shape.
        prob_array = named.get("output")
        if prob_array is None:
            for arr in outputs:
                if arr.ndim <= 2 and arr.size <= 4:
                    prob_array = arr
                    break
        if prob_array is None:
            return 0.0
        prob = float(np.asarray(prob_array).reshape(-1)[0])

        # Update recurrent state and audio context.
        if self._is_v5:
            new_state = named.get("stateN")
            if new_state is not None:
                self._state = new_state
            # Carry over the last 64 samples of the *new* chunk for next step.
            self._context = chunk_tensor[:, -_CONTEXT_SAMPLES:].copy()
        else:
            assert isinstance(self._state, dict)
            if "hn" in named:
                self._state["h"] = named["hn"]
            if "cn" in named:
                self._state["c"] = named["cn"]

        return prob

    def analyze_pcm(self, pcm_bytes: bytes) -> VADResult:
        """Run VAD on raw 16-kHz mono PCM and return a summary."""
        audio = pcm16_bytes_to_float32(pcm_bytes)
        return self.analyze_waveform(audio)

    def analyze_waveform(self, audio: np.ndarray) -> VADResult:
        total_samples = audio.shape[0]
        total_seconds = total_samples / _SAMPLE_RATE
        if total_samples < _WINDOW_SAMPLES:
            return VADResult(total_seconds, 0.0, 0.0, [], 0.0)

        with self._lock:
            self._ensure_session()
            self._reset_state()

            probs: List[float] = []
            for start in range(0, total_samples - _WINDOW_SAMPLES + 1, _WINDOW_SAMPLES):
                chunk = audio[start : start + _WINDOW_SAMPLES]
                probs.append(self._run_chunk(chunk))

        if not probs:
            return VADResult(total_seconds, 0.0, 0.0, [], 0.0)

        chunk_duration = _WINDOW_SAMPLES / _SAMPLE_RATE
        speech_flags = [p >= self.speech_threshold for p in probs]

        # Collapse adjacent speech chunks into contiguous segments.
        segments: List[Tuple[float, float]] = []
        start_idx: Optional[int] = None
        for i, is_speech in enumerate(speech_flags):
            if is_speech and start_idx is None:
                start_idx = i
            elif not is_speech and start_idx is not None:
                segments.append((start_idx * chunk_duration, i * chunk_duration))
                start_idx = None
        if start_idx is not None:
            segments.append(
                (start_idx * chunk_duration, len(speech_flags) * chunk_duration)
            )

        speech_seconds = sum(end - start for start, end in segments)
        speech_ratio = speech_seconds / total_seconds if total_seconds > 0 else 0.0
        peak = max(probs) if probs else 0.0
        return VADResult(
            total_seconds=total_seconds,
            speech_seconds=speech_seconds,
            speech_ratio=speech_ratio,
            segments=segments,
            peak_probability=peak,
        )

    def trim_to_speech(self, pcm_bytes: bytes) -> bytes:
        """Return PCM bytes containing only the concatenated speech segments."""
        audio = pcm16_bytes_to_float32(pcm_bytes)
        result = self.analyze_waveform(audio)
        if not result.segments:
            return pcm_bytes

        parts: List[np.ndarray] = []
        for start_s, end_s in result.segments:
            start = int(start_s * _SAMPLE_RATE)
            end = int(end_s * _SAMPLE_RATE)
            parts.append(audio[start:end])
        if not parts:
            return pcm_bytes
        trimmed = np.concatenate(parts)
        # Back to int16 PCM.
        from .audio import float32_to_pcm16_bytes
        return float32_to_pcm16_bytes(trimmed)
