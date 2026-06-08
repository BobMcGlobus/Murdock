"""CAM++ speaker-embedding extraction via onnxruntime.

Loads the WeSpeaker / 3D-Speaker CAM++ model (exported to ONNX) and turns
mono 16 kHz audio into a 192-dimensional L2-normalized embedding.
"""

from __future__ import annotations

import logging
from pathlib import Path
from threading import Lock
from typing import Optional

import numpy as np

from .audio import pcm16_bytes_to_float32
from .fbank import FBankExtractor

_LOGGER = logging.getLogger("murdock.embeddings")

# Minimum audio length for CAM++ to produce a stable embedding (~0.5 s of speech).
_MIN_FRAMES = 50  # 50 fbank frames @ 10 ms hop = 0.5 s


class CAMPPlusEmbedder:
    """Speaker-embedding extractor backed by CAM++ ONNX.

    The model is loaded lazily on first use so that unit tests and the web
    UI can start up without a working model file.
    """

    EMBEDDING_DIM = 192

    def __init__(self, model_path: Path, sample_rate: int = 16000) -> None:
        self.model_path = Path(model_path)
        self.sample_rate = sample_rate
        self._session = None  # type: ignore[assignment]
        self._input_name: Optional[str] = None
        self._output_name: Optional[str] = None
        self._lock = Lock()
        self._fbank = FBankExtractor(sample_rate=sample_rate)

    def _ensure_session(self) -> None:
        if self._session is not None:
            return
        if not self.model_path.exists():
            raise FileNotFoundError(
                f"CAM++ ONNX model not found at {self.model_path}. "
                "Run scripts/download_models.sh or mount the model file."
            )
        # Imported here so that unit tests without onnxruntime still load the module.
        import onnxruntime as ort

        sess_opts = ort.SessionOptions()
        sess_opts.intra_op_num_threads = 1
        sess_opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        self._session = ort.InferenceSession(
            str(self.model_path),
            sess_options=sess_opts,
            providers=["CPUExecutionProvider"],
        )
        self._input_name = self._session.get_inputs()[0].name
        self._output_name = self._session.get_outputs()[0].name
        _LOGGER.info(
            "Loaded CAM++ embedder from %s (input=%s, output=%s)",
            self.model_path, self._input_name, self._output_name,
        )

    def embed_pcm(self, pcm_bytes: bytes) -> np.ndarray:
        """Return an L2-normalized embedding for raw 16 kHz mono PCM bytes."""
        audio = pcm16_bytes_to_float32(pcm_bytes)
        return self.embed_waveform(audio)

    def embed_waveform(self, audio: np.ndarray) -> np.ndarray:
        """Return an L2-normalized embedding for a float32 mono waveform."""
        if audio.size == 0:
            raise ValueError("Cannot embed empty audio")

        feats = self._fbank(audio)
        if feats.shape[0] < _MIN_FRAMES:
            raise ValueError(
                f"Audio too short for embedding: got {feats.shape[0]} frames, "
                f"need at least {_MIN_FRAMES}"
            )

        with self._lock:
            self._ensure_session()
            input_tensor = feats[np.newaxis, ...].astype(np.float32)
            outputs = self._session.run(
                [self._output_name], {self._input_name: input_tensor}
            )
        embedding = np.asarray(outputs[0]).reshape(-1).astype(np.float32)

        norm = float(np.linalg.norm(embedding))
        if norm < 1e-9:
            raise ValueError("Embedding has near-zero norm")
        return embedding / norm

    @staticmethod
    def cosine_distance(a: np.ndarray, b: np.ndarray) -> float:
        """Cosine distance (0 = identical, 1 = orthogonal, 2 = opposite)."""
        a = a.astype(np.float32)
        b = b.astype(np.float32)
        dot = float(np.dot(a, b))
        denom = float(np.linalg.norm(a) * np.linalg.norm(b))
        if denom < 1e-9:
            return 2.0
        return 1.0 - dot / denom

    @staticmethod
    def average(embeddings: list[np.ndarray]) -> np.ndarray:
        """Average multiple embeddings and re-normalize (speaker centroid)."""
        if not embeddings:
            raise ValueError("No embeddings to average")
        stacked = np.stack(embeddings, axis=0).astype(np.float32)
        mean = stacked.mean(axis=0)
        norm = float(np.linalg.norm(mean))
        if norm < 1e-9:
            raise ValueError("Averaged embedding has near-zero norm")
        return mean / norm
