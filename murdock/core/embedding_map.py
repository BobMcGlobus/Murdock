"""2-D projection of the speaker embedding space for the Web UI.

The CAM++ embeddings live in 192 dimensions; this module flattens them
to 2-D via PCA so the UI can render a scatter map: every enrolled
sample, each speaker's centroid, and (optionally) the untagged unknown
samples. The map answers at a glance:

  * how cleanly the enrolled speakers separate,
  * which samples sit far from their own centroid (drift / bad takes),
  * where the unknown samples cluster relative to known voices.

Everything is recomputed on demand — embeddings are not persisted for
enrolled samples, so a map render re-embeds them (~20 ms each on CPU).
That is deliberate: the map is a diagnostic the user clicks, not a hot
path, and it always reflects the *current* embedder.
"""

from __future__ import annotations

import logging
import time
from typing import List, Optional, Tuple

import numpy as np

from .audio import decode_wav, to_mono_16k_pcm
from .embeddings import CAMPPlusEmbedder

_LOGGER = logging.getLogger("murdock.embedding_map")


def pca_2d(matrix: np.ndarray) -> Tuple[np.ndarray, List[float]]:
    """Project rows of ``matrix`` onto their first two principal axes.

    Returns ``(coords, explained)`` where coords is (n, 2) and explained
    holds the variance fraction captured by each axis. Raises ValueError
    for fewer than two points (nothing to project).
    """
    X = np.asarray(matrix, dtype=np.float32)
    if X.ndim != 2 or X.shape[0] < 2:
        raise ValueError("PCA needs at least two points")
    Xc = X - X.mean(axis=0)
    _U, S, Vt = np.linalg.svd(Xc, full_matrices=False)
    coords = Xc @ Vt[: min(2, Vt.shape[0])].T
    if coords.shape[1] < 2:
        coords = np.pad(coords, ((0, 0), (0, 2 - coords.shape[1])))
    var = (S ** 2).astype(np.float64)
    total = float(var.sum()) or 1.0
    explained = [float(v / total) for v in var[:2]]
    while len(explained) < 2:
        explained.append(0.0)
    return coords.astype(np.float32), explained


def _embed_sample(speakers, sample_id: int) -> Optional[np.ndarray]:
    """Decode + embed one stored sample; None when it can't be embedded."""
    wav = speakers.get_sample_audio(sample_id)
    if wav is None:
        return None
    try:
        pcm, rate, width, channels = decode_wav(wav)
        pcm = to_mono_16k_pcm(pcm, rate, width, channels)
        return speakers.embedder.embed_pcm(pcm)
    except Exception as exc:
        _LOGGER.debug("Map: skipping sample %d: %s", sample_id, exc)
        return None


def compute_embedding_map(
    speakers,
    unknown,
    include_unknown: bool = True,
    max_unknown: int = 100,
) -> dict:
    """Build the 2-D map: re-embed everything, PCA, return plottable points.

    Blocking and embedding-heavy — call via ``asyncio.to_thread``. The
    embedder serialises internally, so this is safe to run while the
    Wyoming handler is active (it just queues behind live verifications).
    """
    t0 = time.monotonic()
    metas: List[dict] = []
    vectors: List[np.ndarray] = []

    for spk in speakers.list_speakers():
        sample_entries: List[Tuple[dict, np.ndarray]] = []
        for sample in speakers.list_samples(spk.id):
            emb = _embed_sample(speakers, sample["id"])
            if emb is None:
                continue
            sample_entries.append((
                {
                    "kind": "sample",
                    "speaker": spk.name,
                    "speaker_id": spk.id,
                    "sample_id": sample["id"],
                    "source": sample.get("source"),
                    "quality": sample.get("quality_score"),
                },
                emb,
            ))
        if not sample_entries:
            continue
        centroid = CAMPPlusEmbedder.average([e for _m, e in sample_entries])
        for meta, emb in sample_entries:
            meta["distance"] = round(
                CAMPPlusEmbedder.cosine_distance(emb, centroid), 4
            )
            metas.append(meta)
            vectors.append(emb)
        metas.append({
            "kind": "centroid",
            "speaker": spk.name,
            "speaker_id": spk.id,
        })
        vectors.append(centroid)

    if include_unknown and unknown is not None:
        for s in unknown.list_samples(include_tagged=False, limit=max_unknown):
            emb = unknown.pop_embedding(s.id)
            if emb is None or emb.size != CAMPPlusEmbedder.EMBEDDING_DIM:
                continue
            metas.append({
                "kind": "unknown",
                "sample_id": s.id,
                "best_speaker": s.best_speaker,
                "distance": round(float(s.best_distance), 4),
            })
            vectors.append(emb.astype(np.float32))

    if len(vectors) < 3:
        return {
            "points": [],
            "explained": [0.0, 0.0],
            "count": len(vectors),
            "note": "need at least 3 embeddings for a meaningful map",
        }

    coords, explained = pca_2d(np.stack(vectors))
    for meta, (x, y) in zip(metas, coords):
        meta["x"] = round(float(x), 4)
        meta["y"] = round(float(y), 4)

    return {
        "points": metas,
        "explained": [round(e, 4) for e in explained],
        "count": len(metas),
        "computed_ms": round((time.monotonic() - t0) * 1000, 1),
    }
