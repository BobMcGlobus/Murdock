"""Cluster untagged unknown samples by voice similarity.

The idea: if the same unfamiliar voice triggered VoiceID five times this
afternoon, we don't want the admin to click *Assign to speaker X* five
times. Group them into one bucket first, let the admin label once.

Algorithm: online greedy clustering in embedding space. For each sample
(newest first, so older recordings tack onto the same cluster), find the
nearest existing cluster centroid by cosine distance; if within
``threshold`` join it, otherwise open a new cluster. Embeddings are
already L2-normalised CAM++ vectors, so cosine distance ≡ 1 - dot.

Why online/greedy instead of KMeans or DBSCAN:
- We don't know ``k`` up front — the number of unknown voices is
  whatever the household happens to produce.
- Unknown lists are small (typically < 100 samples); quadratic cost is
  fine and keeps the module dependency-free.
- Greedy preserves determinism: the same input + threshold always yields
  the same clustering, which matters for the UI (refresh shouldn't
  reshuffle rows).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional

import numpy as np

from .unknown_store import UnknownStore

# Conservative default: slightly looser than a typical verify threshold
# (~0.30) so clusters group voices that *would* match the same speaker
# without pulling in genuinely different voices. Tune via query param.
DEFAULT_CLUSTER_THRESHOLD = 0.25


@dataclass
class ClusterMember:
    sample_id: int
    distance_to_centroid: float
    duration_sec: float
    best_speaker: Optional[str]
    best_distance: float
    liveness_score: Optional[float]
    satellite_id: Optional[str]
    created_at: float
    tag: Optional[str]


@dataclass
class Cluster:
    cluster_id: int
    centroid: np.ndarray
    members: List[ClusterMember] = field(default_factory=list)

    @property
    def size(self) -> int:
        return len(self.members)

    @property
    def avg_distance(self) -> float:
        if not self.members:
            return 0.0
        return float(np.mean([m.distance_to_centroid for m in self.members]))

    @property
    def satellites(self) -> List[str]:
        seen: List[str] = []
        for m in self.members:
            if m.satellite_id and m.satellite_id not in seen:
                seen.append(m.satellite_id)
        return seen


def _load_samples_with_embeddings(
    store: UnknownStore,
    include_tagged: bool,
    limit: int,
) -> List[Dict]:
    """Fetch unknown samples + their float32 embeddings in one pass.

    We go under the hood of ``UnknownStore`` here (raw SQL against the
    same connection) because ``list_samples`` strips the embedding for
    performance, and calling ``pop_embedding`` per sample would be N+1.
    """
    where = "" if include_tagged else "WHERE tag IS NULL"
    with store._lock:  # noqa: SLF001 — intentional, shared sqlite lock
        rows = store.conn.execute(
            "SELECT id, session_id, satellite_id, duration_sec, best_distance, "
            "best_speaker, liveness_score, tag, created_at, embedding "
            f"FROM unknown_samples {where} "
            "ORDER BY created_at DESC LIMIT ?",
            (limit,),
        ).fetchall()
    out: List[Dict] = []
    for row in rows:
        emb_blob = row["embedding"]
        if not emb_blob:
            continue
        emb = np.frombuffer(bytes(emb_blob), dtype=np.float32).copy()
        if emb.size == 0:
            continue
        # Safety: re-normalise in case a legacy row wasn't L2-normalised.
        norm = float(np.linalg.norm(emb))
        if norm > 1e-9:
            emb = emb / norm
        out.append({
            "id": row["id"],
            "session_id": row["session_id"],
            "satellite_id": row["satellite_id"],
            "duration_sec": row["duration_sec"],
            "best_distance": row["best_distance"],
            "best_speaker": row["best_speaker"],
            "liveness_score": row["liveness_score"],
            "tag": row["tag"],
            "created_at": row["created_at"],
            "embedding": emb,
        })
    return out


def cluster_unknown_samples(
    store: UnknownStore,
    threshold: float = DEFAULT_CLUSTER_THRESHOLD,
    *,
    include_tagged: bool = False,
    limit: int = 500,
) -> List[Cluster]:
    """Greedy-cluster unknown samples by cosine distance.

    Returns clusters sorted by size descending, with single-member
    clusters last (they're least useful for bulk-assign).
    """
    samples = _load_samples_with_embeddings(store, include_tagged, limit)
    if not samples:
        return []

    clusters: List[Cluster] = []
    for s in samples:
        emb = s["embedding"]
        best_idx = -1
        best_dist = threshold
        for i, cluster in enumerate(clusters):
            # cosine distance = 1 - dot(a, b) for L2-normalised vectors
            dist = 1.0 - float(np.dot(emb, cluster.centroid))
            if dist < best_dist:
                best_dist = dist
                best_idx = i

        if best_idx >= 0:
            cluster = clusters[best_idx]
            member = ClusterMember(
                sample_id=s["id"],
                distance_to_centroid=best_dist,
                duration_sec=s["duration_sec"],
                best_speaker=s["best_speaker"],
                best_distance=s["best_distance"],
                liveness_score=s["liveness_score"],
                satellite_id=s["satellite_id"],
                created_at=s["created_at"],
                tag=s["tag"],
            )
            cluster.members.append(member)
            # Update centroid as running mean, re-normalised so future
            # distance checks remain on the unit sphere.
            n = len(cluster.members)
            new_centroid = cluster.centroid + (emb - cluster.centroid) / n
            norm = float(np.linalg.norm(new_centroid))
            if norm > 1e-9:
                new_centroid = new_centroid / norm
            cluster.centroid = new_centroid
        else:
            cluster = Cluster(
                cluster_id=len(clusters) + 1,
                centroid=emb.copy(),
            )
            cluster.members.append(
                ClusterMember(
                    sample_id=s["id"],
                    distance_to_centroid=0.0,
                    duration_sec=s["duration_sec"],
                    best_speaker=s["best_speaker"],
                    best_distance=s["best_distance"],
                    liveness_score=s["liveness_score"],
                    satellite_id=s["satellite_id"],
                    created_at=s["created_at"],
                    tag=s["tag"],
                )
            )
            clusters.append(cluster)

    # Stable sort: big clusters first, singletons last. Tie-break by
    # earliest cluster_id so display order is deterministic.
    clusters.sort(key=lambda c: (-c.size, c.cluster_id))
    # Renumber so IDs are 1..N top-down (user-friendly).
    for new_id, cluster in enumerate(clusters, 1):
        cluster.cluster_id = new_id
    return clusters
