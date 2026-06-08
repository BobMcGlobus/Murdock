"""Speaker enrollment, verification, and CRUD against sqlite-vec."""

from __future__ import annotations

import logging
import sqlite3
import struct
import time
from dataclasses import dataclass, field
from threading import RLock
from typing import Dict, List, Optional

import numpy as np

from .audio import encode_wav
from .embeddings import CAMPPlusEmbedder
from .sample_quality import QualityBreakdown, score_sample, speaker_training_quality
from .vad import SileroVAD, VADResult

_LOGGER = logging.getLogger("murdock.speaker_store")


def _embedding_to_blob(embedding: np.ndarray) -> bytes:
    """Pack a float32 vector into the little-endian layout sqlite-vec expects."""
    flat = embedding.astype(np.float32).reshape(-1)
    return struct.pack(f"<{len(flat)}f", *flat)


def _blob_to_embedding(blob: bytes) -> np.ndarray:
    return np.frombuffer(blob, dtype=np.float32)


VALID_ROLES: tuple[str, ...] = (
    "Admin",
    "Mitarbeiter",
    "Familie",
    "Bewohner",
    "Freund",
    "Fremd",
)

VALID_SAMPLE_SOURCES: tuple[str, ...] = (
    "recording",
    "upload",
    "unknown",
    "cli",
    "auto",  # auto-enrolled via aging/re-training
)


@dataclass
class Speaker:
    id: int
    name: str
    ha_user_id: Optional[str]
    role: Optional[str]
    enrollment_count: int
    created_at: float
    updated_at: float


@dataclass
class VerificationResult:
    is_match: bool
    matched_speaker: Optional[str]
    matched_speaker_id: Optional[int]
    distance: float
    threshold: float
    all_distances: Dict[str, float] = field(default_factory=dict)


@dataclass
class EnrollmentResult:
    speaker_id: int
    speaker_name: str
    sample_id: int
    total_samples: int
    vad: Optional[VADResult]
    warnings: List[str]
    quality: Optional[QualityBreakdown] = None


class SpeakerStore:
    """High-level API for speaker enrollment and verification.

    Wraps:
    * CAM++ embedder (CPU ONNX)
    * Silero VAD (for enrollment QC)
    * sqlite-vec backed speaker centroids

    Thread-safe for use from both the Wyoming handler and the FastAPI app.
    """

    def __init__(
        self,
        conn: sqlite3.Connection,
        embedder: CAMPPlusEmbedder,
        vad: Optional[SileroVAD] = None,
        threshold: float = 0.30,
        min_enrollment_speech_ratio: float = 0.6,
    ) -> None:
        self.conn = conn
        self.embedder = embedder
        self.vad = vad
        self.threshold = threshold
        self.min_enrollment_speech_ratio = min_enrollment_speech_ratio
        self._lock = RLock()
        self._quality_weights: Optional[Dict[str, float]] = None  # None = defaults

    # ------------------------------------------------------------------
    # Speaker CRUD
    # ------------------------------------------------------------------

    def list_speakers(self) -> List[Speaker]:
        with self._lock:
            rows = self.conn.execute(
                "SELECT id, name, ha_user_id, role, enrollment_count, "
                "created_at, updated_at FROM speakers ORDER BY name"
            ).fetchall()
        return [
            Speaker(
                id=row["id"],
                name=row["name"],
                ha_user_id=row["ha_user_id"],
                role=row["role"],
                enrollment_count=row["enrollment_count"],
                created_at=row["created_at"],
                updated_at=row["updated_at"],
            )
            for row in rows
        ]

    def get_speaker(self, speaker_id: int) -> Optional[Speaker]:
        with self._lock:
            row = self.conn.execute(
                "SELECT id, name, ha_user_id, role, enrollment_count, "
                "created_at, updated_at FROM speakers WHERE id = ?",
                (speaker_id,),
            ).fetchone()
        if not row:
            return None
        return Speaker(
            id=row["id"],
            name=row["name"],
            ha_user_id=row["ha_user_id"],
            role=row["role"],
            enrollment_count=row["enrollment_count"],
            created_at=row["created_at"],
            updated_at=row["updated_at"],
        )

    def get_speaker_by_name(self, name: str) -> Optional[Speaker]:
        with self._lock:
            row = self.conn.execute(
                "SELECT id FROM speakers WHERE name = ?", (name,)
            ).fetchone()
        if not row:
            return None
        return self.get_speaker(row["id"])

    def delete_speaker(self, speaker_id: int) -> bool:
        with self._lock:
            cur = self.conn.execute(
                "DELETE FROM speakers WHERE id = ?", (speaker_id,)
            )
            self.conn.execute(
                "DELETE FROM speaker_embeddings WHERE speaker_id = ?",
                (speaker_id,),
            )
            self.conn.commit()
            return cur.rowcount > 0

    def list_samples(self, speaker_id: int) -> List[Dict]:
        with self._lock:
            rows = self.conn.execute(
                "SELECT id, duration_sec, source, filename, created_at, "
                "quality_score, satellite_id "
                "FROM speaker_samples WHERE speaker_id = ? "
                "ORDER BY created_at DESC",
                (speaker_id,),
            ).fetchall()
        return [dict(row) for row in rows]

    def get_sample_audio(self, sample_id: int) -> Optional[bytes]:
        with self._lock:
            row = self.conn.execute(
                "SELECT audio FROM speaker_samples WHERE id = ?", (sample_id,)
            ).fetchone()
        return bytes(row["audio"]) if row else None

    def delete_sample(self, sample_id: int) -> bool:
        with self._lock:
            row = self.conn.execute(
                "SELECT speaker_id FROM speaker_samples WHERE id = ?", (sample_id,)
            ).fetchone()
            if not row:
                return False
            speaker_id = row["speaker_id"]
            self.conn.execute("DELETE FROM speaker_samples WHERE id = ?", (sample_id,))
            self._rebuild_centroid(speaker_id)
            self.conn.commit()
            return True

    # ------------------------------------------------------------------
    # Enrollment
    # ------------------------------------------------------------------

    def update_speaker(
        self,
        speaker_id: int,
        name: Optional[str] = None,
        ha_user_id: Optional[str] = None,
        role: Optional[str] = None,
        clear_ha_user_id: bool = False,
        clear_role: bool = False,
    ) -> Speaker:
        """Edit a speaker's metadata. Returns the updated row.

        Pass ``clear_ha_user_id``/``clear_role`` to explicitly NULL the field.
        """
        if role is not None and role not in VALID_ROLES:
            raise ValueError(
                f"Invalid role '{role}'. Allowed: {', '.join(VALID_ROLES)}"
            )
        if name is not None:
            name = name.strip()
            if not name:
                raise ValueError("Speaker name cannot be empty")

        with self._lock:
            current = self.get_speaker(speaker_id)
            if current is None:
                raise ValueError(f"Speaker {speaker_id} not found")

            updates: List[str] = []
            params: List[object] = []
            if name is not None and name != current.name:
                clash = self.conn.execute(
                    "SELECT id FROM speakers WHERE name = ? AND id != ?",
                    (name, speaker_id),
                ).fetchone()
                if clash is not None:
                    raise ValueError(f"Speaker name '{name}' already exists")
                updates.append("name = ?")
                params.append(name)
            if clear_ha_user_id:
                updates.append("ha_user_id = NULL")
            elif ha_user_id is not None:
                updates.append("ha_user_id = ?")
                params.append(ha_user_id)
            if clear_role:
                updates.append("role = NULL")
            elif role is not None:
                updates.append("role = ?")
                params.append(role)

            if updates:
                updates.append("updated_at = ?")
                params.append(time.time())
                params.append(speaker_id)
                self.conn.execute(
                    f"UPDATE speakers SET {', '.join(updates)} WHERE id = ?",
                    params,
                )
                self.conn.commit()

        updated = self.get_speaker(speaker_id)
        assert updated is not None
        return updated

    # ------------------------------------------------------------------
    # Enrollment
    # ------------------------------------------------------------------

    def enroll(
        self,
        speaker_name: str,
        pcm_bytes: bytes,
        duration_sec: float,
        ha_user_id: Optional[str] = None,
        role: Optional[str] = None,
        source: Optional[str] = None,
        filename: Optional[str] = None,
        skip_vad: bool = False,
        satellite_id: Optional[str] = None,
    ) -> EnrollmentResult:
        """Add a new sample for ``speaker_name`` and (re)compute its centroid.

        The speaker is created on demand if it doesn't exist yet. ``role``
        is only applied when creating a brand-new speaker; existing
        speakers keep their role and must be edited via ``update_speaker``.
        """
        if role is not None and role not in VALID_ROLES:
            raise ValueError(
                f"Invalid role '{role}'. Allowed: {', '.join(VALID_ROLES)}"
            )
        if source is not None and source not in VALID_SAMPLE_SOURCES:
            raise ValueError(
                f"Invalid source '{source}'. Allowed: {', '.join(VALID_SAMPLE_SOURCES)}"
            )

        warnings: List[str] = []
        vad_result: Optional[VADResult] = None

        if duration_sec < 1.0:
            raise ValueError(
                f"Sample is only {duration_sec:.1f}s long; need at least "
                "1 second of audio."
            )

        if self.vad is not None and not skip_vad:
            try:
                vad_result = self.vad.analyze_pcm(pcm_bytes)
            except Exception as exc:
                _LOGGER.warning("VAD failed (%s); continuing without QC", exc)
                vad_result = None
            if vad_result is not None:
                if vad_result.peak_probability < 0.1:
                    # Model ran but produced essentially no signal — likely
                    # a model/feature mismatch. Don't block enrollment on it.
                    warnings.append(
                        "VAD returned no speech probability; skipping quality check"
                    )
                elif vad_result.speech_seconds < 1.0:
                    warnings.append(
                        f"Only {vad_result.speech_seconds:.1f}s of detected "
                        "speech — consider a longer sample"
                    )
                elif vad_result.speech_ratio < self.min_enrollment_speech_ratio:
                    warnings.append(
                        f"Low speech ratio: {vad_result.speech_ratio:.0%} "
                        f"(threshold {self.min_enrollment_speech_ratio:.0%})"
                    )

        embedding = self.embedder.embed_pcm(pcm_bytes)
        wav_bytes = encode_wav(pcm_bytes)
        now = time.time()

        # Quality scoring — get existing centroid if speaker exists
        quality: Optional[QualityBreakdown] = None
        existing_centroid: Optional[np.ndarray] = None

        with self._lock:
            row = self.conn.execute(
                "SELECT id FROM speakers WHERE name = ?", (speaker_name,)
            ).fetchone()
            if row is None:
                cur = self.conn.execute(
                    "INSERT INTO speakers(name, ha_user_id, role, "
                    "enrollment_count, created_at, updated_at) "
                    "VALUES(?, ?, ?, 0, ?, ?)",
                    (speaker_name, ha_user_id, role, now, now),
                )
                speaker_id = int(cur.lastrowid)
            else:
                speaker_id = int(row["id"])
                # Only fill ha_user_id / role if the existing values are NULL,
                # so subsequent enrollments don't accidentally clobber edits.
                if ha_user_id is not None:
                    self.conn.execute(
                        "UPDATE speakers SET ha_user_id = COALESCE(ha_user_id, ?) "
                        "WHERE id = ?",
                        (ha_user_id, speaker_id),
                    )
                if role is not None:
                    self.conn.execute(
                        "UPDATE speakers SET role = COALESCE(role, ?) WHERE id = ?",
                        (role, speaker_id),
                    )
                # Fetch existing centroid for quality scoring
                try:
                    existing_centroid = self._get_centroid(speaker_id)
                except Exception:
                    pass

        # Compute quality score (outside lock — may be slow)
        try:
            quality = score_sample(
                pcm_bytes,
                embedder=self.embedder,
                vad=self.vad,
                centroid=existing_centroid,
                weights=self._quality_weights,
            )
        except Exception as exc:
            _LOGGER.warning("Quality scoring failed: %s", exc)

        quality_value = quality.composite if quality else None

        with self._lock:
            cur = self.conn.execute(
                "INSERT INTO speaker_samples("
                "speaker_id, audio, duration_sec, source, filename, created_at, "
                "quality_score, satellite_id) "
                "VALUES(?, ?, ?, ?, ?, ?, ?, ?)",
                (speaker_id, wav_bytes, duration_sec, source, filename, now,
                 quality_value, satellite_id),
            )
            sample_id = int(cur.lastrowid)

            centroid, sample_count = self._compute_centroid(speaker_id)
            self._write_centroid(speaker_id, centroid)

            self.conn.execute(
                "UPDATE speakers SET enrollment_count = ?, updated_at = ? WHERE id = ?",
                (sample_count, now, speaker_id),
            )
            self.conn.commit()

        _LOGGER.info(
            "Enrolled '%s' (id=%d, sample=%d, total=%d, quality=%.3f)",
            speaker_name, speaker_id, sample_id, sample_count,
            quality_value or 0.0,
        )
        _ = embedding  # kept for potential diagnostics
        return EnrollmentResult(
            speaker_id=speaker_id,
            speaker_name=speaker_name,
            sample_id=sample_id,
            total_samples=sample_count,
            vad=vad_result,
            warnings=warnings,
            quality=quality,
        )

    # ------------------------------------------------------------------
    # Aging / auto-enrollment
    # ------------------------------------------------------------------

    def auto_enroll_embedding(
        self,
        speaker_id: int,
        embedding: np.ndarray,
        audio_16k: bytes,
        duration_sec: float,
        max_auto_samples: int = 20,
        satellite_id: Optional[str] = None,
    ) -> bool:
        """Add a fresh embedding as an 'auto' sample to an existing speaker.

        Smart replacement: if at the cap, only replace the worst-scoring
        auto sample when the new sample scores higher.  Recomputes the
        centroid so subsequent verifications benefit from the updated
        profile.  Returns True if a sample was actually added.
        """
        now = time.time()
        wav_bytes = encode_wav(audio_16k)

        # Score the new sample (outside lock — may be slow)
        new_quality: Optional[float] = None
        try:
            centroid_for_score: Optional[np.ndarray] = None
            with self._lock:
                centroid_for_score = self._get_centroid(speaker_id)
            q = score_sample(
                audio_16k,
                embedder=self.embedder,
                vad=self.vad,
                centroid=centroid_for_score,
                weights=self._quality_weights,
            )
            new_quality = q.composite
        except Exception as exc:
            _LOGGER.warning("Quality scoring failed for auto-enroll: %s", exc)

        with self._lock:
            auto_samples = self.conn.execute(
                "SELECT id, quality_score FROM speaker_samples "
                "WHERE speaker_id = ? AND source = 'auto' "
                "ORDER BY created_at ASC",
                (speaker_id,),
            ).fetchall()

            if len(auto_samples) >= max_auto_samples:
                # Smart replacement: find the worst-scoring auto sample
                if new_quality is not None:
                    worst_id = None
                    worst_score = new_quality  # Only replace if we're better
                    for row in auto_samples:
                        s = row["quality_score"]
                        if s is None or s < worst_score:
                            worst_score = s if s is not None else -1.0
                            worst_id = row["id"]
                    if worst_id is None:
                        _LOGGER.debug(
                            "Auto-enroll skipped for speaker_id=%d: "
                            "new sample (%.3f) not better than worst existing",
                            speaker_id, new_quality,
                        )
                        return False
                    self.conn.execute(
                        "DELETE FROM speaker_samples WHERE id = ?",
                        (worst_id,),
                    )
                else:
                    # No quality score — fall back to oldest-delete
                    excess = len(auto_samples) - max_auto_samples + 1
                    if excess > 0:
                        ids_to_delete = [r["id"] for r in auto_samples[:excess]]
                        self.conn.execute(
                            f"DELETE FROM speaker_samples WHERE id IN "
                            f"({','.join('?' * len(ids_to_delete))})",
                            ids_to_delete,
                        )

            self.conn.execute(
                "INSERT INTO speaker_samples("
                "speaker_id, audio, duration_sec, source, filename, created_at, "
                "quality_score, satellite_id) "
                "VALUES(?, ?, ?, 'auto', NULL, ?, ?, ?)",
                (speaker_id, wav_bytes, duration_sec, now, new_quality, satellite_id),
            )

            centroid, sample_count = self._compute_centroid(speaker_id)
            self._write_centroid(speaker_id, centroid)
            self.conn.execute(
                "UPDATE speakers SET enrollment_count = ?, updated_at = ? WHERE id = ?",
                (sample_count, now, speaker_id),
            )
            self.conn.commit()

        _LOGGER.info(
            "Auto-enrolled fresh embedding for speaker_id=%d "
            "(%.2fs, quality=%.3f, total=%d, auto_cap=%d)",
            speaker_id, duration_sec, new_quality or 0.0,
            sample_count, max_auto_samples,
        )
        return True

    # ------------------------------------------------------------------
    # Verification
    # ------------------------------------------------------------------

    def verify_pcm(
        self,
        pcm_bytes: bytes,
        threshold: Optional[float] = None,
    ) -> VerificationResult:
        """Verify PCM audio against all enrolled speakers."""
        try:
            embedding = self.embedder.embed_pcm(pcm_bytes)
        except ValueError as exc:
            _LOGGER.debug("Embedding failed: %s", exc)
            return VerificationResult(
                is_match=False,
                matched_speaker=None,
                matched_speaker_id=None,
                distance=2.0,
                threshold=threshold or self.threshold,
            )
        return self.verify_embedding(embedding, threshold=threshold)

    def verify_embedding(
        self,
        embedding: np.ndarray,
        threshold: Optional[float] = None,
    ) -> VerificationResult:
        effective_threshold = threshold if threshold is not None else self.threshold
        blob = _embedding_to_blob(embedding)

        with self._lock:
            rows = self.conn.execute(
                "SELECT se.speaker_id AS speaker_id, s.name AS name, "
                "       vec_distance_cosine(se.embedding, ?) AS distance "
                "FROM speaker_embeddings se "
                "JOIN speakers s ON s.id = se.speaker_id "
                "ORDER BY distance LIMIT 5",
                (blob,),
            ).fetchall()

        if not rows:
            return VerificationResult(
                is_match=False,
                matched_speaker=None,
                matched_speaker_id=None,
                distance=2.0,
                threshold=effective_threshold,
            )

        all_distances = {row["name"]: float(row["distance"]) for row in rows}
        best = rows[0]
        distance = float(best["distance"])
        is_match = distance <= effective_threshold

        return VerificationResult(
            is_match=is_match,
            matched_speaker=best["name"] if is_match else None,
            matched_speaker_id=int(best["speaker_id"]) if is_match else None,
            distance=distance,
            threshold=effective_threshold,
            all_distances=all_distances,
        )

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    def _get_centroid(self, speaker_id: int) -> Optional[np.ndarray]:
        """Read the current centroid from speaker_embeddings, or None."""
        row = self.conn.execute(
            "SELECT embedding FROM speaker_embeddings WHERE speaker_id = ?",
            (speaker_id,),
        ).fetchone()
        if row is None:
            return None
        return _blob_to_embedding(bytes(row["embedding"]))

    def set_quality_weights(self, weights: Optional[Dict[str, float]]) -> None:
        """Set custom quality scoring weights (None = defaults)."""
        self._quality_weights = weights

    def get_speaker_training_quality(self, speaker_id: int) -> float:
        """Compute aggregate training quality for a speaker."""
        with self._lock:
            rows = self.conn.execute(
                "SELECT quality_score FROM speaker_samples "
                "WHERE speaker_id = ? AND quality_score IS NOT NULL",
                (speaker_id,),
            ).fetchall()
        scores = [float(r["quality_score"]) for r in rows]
        return speaker_training_quality(scores)

    def rescore_all_samples(self, speaker_id: int) -> int:
        """Recompute quality scores for all samples of a speaker.

        Returns the number of samples rescored.
        """
        with self._lock:
            centroid = self._get_centroid(speaker_id)
            rows = self.conn.execute(
                "SELECT id, audio FROM speaker_samples WHERE speaker_id = ? ORDER BY id",
                (speaker_id,),
            ).fetchall()

        count = 0
        for row in rows:
            try:
                from .audio import decode_wav, to_mono_16k_pcm
                pcm, rate, width, channels = decode_wav(bytes(row["audio"]))
                pcm = to_mono_16k_pcm(pcm, rate, width, channels)
                q = score_sample(
                    pcm,
                    embedder=self.embedder,
                    vad=self.vad,
                    centroid=centroid,
                    weights=self._quality_weights,
                )
                with self._lock:
                    self.conn.execute(
                        "UPDATE speaker_samples SET quality_score = ? WHERE id = ?",
                        (q.composite, row["id"]),
                    )
                count += 1
            except Exception as exc:
                _LOGGER.warning("Failed to rescore sample %d: %s", row["id"], exc)

        with self._lock:
            self.conn.commit()
        return count

    def _compute_centroid(self, speaker_id: int) -> tuple[np.ndarray, int]:
        rows = self.conn.execute(
            "SELECT audio FROM speaker_samples WHERE speaker_id = ? ORDER BY id",
            (speaker_id,),
        ).fetchall()
        embeddings: List[np.ndarray] = []
        for row in rows:
            from .audio import decode_wav, to_mono_16k_pcm
            pcm, rate, width, channels = decode_wav(bytes(row["audio"]))
            pcm = to_mono_16k_pcm(pcm, rate, width, channels)
            try:
                embeddings.append(self.embedder.embed_pcm(pcm))
            except ValueError as exc:
                _LOGGER.warning("Skipping sample during centroid rebuild: %s", exc)
        if not embeddings:
            return np.zeros(CAMPPlusEmbedder.EMBEDDING_DIM, dtype=np.float32), 0
        centroid = CAMPPlusEmbedder.average(embeddings)
        return centroid, len(embeddings)

    def _write_centroid(self, speaker_id: int, centroid: np.ndarray) -> None:
        # sqlite-vec's vec0 virtual tables don't support ON CONFLICT / UPSERT,
        # so we always DELETE the existing row before INSERTing the new one.
        self.conn.execute(
            "DELETE FROM speaker_embeddings WHERE speaker_id = ?",
            (speaker_id,),
        )
        if centroid.size == 0 or float(np.linalg.norm(centroid)) < 1e-9:
            return
        blob = _embedding_to_blob(centroid)
        self.conn.execute(
            "INSERT INTO speaker_embeddings(speaker_id, embedding) VALUES(?, ?)",
            (speaker_id, blob),
        )

    def _rebuild_centroid(self, speaker_id: int) -> None:
        centroid, sample_count = self._compute_centroid(speaker_id)
        self._write_centroid(speaker_id, centroid)
        self.conn.execute(
            "UPDATE speakers SET enrollment_count = ?, updated_at = ? WHERE id = ?",
            (sample_count, time.time(), speaker_id),
        )
