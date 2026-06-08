"""Adaptive target-speaker extraction.

The single-embedding verify path averages a CAM++ embedding over the
whole utterance. When the clip contains more than one voice — the target
speaker plus a TV in the background, or two people — that average is a
blend that drifts away from every enrolled profile and gets falsely
rejected.

Extraction fixes this by segmenting the utterance, embedding each speech
region separately, scoring it against the enrolled speakers, and keeping
only the regions that belong to the *dominant* matched speaker. The
concatenation of those regions is a clean, single-speaker clip that the
normal verify gate then judges.

Design goals (see project plan):
  * **Fast-path** — with 0 or 1 speech region there is nothing to
    separate, so we return immediately after the (cheap) VAD pass and add
    zero extra embedding cost in the common case.
  * **Cost proportional to noise** — extra embeddings only happen when
    there are multiple regions, i.e. exactly the noisy / multi-speaker
    case extraction is meant to help.
  * **Never worse** — if no region confidently matches a known speaker,
    or every region is the target, we return the original audio untouched
    and let the verify gate decide.
  * **Min region length** — short regions score unreliably (and fall
    below the embedder's minimum frame count), so they are not used to
    claim a region for a speaker.

``extraction_threshold`` is a cosine *distance* ceiling and should be
**stricter** (smaller) than the verify threshold: we only want to keep
regions we are quite sure belong to an enrolled speaker, then verify the
clean concatenation at the normal, more lenient threshold.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Optional

_LOGGER = logging.getLogger("murdock.extraction")

_SAMPLE_RATE = 16000
_WIDTH = 2  # 16-bit PCM
# The embedder needs ~0.5 s to produce a stable embedding; never hand the
# verify gate less than this after extraction.
_MIN_KEPT_SECONDS = 0.5


@dataclass
class ExtractionResult:
    """Outcome of an extraction pass.

    ``applied`` is True only when we actually dropped some audio. When it
    is False, ``audio`` is the original clip unchanged and the caller
    should behave exactly as it did before extraction existed.
    """

    applied: bool
    audio: bytes
    n_regions: int = 0
    n_kept: int = 0
    target_speaker: Optional[str] = None
    dropped_seconds: float = 0.0


def extract_target_speaker(
    audio_16k: bytes,
    *,
    vad,
    embedder,
    speakers,
    extraction_threshold: float,
    min_region_sec: float = 0.6,
) -> ExtractionResult:
    """Isolate the dominant enrolled speaker's speech in ``audio_16k``.

    Returns an :class:`ExtractionResult`. On the fast-path (≤1 region) or
    when extraction can't confidently improve things, ``applied`` is False
    and ``audio`` is the original clip.

    The collaborators are passed in (not imported) so this stays unit
    testable with light fakes:
      * ``vad.analyze_pcm(bytes) -> result with .segments[(start_s,end_s)]``
      * ``embedder.embed_pcm(bytes) -> np.ndarray`` (may raise ValueError
        for too-short audio)
      * ``speakers.verify_embedding(emb, threshold) -> result with
        .distance, .matched_speaker_id, .matched_speaker``
    """
    if not audio_16k:
        return ExtractionResult(applied=False, audio=audio_16k)

    try:
        vad_result = vad.analyze_pcm(audio_16k)
    except Exception:
        _LOGGER.debug("Extraction VAD pass failed; skipping", exc_info=True)
        return ExtractionResult(applied=False, audio=audio_16k)

    segments = list(getattr(vad_result, "segments", []) or [])

    # --- Fast path: nothing to separate ---
    if len(segments) <= 1:
        return ExtractionResult(
            applied=False, audio=audio_16k, n_regions=len(segments)
        )

    bytes_per_sec = _SAMPLE_RATE * _WIDTH

    # Score every region. Each entry: (start_b, end_b, dur, sid, name).
    regions: list[tuple[int, int, float, Optional[int], Optional[str]]] = []
    for start_s, end_s in segments:
        dur = max(0.0, end_s - start_s)
        start_b = int(start_s * bytes_per_sec)
        end_b = int(end_s * bytes_per_sec)
        start_b -= start_b % _WIDTH
        end_b -= end_b % _WIDTH
        end_b = min(end_b, len(audio_16k))
        if end_b <= start_b:
            continue

        if dur < min_region_sec:
            # Too short to score reliably — record as "not the target".
            regions.append((start_b, end_b, dur, None, None))
            continue

        region_pcm = audio_16k[start_b:end_b]
        try:
            emb = embedder.embed_pcm(region_pcm)
        except Exception:
            regions.append((start_b, end_b, dur, None, None))
            continue

        res = speakers.verify_embedding(emb, threshold=extraction_threshold)
        sid = getattr(res, "matched_speaker_id", None)
        if sid is not None and float(getattr(res, "distance", 2.0)) <= extraction_threshold:
            regions.append((start_b, end_b, dur, int(sid), getattr(res, "matched_speaker", None)))
        else:
            regions.append((start_b, end_b, dur, None, None))

    matched = [r for r in regions if r[3] is not None]
    if not matched:
        # No region confidently belongs to a known speaker — leave it to
        # the verify gate (likely an "unknown"). Don't fabricate a clip.
        return ExtractionResult(
            applied=False, audio=audio_16k, n_regions=len(segments)
        )

    # Dominant speaker = the enrolled speaker covering the most seconds.
    duration_by_speaker: dict[int, float] = {}
    name_by_speaker: dict[int, Optional[str]] = {}
    for _sb, _eb, dur, sid, name in matched:
        duration_by_speaker[sid] = duration_by_speaker.get(sid, 0.0) + dur
        name_by_speaker.setdefault(sid, name)
    target_sid = max(duration_by_speaker, key=lambda k: duration_by_speaker[k])
    target_name = name_by_speaker.get(target_sid)

    kept = [r for r in regions if r[3] == target_sid]

    # Every region already belongs to the target → extraction is a no-op.
    if len(kept) == len(regions):
        return ExtractionResult(
            applied=False,
            audio=audio_16k,
            n_regions=len(segments),
            n_kept=len(kept),
            target_speaker=target_name,
        )

    kept_audio = b"".join(audio_16k[sb:eb] for sb, eb, *_ in kept)
    dropped_seconds = sum(dur for _sb, _eb, dur, sid, _n in regions if sid != target_sid)

    # Guard: never hand back less than the embedder can use.
    if len(kept_audio) < int(_MIN_KEPT_SECONDS * bytes_per_sec):
        return ExtractionResult(
            applied=False, audio=audio_16k, n_regions=len(segments)
        )

    return ExtractionResult(
        applied=True,
        audio=kept_audio,
        n_regions=len(segments),
        n_kept=len(kept),
        target_speaker=target_name,
        dropped_seconds=dropped_seconds,
    )
