#!/usr/bin/env python3
"""CLI enrollment helper — useful when there is no Web UI access yet.

Usage:
    python -m scripts.enroll --speaker jonas sample1.wav sample2.wav sample3.wav
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

from voiceid.core.audio import decode_audio_any, to_mono_16k_pcm
from voiceid.core.context import build_context


def main() -> int:
    parser = argparse.ArgumentParser(description="Enroll a speaker from WAV files")
    parser.add_argument("--speaker", required=True, help="Speaker name")
    parser.add_argument("--ha-user-id", default=None, help="Optional HA user UUID")
    parser.add_argument(
        "--role",
        default=None,
        help="Optional role tag (Admin, Mitarbeiter, Familie, Bewohner, Freund, Fremd)",
    )
    parser.add_argument(
        "--skip-vad", action="store_true", help="Skip Silero VAD quality checks"
    )
    parser.add_argument("samples", nargs="+", help="One or more WAV files")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s [%(name)s] %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    ctx = build_context()
    enrolled = 0
    for path_str in args.samples:
        path = Path(path_str)
        if not path.is_file():
            print(f"skip (not found): {path}", file=sys.stderr)
            continue
        try:
            pcm, rate, width, channels = decode_audio_any(path.read_bytes())
        except Exception as exc:
            print(f"skip ({exc}): {path}", file=sys.stderr)
            continue
        pcm = to_mono_16k_pcm(pcm, rate, width, channels)
        duration = len(pcm) / (16000 * 2)
        try:
            result = ctx.speakers.enroll(
                speaker_name=args.speaker,
                pcm_bytes=pcm,
                duration_sec=duration,
                ha_user_id=args.ha_user_id,
                role=args.role,
                source="cli",
                filename=path.name,
                skip_vad=args.skip_vad,
            )
        except ValueError as exc:
            print(f"skip ({exc}): {path}", file=sys.stderr)
            continue

        enrolled += 1
        msg = (
            f"Enrolled {path.name} -> speaker_id={result.speaker_id}, "
            f"sample_id={result.sample_id}, total={result.total_samples}"
        )
        if result.warnings:
            msg += " [warnings: " + "; ".join(result.warnings) + "]"
        print(msg)

    if enrolled == 0:
        print("No samples enrolled.", file=sys.stderr)
        return 1
    speaker = ctx.speakers.get_speaker_by_name(args.speaker)
    if speaker:
        print(
            f"\n{speaker.name}: {speaker.enrollment_count} total samples, "
            f"updated {speaker.updated_at:.0f}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
