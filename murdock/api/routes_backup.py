"""Speaker backup & restore endpoints."""

from __future__ import annotations

import io
import json
import logging
import time
import zipfile
from typing import Optional

from fastapi import APIRouter, Depends, File, HTTPException, Query, UploadFile
from fastapi.responses import StreamingResponse
from pydantic import BaseModel

from murdock.core.audio import decode_wav, to_mono_16k_pcm
from murdock.core.context import AppContext

from .deps import get_context

_LOGGER = logging.getLogger("murdock.api.backup")

router = APIRouter(prefix="/api/backup", tags=["backup"])

# v2 adds settings.json (full settings table). v1 archives (no settings)
# restore fine — the settings step is simply skipped.
BACKUP_VERSION = 2


def _safe_dirname(name: str) -> str:
    """Sanitise a speaker name for use as a ZIP directory name."""
    return "".join(c if c.isalnum() or c in " _-" else "_" for c in name).strip()


def dump_settings(conn) -> dict:
    """Full settings-table dump: thresholds, MQTT, HA, matrix, calibration.

    Includes secrets (HA token, MQTT password, Mistral key) — the backup
    is a local file and a restore should bring the whole configuration
    back without re-entering credentials. Documented in the wiki.
    """
    rows = conn.execute("SELECT key, value FROM settings ORDER BY key").fetchall()
    return {row["key"]: row["value"] for row in rows}


def apply_settings(conn, data: dict) -> int:
    """Upsert a settings dump back into the table. Returns the row count."""
    from murdock.core.db import set_setting

    count = 0
    for key, value in data.items():
        if not isinstance(key, str) or not isinstance(value, str):
            continue
        set_setting(conn, key, value)
        count += 1
    return count


@router.get("")
async def export_backup(ctx: AppContext = Depends(get_context)):
    """Download a ZIP containing all speakers and their audio samples."""
    speakers = ctx.speakers.list_speakers()
    total_samples = 0
    buf = io.BytesIO()

    with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as zf:
        for spk in speakers:
            dirname = _safe_dirname(spk.name)
            samples = ctx.speakers.list_samples(spk.id)
            sample_entries = []

            for idx, sample in enumerate(samples, 1):
                audio = ctx.speakers.get_sample_audio(sample["id"])
                if not audio:
                    continue
                src = sample.get("source") or "unknown"
                fname = f"sample_{idx:03d}_{src}.wav"
                zf.writestr(f"speakers/{dirname}/{fname}", audio)
                sample_entries.append({
                    "filename": fname,
                    "source": src,
                    "duration_sec": sample["duration_sec"],
                    "original_filename": sample.get("filename"),
                    "satellite_id": sample.get("satellite_id"),
                })
                total_samples += 1

            speaker_meta = {
                "name": spk.name,
                "ha_user_id": spk.ha_user_id,
                "role": spk.role,
                "samples": sample_entries,
            }
            zf.writestr(
                f"speakers/{dirname}/speaker.json",
                json.dumps(speaker_meta, indent=2, ensure_ascii=False),
            )

        # Full settings table: thresholds, MQTT/HA config, media matrix,
        # calibration parameters — so a restore brings back the whole
        # deployment, not just the voices.
        zf.writestr(
            "settings.json",
            json.dumps(dump_settings(ctx.db), indent=2, ensure_ascii=False),
        )

        manifest = {
            "version": BACKUP_VERSION,
            "created_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "speaker_count": len(speakers),
            "sample_count": total_samples,
            "includes_settings": True,
        }
        zf.writestr("manifest.json", json.dumps(manifest, indent=2))

    buf.seek(0)
    ts = time.strftime("%Y%m%d_%H%M%S")
    return StreamingResponse(
        buf,
        media_type="application/zip",
        headers={
            "Content-Disposition": f'attachment; filename="murdock_backup_{ts}.zip"'
        },
    )


class RestoreResult(BaseModel):
    ok: bool
    speakers_created: int = 0
    speakers_skipped: int = 0
    samples_imported: int = 0
    settings_restored: int = 0
    errors: list[str] = []


@router.post("/restore", response_model=RestoreResult)
async def restore_backup(
    file: UploadFile = File(...),
    mode: str = Query("merge", pattern="^(merge|replace)$"),
    restore_settings: bool = Query(True),
    ctx: AppContext = Depends(get_context),
):
    """Upload a backup ZIP to restore speakers (and settings, v2+).

    Modes:
      - ``merge``: add new speakers, skip existing names
      - ``replace``: delete ALL current speakers first, then import

    Settings (thresholds, MQTT/HA config, media matrix, calibration) are
    restored from v2 archives unless ``restore_settings=false``. Runtime
    clients (HA, MQTT, calibration) are reconfigured afterwards so no
    restart is needed.
    """
    data = await file.read()
    if len(data) > 500 * 1024 * 1024:
        raise HTTPException(status_code=413, detail="Backup file too large (max 500 MB)")

    try:
        zf = zipfile.ZipFile(io.BytesIO(data), "r")
    except zipfile.BadZipFile:
        raise HTTPException(status_code=400, detail="Invalid ZIP file")

    # Read manifest
    try:
        manifest = json.loads(zf.read("manifest.json"))
    except (KeyError, json.JSONDecodeError):
        raise HTTPException(status_code=400, detail="Missing or invalid manifest.json")

    if manifest.get("version", 0) > BACKUP_VERSION:
        raise HTTPException(
            status_code=400,
            detail=f"Backup version {manifest.get('version')} is newer than supported ({BACKUP_VERSION})",
        )

    # Find speaker.json files
    speaker_jsons = [
        n for n in zf.namelist()
        if n.endswith("/speaker.json") and n.startswith("speakers/")
    ]
    has_settings = "settings.json" in zf.namelist()
    if not speaker_jsons and not has_settings:
        raise HTTPException(status_code=400, detail="No speakers found in backup")

    # Replace mode: wipe existing speakers
    if mode == "replace":
        for existing in ctx.speakers.list_speakers():
            ctx.speakers.delete_speaker(existing.id)
        _LOGGER.info("Replace mode: deleted all existing speakers")

    result = RestoreResult(ok=True)
    existing_names = {s.name.lower() for s in ctx.speakers.list_speakers()}

    for sp_path in speaker_jsons:
        try:
            sp_meta = json.loads(zf.read(sp_path))
        except json.JSONDecodeError:
            result.errors.append(f"Invalid JSON: {sp_path}")
            continue

        name = sp_meta.get("name", "").strip()
        if not name:
            result.errors.append(f"Empty speaker name in {sp_path}")
            continue

        if mode == "merge" and name.lower() in existing_names:
            result.speakers_skipped += 1
            _LOGGER.info("Skipping existing speaker: %s", name)
            continue

        sp_dir = sp_path.rsplit("/speaker.json", 1)[0]
        samples = sp_meta.get("samples", [])
        first_sample = True

        for sample_info in samples:
            fname = sample_info.get("filename")
            if not fname:
                continue
            wav_path = f"{sp_dir}/{fname}"
            try:
                wav_data = zf.read(wav_path)
            except KeyError:
                result.errors.append(f"Missing audio: {wav_path}")
                continue

            try:
                pcm, rate, width, channels = decode_wav(wav_data)
                pcm = to_mono_16k_pcm(pcm, rate, width, channels)
                duration = len(pcm) / 2 / 16000
                if duration < 1.0:
                    result.errors.append(f"Too short ({duration:.1f}s): {wav_path}")
                    continue

                ctx.speakers.enroll(
                    speaker_name=name,
                    pcm_bytes=pcm,
                    duration_sec=duration,
                    ha_user_id=sp_meta.get("ha_user_id") if first_sample else None,
                    role=sp_meta.get("role") if first_sample else None,
                    source=sample_info.get("source", "upload"),
                    filename=sample_info.get("original_filename"),
                    satellite_id=sample_info.get("satellite_id"),
                    skip_vad=True,
                )
                result.samples_imported += 1
                first_sample = False
            except Exception as exc:
                result.errors.append(f"Failed {wav_path}: {exc}")
                _LOGGER.warning("Restore sample failed: %s — %s", wav_path, exc)

        if not first_sample:
            result.speakers_created += 1
            existing_names.add(name.lower())

    # Settings: apply after speakers so a replace-mode wipe can't clobber
    # what we just restored, then reconfigure the live clients.
    if restore_settings and has_settings:
        try:
            settings_data = json.loads(zf.read("settings.json"))
            if isinstance(settings_data, dict):
                result.settings_restored = apply_settings(ctx.db, settings_data)
                ctx.speakers.threshold = ctx.get_verify_threshold()
                ctx.load_calibration()
                try:
                    await ctx.apply_ha_settings()
                    await ctx.apply_mqtt_settings()
                except Exception:
                    _LOGGER.exception("Client reconfigure after restore failed")
                _LOGGER.info(
                    "Restored %d settings from backup", result.settings_restored
                )
        except (json.JSONDecodeError, KeyError) as exc:
            result.errors.append(f"settings.json unreadable: {exc}")

    zf.close()
    _LOGGER.info(
        "Restore complete: %d speakers created, %d skipped, %d samples, %d errors",
        result.speakers_created, result.speakers_skipped,
        result.samples_imported, len(result.errors),
    )
    return result
