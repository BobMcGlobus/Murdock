"""Runtime configuration loaded from environment variables."""

from __future__ import annotations

from pathlib import Path
from typing import Optional

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """VoiceID runtime settings.

    Values come from environment variables (see docker-compose.yml).
    """

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    # Wyoming proxy
    listen_uri: str = Field(default="tcp://0.0.0.0:10350")
    upstream_uri: str = Field(default="tcp://localhost:10300")
    # Comma-separated language codes to force-advertise in the Wyoming Info
    # event (e.g. "de,en"). When set we skip querying the upstream entirely
    # — useful if the upstream is slow to start, advertises nothing, or if
    # you want VoiceID to appear in HA's pipeline picker for a specific
    # language regardless of the upstream's claimed support.
    advertised_languages: Optional[str] = Field(default=None)
    # How often (seconds) to refresh the upstream language list when no
    # override is configured.
    info_cache_ttl_seconds: float = Field(default=60.0)

    # Web UI
    web_host: str = Field(default="0.0.0.0")
    web_port: int = Field(default=8099)

    # Paths
    data_dir: Path = Field(default=Path("/app/data"))
    model_dir: Path = Field(default=Path("/app/models"))
    db_path: Optional[Path] = Field(default=None)
    cam_model_path: Optional[Path] = Field(default=None)
    vad_model_path: Optional[Path] = Field(default=None)

    # Verification
    verify_threshold: float = Field(default=0.30)
    tv_threshold_boost: float = Field(default=0.05)
    max_verify_seconds: float = Field(default=5.0)
    min_verify_seconds: float = Field(default=1.0)
    # Seconds at the start of each utterance that are discarded before
    # running the speaker embedder. Voice satellites typically play a
    # "now listening" chime which would otherwise pollute the embedding
    # and push the distance toward "unknown". 1.0 s matches the default
    # ESPHome/Wyoming-satellite notification tone. Audio forwarded to
    # the STT upstream is NOT trimmed — the chime is harmless for ASR.
    skip_leading_seconds: float = Field(default=1.0)
    require_speaker_match: bool = Field(default=True)
    # Minimum liveness score (0–1) for the audio to be considered a real
    # voice. Below this the session is classified as TV / background noise
    # and blocked. Set to 0 to disable this gate entirely.
    min_liveness_score: float = Field(default=0.35)
    # Automatically add a fresh embedding to the speaker's profile on
    # every high-confidence match. This lets the speaker model age with
    # the user's voice over time (cold → warm, morning → evening, …).
    auto_enroll: bool = Field(default=True)
    # Only auto-enroll when the match distance is between these bounds.
    # Too close (< min) means we gain no new information; too far (> max)
    # risks contaminating the profile with borderline embeddings.
    auto_enroll_min_distance: float = Field(default=0.15)
    auto_enroll_max_samples: int = Field(default=20)
    # When True AND the speaker store is empty, skip verification entirely and
    # forward every utterance to the upstream STT. Default OFF: the proxy
    # stays strict so deleting all speakers doesn't silently open the gate.
    passthrough_when_no_speakers: bool = Field(default=False)

    # VAD
    vad_min_speech_ratio: float = Field(default=0.6)
    vad_speech_threshold: float = Field(default=0.5)

    # Unknown logging
    unknown_logging: bool = Field(default=True)
    unknown_ttl_hours: int = Field(default=48)
    unknown_cleanup_interval_minutes: int = Field(default=30)

    # Home Assistant
    ha_url: Optional[str] = Field(default=None)
    ha_token: Optional[str] = Field(default=None)
    ha_input_text_entity: str = Field(default="input_text.current_speaker")
    ha_tv_entity: Optional[str] = Field(default=None)

    # Logging
    log_level: str = Field(default="info")

    def resolve_paths(self) -> "Settings":
        """Ensure dependent paths are populated and directories exist."""
        self.data_dir = Path(self.data_dir)
        self.model_dir = Path(self.model_dir)
        if self.db_path is None:
            self.db_path = self.data_dir / "voiceid.db"
        if self.cam_model_path is None:
            self.cam_model_path = self.model_dir / "campplus.onnx"
        if self.vad_model_path is None:
            self.vad_model_path = self.model_dir / "silero_vad.onnx"
        self.data_dir.mkdir(parents=True, exist_ok=True)
        (self.data_dir / "unknown").mkdir(parents=True, exist_ok=True)
        return self


_settings: Optional[Settings] = None


def get_settings() -> Settings:
    """Return the singleton settings instance."""
    global _settings
    if _settings is None:
        _settings = Settings().resolve_paths()
    return _settings


def reset_settings() -> None:
    """Reset cached settings (useful for tests)."""
    global _settings
    _settings = None
