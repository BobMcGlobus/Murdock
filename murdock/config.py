"""Runtime configuration loaded from environment variables."""

from __future__ import annotations

from pathlib import Path
from typing import Optional

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Murdock runtime settings.

    Values come from environment variables (see docker-compose.yml).
    """

    model_config = SettingsConfigDict(
        case_sensitive=False,
        extra="ignore",
    )

    # STT backend: "upstream" (Wyoming proxy) or "voxtral" (Mistral Cloud)
    stt_backend: str = Field(default="upstream")

    # Wyoming proxy
    listen_uri: str = Field(default="tcp://0.0.0.0:10350")
    upstream_uri: str = Field(default="tcp://localhost:10300")
    # Comma-separated language codes to force-advertise in the Wyoming Info
    # event (e.g. "de,en"). When set we skip querying the upstream entirely
    # — useful if the upstream is slow to start, advertises nothing, or if
    # you want Murdock to appear in HA's pipeline picker for a specific
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
    # When True, unknown voices get an empty transcript (hard gate). OFF
    # by default since 0.5: with no/few enrollments a hard gate makes the
    # satellite unusable — recognition results still publish either way,
    # and the early-reject below (opt-in) covers the TV/radio case.
    require_speaker_match: bool = Field(default=False)
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

    # Adaptive target-speaker extraction. When the utterance has multiple
    # speech regions (target + TV / second person), embed each region,
    # keep only the dominant enrolled speaker's regions, and verify the
    # clean concatenation. Fast-path skips it entirely for single-region
    # utterances, so the common case pays only one cheap VAD pass.
    enable_extraction: bool = Field(default=True)
    # Condition the audio sent to a cloud STT endpoint: trim the silence
    # before and after the utterance, and normalise the level. Upload
    # copy only — the speaker embedding is computed from the untouched
    # audio.
    enable_stt_prep: bool = Field(default=True)
    # How much higher the liveness bar sits while something is playing in
    # the satellite's room. Media tightening already existed, but it only
    # ever moved the *verify* threshold — which decides who spoke, not
    # whether anyone did. A playing TV is precisely when a marginal
    # liveness score is most likely to be the TV.
    liveness_media_boost: float = Field(default=0.15, ge=0.0, le=0.9)
    # Phrases that abort the current utterance, comma-separated. Empty
    # disables the feature — a default that silently swallows requests
    # after an upgrade would be worse than having to switch it on.
    cancel_words: str = Field(default="")
    # Give up on a session in which nobody said anything, after this many
    # seconds. Aimed at a wake word that fired on its own: the turn ends
    # without a transcript, so no request is invented from silence.
    # 0 disables it.
    silence_abort_sec: float = Field(default=3.0, ge=0.0, le=30.0)
    # …but only when essentially no speech was detected at all. Someone
    # drawing breath before speaking must not lose their turn.
    silence_abort_speech_floor: float = Field(default=0.2, ge=0.0, le=2.0)
    # Hard ceiling on a cloud transcription request, in seconds. The old
    # hard-coded 30 s meant a stalled provider held the assistant for
    # half a minute. On expiry the local Wyoming fallback takes over when
    # it is enabled; otherwise the utterance returns empty.
    stt_timeout_sec: float = Field(default=8.0, ge=1.0, le=120.0)
    # Language hint used when Home Assistant sends none. Empty means
    # "let the endpoint decide", which in practice means English for
    # most of them — a German sentence then comes back as confident
    # English nonsense rather than an error.
    stt_language: str = Field(default="de")
    # Speech-to-text entity inside Home Assistant, e.g.
    # stt.home_assistant_cloud. Reached over HA's own /api/stt endpoint,
    # which is the only way to use Cloud transcription: it is not a
    # Wyoming service and has no API of its own.
    ha_stt_entity: str = Field(default="")
    # When the primary engine returns an empty transcript, ask the
    # configured A/B shadow instead of handing Home Assistant nothing.
    # Costs latency only in that case — where the alternative is a
    # failed interaction anyway.
    shadow_rescues_empty: bool = Field(default=True)
    # Cosine-distance ceiling for claiming a region for a speaker. Should
    # be stricter (smaller) than verify_threshold — we only keep regions
    # we're confident about, then verify the concatenation at the normal
    # threshold.
    extraction_threshold: float = Field(default=0.25)
    # Regions shorter than this (seconds) are not scored — they fall below
    # the embedder's minimum frame count and score unreliably.
    extraction_min_region_sec: float = Field(default=0.6)

    # Confidence calibration (Platt scaling). When enabled and fitted, the
    # confidence reported to HA/MQTT is a calibrated P(same speaker)
    # instead of the raw 1 - cosine_distance. Gating still uses the
    # distance threshold; calibration only shapes the reported confidence
    # (and is the groundwork for adaptive thresholds / context fusion).
    enable_calibration: bool = Field(default=True)

    # Per-satellite voice sub-profiles: build an extra centroid per
    # (speaker, satellite) from same-mic samples and match against the
    # better of global/sub. Removes systematic microphone bias (e.g. a
    # satellite with fewer mics / no beamforming).
    enable_satellite_profiles: bool = Field(default=True)

    # Adaptive per-speaker thresholds derived from each speaker's own
    # genuine/impostor distributions at calibration time, bounded to a
    # small band around the global threshold. Recomputed on every
    # calibration refit.
    enable_adaptive_thresholds: bool = Field(default=True)

    # Early reject (opt-in): after ~1.5 s of clean voice, kill sessions
    # whose distance is catastrophically far from every profile —
    # distance ≥ effective_threshold + early_reject_margin. Murdock then
    # stops forwarding to the STT immediately and answers with an empty
    # transcript at stream end. Works independently of
    # require_speaker_match, so unknown *humans* can still pass while TV
    # and radio get dropped. When media is playing in the satellite's
    # room (MQTT context), the margin is halved — the reject gets bolder
    # exactly when background audio is the likely culprit.
    enable_early_reject: bool = Field(default=False)
    early_reject_margin: float = Field(default=0.25)

    # How the recognised speaker reaches the conversation agent:
    #   "none"       — Murdock returns the raw transcript untouched; the
    #                  speaker flows out-of-band via MQTT sensors / REST
    #                  and you read it in the HA system prompt. Keeps HA's
    #                  local intent matching ("turn on the light") intact.
    #                  Caveat: HA caches the system prompt within a
    #                  conversation, so the value can lag between turns.
    #   "transcript" — inject the recognition context straight into the
    #                  transcript Murdock returns (templates below), so it
    #                  arrives fresh on every utterance with no MQTT.
    #                  Breaks HA's local intent matching → for LLM agents.
    # A dropdown rather than a bool so a third delivery mode can slot in
    # later. Placeholders in the templates use {{ var }}: transcript
    # (alias tts), speaker, role, confidence (percent), distance,
    # nearest, satellite.
    speaker_context_mode: str = Field(default="none")
    transcript_template_known: str = Field(
        default=(
            "{{ transcript }}\n\n"
            "[Recognized speaker: {{ speaker }} (role: {{ role }}), "
            "confidence {{ confidence }}%.]"
        )
    )
    transcript_template_unknown: str = Field(
        default=(
            "{{ transcript }}\n\n"
            "[Speaker not recognized. Closest match: {{ nearest }} "
            "at {{ confidence }}%.]"
        )
    )

    # VAD
    vad_min_speech_ratio: float = Field(default=0.6)
    vad_speech_threshold: float = Field(default=0.5)

    #
    # The feature is plumbed end-to-end (DB column, handler hook, HA event
    # field, UI toggle) but ships without a model file. A compatible ONNX
    # runtime will actually classify anything — with no model present the
    # experiment with the flag can't accidentally break recognition.
    # Only classify utterances at least this long. Shorter clips produce
    # log. The handler also skips classification if the utterance was
    # rejected (unknown/blocked) so we never pay CPU for a TV sample.

    # Unknown logging
    unknown_logging: bool = Field(default=True)
    unknown_ttl_hours: int = Field(default=48)
    unknown_cleanup_interval_minutes: int = Field(default=30)

    # Home Assistant
    ha_url: Optional[str] = Field(default=None)
    ha_token: Optional[str] = Field(default=None)
    ha_input_text_entity: str = Field(default="input_text.current_speaker")
    ha_tv_entity: Optional[str] = Field(default=None)

    # MQTT — recommended integration path. In the HA addon these are
    # auto-wired from the Mosquitto service (services: mqtt:want), so the
    # user gets working discovery without entering anything. In compose
    # deployments set them via env. The env values seed the DB on first
    # boot; thereafter the Web UI is the source of truth.
    mqtt_enabled: bool = Field(default=False)
    mqtt_host: Optional[str] = Field(default=None)
    mqtt_port: int = Field(default=1883)
    mqtt_username: Optional[str] = Field(default=None)
    mqtt_password: Optional[str] = Field(default=None)
    mqtt_topic_prefix: str = Field(default="murdock")
    mqtt_discovery_prefix: str = Field(default="homeassistant")

    # Voxtral (Mistral Cloud STT)
    mistral_api_key: Optional[str] = Field(default=None)
    mistral_model: str = Field(default="voxtral-mini-latest")

    # OpenAI-compatible STT (stt_backend = "openai"). One backend covers
    # OpenAI itself, Groq, and self-hosted OpenAI-compatible servers:
    #   OpenAI: https://api.openai.com  + gpt-4o-transcribe
    #   Groq:   https://api.groq.com/openai + whisper-large-v3-turbo
    #   local:  http://<host>:8000 (speaches etc.), key may stay empty
    openai_base_url: str = Field(default="https://api.openai.com")
    openai_api_key: Optional[str] = Field(default=None)
    openai_model: str = Field(default="gpt-4o-transcribe")

    # Local fallback: when a *cloud* backend (voxtral/openai) fails —
    # internet down, provider outage — transcribe the buffered audio via
    # the Wyoming upstream instead of returning an empty transcript.
    # Opt-in; needs a reachable upstream_uri.
    stt_local_fallback: bool = Field(default=False)

    # A/B shadow engine: transcribe the same utterance with a second STT
    # in the background. The shadow result is never returned over
    # Wyoming — it only lands next to the primary transcript in the
    # recognition log, so two engines can be compared on real commands.
    #   none | upstream | voxtral | openai
    shadow_stt_backend: str = Field(default="none")
    shadow_upstream_uri: Optional[str] = Field(default=None)
    shadow_mistral_model: str = Field(default="voxtral-small-latest")
    # Own key for the Voxtral shadow (e.g. a second Mistral account /
    # billing separation). Empty = fall back to the primary mistral key.
    shadow_mistral_api_key: Optional[str] = Field(default=None)
    shadow_openai_base_url: str = Field(default="")
    shadow_openai_api_key: Optional[str] = Field(default=None)
    shadow_openai_model: str = Field(default="")

    # --- Transcript quality tiers (all opt-in) ---
    # Tier 1: vocabulary biasing. The terms are sent as the OpenAI
    # `prompt` field to whisper-family endpoints, fixing custom names
    # ("Fehenlichter", "Bed-Lightstrip") at the source. Ignored by
    # backends that don't document the field (Voxtral, OpenRouter).
    enable_stt_vocabulary: bool = Field(default=False)
    stt_vocabulary: str = Field(default="")
    # Tier 2: correction dictionary for known misrecognitions. One entry
    # per line: "wrong -> right" replaces (deterministic, keeps local
    # intents working), "wrong ~> right" annotates with "[oder: right]"
    # for an LLM agent. "#" starts a comment.
    enable_stt_dictionary: bool = Field(default=False)
    stt_dictionary: str = Field(default="")
    # Tier 3: dual transcript. Run the shadow engine *blocking* in
    # parallel with the primary and merge both transcripts, marking
    # disagreements inline as "primary [oder: shadow]". Costs the max of
    # both engines' latency and is LLM-only (breaks rigid intents).
    # Only effective when a shadow engine is configured.
    enable_dual_transcript: bool = Field(default=False)

    # Logging
    log_level: str = Field(default="info")

    def resolve_paths(self) -> "Settings":
        """Ensure dependent paths are populated and directories exist."""
        self.data_dir = Path(self.data_dir)
        self.model_dir = Path(self.model_dir)
        if self.db_path is None:
            self.db_path = self.data_dir / "murdock.db"
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
