"""
Pipecat Voice Agent Configuration

Unified configuration for the standalone Pipecat voice agent.
All settings are loaded from environment variables via .env file.
"""

from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Optional

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict

logger = logging.getLogger("voice_agent.config")


class Settings(BaseSettings):
    """Settings for Pipecat-based voice agent."""
    
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
        populate_by_name=True,
        case_sensitive=False,
    )
    
    # ──────────────────────────────────────────────────────────────────────────
    # OpenAI LLM Configuration
    # ──────────────────────────────────────────────────────────────────────────
    openai_api_key: str = Field("", alias="OPENAI_API_KEY")
    openai_model: str = Field("gpt-5-nano", alias="OPENAI_MODEL")
    openai_base_url: str = Field("https://api.openai.com/v1", alias="OPENAI_BASE_URL")
    openai_max_output_tokens: int = Field(150, alias="OPENAI_MAX_OUTPUT_TOKENS")
    
    # ──────────────────────────────────────────────────────────────────────────
    # Cartesia TTS/STT Configuration
    # ──────────────────────────────────────────────────────────────────────────
    cartesia_api_key: str = Field("", alias="CARTESIA_API_KEY")
    cartesia_voice_id: str = Field("", alias="CARTESIA_VOICE_ID")
    cartesia_tts_model: str = Field("sonic-2", alias="CARTESIA_TTS_MODEL")
    cartesia_stt_model: str = Field("ink-whisper", alias="CARTESIA_STT_MODEL")
    cartesia_stt_language: str = Field("pt", alias="CARTESIA_STT_LANGUAGE")
    cartesia_sample_rate: int = Field(8000, alias="CARTESIA_SAMPLE_RATE")
    
    # ──────────────────────────────────────────────────────────────────────────
    # VAD (Silero) Configuration
    # ──────────────────────────────────────────────────────────────────────────
    vad_threshold: float = Field(0.4, alias="VAD_SILERO_THRESHOLD")
    vad_min_silence_ms: int = Field(120, alias="VAD_SILERO_MIN_SILENCE_MS")
    
    # ──────────────────────────────────────────────────────────────────────────
    # Turn Detection Configuration
    # ──────────────────────────────────────────────────────────────────────────
    user_idle_timeout_seconds: float = Field(15.0, alias="USER_IDLE_TIMEOUT_SECONDS")
    user_idle_warning_seconds: float = Field(8.0, alias="USER_IDLE_WARNING_SECONDS")
    user_idle_max_warnings: int = Field(2, alias="USER_IDLE_MAX_WARNINGS")
    mute_during_function_call: bool = Field(True, alias="MUTE_DURING_FUNCTION_CALL")
    mute_function_call_duration: float = Field(3.0, alias="MUTE_FUNCTION_CALL_DURATION")
    mute_until_first_bot_speech: bool = Field(False, alias="MUTE_UNTIL_FIRST_BOT_SPEECH")
    
    # ──────────────────────────────────────────────────────────────────────────
    # Twilio Configuration
    # ──────────────────────────────────────────────────────────────────────────
    twilio_ws_path: str = Field("/media-stream", alias="TWILIO_WS_PATH")
    twilio_sample_rate: int = Field(8000, alias="TWILIO_MEDIA_SAMPLE_RATE_HZ")
    twilio_account_sid: str = Field("", alias="TWILIO_ACCOUNT_SID")
    twilio_auth_token: str = Field("", alias="TWILIO_AUTH_TOKEN")
    twilio_phone_number: str = Field("", alias="TWILIO_PHONE_NUMBER")
    
    # ──────────────────────────────────────────────────────────────────────────
    # Voicemail Detection (Twilio AMD)
    # ──────────────────────────────────────────────────────────────────────────
    voicemail_detection_enabled: bool = Field(False, alias="VOICEMAIL_DETECTION_ENABLED")
    voicemail_detection_timeout: int = Field(5, alias="VOICEMAIL_DETECTION_TIMEOUT")
    voicemail_speech_threshold: int = Field(2500, alias="VOICEMAIL_SPEECH_THRESHOLD")
    voicemail_message_file: str = Field("VOICEMAIL_PROMPT.md", alias="VOICEMAIL_MESSAGE_FILE")
    
    # ──────────────────────────────────────────────────────────────────────────
    # Greeting Audio Configuration
    # ──────────────────────────────────────────────────────────────────────────
    greeting_audio_enabled: bool = Field(True, alias="GREETING_AUDIO_ENABLED")
    greeting_audio_clip: str = Field("greetings", alias="GREETING_AUDIO_CLIP")
    
    # ──────────────────────────────────────────────────────────────────────────
    # Silence Fill Configuration
    # ──────────────────────────────────────────────────────────────────────────
    silence_fill_enabled: bool = Field(False, alias="ENABLE_SILENCE_FILL")
    silence_fill_audio_clip: str = Field("random_silence_filling", alias="SILENCE_FILL_AUDIO_CLIP")
    
    # ──────────────────────────────────────────────────────────────────────────
    # Pipeline Settings
    # ──────────────────────────────────────────────────────────────────────────
    enable_metrics: bool = Field(True, alias="PIPECAT_ENABLE_METRICS")
    enable_usage_metrics: bool = Field(True, alias="PIPECAT_ENABLE_USAGE_METRICS")
    log_level: str = Field("INFO", alias="LOG_LEVEL")
    log_json_file: Optional[str] = Field(None, alias="LOG_JSON_FILE")
    log_color: str = Field("auto", alias="LOG_COLOR")
    
    # ──────────────────────────────────────────────────────────────────────────
    # Supabase Database
    # ──────────────────────────────────────────────────────────────────────────
    supabase_url: str = Field("", alias="SUPABASE_URL")
    supabase_service_role_key: str = Field("", alias="SERVICE_ROLE_KEY")
    
    # ──────────────────────────────────────────────────────────────────────────
    # System Prompt
    # ──────────────────────────────────────────────────────────────────────────
    custom_prompt_file: str = Field("PROMPT.md", alias="CUSTOM_PROMPT_FILE")
    
    @property
    def prompts_dir(self) -> Path:
        """Get the prompts directory (relative to package or project root)."""
        # Try package-relative first
        pkg_prompts = Path(__file__).parent.parent.parent.parent / "prompts"
        if pkg_prompts.exists():
            return pkg_prompts
        # Fallback to cwd
        return Path.cwd() / "prompts"
    
    # ──────────────────────────────────────────────────────────────────────────
    # Evolution API (WhatsApp)
    # ──────────────────────────────────────────────────────────────────────────
    evolution_api_url: str = Field("", alias="EVOLUTION_API_URL")
    evolution_api_key: str = Field("", alias="EVOLUTION_API_KEY")
    evolution_instance_name: str = Field("ai_agent_01", alias="EVOLUTION_INSTANCE_NAME")
    evolution_instance_token: str = Field("", alias="EVOLUTION_INSTANCE_TOKEN")
    evolution_whatsapp_from: str = Field("", alias="EVOLUTION_WHATSAPP_FROM")
    
    # ──────────────────────────────────────────────────────────────────────────
    # WhatsApp Settings
    # ──────────────────────────────────────────────────────────────────────────
    whatsapp_backend: str = Field("evolution", alias="WHATSAPP_BACKEND")
    whatsapp_media_base_url: str = Field("", alias="WHATSAPP_MEDIA_BASE_URL")
    whatsapp_media_root: str = Field("data/media/whatsapp", alias="WHATSAPP_MEDIA_ROOT")
    communication_webhook_base: str = Field("", alias="COMMUNICATION_WEBHOOK_BASE")
    twilio_media_sample_rate_hz: int = Field(8000, alias="TWILIO_MEDIA_SAMPLE_RATE_HZ")
    
    # ──────────────────────────────────────────────────────────────────────────
    # WhatsApp Calling (Meta Business API with WebRTC)
    # ──────────────────────────────────────────────────────────────────────────
    whatsapp_calling_enabled: bool = Field(False, alias="WHATSAPP_CALLING_ENABLED")
    whatsapp_phone_number_id: str = Field("", alias="WHATSAPP_PHONE_NUMBER_ID")
    whatsapp_business_account_id: str = Field("", alias="WHATSAPP_BUSINESS_ACCOUNT_ID")
    meta_access_token: str = Field("", alias="META_ACCESS_TOKEN")
    meta_app_secret: str = Field("", alias="META_APP_SECRET")
    whatsapp_calling_webhook_verify_token: str = Field("", alias="WHATSAPP_CALLING_WEBHOOK_VERIFY_TOKEN")
    webrtc_stun_server: str = Field("stun:stun.l.google.com:19302", alias="WEBRTC_STUN_SERVER")
    webrtc_audio_codec: str = Field("opus", alias="WEBRTC_AUDIO_CODEC")
    
    # ──────────────────────────────────────────────────────────────────────────
    # Google Calendar
    # ──────────────────────────────────────────────────────────────────────────
    google_service_account_path: str = Field(
        "data/google_calendar_service_account.json", 
        alias="GOOGLE_SERVICE_ACCOUNT_PATH"
    )
    google_calendar_default_org_id: str = Field("", alias="GOOGLE_CALENDAR_DEFAULT_ORG_ID")
    google_calendar_auth_mode: str = Field("service_account", alias="GOOGLE_CALENDAR_AUTH_MODE")
    google_calendar_delegation_email: str = Field("", alias="GOOGLE_CALENDAR_DELEGATION_EMAIL")
    google_calendar_enable_meet: bool = Field(False, alias="GOOGLE_CALENDAR_ENABLE_MEET")
    google_calendar_enable_attendee_invites: bool = Field(False, alias="GOOGLE_CALENDAR_ENABLE_ATTENDEE_INVITES")
    
    # ──────────────────────────────────────────────────────────────────────────
    # Timezone
    # ──────────────────────────────────────────────────────────────────────────
    timezone: str = Field("America/Sao_Paulo", alias="TIMEZONE")
    
    # ──────────────────────────────────────────────────────────────────────────
    # Timeouts
    # ──────────────────────────────────────────────────────────────────────────
    prefetch_timeout: float = Field(2.0, alias="PREFETCH_TIMEOUT")
    smtp_timeout: float = Field(15.0, alias="SMTP_TIMEOUT")
    evolution_timeout: float = Field(30.0, alias="EVOLUTION_TIMEOUT")
    
    # ──────────────────────────────────────────────────────────────────────────
    # Startup Optimization Flags
    # ──────────────────────────────────────────────────────────────────────────
    eager_tool_loading: bool = Field(True, alias="EAGER_TOOL_LOADING")
    direct_greeting_injection: bool = Field(True, alias="DIRECT_GREETING_INJECTION")
    parallel_context_prefetch: bool = Field(True, alias="PARALLEL_CONTEXT_PREFETCH")
    
    # ──────────────────────────────────────────────────────────────────────────
    # Infrastructure / Networking
    # ──────────────────────────────────────────────────────────────────────────
    app_public_domain: Optional[str] = Field(None, alias="APP_PUBLIC_DOMAIN")
    app_scheme_override: Optional[str] = Field(None, alias="APP_SCHEME_OVERRIDE")
    
    # ──────────────────────────────────────────────────────────────────────────
    # Recording Configuration
    # ──────────────────────────────────────────────────────────────────────────
    recording_enabled: bool = Field(False, alias="RECORDING_ENABLED")
    recording_path: str = Field("data/recordings", alias="RECORDING_PATH")
    recording_format: str = Field("wav", alias="RECORDING_FORMAT")
    recording_sample_rate: int = Field(16000, alias="RECORDING_SAMPLE_RATE")
    recording_include_bot_audio: bool = Field(True, alias="RECORDING_INCLUDE_BOT_AUDIO")
    
    # ──────────────────────────────────────────────────────────────────────────
    # Post-Call Processing Configuration
    # ──────────────────────────────────────────────────────────────────────────
    auto_save_call_summary: bool = Field(True, alias="AUTO_SAVE_CALL_SUMMARY")
    min_call_duration_to_save: float = Field(5.0, alias="MIN_CALL_DURATION_TO_SAVE")
    use_ai_summary_generation: bool = Field(True, alias="USE_AI_SUMMARY_GENERATION")
    
    # ──────────────────────────────────────────────────────────────────────────
    # Derived Properties
    # ──────────────────────────────────────────────────────────────────────────
    @property
    def system_prompt_path(self) -> Path:
        """Get the full path to the system prompt file."""
        return self.prompts_dir / self.custom_prompt_file
    
    def get_system_prompt(self) -> str:
        """Load and return the system prompt content."""
        prompt_path = self.system_prompt_path
        if prompt_path.exists():
            return prompt_path.read_text(encoding="utf-8")
        logger.warning("System prompt file not found: %s", prompt_path)
        return "You are a helpful AI assistant."


# Global settings instance
SETTINGS = Settings()
