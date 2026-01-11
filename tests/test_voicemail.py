"""Tests for voicemail detection and outbound calling."""

from unittest.mock import MagicMock, patch

import pytest


class TestAnsweredByEnum:
    """Test AnsweredBy enum."""

    def test_from_twilio_human(self):
        """Test human detection."""
        from voice_agent.services.outbound import AnsweredBy
        
        result = AnsweredBy.from_twilio("human")
        assert result == AnsweredBy.HUMAN
        assert not result.is_voicemail

    def test_from_twilio_machine_start(self):
        """Test machine_start detection."""
        from voice_agent.services.outbound import AnsweredBy
        
        result = AnsweredBy.from_twilio("machine_start")
        assert result == AnsweredBy.MACHINE_START
        assert result.is_voicemail

    def test_from_twilio_machine_end_beep(self):
        """Test machine_end_beep detection."""
        from voice_agent.services.outbound import AnsweredBy
        
        result = AnsweredBy.from_twilio("machine_end_beep")
        assert result == AnsweredBy.MACHINE_END_BEEP
        assert result.is_voicemail

    def test_from_twilio_fax(self):
        """Test fax detection."""
        from voice_agent.services.outbound import AnsweredBy
        
        result = AnsweredBy.from_twilio("fax")
        assert result == AnsweredBy.FAX
        assert not result.is_voicemail

    def test_from_twilio_unknown_value(self):
        """Test unknown value falls back to UNKNOWN."""
        from voice_agent.services.outbound import AnsweredBy
        
        result = AnsweredBy.from_twilio("some_new_value")
        assert result == AnsweredBy.UNKNOWN

    def test_from_twilio_with_dashes(self):
        """Test conversion of dashed values."""
        from voice_agent.services.outbound import AnsweredBy
        
        result = AnsweredBy.from_twilio("machine-end-beep")
        assert result == AnsweredBy.MACHINE_END_BEEP


class TestOutboundCallResult:
    """Test OutboundCallResult dataclass."""

    def test_success_result(self):
        """Test successful result."""
        from voice_agent.services.outbound import OutboundCallResult
        
        result = OutboundCallResult(success=True, call_sid="CA123")
        assert result.success is True
        assert result.call_sid == "CA123"
        assert result.error is None

    def test_failure_result(self):
        """Test failed result."""
        from voice_agent.services.outbound import OutboundCallResult
        
        result = OutboundCallResult(success=False, error="Connection failed")
        assert result.success is False
        assert result.call_sid is None
        assert result.error == "Connection failed"


class TestOutboundCallService:
    """Test OutboundCallService."""

    def test_get_client_missing_credentials(self):
        """Test error when credentials not configured."""
        from voice_agent.services.outbound import OutboundCallService
        
        with patch("voice_agent.services.outbound.SETTINGS") as mock_settings:
            mock_settings.twilio_account_sid = ""
            mock_settings.twilio_auth_token = ""
            
            service = OutboundCallService()
            
            with pytest.raises(ValueError, match="credentials not configured"):
                service._get_client()

    def test_initiate_call_success(self):
        """Test successful call initiation."""
        from voice_agent.services.outbound import OutboundCallService
        
        with patch("voice_agent.services.outbound.SETTINGS") as mock_settings:
            mock_settings.twilio_account_sid = "AC123"
            mock_settings.twilio_auth_token = "auth_token"
            mock_settings.twilio_phone_number = "+1234567890"
            mock_settings.voicemail_detection_enabled = False
            
            service = OutboundCallService()
            
            # Mock the Twilio Client
            mock_client = MagicMock()
            mock_call = MagicMock()
            mock_call.sid = "CA123456"
            mock_client.calls.create.return_value = mock_call
            service._client = mock_client
            
            result = service.initiate_call(
                to_number="+9876543210",
                callback_base_url="https://example.com",
            )
            
            assert result.success is True
            assert result.call_sid == "CA123456"
            mock_client.calls.create.assert_called_once()

    def test_initiate_call_with_amd(self):
        """Test call initiation with AMD enabled."""
        from voice_agent.services.outbound import OutboundCallService
        
        with patch("voice_agent.services.outbound.SETTINGS") as mock_settings:
            mock_settings.twilio_account_sid = "AC123"
            mock_settings.twilio_auth_token = "auth_token"
            mock_settings.twilio_phone_number = "+1234567890"
            mock_settings.voicemail_detection_enabled = True
            mock_settings.voicemail_detection_timeout = 5
            mock_settings.voicemail_speech_threshold = 2500
            
            service = OutboundCallService()
            
            mock_client = MagicMock()
            mock_call = MagicMock()
            mock_call.sid = "CA789"
            mock_client.calls.create.return_value = mock_call
            service._client = mock_client
            
            result = service.initiate_call(
                to_number="+9876543210",
                callback_base_url="https://example.com",
            )
            
            assert result.success is True
            
            # Verify AMD parameters were passed
            call_kwargs = mock_client.calls.create.call_args[1]
            assert call_kwargs["machine_detection"] == "DetectMessageEnd"
            assert call_kwargs["async_amd"] is True
            assert "async_amd_status_callback" in call_kwargs


class TestGetOutboundService:
    """Test singleton getter."""

    def test_singleton(self):
        """Test that get_outbound_service returns singleton."""
        with patch("voice_agent.services.outbound.SETTINGS"):
            import voice_agent.services.outbound as module
            module._outbound_service = None
            
            from voice_agent.services.outbound import get_outbound_service
            
            service1 = get_outbound_service()
            service2 = get_outbound_service()
            
            assert service1 is service2


class TestConfigIntegration:
    """Test config has voicemail settings."""

    def test_config_has_voicemail_settings(self):
        """Test config has all voicemail detection settings."""
        from voice_agent.config import Settings
        
        with patch.dict("os.environ", {}, clear=False):
            settings = Settings()
            
            assert hasattr(settings, "twilio_account_sid")
            assert hasattr(settings, "twilio_auth_token")
            assert hasattr(settings, "twilio_phone_number")
            assert hasattr(settings, "voicemail_detection_enabled")
            assert hasattr(settings, "voicemail_detection_timeout")
            assert hasattr(settings, "voicemail_speech_threshold")
            assert hasattr(settings, "voicemail_message_file")
            
            # Check defaults
            assert settings.voicemail_detection_enabled is False
            assert settings.voicemail_detection_timeout == 5
