"""Tests for WhatsApp Business Calling API integration."""

from unittest.mock import MagicMock, patch, AsyncMock

import pytest


class TestCallAction:
    """Test CallAction enum."""

    def test_actions_exist(self):
        """Test call actions are defined."""
        from voice_agent.services.meta_calling import CallAction
        
        assert CallAction.ACCEPT.value == "accept"
        assert CallAction.REJECT.value == "reject"
        assert CallAction.TERMINATE.value == "terminate"


class TestCallInfo:
    """Test CallInfo dataclass."""

    def test_call_info_creation(self):
        """Test creating CallInfo."""
        from voice_agent.services.meta_calling import CallInfo
        
        info = CallInfo(
            call_id="call123",
            from_number="+1234567890",
            to_number="+0987654321",
            sdp_offer="v=0...",
        )
        
        assert info.call_id == "call123"
        assert info.from_number == "+1234567890"
        assert info.sdp_offer == "v=0..."


class TestCallActionResult:
    """Test CallActionResult dataclass."""

    def test_success_result(self):
        """Test successful result."""
        from voice_agent.services.meta_calling import CallActionResult
        
        result = CallActionResult(success=True, call_id="call123")
        assert result.success is True
        assert result.call_id == "call123"
        assert result.error is None

    def test_failure_result(self):
        """Test failed result."""
        from voice_agent.services.meta_calling import CallActionResult
        
        result = CallActionResult(success=False, error="API error")
        assert result.success is False
        assert result.error == "API error"


class TestMetaCallingClient:
    """Test MetaCallingClient."""

    def test_get_client_missing_token(self):
        """Test error when token not configured."""
        from voice_agent.services.meta_calling import MetaCallingClient
        
        async def run_test():
            with patch("voice_agent.services.meta_calling.SETTINGS") as mock_settings:
                mock_settings.meta_access_token = ""
                
                client = MetaCallingClient()
                
                with pytest.raises(ValueError, match="META_ACCESS_TOKEN not configured"):
                    await client._get_client()
        
        import asyncio
        asyncio.run(run_test())

    def test_verify_webhook_signature_valid(self):
        """Test valid webhook signature verification."""
        from voice_agent.services.meta_calling import MetaCallingClient
        import hashlib
        import hmac
        
        with patch("voice_agent.services.meta_calling.SETTINGS") as mock_settings:
            mock_settings.meta_app_secret = "test_secret"
            
            payload = b'{"test": "data"}'
            expected_sig = hmac.new(
                b"test_secret",
                payload,
                hashlib.sha256,
            ).hexdigest()
            
            result = MetaCallingClient.verify_webhook_signature(
                payload,
                f"sha256={expected_sig}",
            )
            
            assert result is True

    def test_verify_webhook_signature_invalid(self):
        """Test invalid webhook signature verification."""
        from voice_agent.services.meta_calling import MetaCallingClient
        
        with patch("voice_agent.services.meta_calling.SETTINGS") as mock_settings:
            mock_settings.meta_app_secret = "test_secret"
            
            result = MetaCallingClient.verify_webhook_signature(
                b'{"test": "data"}',
                "sha256=invalid_signature",
            )
            
            assert result is False

    def test_verify_webhook_signature_no_secret(self):
        """Test signature verification when no secret configured."""
        from voice_agent.services.meta_calling import MetaCallingClient
        
        with patch("voice_agent.services.meta_calling.SETTINGS") as mock_settings:
            mock_settings.meta_app_secret = ""
            
            # Should return True (skip verification)
            result = MetaCallingClient.verify_webhook_signature(
                b'{"test": "data"}',
                "sha256=any",
            )
            
            assert result is True


class TestWhatsAppWebRTCTransport:
    """Test WhatsApp WebRTC transport components."""

    def test_input_processor_init(self):
        """Test input processor initialization."""
        from voice_agent.transports.whatsapp_webrtc import WhatsAppWebRTCInputProcessor
        
        processor = WhatsAppWebRTCInputProcessor(sample_rate=16000)
        assert processor._sample_rate == 16000
        assert processor._running is False

    def test_output_processor_init(self):
        """Test output processor initialization."""
        from voice_agent.transports.whatsapp_webrtc import WhatsAppWebRTCOutputProcessor
        
        callback = MagicMock()
        processor = WhatsAppWebRTCOutputProcessor(
            sample_rate=16000,
            on_audio=callback,
        )
        assert processor._sample_rate == 16000
        assert processor._on_audio is callback

    def test_transport_init(self):
        """Test transport initialization."""
        from voice_agent.transports.whatsapp_webrtc import WhatsAppWebRTCTransport
        
        transport = WhatsAppWebRTCTransport(
            call_id="call123",
            sample_rate=16000,
        )
        
        assert transport._call_id == "call123"
        assert transport._sample_rate == 16000

    def test_transport_input_output_processors(self):
        """Test transport returns input/output processors."""
        from voice_agent.transports.whatsapp_webrtc import WhatsAppWebRTCTransport
        
        transport = WhatsAppWebRTCTransport(call_id="call123")
        
        input_proc = transport.input_processor()
        output_proc = transport.output_processor()
        
        assert input_proc is not None
        assert output_proc is not None


class TestTransportRegistry:
    """Test transport registry functions."""

    def test_register_and_get_transport(self):
        """Test registering and getting transport."""
        from voice_agent.transports.whatsapp_webrtc import (
            WhatsAppWebRTCTransport,
            register_transport,
            get_transport,
            unregister_transport,
        )
        
        transport = WhatsAppWebRTCTransport(call_id="call456")
        register_transport("call456", transport)
        
        retrieved = get_transport("call456")
        assert retrieved is transport
        
        unregister_transport("call456")
        assert get_transport("call456") is None


class TestConfigIntegration:
    """Test config has WhatsApp calling settings."""

    def test_config_has_whatsapp_calling_settings(self):
        """Test config has all WhatsApp calling settings."""
        from voice_agent.config import Settings
        
        with patch.dict("os.environ", {}, clear=False):
            settings = Settings()
            
            assert hasattr(settings, "whatsapp_calling_enabled")
            assert hasattr(settings, "whatsapp_phone_number_id")
            assert hasattr(settings, "whatsapp_business_account_id")
            assert hasattr(settings, "meta_access_token")
            assert hasattr(settings, "meta_app_secret")
            assert hasattr(settings, "whatsapp_calling_webhook_verify_token")
            assert hasattr(settings, "webrtc_stun_server")
            assert hasattr(settings, "webrtc_audio_codec")
            
            # Check defaults
            assert settings.whatsapp_calling_enabled is False
            assert settings.webrtc_audio_codec == "opus"
