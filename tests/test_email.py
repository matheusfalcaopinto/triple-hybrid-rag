from __future__ import annotations

import asyncio
import os
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from voice_agent.tools.email import (
    TOOL_DEFINITIONS,
    get_smtp_config,
    send_bulk_email,
    send_bulk_email_async,
    send_email,
    send_email_async,
    send_email_template,
    send_email_template_async,
    send_html_email,
)


class TestEmailConfiguration:
    """Tests for SMTP configuration"""
    
    @patch.dict(os.environ, {
        "SMTP_HOST": "smtp.gmail.com",
        "SMTP_PORT": "587",
        "SMTP_USER": "test@example.com",
        "SMTP_PASSWORD": "test_password",
        "SMTP_FROM_EMAIL": "noreply@example.com"
    })
    def test_get_smtp_config_success(self):
        """Test successful SMTP configuration retrieval"""
        config = get_smtp_config()
        
        assert config["host"] == "smtp.gmail.com"
        assert config["port"] == 587
        assert config["user"] == "test@example.com"
        assert config["password"] == "test_password"
        assert config["from_email"] == "noreply@example.com"
    
    @patch.dict(os.environ, {
        "SMTP_HOST": "smtp.gmail.com",
        "SMTP_USER": "test@example.com",
        "SMTP_PASSWORD": "test_password",
    }, clear=True)
    def test_get_smtp_config_default_from_email(self):
        """Test SMTP config uses user email as default from address"""
        config = get_smtp_config()
        
        assert config["from_email"] == "test@example.com"
    
    @patch.dict(os.environ, {}, clear=True)
    def test_get_smtp_config_missing_host(self):
        """Test error when SMTP_HOST is missing"""
        with pytest.raises(ValueError, match="SMTP_HOST not configured"):
            get_smtp_config()
    
    @patch.dict(os.environ, {"SMTP_HOST": "smtp.gmail.com"}, clear=True)
    def test_get_smtp_config_missing_user(self):
        """Test error when SMTP_USER is missing"""
        with pytest.raises(ValueError, match="SMTP_USER not configured"):
            get_smtp_config()


class TestSendEmail:
    """Tests for send_email function"""
    
    @patch("voice_agent.tools.email.smtplib.SMTP")
    @patch.dict(os.environ, {
        "SMTP_HOST": "smtp.gmail.com",
        "SMTP_PORT": "587",
        "SMTP_USER": "test@example.com",
        "SMTP_PASSWORD": "test_password",
        "SMTP_FROM_EMAIL": "noreply@example.com"
    })
    def test_send_email_success(self, mock_smtp):
        """Test successful email sending"""
        mock_server = MagicMock()
        mock_smtp.return_value.__enter__.return_value = mock_server
        
        result = send_email(
            to="customer@example.com",
            subject="Test Email",
            body="This is a test message"
        )
        
        assert result["success"] is True
        assert result["message"] == "Email sent successfully"
        assert result["to"] == "customer@example.com"
        assert result["subject"] == "Test Email"
        
        mock_server.starttls.assert_called_once()
        mock_server.login.assert_called_once_with("test@example.com", "test_password")
        mock_server.send_message.assert_called_once()
    
    @patch("voice_agent.tools.email.smtplib.SMTP")
    @patch.dict(os.environ, {
        "SMTP_HOST": "smtp.gmail.com",
        "SMTP_USER": "test@example.com",
        "SMTP_PASSWORD": "test_password",
    })
    def test_send_email_with_cc_bcc(self, mock_smtp):
        """Test email with CC and BCC recipients"""
        mock_server = MagicMock()
        mock_smtp.return_value.__enter__.return_value = mock_server
        
        result = send_email(
            to="customer@example.com",
            subject="Test",
            body="Test message",
            cc="cc@example.com",
            bcc="bcc@example.com"
        )
        
        assert result["success"] is True
        mock_server.send_message.assert_called_once()
    
    @patch("voice_agent.tools.email.smtplib.SMTP")
    @patch.dict(os.environ, {
        "SMTP_HOST": "smtp.gmail.com",
        "SMTP_USER": "test@example.com",
        "SMTP_PASSWORD": "test_password",
    })
    def test_send_email_html(self, mock_smtp):
        """Test HTML email sending"""
        mock_server = MagicMock()
        mock_smtp.return_value.__enter__.return_value = mock_server
        
        result = send_email(
            to="customer@example.com",
            subject="HTML Test",
            body="<h1>Hello</h1><p>This is HTML</p>",
            html=True
        )
        
        assert result["success"] is True
        mock_server.send_message.assert_called_once()
    
    @patch("voice_agent.tools.email.smtplib.SMTP")
    @patch.dict(os.environ, {
        "SMTP_HOST": "smtp.gmail.com",
        "SMTP_USER": "test@example.com",
        "SMTP_PASSWORD": "wrong_password",
    })
    def test_send_email_authentication_error(self, mock_smtp):
        """Test SMTP authentication error"""
        import smtplib
        mock_server = MagicMock()
        mock_server.login.side_effect = smtplib.SMTPAuthenticationError(
            535,
            b"Authentication failed",
        )
        mock_smtp.return_value.__enter__.return_value = mock_server
        
        result = send_email(
            to="customer@example.com",
            subject="Test",
            body="Test"
        )
        
        assert "error" in result
        assert "authentication failed" in result["error"].lower()
        assert result["setup_required"] is True
    
    @patch.dict(os.environ, {}, clear=True)
    def test_send_email_missing_config(self):
        """Test error when SMTP config is missing"""
        result = send_email(
            to="customer@example.com",
            subject="Test",
            body="Test"
        )
        
        assert "error" in result
        assert result["setup_required"] is True


class TestSendHtmlEmail:
    """Tests for send_html_email function"""
    
    @patch("voice_agent.tools.email.smtplib.SMTP")
    @patch.dict(os.environ, {
        "SMTP_HOST": "smtp.gmail.com",
        "SMTP_USER": "test@example.com",
        "SMTP_PASSWORD": "test_password",
    })
    def test_send_html_email_success(self, mock_smtp):
        """Test HTML email sending"""
        mock_server = MagicMock()
        mock_smtp.return_value.__enter__.return_value = mock_server
        
        html_body = "<html><body><h1>Hello</h1></body></html>"
        result = send_html_email(
            to="customer@example.com",
            subject="HTML Email",
            html_body=html_body
        )
        
        assert result["success"] is True
        mock_server.send_message.assert_called_once()


class TestSendBulkEmail:
    """Tests for send_bulk_email function"""
    
    @patch("voice_agent.tools.email.smtplib.SMTP")
    @patch.dict(os.environ, {
        "SMTP_HOST": "smtp.gmail.com",
        "SMTP_USER": "test@example.com",
        "SMTP_PASSWORD": "test_password",
    })
    def test_send_bulk_email_success(self, mock_smtp):
        """Test bulk email to multiple recipients"""
        mock_server = MagicMock()
        mock_smtp.return_value.__enter__.return_value = mock_server
        
        result = send_bulk_email(
            recipients="user1@example.com,user2@example.com,user3@example.com",
            subject="Newsletter",
            body="Monthly update"
        )
        
        assert result["success"] is True
        assert result["sent_count"] == 3
        assert result["total_count"] == 3
        assert "3/3" in result["message"]
        assert mock_server.send_message.call_count == 3


class TestEmailAsyncWrappers:
    """Tests for async wrappers delegating to synchronous implementations."""

    def test_send_email_async_wrapper(self, monkeypatch):
        async_mock = AsyncMock(return_value={"success": True})
        monkeypatch.setattr("voice_agent.tools.email.asyncio.to_thread", async_mock)
        result = asyncio.run(send_email_async("to@example.com", "Subject", "Body"))
        async_mock.assert_awaited_once()
        func = async_mock.call_args.args[0]
        assert func is send_email
        assert result["success"] is True

    def test_send_bulk_email_async_wrapper(self, monkeypatch):
        async_mock = AsyncMock(return_value={"success": True})
        monkeypatch.setattr("voice_agent.tools.email.asyncio.to_thread", async_mock)
        result = asyncio.run(
            send_bulk_email_async("user@example.com", "Subject", "Body")
        )
        async_mock.assert_awaited_once()
        func = async_mock.call_args.args[0]
        assert func is send_bulk_email
        assert result["success"] is True

    def test_send_email_template_async_wrapper(self, monkeypatch):
        async_mock = AsyncMock(return_value={"success": True})
        monkeypatch.setattr("voice_agent.tools.email.asyncio.to_thread", async_mock)
        result = asyncio.run(send_email_template_async("user@example.com", "welcome"))
        async_mock.assert_awaited_once()
        func = async_mock.call_args.args[0]
        assert func is send_email_template
        assert result["success"] is True


class TestToolDefinitions:

    """Tests for MCP tool definitions"""
    
    def test_tool_definitions_format(self):
        """Test that all tool definitions have correct format"""
        assert len(TOOL_DEFINITIONS) == 4
        
        for tool in TOOL_DEFINITIONS:
            assert "name" in tool
            assert "description" in tool
            assert "parameters" in tool
            assert "required" in tool
            assert "handler" in tool
            
            assert isinstance(tool["name"], str)
            assert isinstance(tool["description"], str)
            assert isinstance(tool["parameters"], dict)
            assert isinstance(tool["required"], list)
            assert callable(tool["handler"])
    
    def test_tool_names_unique(self):
        """Test that tool names are unique"""
        names = [tool["name"] for tool in TOOL_DEFINITIONS]
        assert len(names) == len(set(names))
    
    def test_tool_handlers_valid(self):
        """Test that all handlers are valid functions"""
        expected_handlers = {
            "send_email": send_email,
            "send_html_email": send_html_email,
            "send_bulk_email": send_bulk_email,
            "send_email_template": send_email_template,
        }
        
        for tool in TOOL_DEFINITIONS:
            assert tool["handler"] == expected_handlers[tool["name"]]


class TestEmailIntegration:
    """Integration tests for email functionality"""
    
    @patch("voice_agent.tools.email.smtplib.SMTP")
    @patch.dict(os.environ, {
        "SMTP_HOST": "smtp.gmail.com",
        "SMTP_USER": "test@example.com",
        "SMTP_PASSWORD": "test_password",
    })
    def test_multiple_email_types(self, mock_smtp):
        """Test sending multiple types of emails"""
        mock_server = MagicMock()
        mock_smtp.return_value.__enter__.return_value = mock_server
        
        # Plain text
        result1 = send_email(
            to="test@example.com",
            subject="Plain",
            body="Plain text"
        )
        assert result1["success"] is True
        
        # HTML
        result2 = send_html_email(
            to="test@example.com",
            subject="HTML",
            html_body="<h1>HTML</h1>"
        )
        assert result2["success"] is True
        
        # Template
        result3 = send_email_template(
            to="test@example.com",
            template="welcome",
            variables='{"name": "John", "company": "ACME"}'
        )
        assert result3["success"] is True
        
        assert mock_server.send_message.call_count == 3
