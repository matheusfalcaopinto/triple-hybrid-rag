"""
Evolution API Client for WhatsApp Integration.

Provides async HTTP client for Evolution API endpoints using httpx.
Supports text messaging, media sending, presence updates, and instance management.

Reference: https://doc.evolution-api.com/v2/
"""

from __future__ import annotations

import base64
import logging
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import httpx

from voice_agent.config import SETTINGS

logger = logging.getLogger("voice_agent_v4.evolution_client")

# Evolution API availability flag
EVOLUTION_AVAILABLE = True

# Default timeout for API requests
DEFAULT_TIMEOUT = 30.0


class EvolutionAPIError(Exception):
    """Exception raised for Evolution API errors."""

    def __init__(
        self,
        message: str,
        status_code: Optional[int] = None,
        response_data: Optional[Dict[str, Any]] = None,
    ):
        super().__init__(message)
        self.status_code = status_code
        self.response_data = response_data or {}


class EvolutionClient:
    """
    Async client for Evolution API WhatsApp messaging.
    
    Uses httpx.AsyncClient for non-blocking HTTP requests.
    Handles authentication, phone number formatting, and error handling.
    """

    def __init__(
        self,
        base_url: Optional[str] = None,
        api_key: Optional[str] = None,
        instance_name: Optional[str] = None,
        instance_token: Optional[str] = None,
    ):
        self.base_url = (base_url or SETTINGS.evolution_api_url).rstrip("/")
        self.api_key = api_key or SETTINGS.evolution_api_key
        self.instance_name = instance_name or SETTINGS.evolution_instance_name
        self.instance_token = instance_token or SETTINGS.evolution_instance_token
        self._client: Optional[httpx.AsyncClient] = None

    def _get_headers(self) -> Dict[str, str]:
        """Get authentication headers for API requests."""
        headers = {
            "Content-Type": "application/json",
            "apikey": self.api_key,
        }
        if self.instance_token:
            headers["Authorization"] = f"Bearer {self.instance_token}"
        return headers

    @staticmethod
    def format_phone_number(phone: str) -> str:
        """
        Format phone number for Evolution API.
        
        Args:
            phone: Phone number (e.g., '+5511999990001' or '5511999990001')
            
        Returns:
            Formatted number for WhatsApp (e.g., '5511999990001')
        """
        # Remove common prefixes
        cleaned = phone.replace("whatsapp:", "").replace("+", "").strip()
        # Remove any non-digit characters
        cleaned = "".join(c for c in cleaned if c.isdigit())
        return cleaned

    async def _get_client(self) -> httpx.AsyncClient:
        """Get or create async HTTP client."""
        if self._client is None or self._client.is_closed:
            self._client = httpx.AsyncClient(
                base_url=self.base_url,
                headers=self._get_headers(),
                timeout=SETTINGS.evolution_timeout,
            )
        return self._client

    async def close(self) -> None:
        """Close the HTTP client."""
        if self._client and not self._client.is_closed:
            await self._client.aclose()
            self._client = None

    async def _request(
        self,
        method: str,
        endpoint: str,
        data: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """
        Make an API request to Evolution API.
        
        Args:
            method: HTTP method (GET, POST, etc.)
            endpoint: API endpoint path
            data: Request payload (for POST/PUT)
            
        Returns:
            Response JSON data
            
        Raises:
            EvolutionAPIError: If the request fails
        """
        client = await self._get_client()
        url = f"{endpoint}"
        
        try:
            if method.upper() == "GET":
                response = await client.get(url)
            elif method.upper() == "POST":
                response = await client.post(url, json=data)
            elif method.upper() == "PUT":
                response = await client.put(url, json=data)
            elif method.upper() == "DELETE":
                response = await client.delete(url)
            else:
                raise ValueError(f"Unsupported HTTP method: {method}")

            if response.status_code >= 400:
                try:
                    error_data = response.json()
                except Exception:
                    error_data = {"raw": response.text}
                raise EvolutionAPIError(
                    f"Evolution API error: {response.status_code}",
                    status_code=response.status_code,
                    response_data=error_data,
                )

            if response.status_code == 204:
                return {"success": True}
            
            return response.json()

        except httpx.RequestError as e:
            raise EvolutionAPIError(f"Request failed: {str(e)}") from e

    # =========================================================================
    # Instance Management
    # =========================================================================

    async def get_instance_info(self) -> Dict[str, Any]:
        """Get information about the current instance."""
        return await self._request("GET", f"/instance/fetchInstances")

    async def get_connection_state(self) -> Dict[str, Any]:
        """Get the connection state of the instance."""
        return await self._request(
            "GET", f"/instance/connectionState/{self.instance_name}"
        )

    # =========================================================================
    # Text Messages
    # =========================================================================

    async def send_text(
        self,
        number: str,
        text: str,
        *,
        delay: int = 0,
        link_preview: bool = True,
    ) -> Dict[str, Any]:
        """
        Send a text message via WhatsApp.
        
        Args:
            number: Recipient phone number (with country code)
            text: Message text content
            delay: Optional delay in milliseconds before sending
            link_preview: Whether to generate link previews
            
        Returns:
            API response with message ID and status
        """
        payload = {
            "number": self.format_phone_number(number),
            "text": text,
            "delay": delay,
            "linkPreview": link_preview,
        }
        
        logger.info(
            "Sending text message via Evolution API: to=%s, text_preview=%s",
            number,
            text[:50] + "..." if len(text) > 50 else text,
        )
        
        return await self._request(
            "POST", f"/message/sendText/{self.instance_name}", payload
        )

    # =========================================================================
    # Media Messages
    # =========================================================================

    async def send_media(
        self,
        number: str,
        media_type: str,
        media_url: str,
        *,
        caption: Optional[str] = None,
        filename: Optional[str] = None,
        mimetype: Optional[str] = None,
        delay: int = 0,
    ) -> Dict[str, Any]:
        """
        Send a media message (image, video, document, audio).
        
        Args:
            number: Recipient phone number
            media_type: Type of media (image, video, document, audio)
            media_url: Public URL of the media file
            caption: Optional caption for the media
            filename: Optional filename for documents
            mimetype: Optional MIME type override
            delay: Optional delay in milliseconds
            
        Returns:
            API response with message ID
        """
        payload: Dict[str, Any] = {
            "number": self.format_phone_number(number),
            "mediatype": media_type.lower(),
            "media": media_url,
            "delay": delay,
        }
        
        if caption:
            payload["caption"] = caption
        if filename:
            payload["fileName"] = filename
        if mimetype:
            payload["mimetype"] = mimetype

        logger.info(
            "Sending %s via Evolution API: to=%s, url=%s",
            media_type,
            number,
            media_url[:80],
        )

        return await self._request(
            "POST", f"/message/sendMedia/{self.instance_name}", payload
        )

    async def send_audio_ptt(
        self,
        number: str,
        audio_base64: str,
        *,
        mimetype: str = "audio/mp4",
        delay: int = 0,
    ) -> Dict[str, Any]:
        """
        Send an audio message as Push-To-Talk (Voice Note).
        
        For PTT to display correctly in WhatsApp (blue microphone icon),
        audio must be in AAC format (audio/mp4 or audio/ogg with opus).
        
        Args:
            number: Recipient phone number
            audio_base64: Base64-encoded audio data
            mimetype: Audio MIME type (default: audio/mp4 for AAC)
            delay: Optional delay in milliseconds
            
        Returns:
            API response with message ID
        """
        # Ensure proper data URI format
        if not audio_base64.startswith("data:"):
            audio_base64 = f"data:{mimetype};base64,{audio_base64}"
        
        payload = {
            "number": self.format_phone_number(number),
            "audio": audio_base64,
            "delay": delay,
        }

        logger.info("Sending PTT audio via Evolution API: to=%s", number)

        return await self._request(
            "POST", f"/message/sendWhatsAppAudio/{self.instance_name}", payload
        )

    # =========================================================================
    # Presence & Status
    # =========================================================================

    async def set_presence(
        self,
        number: str,
        presence: str = "composing",
        *,
        delay: int = 1200,
    ) -> Dict[str, Any]:
        """
        Set presence status for a chat (typing indicator, recording, etc.).
        
        Args:
            number: Chat phone number
            presence: Status type:
                - "composing" (typing...)
                - "recording" (recording audio...)
                - "paused" (clear status)
            delay: Duration in milliseconds to show the status
            
        Returns:
            API response
        """
        payload = {
            "number": self.format_phone_number(number),
            "presence": presence,
            "delay": delay,
        }

        return await self._request(
            "POST", f"/chat/presence/{self.instance_name}", payload
        )

    # =========================================================================
    # Utility Methods
    # =========================================================================

    async def download_media(self, media_url: str) -> Tuple[bytes, str]:
        """
        Download media from Evolution API.
        
        Args:
            media_url: URL of the media to download
            
        Returns:
            Tuple of (media_bytes, content_type)
        """
        client = await self._get_client()
        response = await client.get(media_url)
        response.raise_for_status()
        content_type = response.headers.get("Content-Type", "application/octet-stream")
        return response.content, content_type


def get_evolution_client() -> EvolutionClient:
    """
    Get an Evolution API client instance.
    
    Returns:
        Configured EvolutionClient instance
        
    Raises:
        ValueError: If Evolution API is not configured
    """
    if not SETTINGS.evolution_api_url:
        raise ValueError(
            "EVOLUTION_API_URL not configured. "
            "Set your Evolution API URL in .env"
        )
    
    if not SETTINGS.evolution_api_key:
        raise ValueError(
            "EVOLUTION_API_KEY not configured. "
            "Set your Evolution API key in .env"
        )
    
    return EvolutionClient()


# Singleton instance for reuse
_client_instance: Optional[EvolutionClient] = None


async def get_shared_client() -> EvolutionClient:
    """Get a shared Evolution client instance."""
    global _client_instance
    if _client_instance is None:
        _client_instance = get_evolution_client()
    return _client_instance
