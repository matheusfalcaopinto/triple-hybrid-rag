#!/usr/bin/env python3
"""
Setup Ngrok for Local Development

This script automates the setup of ngrok for the Voice Agent.
It performs the following:
1. Ensures ngrok is running (starts it if pyngrok is installed, or finds existing).
2. Detects the public HTTPS URL of the tunnel.
3. Updates the .env file with:
    - APP_PUBLIC_DOMAIN (for Twilio WebSockets)
    - WHATSAPP_MEDIA_BASE_URL (for media storage)
    - COMMUNICATION_WEBHOOK_BASE (legacy/compatibility)
4. Updates the Twilio Phone Number's Voice URL to point to this tunnel.

Usage:
    python scripts/setup_ngrok.py
"""

from __future__ import annotations

import base64
import json
import os
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Optional

import requests
from dotenv import load_dotenv

# Path to .env file (relative to this script: ../.env)
ENV_PATH = Path(__file__).resolve().parent.parent / ".env"

NGROK_API = "http://127.0.0.1:4040/api/tunnels"
TWILIO_BASE = "https://api.twilio.com/2010-04-01"


def _load_env() -> None:
    load_dotenv(dotenv_path=str(ENV_PATH))


def _h1(msg: str) -> None:
    print("\n" + "=" * len(msg))
    print(msg)
    print("=" * len(msg))


def _info(msg: str) -> None:
    print(f"[INFO] {msg}")


def _warn(msg: str) -> None:
    print(f"[WARN] {msg}")


def _err(msg: str) -> None:
    print(f"[ERROR] {msg}")


def _update_env_file(entries: dict[str, str]) -> None:
    """Update keys in the .env file, preserving comments and structure."""
    if not ENV_PATH.exists():
        _warn(f"Expected .env at {ENV_PATH} but file not found. Skipping update.")
        return

    try:
        lines = ENV_PATH.read_text().splitlines()
    except OSError as exc:
        _warn(f"Unable to read .env ({exc}). Skipping update.")
        return

    updated_keys = set()
    new_lines: list[str] = []

    for line in lines:
        stripped = line.strip()
        # Keep comments and empty lines
        if not stripped or stripped.startswith("#"):
            new_lines.append(line)
            continue

        key, sep, _ = line.partition("=")
        if key in entries and sep:
            new_value = entries[key]
            new_lines.append(f"{key}={new_value}")
            updated_keys.add(key)
        else:
            new_lines.append(line)

    # Append new keys if they weren't in the file
    for key, value in entries.items():
        if key not in updated_keys:
            new_lines.append(f"{key}={value}")

    try:
        ENV_PATH.write_text("\n".join(new_lines) + "\n")
    except OSError as exc:
        _warn(f"Failed writing updated .env ({exc}).")
        return

    for key in entries:
        _info(f".env updated: {key}={entries[key]}")


def _ensure_ngrok_running(port: int) -> bool:
    """Check if ngrok is running; start it via pyngrok if possible."""
    try:
        from pyngrok import ngrok  # type: ignore
    except ImportError:
        _warn(f"pyngrok not installed. Please run `ngrok http {port}` in another terminal.")
        return False
    except Exception as e:
        _warn(f"Error importing pyngrok: {e}")
        return False

    # Check if we already have tunnels
    tunnels = ngrok.get_tunnels()
    if any(t.proto == "https" for t in tunnels):
        return True

    _info(f"Starting ngrok tunnel on port {port} via pyngrok...")
    try:
        # pyngrok 5+
        ngrok.connect(addr=port, proto="http")
    except TypeError:
        # Older pyngrok signature
        ngrok.connect(port, "http")
    return True


def _fetch_ngrok_https_url() -> Optional[str]:
    """Fetch the public HTTPS URL from the local ngrok API."""
    try:
        with urllib.request.urlopen(NGROK_API, timeout=2.0) as resp:
            data = json.load(resp)
    except urllib.error.URLError:
        # This usually means ngrok isn't running
        return None

    for tunnel in data.get("tunnels", []):
        if tunnel.get("proto") == "https":
            return tunnel.get("public_url")
    return None


def _twilio_auth_header(account_sid: str, auth_token: str) -> dict[str, str]:
    token = base64.b64encode(f"{account_sid}:{auth_token}".encode()).decode()
    return {"Authorization": f"Basic {token}"}


def _find_twilio_number_sid(account_sid: str, auth_token: str, phone_number: str) -> Optional[str]:
    """Find the SID for a given Twilio phone number."""
    headers = _twilio_auth_header(account_sid, auth_token)
    url = f"{TWILIO_BASE}/Accounts/{account_sid}/IncomingPhoneNumbers.json"
    params = {"PhoneNumber": phone_number}
    try:
        resp = requests.get(url, headers=headers, params=params, timeout=10)
    except requests.RequestException as exc:
        _err(f"Unable to query Twilio incoming numbers: {exc}")
        return None

    if resp.status_code >= 400:
        _err(f"Twilio API returned {resp.status_code}: {resp.text}")
        return None

    payload = resp.json()
    numbers = payload.get("incoming_phone_numbers", [])
    if not numbers:
        _err(f"No Twilio number matched {phone_number}. Check TWILIO_PHONE_NUMBER.")
        return None
    return numbers[0].get("sid")


def _update_twilio_webhook(
    account_sid: str,
    auth_token: str,
    number_sid: str,
    webhook_url: str,
) -> bool:
    """Update the voice webhook for the specified number."""
    headers = _twilio_auth_header(account_sid, auth_token)
    url = f"{TWILIO_BASE}/Accounts/{account_sid}/IncomingPhoneNumbers/{number_sid}.json"
    # We set both VoiceUrl (for inbound calls) and ensure method is POST
    data = {
        "VoiceUrl": webhook_url,
        "VoiceMethod": "POST",
    }
    try:
        resp = requests.post(url, headers=headers, data=data, timeout=10)
    except requests.RequestException as exc:
        _err(f"Failed updating Twilio webhook: {exc}")
        return False

    if resp.status_code >= 400:
        _err(f"Twilio update failed with {resp.status_code}: {resp.text}")
        return False
    return True


def main() -> None:
    _load_env()

    # Default to 8000 for Pipecat standalone (or read from PORT env)
    port_env = os.getenv("PORT", "8000")
    try:
        port = int(port_env)
    except ValueError:
        port = 8000
        
    account_sid = os.getenv("TWILIO_ACCOUNT_SID")
    auth_token = os.getenv("TWILIO_AUTH_TOKEN")
    phone_number = os.getenv("TWILIO_PHONE_NUMBER")

    _h1("Voice Agent Pipecat — Ngrok & Twilio Setup")
    _info(f"Targeting local port: {port}")

    # 1. Start/Find Ngrok
    started_ngrok = _ensure_ngrok_running(port)
    
    https_url: Optional[str] = None
    _info("Waiting for ngrok tunnel...")
    
    for _ in range(10):
        https_url = _fetch_ngrok_https_url()
        if https_url:
            break
        time.sleep(1.0)

    if not https_url:
        _err("Unable to find an active ngrok HTTPS tunnel.")
        _info("Please ensure ngrok is running: `ngrok http 8000`")
        sys.exit(1)

    # 2. Extract domain and build URLs
    # https_url is like "https://1234.ngrok.io"
    base_domain = https_url.replace("https://", "").replace("http://", "")
    webhook_url = f"{https_url}/incoming-call"
    
    _info(f"Tunnel Active: {https_url}")
    _info(f"Domain:        {base_domain}")
    _info(f"Webhook:       {webhook_url}")

    # 3. Update .env
    # We update APP_PUBLIC_DOMAIN so the app knows its own address
    env_updates = {
        "APP_PUBLIC_DOMAIN": base_domain,
        # Update media URL for WhatsApp (Evolution API requires public URL)
        "WHATSAPP_MEDIA_BASE_URL": f"{https_url}/media/whatsapp",
        # Legacy compatibility (optional)
        "COMMUNICATION_WEBHOOK_BASE": https_url,
    }
    _update_env_file(env_updates)

    # 4. Update Twilio
    if not all([account_sid, auth_token, phone_number]):
        _warn("Missing Twilio credentials. Skipping Webhook update.")
        print(">> Note: You must manually update your Twilio Number's Voice URL to:")
        print(f"   {webhook_url}")
    else:
        _info(f"Updating Twilio number {phone_number}...")
        number_sid = _find_twilio_number_sid(account_sid, auth_token, phone_number)
        if not number_sid:
            _warn("Could not find Twilio Number SID. Update manually.")
        elif _update_twilio_webhook(account_sid, auth_token, number_sid, webhook_url):
            _info("Twilio webhook updated successfully! ✅")
        else:
            _err("Twilio update failed.")

    # 5. Keep alive if we started it
    if started_ngrok:
        try:
            from pyngrok import ngrok  # type: ignore
        except ImportError:
            return
            
        _info("ngrok is running in this process.")
        _info("Press Ctrl+C to stop the tunnel.")
        try:
            process = ngrok.get_ngrok_process()
            process.proc.wait()
        except KeyboardInterrupt:
            _info("Stopping ngrok...")
            ngrok.kill()

if __name__ == "__main__":
    main()
