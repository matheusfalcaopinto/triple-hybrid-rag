#!/usr/bin/env python3
"""
Make Outbound Call

Initiates an outbound call using Twilio.
Automatically uses the APP_PUBLIC_DOMAIN from .env to construct the webhook URL.

Usage:
    python scripts/make_call.py <phone_number>
    
Example:
    python scripts/make_call.py +5511999999999
"""

from __future__ import annotations

import os
import sys
import base64
from pathlib import Path
from typing import Tuple

import requests
from dotenv import load_dotenv

# Path to .env file (relative to this script: ../.env)
ENV_PATH = Path(__file__).resolve().parent.parent / ".env"
TWILIO_BASE = "https://api.twilio.com/2010-04-01"


def _load_env() -> None:
    load_dotenv(dotenv_path=str(ENV_PATH))


def _auth(account_sid: str, auth_token: str) -> Tuple[str, str]:
    return account_sid, auth_token


def _place_call(
    account_sid: str,
    auth_token: str,
    from_number: str,
    to_number: str,
    webhook: str,
) -> None:
    url = f"{TWILIO_BASE}/Accounts/{account_sid}/Calls.json"
    data = {
        "To": to_number,
        "From": from_number,
        "Url": webhook,
        "Method": "POST",
    }
    
    print(f"Calling {to_number}...")
    try:
        resp = requests.post(url, data=data, auth=_auth(account_sid, auth_token), timeout=10)
    except requests.RequestException as exc:
        print(f"[ERROR] Twilio call create failed: {exc}")
        sys.exit(1)

    if resp.status_code >= 400:
        print(f"[ERROR] Twilio responded {resp.status_code}: {resp.text}")
        sys.exit(1)

    payload = resp.json()
    print("\n📞 Outbound call initiated successfully!")
    print(f"  Call SID: {payload.get('sid')}")
    print(f"  Status:   {payload.get('status')}")
    print(f"  To:       {payload.get('to')}")
    print(f"  Webhook:  {webhook}")


def main_cli() -> None:
    _load_env()

    # 1. Credentials
    account_sid = os.getenv("TWILIO_ACCOUNT_SID")
    auth_token = os.getenv("TWILIO_AUTH_TOKEN")
    from_number = os.getenv("TWILIO_PHONE_NUMBER")

    if not all([account_sid, auth_token, from_number]):
        print("[ERROR] Missing Twilio credentials in .env file.")
        print("Please ensure TWILIO_ACCOUNT_SID, TWILIO_AUTH_TOKEN, and TWILIO_PHONE_NUMBER are set.")
        sys.exit(1)

    # 2. Parse Arguments
    if len(sys.argv) < 2:
        print("Usage: python scripts/make_call.py <phone_number>")
        # Optional: could support interactive input
        to_number = input("Enter phone number to call (E.164 format): ").strip()
    else:
        to_number = sys.argv[1].strip()

    if not to_number.startswith("+"):
        to_number = "+" + to_number

    # 3. Determine Webhook URL
    # We use the new configuration: APP_PUBLIC_DOMAIN
    public_domain = os.getenv("APP_PUBLIC_DOMAIN")
    
    if public_domain:
        # Default to https/wss for public domains
        webhook_url = f"https://{public_domain}/incoming-call"
        print(f"[INFO] Using configured public domain: {public_domain}")
    else:
        # Fallback: Ask user if not configured
        print("[WARN] APP_PUBLIC_DOMAIN not set in .env.")
        print("This usually means you haven't run 'scripts/setup_ngrok.py' or configured a cloud domain.")
        webhook_url = input("Enter full Webhook URL (e.g. https://xyz.ngrok.io/incoming-call): ").strip()

    if not webhook_url.startswith("http"):
        webhook_url = "https://" + webhook_url

    # 4. Execute
    print("\n🤖 Voice Agent Pipecat — Outbound Call")
    print("=" * 32)
    print(f"From:    {from_number}")
    print(f"Target:  {to_number}")
    print(f"Webhook: {webhook_url}")
    print("-" * 32)
    
    _place_call(account_sid, auth_token, from_number, to_number, webhook_url)


if __name__ == "__main__":
    main_cli()
