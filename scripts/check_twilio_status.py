from __future__ import annotations

import os
import sys

from dotenv import load_dotenv
from twilio.rest import Client


def main() -> None:
    if len(sys.argv) != 2:
        print("Usage: python scripts/check_twilio_status.py <E.164 number>", file=sys.stderr)
        raise SystemExit(1)

    target_number = sys.argv[1]
    if not target_number.startswith("+"):
        print("Phone number must be in E.164 format, e.g. +5517991385892", file=sys.stderr)
        raise SystemExit(1)

    load_dotenv(".env")

    account_sid = os.getenv("TWILIO_ACCOUNT_SID")
    auth_token = os.getenv("TWILIO_AUTH_TOKEN")
    if not account_sid or not auth_token:
        print("Missing TWILIO_ACCOUNT_SID or TWILIO_AUTH_TOKEN in environment.", file=sys.stderr)
        raise SystemExit(1)

    client = Client(account_sid, auth_token)
    messages = client.messages.list(to=f"whatsapp:{target_number}", limit=10)
    if not messages:
        print("No messages found for the given number.")
        return

    for msg in messages:
        print(
            f"{msg.date_sent or msg.date_created} | SID={msg.sid} | status={msg.status} "
            f"| error_code={msg.error_code} | error_message={msg.error_message}"
        )


if __name__ == "__main__":
    main()
