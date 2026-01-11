
import asyncio
import os
import logging
from urllib.parse import urlencode

try:
    from websockets.asyncio.client import connect
except ImportError:
    print("Could not import websockets.asyncio.client, trying legacy")
    from websockets.client import connect

logging.basicConfig(level=logging.DEBUG)

API_KEY = os.getenv("CARTESIA_API_KEY")
if not API_KEY:
    print("Missing CARTESIA_API_KEY")
    exit(1)

WS_URL = "wss://api.cartesia.ai/stt/websocket"

async def test_connect():
    params = {
        "api_key": API_KEY.strip(),  # Ensure no whitespace
        "cartesia_version": "2025-04-16",
        "model": "ink-whisper",
        "encoding": "pcm_s16le", # Testing if this causes timeout
        "sample_rate": "8000",
        "language": "pt",
        # "min_volume": "0.005000",
        "max_silence_duration_secs": "0.5"
    }
    url = f"{WS_URL}?{urlencode(params)}"
    print(f"Connecting to: {url.replace(API_KEY, 'REDACTED')}")
    
    try:
        async with connect(
            url,
            additional_headers={"X-Cartesia-Version": "2025-04-16"},
            open_timeout=10,
            close_timeout=5
        ) as ws:
            print("Successfully connected!")
            await ws.send("test")
            print("Sent test message")
            # Wait a bit
            await asyncio.sleep(1)
            print("Closing...")
    except Exception as e:
        print(f"Connection failed: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    asyncio.run(test_connect())
