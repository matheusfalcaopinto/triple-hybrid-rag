# Voice Agent Pipecat

Standalone Pipecat-based voice agent with 63 MCP tools for CRM, Calendar, and WhatsApp integration.

## Quick Start

```bash
# Create virtual environment
python -m venv venv
source venv/bin/activate

# Install dependencies
pip install -e .

# For WhatsApp Calling (WebRTC), also install:
pip install -e ".[webrtc]"

# Copy and configure environment
cp .env.example .env
# Edit .env with your API keys

# Run the chat interface
./scripts/run_chat.sh
```

## Features

### Core Pipeline

- **Pipecat Framework**: Modern async pipeline architecture
- **63 MCP Tools**: Full CRM, Calendar, and Communication toolset
- **Cartesia TTS/STT**: High-quality Portuguese voice
- **Silero VAD**: Voice activity detection
- **Twilio Integration**: WebSocket media streaming

### 📼 Call Recording

Record both sides of the conversation for quality assurance and training.

```bash
RECORDING_ENABLED=true
RECORDING_PATH=data/recordings
```

### ⏱️ Smart Turn Detection

Handle user inactivity with configurable warnings and graceful call termination.

```bash
USER_IDLE_WARNING_SECONDS=8.0   # Warn after silence
USER_IDLE_MAX_WARNINGS=2        # End after N warnings
```

### 📞 Voicemail Detection

Detect voicemail/answering machines on outbound calls using Twilio AMD.

```bash
VOICEMAIL_DETECTION_ENABLED=true
TWILIO_ACCOUNT_SID=your-sid
TWILIO_AUTH_TOKEN=your-token
```

**API Endpoint:**

```bash
curl -X POST https://your-domain/api/outbound-call \
  -H "Content-Type: application/json" \
  -d '{"to_number": "+1234567890"}'
```

### 💬 WhatsApp Business Calling

Receive voice calls via WhatsApp using Meta's Business Calling API.

```bash
WHATSAPP_CALLING_ENABLED=true
META_ACCESS_TOKEN=your-token
WHATSAPP_PHONE_NUMBER_ID=your-id
```

> **Note:** Requires `aiortc` package (`pip install -e ".[webrtc]"`) and Meta Business Calling API beta access.

## Directory Structure

```
src/voice_agent/
├── app.py              # FastAPI server with Twilio webhooks
├── bot.py              # Pipecat pipeline builder
├── config.py           # Settings from .env
├── context.py          # Customer prefetch & prompts
├── tools/              # 63 MCP tools
├── services/           # WhatsApp, Evolution, Outbound, Meta Calling
├── processors/         # Idle handler, custom processors
├── communication/      # Event storage
├── transports/         # Twilio WebSocket, WhatsApp WebRTC
└── utils/              # Database client
```

## API Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/incoming-call` | POST | Twilio incoming call webhook |
| `/media-stream` | WS | Twilio media stream WebSocket |
| `/api/outbound-call` | POST | Initiate outbound call |
| `/amd-callback` | POST | Twilio AMD result webhook |
| `/call-status` | POST | Call status updates |
| `/whatsapp/call/webhook` | GET/POST | Meta WhatsApp calling webhook |

## Tools Available

| Category | Tools |
|----------|-------|
| CRM | Customer CRUD, Facts, Knowledge, Tasks, Calls, Scripts |
| Calendar | Google Calendar scheduling, availability |
| Communication | WhatsApp, Email |
| Utility | Time, calculations |

## Environment Variables

See `.env.example` for all configuration options. Key sections:

| Section | Variables |
|---------|-----------|
| OpenAI LLM | 3 |
| Cartesia TTS/STT | 3 |
| Supabase | 2 |
| Recording | 5 |
| Turn Detection | 5 |
| Twilio/Voicemail | 7 |
| WhatsApp Calling | 8 |

## Scripts

| Script | Description |
|--------|-------------|
| `scripts/run_chat.sh` | Interactive terminal chat |
| `scripts/run_local.py` | Start local server |
| `scripts/pipecat_chat.py` | Chat without voice |

## Testing

```bash
# Run all feature tests
PYTHONPATH=src python -m pytest tests/ -v

# Run specific feature tests
pytest tests/test_recording.py       # 8 tests
pytest tests/test_turn_detection.py  # 9 tests
pytest tests/test_voicemail.py       # 13 tests
pytest tests/test_whatsapp_calling.py # 14 tests
```

## Database

The agent uses Supabase for CRM data. See `database/schema.sql` for the schema.

```bash
cd database
docker-compose up -d
```

## License

Proprietary
