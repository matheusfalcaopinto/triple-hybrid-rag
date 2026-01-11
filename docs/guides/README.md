# User Guides

## 1. Quick Start & Usage

### 1.1 Prerequisites
- Python 3.12+
- `uv` (recommended) or `pip`
- Credentials for Cartesia, OpenAI, and Twilio in `.env`

### 1.2 Installation
```bash
# Using uv (recommended)
uv sync

# Or using pip
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

### 1.3 Running the Server
```bash
# Start the FastAPI server
uv run uvicorn voice_agent_v4.app:app --host 0.0.0.0 --port 5050
```

### 1.4 Exposing to Twilio
Use `ngrok` to expose your local server:
```bash
ngrok http 5050
```
Then configure your Twilio Phone Number's Voice URL to: `https://<your-ngrok-domain>/incoming-call`

---

## 2. CRM Integration

### 2.1 Setup
Initialize the SQLite database:
```bash
python scripts/init_crm_db.py
```

### 2.2 Features
- **Customer Recognition**: Automatically identifies callers by phone number.
- **Memory**: Remembers name, preferences, and past interactions.
- **Tools**: Can schedule appointments, send WhatsApp/Email confirmations.

### 2.3 Testing
1.  **First Call**: Agent asks for name -> Saves profile.
2.  **Second Call**: Agent greets by name -> Recalls history.
3.  **Commands**: "Schedule a meeting" -> Triggers calendar tools.

---

## 3. VAD Calibration (Voice Activity Detection)

### 3.1 Recommended Settings (Silero VAD)
For natural conversation, use these settings in `.env`:
```bash
VAD_IMPL=silero
VAD_SILERO_THRESHOLD=0.40           # Sensitivity (0.35=Fast, 0.5=Conservative)
VAD_SILERO_MIN_SILENCE_MS=180       # Silence before turn end
VAD_SILERO_WINDOW_MS=30             # Processing window
VAD_SILERO_DEVICE=cpu               # 'cuda' if GPU available
```

### 3.2 Troubleshooting
- **Cuts off too early?** Increase `VAD_SILERO_MIN_SILENCE_MS` to 250ms.
- **Takes too long to reply?** Decrease `VAD_SILERO_MIN_SILENCE_MS` to 120ms.
- **Background noise triggers?** Increase `VAD_SILERO_THRESHOLD` to 0.50.

---

## 4. Custom Prompts

### 4.1 Configuration
You can switch agent personas without code changes using `CUSTOM_PROMPT_FILE`:
```bash
export CUSTOM_PROMPT_FILE=prompts/clinic_secretary.md
make watch
```

### 4.2 Creating a Persona
1.  Copy `prompts/clinic_secretary.md` as a template.
2.  Define **Role**, **Workflow** (Step 1, 2, 3...), and **Tools**.
3.  Test with `scripts/manual_token_harness.py`.

---

## 5. Open Source LLMs (Tool Calling)

### 5.1 Supported Models
- **Qwen 2.5**: Excellent tool calling support.
- **Llama 3.1**: Good support, requires strict prompting.
- **DeepSeek R1**: Capable, but check template compatibility.

### 5.2 Best Practices
- **Streaming**: Buffer JSON tokens until valid before parsing.
- **System Prompt**: Explicitly list tools and enforce JSON format.
- **Parameters**: Use `temperature=0.4` for reliable structured output.
