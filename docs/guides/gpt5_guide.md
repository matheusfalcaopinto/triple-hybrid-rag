# GPT-5 Complete Guide

This guide consolidates all GPT-5 integration documentation for Voice Agent v4.

---

## 1. Overview

Voice Agent v4 supports both GPT-4 and GPT-5 model families:

| Family | Models | Status |
|--------|--------|--------|
| **GPT-5** | `gpt-5-nano`, `gpt-5-mini`, `gpt-5`, `gpt-5-large` | Default |
| **GPT-4** | `gpt-4o-mini`, `gpt-4o`, `gpt-4-turbo` | Fallback |

### Check Model Availability

```bash
python scripts/check_openai_models.py
# Or: curl https://api.openai.com/v1/models -H "Authorization: Bearer $OPENAI_API_KEY" | grep gpt
```

---

## 2. API Differences

| Feature | GPT-4 | GPT-5 |
|---------|-------|-------|
| **Primary API** | Chat Completions | Responses API |
| **Max tokens param** | `max_tokens` | `max_completion_tokens` |
| **Temperature** | Configurable | Fixed at 1.0 |
| **Verbosity** | N/A | `low`, `medium`, `high` |
| **Reasoning Effort** | N/A | `low`, `medium`, `high` |

### GPT-5 Specific Parameters

- **Verbosity**: Controls response detail (`low` for voice agents)
- **Reasoning Effort**: Controls thinking depth (`low`/`medium` for real-time)

---

## 3. Configuration

### Recommended for Voice (GPT-5)

```bash
OPENAI_MODEL=gpt-5-nano
OPENAI_VERBOSITY=low           # Concise responses
OPENAI_REASONING_EFFORT=low    # Fast responses
OPENAI_USE_RESPONSES=true      # Responses API with fallback
```

### Fallback (GPT-4)

```bash
OPENAI_MODEL=gpt-4o-mini
OPENAI_USE_RESPONSES=false
```

---

## 4. Function Calling with GPT-5-Nano

### Key Concepts

1. Define tools as JSON schemas
2. Send tool definitions in API call
3. Model replies with `function_call` (name + args)
4. Execute function, send result back

### Minimal Example

```python
from openai import OpenAI

functions = [{
    "name": "get_customer_by_phone",
    "description": "Look up customer by phone",
    "parameters": {
        "type": "object",
        "properties": {
            "phone": {"type": "string", "description": "Phone number"}
        },
        "required": ["phone"]
    }
}]

resp = client.chat.completions.create(
    model="gpt-5-nano",
    messages=messages,
    functions=functions,
    function_call="auto",
    temperature=0,
)

if resp.choices[0].message.function_call:
    tool_name = resp.choices[0].message.function_call.name
    tool_args = json.loads(resp.choices[0].message.function_call.arguments)
    # Execute tool...
```

### Streaming Function Calls

```python
tool_name = None
tool_args_chunks = []

for chunk in stream:
    delta = chunk.choices[0].delta
    if delta.function_call and delta.function_call.name:
        tool_name = delta.function_call.name
    if delta.function_call and delta.function_call.arguments:
        tool_args_chunks.append(delta.function_call.arguments)

# After stream ends
if tool_name:
    args = json.loads("".join(tool_args_chunks))
```

### System Prompt Template

```
You are a voice assistant with access to tools.
RULES:
1. If you need real-time data, CALL A TOOL.
2. Do NOT invent data that a tool can provide.
3. Respond with function call ONLY when tool is needed.
4. Use exact JSON schema for arguments.
```

---

## 5. Troubleshooting

### "Responses API failed" (400 Bad Request)

**Symptom**: Logs show fallback to Chat Completions

**Cause**: GPT-5 not available for your account

**Solution**: Expected behavior - system falls back automatically. Or set `OPENAI_USE_RESPONSES=false`.

### Silent Responses (No Audio)

**Symptom**: `llm_done` but no TTS events

**Fixed in**: `openai_adapter.py:99-109` - fallback now uses standard parameters

### Model Not Found

**Solution**:

1. Verify model name spelling
2. Run `python scripts/check_openai_models.py`
3. Use `gpt-4o-mini` as fallback

### Higher Latency Than Expected

Check:

- `OPENAI_VERBOSITY=low`
- `OPENAI_REASONING_EFFORT=low`
- Enable `LOG_TIMING=true`

---

## 6. Migration from GPT-4

### Key Changes

| GPT-4 | GPT-5 |
|-------|-------|
| `temperature`, `top_p` | `verbosity`, `reasoning.effort` |
| `frequency_penalty` | Remove (use instructions) |
| `system` message | Developer preamble in `instructions` |
| JSON function schemas | Plaintext tools with grammar |

### Migration Steps

1. Inventory Chat Completions usage
2. Build Responses API adapter
3. Map prompts to verbosity/reasoning defaults
4. Test with feature flags
5. Monitor latency and quality metrics

---

## 7. Pricing Reference

| Model | Input $/1M | Output $/1M | Reasoning $/1M |
|-------|------------|-------------|----------------|
| GPT-5 | 1.25 | 5.00 | 15.00 |
| GPT-5-mini | 0.25 | 1.00 | 3.00 |
| GPT-5-nano | 0.05 | 0.20 | 0.60 |

Use `reasoning_effort=low` for latency-critical voice apps.

---

## 8. Code References

- **Model detection**: `providers/openai_adapter.py:191-202`
- **Parameter handling**: `providers/openai_adapter.py:97-113`
- **Responses API**: `providers/openai_adapter.py:204-253`
- **Configuration**: `config.py`
- **Model checker**: `scripts/check_openai_models.py`
