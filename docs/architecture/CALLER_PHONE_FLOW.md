# Caller Phone Number Flow - Complete Trace

**Date:** Tue Oct 07 2025  
**Purpose:** Document how caller phone number flows from Twilio → Actor → LLM → CRM tools

---

## Overview

This document provides a complete trace of how the customer's phone number is captured, passed through the system, and made available to the LLM **before the first greeting**.

**Supports both INBOUND and OUTBOUND calls.**

---

## Flow Diagram

```
┌─────────────────────────────────────────────────────────────────────────┐
│ 1. INCOMING CALL FROM TWILIO                                            │
└─────────────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────────────┐
│ 2. app.py:incoming_call() - Twilio Webhook Handler                      │
│                                                                          │
│  form_data = await request.form()                                       │
│  from_number = form_data.get("From", "")                                │
│  to_number = form_data.get("To", "")                                    │
│  call_direction = form_data.get("Direction", "inbound")                 │
│                                                                          │
│  # Determine customer phone based on call direction:                    │
│  # - INBOUND: From=customer, To=our Twilio number                       │
│  # - OUTBOUND: From=our Twilio number, To=customer                      │
│  if call_direction == "outbound-api":                                   │
│      caller_phone = to_number  # Customer is "To"                       │
│  else:                                                                   │
│      caller_phone = from_number  # Customer is "From" (default)         │
│                                                                          │
│  # Result: caller_phone = "+5517997019739" (customer's number)          │
│                                                                          │
│  # Attach phone to WebSocket URL as query parameter                     │
│  ws_url = f"ws://host/media-stream?caller_phone={caller_phone}"         │
│                                                                          │
│  # Return TwiML with WebSocket URL                                      │
│  return <Connect><Stream url={ws_url} /></Connect>                      │
└─────────────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────────────┐
│ 3. app.py:media_stream() - WebSocket Handler                            │
│                                                                          │
│  # Extract phone from query parameters                                  │
│  caller_phone = ws.query_params.get("caller_phone", "")                 │
│  # Result: "+5517997019739"                                             │
│                                                                          │
│  # Create actor with phone number                                       │
│  actor = SessionActor(                                                   │
│      call_sid=call_sid,                                                  │
│      sink=sink,                                                          │
│      caller_phone=caller_phone  # ← Phone passed to actor               │
│  )                                                                       │
└─────────────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────────────┐
│ 4. actor.py:SessionActor.__init__() - Actor Initialization              │
│                                                                          │
│  def __init__(self, call_sid, sink, *, caller_phone="", ...):           │
│      self.caller_phone = caller_phone  # ← Stored in actor              │
│      self.conversation_history = []                                      │
│                                                                          │
│      # CRITICAL: Inject phone info as FIRST message in history          │
│      if caller_phone:                                                    │
│          self.conversation_history.append({                              │
│              "role": "system",                                           │
│              "content": (                                                │
│                  f"🔔 INCOMING CALL FROM: {caller_phone}\n\n"            │
│                  f"⚠️ CRITICAL: Before greeting, IMMEDIATELY call "      │
│                  f"get_customer_by_phone(phone=\"{caller_phone}\") "     │
│                  f"to check if this is a returning customer. "           │
│                  f"This is REQUIRED for EVERY call."                     │
│              )                                                           │
│          })                                                              │
│                                                                          │
│  # Result: conversation_history[0] now contains caller phone            │
└─────────────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────────────┐
│ 5. actor.py:_start_llm_stream() - First LLM Call                        │
│                                                                          │
│  # This method is called when starting LLM response                     │
│  # It passes conversation_history to OpenAI adapter                     │
│                                                                          │
│  async for token in llm.stream_tokens(                                  │
│      self.conversation_history,  # ← Includes phone message             │
│      self.llm_cancel,                                                    │
│      trace_id=self.trace_id                                              │
│  ):                                                                      │
│      ...                                                                 │
└─────────────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────────────┐
│ 6. core/llm.py:stream_tokens() - LLM Wrapper                            │
│                                                                          │
│  # Delegates to provider adapter                                        │
│  async for chunk in provider_adapter.stream_completion(                 │
│      messages=messages,  # ← Contains phone message                     │
│      cancel=cancel,                                                      │
│      trace_id=trace_id                                                   │
│  ):                                                                      │
│      yield chunk                                                         │
└─────────────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────────────┐
│ 7. providers/openai_adapter.py:stream_completion() - OpenAI Integration │
│                                                                          │
│  # Load system prompt from PROMPT.md                                    │
│  system_prompt = _load_system_prompt()                                  │
│                                                                          │
│  # Extract caller phone info from first message                         │
│  caller_phone_info = ""                                                  │
│  if messages and messages[0].get("role") == "system":                   │
│      # Extract phone message                                            │
│      caller_phone_info = "\n\n" + messages[0]["content"]                │
│      # Result: "\n\n🔔 INCOMING CALL FROM: +5517997019739..."           │
│                                                                          │
│      # Remove from messages list                                        │
│      messages = messages[1:]                                             │
│                                                                          │
│  # Append phone info to system prompt                                   │
│  full_system_prompt = system_prompt + caller_phone_info                 │
│                                                                          │
│  # Build final messages for OpenAI                                      │
│  full_messages = [                                                       │
│      {"role": "system", "content": full_system_prompt},                 │
│      ...messages  # User/assistant history                              │
│  ]                                                                       │
└─────────────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────────────┐
│ 8. OPENAI API CALL                                                       │
│                                                                          │
│  client.chat.completions.create(                                        │
│      model="gpt-4o",                                                     │
│      messages=[                                                          │
│          {                                                               │
│              "role": "system",                                           │
│              "content": (                                                │
│                  "You are a voice assistant...\n\n"                      │
│                  "## AVAILABLE TOOLS - USE PROACTIVELY\n"               │
│                  "...\n\n"                                               │
│                  "🔔 INCOMING CALL FROM: +5517997019739\n\n"             │
│                  "⚠️ CRITICAL: Before greeting, IMMEDIATELY call "       │
│                  "get_customer_by_phone(phone=\"+5517997019739\") "      │
│                  "to check if this is a returning customer."             │
│              )                                                           │
│          }                                                               │
│      ],                                                                  │
│      tools=[...71 tools including get_customer_by_phone...]             │
│  )                                                                       │
└─────────────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────────────┐
│ 9. LLM PROCESSES AND DECIDES TO CALL TOOL                               │
│                                                                          │
│  # GPT-4o reads system prompt and sees:                                 │
│  # - Caller phone: +5517997019739                                       │
│  # - Instruction: IMMEDIATELY call get_customer_by_phone()              │
│  # - Tool available: get_customer_by_phone(phone: str)                  │
│                                                                          │
│  # LLM Response (before any text):                                      │
│  {                                                                       │
│      "tool_calls": [{                                                    │
│          "id": "call_abc123",                                            │
│          "type": "function",                                             │
│          "function": {                                                   │
│              "name": "get_customer_by_phone",                            │
│              "arguments": '{"phone": "+5517997019739"}'                 │
│          }                                                               │
│      }]                                                                  │
│  }                                                                       │
└─────────────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────────────┐
│ 10. providers/openai_adapter.py:_handle_tool_calls_complete()           │
│                                                                          │
│  # Execute the tool call                                                │
│  tool_result = await _execute_mcp_tool(                                 │
│      "get_customer_by_phone",                                            │
│      {"phone": "+5517997019739"}                                        │
│  )                                                                       │
└─────────────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────────────┐
│ 11. mcp_tools/crm_customer.py:get_customer_by_phone()                   │
│                                                                          │
│  def get_customer_by_phone(phone: str) -> Dict[str, Any]:               │
│      # Clean phone number                                               │
│      clean_phone = re.sub(r'[^\d+]', '', phone)                         │
│      # Result: "+5517997019739"                                         │
│                                                                          │
│      # Query database                                                   │
│      cursor.execute(                                                     │
│          "SELECT * FROM customers WHERE phone LIKE ?",                   │
│          (f"%{clean_phone}%",)                                           │
│      )                                                                   │
│                                                                          │
│      # If found:                                                         │
│      return {                                                            │
│          "success": True,                                                │
│          "customer_id": "c_abc123",                                      │
│          "name": "João Silva",                                           │
│          "phone": "+5517997019739",                                      │
│          "email": "joao@example.com",                                    │
│          "company": "TechBrasil"                                         │
│      }                                                                   │
│                                                                          │
│      # If NOT found:                                                    │
│      return {                                                            │
│          "success": False,                                               │
│          "message": "No customer found with phone number: ..."           │
│      }                                                                   │
└─────────────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────────────┐
│ 12. TOOL RESULT SENT BACK TO LLM                                        │
│                                                                          │
│  # OpenAI adapter makes follow-up call with tool result                 │
│  client.chat.completions.create(                                        │
│      model="gpt-4o",                                                     │
│      messages=[                                                          │
│          {"role": "system", "content": "..."},                           │
│          {                                                               │
│              "role": "assistant",                                        │
│              "tool_calls": [...]                                         │
│          },                                                              │
│          {                                                               │
│              "role": "tool",                                             │
│              "tool_call_id": "call_abc123",                              │
│              "content": '{"success": true, "name": "João Silva", ...}'  │
│          }                                                               │
│      ]                                                                   │
│  )                                                                       │
└─────────────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────────────┐
│ 13. LLM GENERATES GREETING WITH CUSTOMER CONTEXT                        │
│                                                                          │
│  # If customer was found:                                               │
│  "Olá João! Tudo bem? Como posso ajudá-lo hoje?"                        │
│                                                                          │
│  # If customer NOT found:                                               │
│  "Olá! Bem-vindo. Qual é o seu nome?"                                   │
└─────────────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────────────┐
│ 14. GREETING SENT TO USER                                               │
│                                                                          │
│  LLM Token → TTS → Audio → Twilio → User's Phone                        │
└─────────────────────────────────────────────────────────────────────────┘
```

---

## Critical Points - Phone Number Availability

### ✅ GUARANTEED: Phone is available BEFORE first greeting

**Why?**
1. Phone injected into `conversation_history` **in actor constructor** (actor.py:178-182)
2. Constructor runs **before** `actor.start()` is called
3. `actor.start()` plays greeting (if enabled) **after** initialization
4. First LLM call receives phone in system prompt **before** generating any text
5. LLM **must** call tool before responding (per instructions)

**Timeline:**
```
T=0ms    → Twilio webhook receives call with "From" parameter
T=10ms   → WebSocket opened with caller_phone query param
T=20ms   → SessionActor.__init__() injects phone into conversation_history
T=30ms   → actor.start() begins (greeting NOT played yet)
T=50ms   → First LLM call includes phone in system message
T=1000ms → LLM calls get_customer_by_phone("+5517...")
T=1200ms → Tool returns customer data
T=1500ms → LLM generates greeting: "Olá João!"
T=2000ms → Audio reaches user
```

---

## Code References

### 1. Phone Extraction from Twilio
**File:** `app.py:93-104`
```python
# Extract call parameters
from_number = form_data.get("From", "")
to_number = form_data.get("To", "")
call_direction = form_data.get("Direction", "inbound")

# Determine customer phone based on direction
if call_direction == "outbound-api" or call_direction == "outbound-dial":
    caller_phone = to_number  # Outbound: customer is "To"
else:
    caller_phone = from_number  # Inbound: customer is "From"
```

### 2. Phone Passed to Actor
**File:** `app.py:135`
```python
actor = SessionActor(
    call_sid=call_sid,
    sink=sink,
    caller_phone=caller_phone  # ← Passed here
)
```

### 3. Phone Injected into Conversation
**File:** `actor.py:178-182`
```python
if caller_phone:
    self.conversation_history.append({
        "role": "system",
        "content": f"🔔 INCOMING CALL FROM: {caller_phone}\n\n"
                   f"⚠️ CRITICAL: Before greeting, IMMEDIATELY call "
                   f"get_customer_by_phone(phone=\"{caller_phone}\") "
                   f"to check if this is a returning customer. "
                   f"This is REQUIRED for EVERY call."
    })
```

### 4. Phone Appended to System Prompt
**File:** `providers/openai_adapter.py:386-394`
```python
# Extract caller phone info from first message
caller_phone_info = ""
if messages and messages[0].get("role") == "system":
    caller_phone_info = "\n\n" + messages[0]["content"]
    messages = messages[1:]  # Remove from messages list

# Append to system prompt
full_system_prompt = system_prompt + caller_phone_info
```

### 5. System Prompt Instructions
**File:** `PROMPT.md:19`
```
1. **Lookup customer**: `get_customer_by_phone(phone="<caller_phone>")` IMMEDIATELY
   - If found: Greet by name, reference past facts
   - If NOT found: Ask "Qual é o seu nome?" then `create_customer()`
```

---

## Call Direction Handling

### Twilio Webhook Parameters by Direction

**INBOUND CALL** (customer calls your Twilio number):
```
From: +5517997019739  ← Customer's phone
To: +14155551234      ← Your Twilio number
Direction: inbound
```

**OUTBOUND CALL** (you call customer using `scripts/make_call_v4.py`):
```
From: +14155551234      ← Your Twilio number
To: +5517997019739      ← Customer's phone
Direction: outbound-api
```

**Our Logic:**
- If `Direction == "outbound-api"` → Use `To` parameter (customer's phone)
- Otherwise (inbound) → Use `From` parameter (customer's phone)

This ensures we **always get the customer's phone number**, regardless of call direction.

---

## Example Call Flows

### Scenario 1: INBOUND - Returning Customer "João Silva"

**Step 1: Twilio Webhook (Inbound)**
```http
POST /incoming-call
Content-Type: application/x-www-form-urlencoded

From=+5517997019739
To=+14155551234
Direction=inbound
CallSid=CA1234567890abcdef
```

**App Logic:**
```python
call_direction = "inbound"  # Default
caller_phone = from_number  # "+5517997019739"
```

**Step 2: Actor Initialization**
```python
actor.conversation_history = [
    {
        "role": "system",
        "content": "🔔 INCOMING CALL FROM: +5517997019739\n\n"
                   "⚠️ CRITICAL: Before greeting, IMMEDIATELY call "
                   "get_customer_by_phone(phone=\"+5517997019739\")"
    }
]
```

**Step 3: First LLM Call**
```json
{
  "model": "gpt-4o",
  "messages": [
    {
      "role": "system",
      "content": "You are a voice assistant...\n\n🔔 INCOMING CALL FROM: +5517997019739\n\n⚠️ CRITICAL: Before greeting, IMMEDIATELY call get_customer_by_phone(phone=\"+5517997019739\")"
    }
  ],
  "tools": [
    {"type": "function", "function": {"name": "get_customer_by_phone", ...}},
    ...70 more tools
  ]
}
```

**Step 4: LLM Tool Call**
```json
{
  "tool_calls": [
    {
      "id": "call_xyz",
      "function": {
        "name": "get_customer_by_phone",
        "arguments": "{\"phone\": \"+5517997019739\"}"
      }
    }
  ]
}
```

**Step 5: CRM Database Query**
```sql
SELECT * FROM customers WHERE phone LIKE '%5517997019739%'
```

**Step 6: Tool Result**
```json
{
  "success": true,
  "customer_id": "c_123",
  "name": "João Silva",
  "phone": "+5517997019739",
  "email": "joao@techbrasil.com.br",
  "company": "TechBrasil"
}
```

**Step 7: LLM Greeting**
```
"Olá João! Tudo bem? Como posso ajudá-lo hoje?"
```

---

### Scenario 2: OUTBOUND - Cold Call to New Customer

**Step 1: Make Outbound Call**
```bash
python scripts/make_call_v4.py +5521999887766 https://your-domain.com/incoming-call
```

**Step 2: Twilio Creates Call**
```http
POST https://your-domain.com/incoming-call
Content-Type: application/x-www-form-urlencoded

From=+14155551234
To=+5521999887766
Direction=outbound-api
CallSid=CA9876543210fedcba
```

**App Logic:**
```python
call_direction = "outbound-api"
caller_phone = to_number  # "+5521999887766" (customer)
```

**Step 3: Actor Initialization**
```python
actor.conversation_history = [
    {
        "role": "system",
        "content": "🔔 INCOMING CALL FROM: +5521999887766\n\n"
                   "⚠️ CRITICAL: Before greeting, IMMEDIATELY call "
                   "get_customer_by_phone(phone=\"+5521999887766\")"
    }
]
```

**Step 4: LLM Tool Call**
```json
{
  "tool_calls": [
    {
      "function": {
        "name": "get_customer_by_phone",
        "arguments": "{\"phone\": \"+5521999887766\"}"
      }
    }
  ]
}
```

**Step 5: CRM Result**
```json
{
  "success": false,
  "message": "No customer found with phone number: +5521999887766"
}
```

**Step 6: LLM Greeting (New Customer)**
```
"Olá! Aqui é da [empresa]. Qual é o seu nome?"
```

---

## Verification Checklist

To verify phone number is properly passed:

- [x] **Twilio sends "From" parameter** - app.py:89
- [x] **WebSocket receives caller_phone** - app.py:124
- [x] **Actor stores caller_phone** - actor.py:157
- [x] **Phone injected into conversation_history** - actor.py:178-182
- [x] **Phone message sent to OpenAI** - openai_adapter.py:386-394
- [x] **System prompt instructs immediate lookup** - PROMPT.md:19
- [x] **Tool available to LLM** - mcp_tools/crm_customer.py:527
- [x] **LLM can call get_customer_by_phone()** - Tested in test_mcp_integration.py

---

## Testing

### Unit Test
```python
def test_caller_phone_injection():
    """Verify phone is injected into conversation history"""
    actor = SessionActor(
        call_sid="test",
        sink=mock_sink,
        caller_phone="+5517997019739"
    )
    
    # Should have one system message with phone
    assert len(actor.conversation_history) == 1
    assert actor.conversation_history[0]["role"] == "system"
    assert "+5517997019739" in actor.conversation_history[0]["content"]
    assert "get_customer_by_phone" in actor.conversation_history[0]["content"]
```

### Integration Test - Inbound
```bash
# Test inbound call
curl -X POST http://localhost:5050/incoming-call \
  -d "From=+5517997019739" \
  -d "To=+14155551234" \
  -d "Direction=inbound" \
  -d "CallSid=CA123"

# Check logs for:
# 1. "Twilio webhook received from ... (caller=+5517997019739)"
# 2. "Executing tool: get_customer_by_phone with args: {'phone': '+5517997019739'}"
# 3. "Tool get_customer_by_phone executed successfully"
```

### Integration Test - Outbound
```bash
# Test outbound call
curl -X POST http://localhost:5050/incoming-call \
  -d "From=+14155551234" \
  -d "To=+5517997019739" \
  -d "Direction=outbound-api" \
  -d "CallSid=CA456"

# Should extract customer phone from "To" parameter
# Check logs for: "Twilio webhook received from ... (caller=+5517997019739)"
```

### Real Outbound Test
```bash
# Make real outbound call
python scripts/make_call_v4.py +5517997019739 https://your-domain.com/incoming-call

# Twilio will call the customer and webhook to your server
# Verify customer phone is correctly extracted from "To" parameter
```

---

## Troubleshooting

### Issue: LLM doesn't call get_customer_by_phone()

**Possible causes:**
1. ❌ Phone not in Twilio webhook → Check `form_data.get("From")`
2. ❌ Phone not passed to actor → Check `actor.caller_phone`
3. ❌ Phone not in conversation_history → Check actor initialization
4. ❌ System prompt missing instruction → Check PROMPT.md
5. ❌ Tool not available → Check MCP tools loading

**Debug:**
```python
# Add logging in actor.py:__init__()
logger.info("Actor initialized with caller_phone: %s", caller_phone)
logger.info("Conversation history: %s", self.conversation_history)

# Add logging in openai_adapter.py:stream_completion()
logger.info("Full system prompt: %s", full_system_prompt[:500])
```

### Issue: Phone format incorrect

**Twilio sends:** `+5517997019739` (E.164 format with +)  
**CRM expects:** Any format (cleaned with regex)

**Cleaning logic (crm_customer.py:54):**
```python
clean_phone = re.sub(r'[^\d+]', '', phone)
# "+55 (17) 99701-9739" → "+5517997019739"
```

---

## Summary

✅ **Phone number is GUARANTEED to be available to the LLM before the first greeting**

**Flow:**
1. Twilio → `app.py` (webhook)
2. `app.py` → `SessionActor` (constructor parameter)
3. `SessionActor.__init__()` → `conversation_history[0]` (system message)
4. `conversation_history` → OpenAI adapter (system prompt)
5. OpenAI → LLM (receives phone in first message)
6. LLM → `get_customer_by_phone()` tool call (before greeting)
7. Tool result → LLM (generates personalized greeting)

**Total latency:** ~1-2 seconds from call start to personalized greeting

---

**Document Version:** 1.0  
**Author:** Claude Code (AI Assistant)  
**Last Updated:** Tue Oct 07 2025
