# Post-Call Processing - Technical Documentation

**Feature:** Automatic and Manual Call Summary & Recording  
**Version:** Pipecat Migration (v5)  
**Status:** ⚠️ Partially Implemented (Manual only)  
**Last Updated:** January 12, 2026

---

## Overview

Post-call processing refers to actions that happen **after** a voice call ends, such as:
- Saving call summaries to CRM
- Updating call transcripts
- Saving audio recordings
- Creating follow-up tasks
- Updating customer facts learned during conversation

### Current State: Manual vs Automatic

| Feature | Status | Trigger | Implementation |
|---------|--------|---------|----------------|
| **Audio Recording** | ✅ Implemented | Automatic | `bot.py` finally block |
| **Call Summary** | ⚠️ Manual only | Agent must call tool | `save_call_summary` tool |
| **Transcript Save** | ⚠️ Manual only | Agent must call tool | `update_call_transcript` tool |
| **Task Creation** | ⚠️ Manual only | Agent must call tool | `create_task` tool |
| **Customer Facts** | ⚠️ Manual only | Agent must call tool | `add_customer_fact` tool |

---

## Automatic Post-Call: Audio Recording

### Current Implementation

**Location:** `src/voice_agent/bot.py` (lines 410-425)

```python
finally:
    # Save recording on call end
    if audio_buffer and SETTINGS.recording_enabled:
        try:
            from .services.recording import get_recording_service
            recording_service = get_recording_service()
            recording_path = await recording_service.save_recording(
                audio_buffer,
                call_sid=call_sid,
                caller_phone=caller_phone,
            )
            if recording_path:
                logger.info("Recording saved: %s", recording_path)
        except Exception as e:
            logger.error("Failed to save recording: %s", e)
    
    logger.info("Bot pipeline completed for call_sid=%s", call_sid)
```

### How It Works

1. **During Call:** `AudioBufferProcessor` captures all audio (user + bot)
2. **Call Ends:** Pipeline enters `finally` block (always runs)
3. **Check Enabled:** Verifies `SETTINGS.recording_enabled=True`
4. **Save Audio:** Writes WAV file to disk
5. **Log Path:** Records file location in logs

### Configuration

```bash
# .env
RECORDING_ENABLED=true                    # Enable/disable recording
RECORDING_PATH=recordings                 # Directory to save recordings
RECORDING_FORMAT=wav                      # Audio format (wav only for now)
RECORDING_INCLUDE_BOT_AUDIO=true         # Include bot responses in recording
```

### Output

**Filename Format:** `{call_sid}_{timestamp}.wav`  
**Example:** `ca93aeaa-9d2b-4b80-b5a7_20260112_150117.wav`

**Location:** `/home/matheus/repos/voice-agent-v5/recordings/`

**Audio Properties:**
- Sample Rate: 8000 Hz (Twilio standard)
- Channels: 1 (mono) or 2 (stereo if bot audio included)
- Format: 16-bit PCM WAV

---

## Manual Post-Call: CRM Tools

These tools are **available** during the call but require the **agent to explicitly call them**. There's no automatic execution.

### 1️⃣ Save Call Summary (`save_call_summary`)

**Purpose:** Record call outcome and summary to CRM

**Tool Definition:**
```python
{
    "name": "save_call_summary",
    "description": "Save a summary of the current call to the CRM",
    "parameters": {
        "customer_id": "Customer identifier (required)",
        "summary": "Brief summary of the call (required)",
        "outcome": "Call outcome (interested, not_interested, callback, demo_scheduled, etc.)",
        "call_type": "Type of call (inbound_support, outbound_cold_call, outbound_followup)",
        "duration_seconds": "Call duration in seconds",
        "sentiment": "Customer sentiment (positive, neutral, negative)",
        "next_action": "What should happen next"
    }
}
```

**Database Impact:**
```sql
INSERT INTO calls (
    id,
    org_id,
    customer_id,
    call_date,
    duration_seconds,
    call_type,
    outcome,
    summary,
    sentiment,
    next_action
) VALUES (...);
```

**Example Agent Usage:**
```
User: "Thanks, that helps. Talk to you later!"
Agent: [Internally calls save_call_summary]
       {
         "customer_id": "cust_123",
         "summary": "Customer asked about Premium Plan pricing. Sent proposal via email.",
         "outcome": "interested",
         "sentiment": "positive",
         "next_action": "Follow up in 2 days if no response"
       }
Agent: "Great! I've saved our conversation. Have a nice day!"
```

**Current Issue:** ⚠️ Agent doesn't automatically call this - must be prompted or configured

---

### 2️⃣ Update Call Transcript (`update_call_transcript`)

**Purpose:** Save full conversation transcript

**Tool Definition:**
```python
{
    "name": "update_call_transcript",
    "description": "Update the full transcript for a call",
    "parameters": {
        "call_id": "Call identifier (from save_call_summary)",
        "transcript": "Full conversation transcript"
    }
}
```

**Database Impact:**
```sql
UPDATE calls 
SET transcript = '...' 
WHERE id = 'call_xyz';
```

**Current Issue:** ⚠️ Requires `call_id` from `save_call_summary`, which must be called first

---

### 3️⃣ Create Follow-Up Task (`create_task`)

**Purpose:** Schedule follow-up actions

**Tool Definition:**
```python
{
    "name": "create_task",
    "description": "Create a new task for a customer",
    "parameters": {
        "customer_id": "Customer identifier",
        "title": "Task title (required)",
        "description": "Task details",
        "due_date": "When task is due (YYYY-MM-DD)",
        "priority": "Task priority (high, medium, low)"
    }
}
```

**Example Agent Usage:**
```
User: "Can you send me the proposal and follow up next week?"
Agent: [Internally calls create_task]
       {
         "customer_id": "cust_123",
         "title": "Send Premium Plan proposal",
         "due_date": "2026-01-13",
         "priority": "high"
       }
       [Then calls another create_task]
       {
         "customer_id": "cust_123",
         "title": "Follow up on proposal",
         "due_date": "2026-01-19",
         "priority": "medium"
       }
Agent: "Done! I'll send the proposal today and check in with you next week."
```

---

### 4️⃣ Add Customer Facts (`add_customer_fact`)

**Purpose:** Record learnings about customer

**Tool Definition:**
```python
{
    "name": "add_customer_fact",
    "description": "Add a fact about a customer (preferences, notes, requirements)",
    "parameters": {
        "customer_id": "Customer identifier",
        "fact_type": "Category (preference, note, requirement, etc.)",
        "fact_value": "The actual information"
    }
}
```

**Example Agent Usage:**
```
User: "I prefer calls after 2pm, and I'm only interested in the enterprise tier"
Agent: [Internally calls add_customer_fact twice]
       1. {"customer_id": "cust_123", "fact_type": "preferred_contact_time", "fact_value": "after 2pm"}
       2. {"customer_id": "cust_123", "fact_type": "product_interest", "fact_value": "enterprise tier"}
Agent: "Got it! I've noted your preference for afternoon calls and interest in enterprise."
```

---

## The Problem: No Automatic Post-Call Workflow

### What's Missing

Currently, when a call ends:

1. ✅ Recording is saved automatically
2. ❌ Call summary is **NOT** saved unless agent calls tool
3. ❌ Transcript is **NOT** saved unless agent calls tool
4. ❌ Tasks are **NOT** created unless agent calls tool
5. ❌ Facts are **NOT** recorded unless agent calls tool

### Why This is a Problem

**Scenario 1: Agent Forgets**
```
User talks for 10 minutes about their needs
Agent provides helpful information
Call ends
❌ No summary saved
❌ No context for next call
❌ Lost conversation data
```

**Scenario 2: Connection Drops**
```
User is mid-sentence
Network issue causes disconnect
Call ends abruptly
❌ Agent never had chance to save summary
❌ All context lost except audio recording
```

**Scenario 3: User Hangs Up Quickly**
```
User: "Thanks, bye!" *click*
Agent: "Wait, let me save—"
❌ Call already disconnected
❌ Summary not saved
```

---

## Solution: Automatic Post-Call Hook

### Proposed Implementation

**Add to `bot.py` finally block:**

```python
finally:
    # 1. Save recording (already implemented)
    if audio_buffer and SETTINGS.recording_enabled:
        recording_path = await recording_service.save_recording(...)
    
    # 2. NEW: Automatic post-call processing
    if SETTINGS.auto_save_call_summary:
        await _save_call_summary_automatically(
            caller_phone=caller_phone,
            call_sid=call_sid,
            duration=time.monotonic() - call_start_time,
            transcript=await _extract_transcript_from_context(context_aggregator),
            customer_context=customer_context,
        )
    
    logger.info("Bot pipeline completed for call_sid=%s", call_sid)
```

### New Function: `_save_call_summary_automatically()`

**Location:** `src/voice_agent/bot.py` (new function)

```python
async def _save_call_summary_automatically(
    caller_phone: str,
    call_sid: str,
    duration: float,
    transcript: str,
    customer_context: Optional[CustomerContext],
) -> None:
    """
    Automatically save call summary after call ends.
    
    This runs in the finally block to ensure it always executes,
    even if call disconnects abruptly.
    """
    try:
        from voice_agent.tools import get_mcp_server
        
        server = get_mcp_server()
        
        # Get or create customer
        customer_id = None
        if customer_context and customer_context.customer_id:
            customer_id = customer_context.customer_id
        else:
            # Try to find/create customer by phone
            result = await server.call_tool_async(
                "get_customer_by_phone",
                {"phone": caller_phone}
            )
            if result.get("success") and result.get("result"):
                customer_id = result["result"].get("customer_id")
        
        if not customer_id:
            logger.warning("No customer ID for post-call save: %s", caller_phone)
            return
        
        # Generate AI summary from transcript
        summary = await _generate_summary_from_transcript(transcript)
        
        # Detect outcome and sentiment
        outcome, sentiment = await _analyze_conversation_outcome(transcript)
        
        # Save call summary
        result = await server.call_tool_async(
            "save_call_summary",
            {
                "customer_id": customer_id,
                "summary": summary,
                "outcome": outcome,
                "call_type": "inbound_support",  # or detect from context
                "duration_seconds": int(duration),
                "sentiment": sentiment,
            }
        )
        
        if result.get("success"):
            call_id = result.get("result", {}).get("call_id")
            logger.info("Auto-saved call summary: call_id=%s", call_id)
            
            # Update transcript if we have a call_id
            if call_id and transcript:
                await server.call_tool_async(
                    "update_call_transcript",
                    {"call_id": call_id, "transcript": transcript}
                )
        else:
            logger.error("Failed to auto-save call summary: %s", result.get("error"))
            
    except Exception as e:
        logger.exception("Error in automatic post-call save: %s", e)
        # Don't raise - we don't want post-call failures to crash the app
```

### Helper: Generate Summary from Transcript

**Using OpenAI to summarize:**

```python
async def _generate_summary_from_transcript(transcript: str) -> str:
    """Generate concise summary from full transcript using LLM."""
    if not transcript or len(transcript) < 50:
        return "Brief call - no significant content"
    
    try:
        from openai import AsyncOpenAI
        from voice_agent.config import SETTINGS
        
        client = AsyncOpenAI(
            base_url=SETTINGS.openai_base_url,
            api_key=SETTINGS.openai_api_key,
        )
        
        response = await client.chat.completions.create(
            model=SETTINGS.openai_model,
            messages=[
                {
                    "role": "system",
                    "content": "Summarize this phone conversation in 1-2 sentences. Focus on: customer needs, information provided, next steps."
                },
                {
                    "role": "user",
                    "content": transcript
                }
            ],
            max_tokens=100,
            temperature=0.3,
        )
        
        return response.choices[0].message.content.strip()
        
    except Exception as e:
        logger.error("Failed to generate summary: %s", e)
        return transcript[:200] + "..." if len(transcript) > 200 else transcript
```

### Helper: Analyze Outcome & Sentiment

```python
async def _analyze_conversation_outcome(transcript: str) -> tuple[str, str]:
    """
    Analyze conversation to determine outcome and sentiment.
    
    Returns:
        (outcome, sentiment) tuple
        outcome: interested, not_interested, callback, demo_scheduled, issue_resolved, etc.
        sentiment: positive, neutral, negative
    """
    # Simple keyword-based detection (can be improved with LLM)
    transcript_lower = transcript.lower()
    
    # Detect outcome
    if any(word in transcript_lower for word in ["interested", "sounds good", "sign me up", "let's do it"]):
        outcome = "interested"
    elif any(word in transcript_lower for word in ["not interested", "no thank you", "not now"]):
        outcome = "not_interested"
    elif any(word in transcript_lower for word in ["call back", "follow up", "next week"]):
        outcome = "callback"
    elif any(word in transcript_lower for word in ["schedule", "book", "appointment"]):
        outcome = "demo_scheduled"
    elif any(word in transcript_lower for word in ["solved", "fixed", "resolved", "helped"]):
        outcome = "issue_resolved"
    else:
        outcome = "completed"
    
    # Detect sentiment
    positive_words = ["great", "excellent", "perfect", "thank you", "helpful", "appreciate"]
    negative_words = ["frustrated", "disappointed", "angry", "unhappy", "terrible"]
    
    positive_count = sum(1 for word in positive_words if word in transcript_lower)
    negative_count = sum(1 for word in negative_words if word in transcript_lower)
    
    if positive_count > negative_count + 1:
        sentiment = "positive"
    elif negative_count > positive_count:
        sentiment = "negative"
    else:
        sentiment = "neutral"
    
    return outcome, sentiment
```

---

## Configuration

### New Environment Variables

```bash
# .env

# Enable automatic call summary save
AUTO_SAVE_CALL_SUMMARY=true              # Default: false (not implemented yet)

# Minimum call duration to save (seconds)
MIN_CALL_DURATION_TO_SAVE=10             # Skip very short calls

# Use AI for summary generation
USE_AI_SUMMARY_GENERATION=true           # Use LLM vs simple truncation
```

---

## Current Workaround: Prompt Engineering

Until automatic post-call is implemented, you can configure the agent to save summaries:

### System Prompt Addition

**Add to `prompts/PROMPT_v3.md`:**

```markdown
## Post-Call Responsibilities

At the end of EVERY call, you MUST:

1. **Save Call Summary** - Use `save_call_summary` tool with:
   - Brief summary of conversation
   - Call outcome (interested, not_interested, callback, etc.)
   - Customer sentiment (positive, neutral, negative)
   - Recommended next action

2. **Create Tasks** - If follow-up needed, use `create_task` tool

3. **Record Facts** - If you learned preferences, use `add_customer_fact` tool

Example:
User: "Thanks, bye!"
You: [BEFORE responding, call tools]
     1. save_call_summary(customer_id="...", summary="...", outcome="...")
     2. create_task(if needed)
     3. add_customer_fact(if learned something)
You: "Thank you for calling! I've saved our conversation. Have a great day!"
```

### LLM Configuration

**Increase function calling reliability:**

```python
# In bot.py, when creating LLM service
llm = OpenAILLMService(
    ...
    run_on_config=True,  # Allow LLM to call tools proactively
)
```

---

## Testing Post-Call Features

### Test 1: Manual Call Summary (Current)

```bash
# Make a test call
python scripts/make_call.py +5517991385892

# During call, ask agent:
"Can you save a summary of our conversation?"

# Agent should call save_call_summary tool
# Check database:
python -c "
import asyncio, sys
sys.path.insert(0, 'src')
from voice_agent.tools import get_mcp_server

async def check():
    server = get_mcp_server()
    result = await server.call_tool_async('get_call_history', {
        'customer_id': '2e54fd8a-ec83-439c-b187-f6aa1ab1cdfd',
        'limit': 1
    })
    print(result)

asyncio.run(check())
"
```

### Test 2: Automatic Recording (Current)

```bash
# Make a test call
python scripts/make_call.py +5517991385892

# After call ends, check recordings directory:
ls -lh recordings/

# Should see: ca93aeaa-*.wav file
```

### Test 3: Automatic Summary (Not Implemented Yet)

```bash
# Would work once implementation is complete:
# 1. Make call
# 2. Hang up
# 3. Check database automatically has summary
```

---

## Migration Path

### Phase 1: ✅ Current State
- Audio recording works automatically
- Call summary tools available but manual
- Agent can save if prompted

### Phase 2: 🔄 Prompt Engineering (Interim Solution)
- Update system prompt to require post-call saves
- Configure LLM to call tools proactively
- Add post-call validation to tests

### Phase 3: 🎯 Automatic Implementation (Recommended)
- Add automatic post-call hook to `bot.py`
- Generate summaries using AI
- Save to database in finally block
- Always executes, even on disconnect

### Phase 4: 🚀 Advanced Features
- Automatic task creation from conversation
- Sentiment analysis and alerts
- Post-call webhooks to external systems
- Call analytics and reporting dashboard

---

## Comparison: Pre-Pipecat vs Pipecat

| Feature | Pre-Pipecat (v4) | Pipecat (v5) | Status |
|---------|-----------------|--------------|--------|
| **Recording** | Manual save | Automatic in finally block | ✅ Improved |
| **Call Summary** | ??? | Manual (tools available) | ⚠️ Needs implementation |
| **Transcript** | ??? | Manual (tools available) | ⚠️ Needs implementation |
| **Post-Call Hook** | ??? | Not implemented | ❌ Missing |

---

## Related Documentation

- `docs/artifacts/TOOLS_INVENTORY_REPORT.md` - Complete tool catalog
- `docs/artifacts/CUSTOMER_PREFETCH_FEATURE.md` - Pre-call context loading
- `src/voice_agent/tools/crm_calls.py` - Call management tools
- `src/voice_agent/services/recording.py` - Recording service
- `src/voice_agent/bot.py` - Pipeline and lifecycle management

---

## Summary & Recommendations

### ✅ What Works Today
1. **Audio recordings** - Automatically saved after every call
2. **Manual summaries** - Agent can save if prompted or asked
3. **CRM tools** - All necessary tools exist and work

### ⚠️ What Needs Improvement
1. **Automatic summary save** - Should happen in finally block
2. **Transcript extraction** - Need to build from context aggregator
3. **AI-powered summarization** - Use LLM for better summaries
4. **Outcome detection** - Analyze conversation automatically

### 🎯 Recommended Next Steps
1. **Short-term:** Update system prompt to require post-call saves
2. **Medium-term:** Implement automatic post-call hook in bot.py
3. **Long-term:** Add AI analysis, webhooks, and analytics

---

## Document Metadata

**Version:** 1.0  
**Author:** AI Assistant (GitHub Copilot)  
**Created:** January 12, 2026  
**Status:** Documentation complete, implementation needed

**Change History:**
- 2026-01-12 15:30: Initial documentation created
- 2026-01-12 15:30: Analyzed current state and proposed solutions
