# CRM Call Flow Examples

## Example 1: New Customer (First Call)

### What Happens:
```
📞 Call arrives from +5511987654321

Agent thinks:
1. "Let me check if I know this caller"
   → Calls get_customer_by_phone("+5511987654321")
   → Result: {"found": false}

2. "New caller, I should ask their name"
   
Agent says: "Hello! Thanks for calling. May I ask your name?"

Caller: "Hi, I'm Maria from Tech Startup Inc."

Agent thinks:
3. "Great, now I'll save this new customer"
   → Calls create_customer(
       phone="+5511987654321",
       name="Maria",
       company="Tech Startup Inc",
       status="new"
     )
   → Gets customer_id back

Agent says: "Nice to meet you Maria! How can I help you today?"

Caller: "We're struggling with manual data entry. Do you have a solution?"

Agent thinks:
4. "This is a pain point - I should remember this"
   → Calls add_customer_fact(
       customer_id=customer_id,
       fact_type="pain_point",
       content="Struggling with manual data entry"
     )

5. "Let me search for relevant information"
   → Calls search_knowledge_base("automated data entry solution")

Agent says: "Absolutely! Our Enterprise plan includes automated data import..."

Caller: "Sounds interesting. Can you send me more info?"

Agent thinks:
6. "I need to create a follow-up task"
   → Calls create_task(
       customer_id=customer_id,
       description="Send Enterprise plan info and pricing",
       task_type="send_email",
       priority="high",
       due_date="2025-10-07"
     )

Agent says: "Of course! I'll email you the details today. What's the best email?"

Caller: "maria@techstartup.com"

Agent thinks:
7. "Update customer with email"
   → Calls update_customer_info(
       customer_id=customer_id,
       email="maria@techstartup.com"
     )

Agent says: "Perfect, I'll send that over. Can I follow up with you Thursday?"

Caller: "Yes, Thursday afternoon works."

Agent thinks:
8. "Save this conversation"
   → Calls save_call_summary(
       customer_id=customer_id,
       call_type="inbound",
       outcome="interested",
       summary="New lead Maria from Tech Startup Inc. Main pain point: manual data entry. Interested in Enterprise plan. Will email info today and follow up Thursday.",
       duration_seconds=180
     )

Agent says: "Great! I'll call you Thursday afternoon. Have a great day Maria!"
```

---

## Example 2: Returning Customer (Second Call)

### What Happens:
```
📞 Call arrives from +5511987654321 (Maria calling back)

Agent thinks:
1. "Let me check who this is"
   → Calls get_customer_by_phone("+5511987654321")
   → Result: {"found": true, "name": "Maria", "company": "Tech Startup Inc"}

2. "It's Maria! Let me get context"
   → Calls get_call_history(customer_id)
   → Sees: Last call was about manual data entry, Enterprise plan
   
   → Calls get_customer_facts(customer_id)
   → Sees: Pain point = "manual data entry"
   
   → Calls get_pending_tasks(customer_id)
   → Sees: "Send Enterprise plan info" task (should be completed)

Agent says: "Hi Maria! Good to hear from you again. Did you get the Enterprise plan information I sent?"

Caller: "Yes! I did. Quick question - how much is it per month?"

Agent thinks:
3. "Price question - search knowledge base"
   → Calls search_knowledge_base("Enterprise pricing")
   → Gets: "R$ 499/month with unlimited users..."

Agent says: "The Enterprise plan is R$ 499 per month, which includes unlimited users, full API access, and 24/7 support..."

Caller: "That's a bit high for us right now."

Agent thinks:
4. "Price objection - get handler"
   → Calls get_objection_handler(
       call_type="outbound_cold_call",
       objection_key="muito_caro"
     )
   → Gets suggested response

Agent says: "I understand. Compared to hiring a full-time developer, it actually saves about R$5,000 per month. Plus we have a 20% discount if you pay annually. Would that work better?"

Caller: "Hmm, let me think about it. Can you call me next week?"

Agent thinks:
5. "Schedule callback task"
   → Calls create_task(
       customer_id=customer_id,
       description="Follow up on pricing - mentioned annual discount",
       task_type="follow_up",
       due_date="2025-10-13",
       priority="medium"
     )

6. "Save this call"
   → Calls save_call_summary(
       customer_id=customer_id,
       call_type="follow_up",
       outcome="callback",
       summary="Maria interested but concerned about price. Mentioned annual discount (20% off). Will follow up next week.",
       duration_seconds=120
     )

Agent says: "Of course! I'll call you next Monday. Have a great week!"
```

---

## Example 3: Your Specific Case

### When Agent Calls You:

```python
# The agent receives your phone number from Twilio
# Let's say it's +5511987654321

# 1. Agent IMMEDIATELY calls (automatic):
get_customer_by_phone("+5511987654321")
→ Returns: {"found": false} (first time)

# 2. Agent says something like:
"Hi! Thanks for taking my call. May I know your name?"

# 3. You say:
"Hi, I'm Matheus"

# 4. Agent calls (automatic):
create_customer(
    phone="+5511987654321",
    name="Matheus",
    status="contacted"
)
→ Saves you to database, returns customer_id

# 5. During conversation, if you mention preferences:
You: "I prefer to communicate via email"

# Agent calls (automatic):
add_customer_fact(
    customer_id=customer_id,
    fact_type="preference",
    content="Prefers email communication"
)

# 6. At end of call, agent calls (automatic):
save_call_summary(
    customer_id=customer_id,
    call_type="outbound_cold_call",
    outcome="interested",
    summary="First call with Matheus. Discussed product features. Prefers email.",
    duration_seconds=300
)
```

### Next Time Agent Calls You:

```python
# 1. Agent calls (automatic):
get_customer_by_phone("+5511987654321")
→ Returns: {"found": true, "name": "Matheus"}

get_call_history(customer_id)
→ Shows previous conversation

get_customer_facts(customer_id)
→ Shows: "Prefers email communication"

# 2. Agent says:
"Hi Matheus! Hope you're doing well. I'm following up on our conversation about..."
# (Uses your name + references past conversation naturally)
```

---

## What Gets Saved Automatically:

With the updated prompt, the agent will automatically:

✅ **Always lookup** customer at call start  
✅ **Ask your name** if not found  
✅ **Create profile** with your phone + name  
✅ **Store facts** when you mention preferences/pain points  
✅ **Save call summary** at end of every call  
✅ **Create tasks** for follow-ups  

## You Don't Need To:

❌ Tell agent to "remember this"  
❌ Ask agent to "save my info"  
❌ Manually trigger any CRM functions  

The agent does it all automatically based on the system prompt!

---

## Testing Your Setup:

1. **First Call**: Agent calls you
   - Agent asks your name
   - You say "Matheus"
   - ✓ Agent saves `create_customer(phone, name="Matheus")`

2. **During Call**: You mention something
   - "I prefer email"
   - ✓ Agent saves `add_customer_fact(type="preference", content="Prefers email")`

3. **End of Call**: 
   - ✓ Agent saves `save_call_summary(outcome="interested", summary="...")`

4. **Next Call**: Agent calls again
   - ✓ Agent says "Hi Matheus!" (remembers your name)
   - ✓ Agent references previous conversation

---

## Verify It's Working:

After the call, check the database:

```bash
sqlite3 data/crm.db << 'SQL'
SELECT 'YOUR PROFILE:' as info;
SELECT name, phone, status, created_at FROM customers WHERE phone LIKE '%987654321%';

SELECT '' as info;
SELECT 'CALL HISTORY:' as info;
SELECT call_date, call_type, outcome, summary FROM calls WHERE customer_id IN (SELECT customer_id FROM customers WHERE phone LIKE '%987654321%');

SELECT '' as info;
SELECT 'LEARNED FACTS:' as info;
SELECT fact_type, content FROM customer_facts WHERE customer_id IN (SELECT customer_id FROM customers WHERE phone LIKE '%987654321%');
SQL
```

You should see your name, call history, and any facts the agent learned!
