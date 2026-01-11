# CRM Database Summary

## Overview

The voice agent v4 uses a **SQLite database** for CRM functionality, storing customer data, call history, facts learned during conversations, action items, call scripts, and knowledge base articles.

**Database Location:** `data/crm.db`  
**Database Type:** SQLite 3  
**Initialization Script:** `scripts/init_crm_db.py`

---

## Database Statistics

| Table | Row Count | Purpose |
|-------|-----------|---------|
| `customers` | 7 | Customer profiles and contact information |
| `calls` | 0 | Call history and transcripts |
| `customer_facts` | 0 | Facts learned about customers during calls |
| `action_items` | 2 | Tasks and follow-ups from calls |
| `call_scripts` | 1 | Scripted call flows for different scenarios |
| `knowledge_base` | 3 | Product info, FAQs, pricing for agent reference |
| `knowledge_base_fts` | - | Full-text search index (FTS5) |

---

## Table Schemas

### 1. `customers` Table

**Purpose:** Store customer profiles, contact details, and sales pipeline status.

| Column | Type | Constraints | Default | Description |
|--------|------|-------------|---------|-------------|
| `customer_id` | TEXT | PRIMARY KEY | - | UUID identifier |
| `phone` | TEXT | UNIQUE NOT NULL | - | Phone number (E.164 format) |
| `name` | TEXT | - | - | Customer full name |
| `email` | TEXT | - | - | Email address |
| `company` | TEXT | - | - | Company name |
| `role` | TEXT | - | - | Job title/role |
| `status` | TEXT | - | `'new'` | Pipeline status: new, contacted, qualified, customer, lost |
| `lead_source` | TEXT | - | - | How they found us: website_form, referral, cold_call, etc. |
| `assigned_to` | TEXT | - | - | Sales rep assigned |
| `preferred_language` | TEXT | - | `'pt-BR'` | Language preference (pt-BR, en-US, etc.) |
| `timezone` | TEXT | - | `'America/Sao_Paulo'` | Timezone for scheduling |
| `has_whatsapp` | BOOLEAN | - | `NULL` | Whether phone has WhatsApp (NULL=unknown, 0=no, 1=yes) |
| `whatsapp_number` | TEXT | - | `NULL` | Alternative WhatsApp number if different from phone |
| `created_at` | TIMESTAMP | - | `CURRENT_TIMESTAMP` | Record creation time |
| `updated_at` | TIMESTAMP | - | `CURRENT_TIMESTAMP` | Last update time |
| `notes` | TEXT | - | - | Free-form notes |

**Indexes:**
- `idx_customers_phone` on `phone`
- `idx_customers_status` on `status`
- `idx_customers_whatsapp` on `has_whatsapp`

**Sample Data:**
```
customer_id: b45ed16f-f4f8-4aac-8b80-8dd33e192965
phone: +55 11 99999-0001
name: João Silva
email: joao.silva@acme.com
company: Acme Corp
role: CEO
status: customer
lead_source: website_form
has_whatsapp: 1 (yes, phone has WhatsApp)
whatsapp_number: NULL (uses same number as phone)
```

**WhatsApp Integration:**
- `has_whatsapp = 1` → Customer can receive WhatsApp messages on their phone
- `has_whatsapp = 0` → Phone does NOT have WhatsApp, use voice calls only
- `has_whatsapp = NULL` → Unknown status, needs verification
- `whatsapp_number` → Use only if customer has different WhatsApp number than phone

---

### 2. `calls` Table

**Purpose:** Track call history, transcripts, outcomes, and next actions.

| Column | Type | Constraints | Default | Description |
|--------|------|-------------|---------|-------------|
| `call_id` | TEXT | PRIMARY KEY | - | UUID identifier |
| `customer_id` | TEXT | FOREIGN KEY → customers | - | Customer who called |
| `call_date` | TIMESTAMP | - | `CURRENT_TIMESTAMP` | When call occurred |
| `duration_seconds` | INTEGER | - | - | Call duration in seconds |
| `call_type` | TEXT | - | - | Type: outbound_cold_call, inbound_support, demo, etc. |
| `outcome` | TEXT | - | - | Result: interested_schedule_demo, not_interested, callback_later |
| `transcript` | TEXT | - | - | Full conversation transcript |
| `summary` | TEXT | - | - | AI-generated call summary |
| `sentiment` | TEXT | - | - | Sentiment: positive, neutral, negative |
| `next_action` | TEXT | - | - | What to do next |
| `next_action_date` | TIMESTAMP | - | - | When to follow up |

**Indexes:**
- `idx_calls_customer` on `customer_id`
- `idx_calls_date` on `call_date DESC`
- `idx_calls_type` on `call_type`

**Foreign Keys:**
- `customer_id` → `customers(customer_id)`

---

### 3. `customer_facts` Table

**Purpose:** Store facts learned about customers during conversations (memory/context building).

| Column | Type | Constraints | Default | Description |
|--------|------|-------------|---------|-------------|
| `fact_id` | TEXT | PRIMARY KEY | - | UUID identifier |
| `customer_id` | TEXT | FOREIGN KEY → customers | - | Customer this fact relates to |
| `fact_type` | TEXT | - | - | Type: preference, pain_point, business_context, goal, objection |
| `content` | TEXT | NOT NULL | - | The actual fact/insight |
| `confidence` | REAL | - | `1.0` | Confidence score (0.0-1.0) |
| `learned_from_call` | TEXT | FOREIGN KEY → calls | - | Which call this was learned from |
| `created_at` | TIMESTAMP | - | `CURRENT_TIMESTAMP` | When fact was learned |

**Indexes:**
- `idx_facts_customer` on `customer_id`
- `idx_facts_type` on `fact_type`

**Foreign Keys:**
- `customer_id` → `customers(customer_id)`
- `learned_from_call` → `calls(call_id)`

**Example Facts:**
```
fact_type: preference
content: "Prefers email over phone for documentation"

fact_type: pain_point
content: "Current CRM is too slow, losing sales opportunities"

fact_type: business_context
content: "Company has 50 employees, planning to expand to 100 by end of year"
```

---

### 4. `action_items` Table

**Purpose:** Track tasks and follow-ups generated from calls.

| Column | Type | Constraints | Default | Description |
|--------|------|-------------|---------|-------------|
| `task_id` | TEXT | PRIMARY KEY | - | UUID identifier |
| `customer_id` | TEXT | FOREIGN KEY → customers | - | Customer this task relates to |
| `created_in_call` | TEXT | FOREIGN KEY → calls | - | Call that generated this task |
| `task_type` | TEXT | - | - | Type: send_proposal, schedule_demo, follow_up_call, send_contract |
| `description` | TEXT | NOT NULL | - | Task description |
| `due_date` | TIMESTAMP | - | - | When task is due |
| `status` | TEXT | - | `'pending'` | Status: pending, in_progress, completed, cancelled |
| `priority` | TEXT | - | `'medium'` | Priority: low, medium, high, urgent |
| `assigned_to` | TEXT | - | - | Who should do this task |
| `completed_at` | TIMESTAMP | - | - | When task was completed |

**Indexes:**
- `idx_tasks_customer` on `customer_id`
- `idx_tasks_status` on `status`
- `idx_tasks_due` on `due_date`

**Foreign Keys:**
- `customer_id` → `customers(customer_id)`
- `created_in_call` → `calls(call_id)`

---

### 5. `call_scripts` Table

**Purpose:** Store call scripts/flows for different scenarios (cold calls, demos, support).

| Column | Type | Constraints | Default | Description |
|--------|------|-------------|---------|-------------|
| `script_id` | TEXT | PRIMARY KEY | - | UUID identifier |
| `name` | TEXT | NOT NULL | - | Script name (e.g., "Cold Call - SaaS B2B") |
| `call_type` | TEXT | NOT NULL | - | When to use: outbound_cold_call, inbound_support, demo |
| `industry` | TEXT | - | - | Industry-specific script (optional) |
| `greeting` | TEXT | - | - | Opening line with variables: {customer_name}, {agent_name} |
| `qualification_questions` | TEXT | - | - | JSON array of questions to ask |
| `objection_handlers` | TEXT | - | - | JSON object of common objections and responses |
| `call_goals` | TEXT | - | - | JSON array of call objectives |
| `closing_statements` | TEXT | - | - | How to end the call |
| `active` | BOOLEAN | - | `1` | Whether script is currently active |
| `created_at` | TIMESTAMP | - | `CURRENT_TIMESTAMP` | When script was created |

**Sample Script Structure:**
```json
{
  "greeting": "Olá {customer_name}, aqui é {agent_name} da {company}.",
  "qualification_questions": [
    "Vocês estão usando algum CRM atualmente?",
    "Qual o maior desafio com a solução atual?",
    "Quantos usuários precisariam ter acesso?"
  ],
  "objection_handlers": {
    "muito_caro": "Nossa solução economiza R$5.000/mês comparado com dev full-time",
    "não_interessado": "Posso perguntar qual solução vocês usam atualmente?"
  },
  "call_goals": [
    "Qualificar lead (BANT: Budget, Authority, Need, Timeline)",
    "Agendar demonstração"
  ]
}
```

---

### 6. `knowledge_base` Table

**Purpose:** Store product information, FAQs, pricing, technical details for agent to reference.

| Column | Type | Constraints | Default | Description |
|--------|------|-------------|---------|-------------|
| `kb_id` | TEXT | PRIMARY KEY | - | UUID identifier |
| `category` | TEXT | NOT NULL | - | Category: pricing, technical, faq, product, integration |
| `title` | TEXT | NOT NULL | - | Article title |
| `content` | TEXT | NOT NULL | - | Full article content |
| `keywords` | TEXT | - | - | Space-separated keywords for search |
| `source_document` | TEXT | - | - | Original document reference |
| `created_at` | TIMESTAMP | - | `CURRENT_TIMESTAMP` | Creation time |
| `updated_at` | TIMESTAMP | - | `CURRENT_TIMESTAMP` | Last update time |
| `access_count` | INTEGER | - | `0` | How many times accessed |

**Indexes:**
- `idx_kb_category` on `category`

**Sample Knowledge Base Entry:**
```
kb_id: 550e8400-e29b-41d4-a716-446655440000
category: pricing
title: Plano Enterprise - Preços e Recursos
content: O Plano Enterprise custa R$ 499,00 por mês e inclui:
- Usuários ilimitados
- Acesso à API completa
- Suporte 24/7 em português
- Integração com Salesforce, HubSpot, RD Station
...
keywords: enterprise preço valor custo plano api integração salesforce
```

---

### 7. `knowledge_base_fts` Virtual Table

**Purpose:** Full-text search index using SQLite FTS5 for fast knowledge base searches.

**Structure:** Virtual table using FTS5 tokenizer
```sql
CREATE VIRTUAL TABLE knowledge_base_fts 
USING fts5(
    kb_id UNINDEXED,      -- Not searchable, just returned
    title,                 -- Searchable field
    content,              -- Searchable field
    keywords,             -- Searchable field
    content='knowledge_base',
    content_rowid='rowid'
)
```

**Auto-Sync Triggers:**
- `knowledge_base_ai` - Inserts into FTS on new KB entry
- `knowledge_base_ad` - Deletes from FTS when KB entry deleted
- `knowledge_base_au` - Updates FTS when KB entry updated

**Usage Example:**
```sql
-- Search for "api integration"
SELECT kb_id, title 
FROM knowledge_base_fts 
WHERE knowledge_base_fts MATCH 'api integration'
ORDER BY rank;
```

---

## Relationships Diagram

```
customers (1) ──── (N) calls
    │                   │
    │                   │
   (N)                 (N)
    │                   │
customer_facts    action_items
```

**Key Relationships:**
- One customer can have many calls
- One customer can have many facts learned
- One customer can have many action items
- One call can generate many facts
- One call can generate many action items

---

## Database Initialization

### Setup Script

**File:** `scripts/init_crm_db.py`

**Run:**
```bash
cd /home/matheus/repos/agno_cartesia/voice_agent_v4
python scripts/init_crm_db.py
```

**What it does:**
1. Creates all 7 tables
2. Creates 11 indexes for performance
3. Sets up FTS5 virtual table and triggers
4. Inserts sample data:
   - 3 sample customers (João Silva, Maria Santos, Pedro Costa)
   - 1 call history
   - 3 customer facts
   - 1 action item
   - 1 call script (Cold Call - SaaS B2B in Portuguese)
   - 3 knowledge base articles (pricing, technical, FAQ)

**Sample Data Generated:**
- **Customers:** Brazilian B2B leads (Acme Corp, TechStart, Innovate Solutions)
- **Call Script:** Portuguese SaaS cold call template with BANT qualification
- **Knowledge Base:** Pricing (R$ 499/month Enterprise plan), Technical setup, Security FAQ

---

## Database Access Patterns

### From MCP Tools

The CRM tools in `mcp_tools/` interact with this database:

**Customer Tools** (`mcp_tools/crm_customer.py`):
- `get_customer_by_phone()` - Lookup by phone (UNIQUE index)
- `get_customer_by_id()` - Fetch by UUID
- `create_customer()` - Insert new lead
- `update_customer_status()` - Move through pipeline
- `update_customer_info()` - Update fields
- `search_customers()` - Text search on name/company/email

**Call Tools** (`mcp_tools/crm_calls.py`):
- Record call history
- Store transcripts
- Track outcomes and sentiment

**Facts Tools** (`mcp_tools/crm_facts.py`):
- Save facts learned during calls
- Retrieve customer context for future calls

**Tasks Tools** (`mcp_tools/crm_tasks.py`):
- Create follow-up tasks
- Track task completion
- Query pending tasks by customer

**Knowledge Base Tools** (`mcp_tools/crm_knowledge.py`):
- Search KB using FTS5
- Retrieve articles by category
- Track access patterns

---

## Performance Optimizations

### Indexes Created

| Index Name | Table | Column(s) | Purpose |
|------------|-------|-----------|---------|
| `idx_customers_phone` | customers | phone | Fast phone number lookup |
| `idx_customers_status` | customers | status | Filter by pipeline stage |
| `idx_customers_whatsapp` | customers | has_whatsapp | Filter by WhatsApp availability |
| `idx_calls_customer` | calls | customer_id | Get all calls for a customer |
| `idx_calls_date` | calls | call_date DESC | Recent calls first |
| `idx_calls_type` | calls | call_type | Filter by call type |
| `idx_facts_customer` | customer_facts | customer_id | Get all facts for customer |
| `idx_facts_type` | customer_facts | fact_type | Filter facts by type |
| `idx_tasks_customer` | action_items | customer_id | Get tasks for customer |
| `idx_tasks_status` | action_items | status | Filter pending/completed |
| `idx_tasks_due` | action_items | due_date | Sort by due date |
| `idx_kb_category` | knowledge_base | category | Browse by category |

### Full-Text Search

**Technology:** SQLite FTS5 (Full-Text Search version 5)

**Benefits:**
- Fast keyword search across title, content, keywords
- Relevance ranking with BM25
- Phrase matching and proximity search
- Automatic tokenization

**Performance:**
- Sub-millisecond searches on 1000s of KB articles
- Indexes updated automatically via triggers

---

## Data Types and Conventions

### UUID Format
All primary keys use UUID v4:
```
b45ed16f-f4f8-4aac-8b80-8dd33e192965
```

### Phone Format
E.164 international format with spaces:
```
+55 11 99999-0001  (Brazil)
+1 415 555-1234    (USA)
```

### Timestamps
ISO 8601 format:
```
2025-10-06T18:00:49.011612
```

### JSON Fields
Some TEXT fields store JSON:
- `call_scripts.qualification_questions` - Array of strings
- `call_scripts.objection_handlers` - Object with key-value pairs
- `call_scripts.call_goals` - Array of goal strings

**Example:**
```json
{
  "muito_caro": "Nossa solução economiza R$5.000/mês",
  "não_interessado": "Qual solução vocês usam atualmente?"
}
```

---

## Security Considerations

### Credentials
- **Never stored in database:** API keys, passwords, tokens
- **Environment variables:** Use `.env` for sensitive config
- **Database location:** `data/crm.db` is in `.gitignore`

### Data Privacy
- **LGPD Compliance:** Customer data can be exported/deleted
- **Access Control:** Implement in application layer (not DB level)
- **Encryption:** SQLite supports encryption via SQLCipher (optional)

### Backup Strategy
```bash
# Backup database
sqlite3 data/crm.db ".backup data/crm_backup_$(date +%Y%m%d).db"

# Export to SQL
sqlite3 data/crm.db .dump > data/crm_backup.sql
```

---

## Query Examples

### Get customer with all related data
```sql
SELECT 
    c.*,
    (SELECT COUNT(*) FROM calls WHERE customer_id = c.customer_id) as call_count,
    (SELECT COUNT(*) FROM customer_facts WHERE customer_id = c.customer_id) as fact_count,
    (SELECT COUNT(*) FROM action_items WHERE customer_id = c.customer_id AND status = 'pending') as pending_tasks
FROM customers c
WHERE c.phone = '+55 11 99999-0001';
```

### Search knowledge base
```sql
SELECT kb.*, rank
FROM knowledge_base kb
JOIN knowledge_base_fts fts ON kb.rowid = fts.rowid
WHERE knowledge_base_fts MATCH 'api integração'
ORDER BY rank
LIMIT 5;
```

### Get recent calls with customer info
```sql
SELECT 
    c.call_date,
    c.duration_seconds,
    c.outcome,
    cust.name,
    cust.company
FROM calls c
JOIN customers cust ON c.customer_id = cust.customer_id
ORDER BY c.call_date DESC
LIMIT 10;
```

### Pipeline report
```sql
SELECT 
    status,
    COUNT(*) as count,
    GROUP_CONCAT(name, ', ') as customers
FROM customers
GROUP BY status
ORDER BY 
    CASE status
        WHEN 'new' THEN 1
        WHEN 'contacted' THEN 2
        WHEN 'qualified' THEN 3
        WHEN 'customer' THEN 4
        WHEN 'lost' THEN 5
    END;
```

### Get customers with WhatsApp for messaging campaign
```sql
-- All customers with confirmed WhatsApp
SELECT name, phone, whatsapp_number
FROM customers
WHERE has_whatsapp = 1;

-- Customers with WhatsApp and specific status
SELECT name, phone, company, 
       COALESCE(whatsapp_number, phone) as whatsapp_contact
FROM customers
WHERE has_whatsapp = 1 AND status IN ('qualified', 'contacted');

-- Customers WITHOUT WhatsApp (voice call only)
SELECT name, phone, status
FROM customers
WHERE has_whatsapp = 0 OR has_whatsapp IS NULL;
```

---

## File Locations

```
voice_agent_v4/
├── data/
│   ├── crm.db                    # SQLite database file
│   └── .gitignore               # Excludes database from git
├── scripts/
│   └── init_crm_db.py           # Database initialization script
└── mcp_tools/
    ├── crm_customer.py          # Customer CRUD operations
    ├── crm_calls.py             # Call history management
    ├── crm_facts.py             # Customer facts/memory
    ├── crm_tasks.py             # Action items/tasks
    ├── crm_scripts.py           # Call script templates
    └── crm_knowledge.py         # Knowledge base search
```

---

## Summary

The CRM database provides a complete customer relationship management system for the voice agent:

✅ **7 main tables** + FTS5 virtual table  
✅ **11 indexes** for fast queries  
✅ **Full-text search** for knowledge base  
✅ **Sample data** with Brazilian B2B context  
✅ **Relationship tracking** (customers → calls → facts → tasks)  
✅ **MCP tool integration** via `mcp_tools/crm_*.py`  
✅ **Easy initialization** via `scripts/init_crm_db.py`

**Current State:** 7 customers, 0 calls, 2 tasks, 3 KB articles
