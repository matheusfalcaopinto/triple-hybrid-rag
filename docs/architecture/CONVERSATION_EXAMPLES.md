# Conversation Examples - Enhanced Tool Usage

**Branch:** tools/email  
**Date:** Tue Oct 07 2025  
**Total Tools:** 71 (33 CRM + 11 Communication + 7 Calendar + 20 System/Utilities)

---

## Overview

This document demonstrates real-world conversation flows showing the LLM's enhanced tool usage based on the updated `PROMPT.md`. Each example shows:

1. Customer input
2. Tools the LLM should automatically call
3. Natural response to customer

---

## Example 1: First-Time Caller (Cold Lead)

### Conversation Flow

**[Call Start - System sends caller phone: +5517997019739]**

**LLM Auto-Actions:**
```python
# 1. Immediately lookup customer (REQUIRED)
get_customer_by_phone(phone="+5517997019739")
# Returns: {"success": False, "message": "Customer not found"}
```

**Agent:** "Olá! Bem-vindo à nossa empresa. Qual é o seu nome?"

**Customer:** "João Silva, da empresa TechBrasil"

**LLM Auto-Actions:**
```python
# 2. Create customer profile
create_customer(
    phone="+5517997019739",
    name="João Silva", 
    company="TechBrasil"
)
# Returns: {"success": True, "customer_id": "c_abc123"}

# 3. Save business fact
add_customer_fact(
    customer_id="c_abc123",
    fact_type="business_context",
    content="Works at TechBrasil company"
)
```

**Agent:** "Prazer João! Como posso ajudá-lo hoje?"

**Customer:** "Quero saber sobre o plano enterprise"

**LLM Auto-Actions:**
```python
# 4. Search knowledge base
search_knowledge_base(
    query="plano enterprise preço",
    category="pricing"
)
# Returns: {"success": True, "results": [{"title": "Enterprise Plan", "content": "R$499/mês, até 100 usuários..."}]}
```

**Agent:** "Nosso plano enterprise custa R$499 por mês e inclui até 100 usuários. Posso agendar uma demonstração?"

**Customer:** "Está muito caro"

**LLM Auto-Actions:**
```python
# 5. Get objection handler
get_objection_handler(
    call_type="cold_call",
    objection_key="too_expensive"
)
# Returns: {"success": True, "response": "Entendo. Considere o ROI - clientes economizam 40% em processos..."}

# 6. Save objection as fact
add_customer_fact(
    customer_id="c_abc123",
    fact_type="pain_point",
    content="Price sensitivity - mentioned enterprise plan too expensive"
)
```

**Agent:** "Entendo João. Nossos clientes economizam em média 40% em custos operacionais. Que tal uma demonstração gratuita para ver o valor?"

**Customer:** "Ok, pode agendar para amanhã 14h"

**LLM Auto-Actions:**
```python
# 7. Check calendar availability
check_calendar_availability(
    start_time="2025-10-08T14:00:00",
    end_time="2025-10-08T15:00:00"
)
# Returns: {"success": True, "available": True}

# 8. Create calendar event
create_calendar_event(
    summary="Demo - João Silva (TechBrasil)",
    start_time="2025-10-08T14:00:00",
    end_time="2025-10-08T15:00:00",
    description="Enterprise plan demo - price sensitive",
    attendees=[]
)
# Returns: {"success": True, "event_id": "evt_123"}

# 9. Create follow-up task
create_task(
    customer_id="c_abc123",
    description="Prepare enterprise demo focusing on ROI",
    task_type="demo",
    due_date="2025-10-08",
    priority="high"
)
# Returns: {"success": True, "task_id": "t_456"}
```

**Agent:** "Perfeito! Agendei para amanhã 8 de outubro às 14h. Qual seu email para enviar a confirmação?"

**Customer:** "joao@techbrasil.com.br"

**LLM Auto-Actions:**
```python
# 10. Update customer with email
update_customer_info(
    customer_id="c_abc123",
    email="joao@techbrasil.com.br"
)

# 11. Send confirmation email
send_email(
    to="joao@techbrasil.com.br",
    subject="Demonstração Enterprise - 08/10 às 14h",
    body="Olá João,\n\nConfirmamos sua demonstração do plano Enterprise para amanhã, 8 de outubro às 14h.\n\nAtenciosamente,\nEquipe"
)
# Returns: {"success": True, "message_id": "msg_789"}

# 12. Send WhatsApp confirmation
send_whatsapp_message(
    to="+5517997019739",
    body="Oi João! Confirmamos sua demo Enterprise para amanhã 08/10 às 14h. Enviamos os detalhes por email. Até lá!"
)
# Returns: {"success": True, "sid": "SM123"}
```

**Agent:** "Pronto João! Te enviei email e WhatsApp com a confirmação. Até amanhã às 14h!"

**[Call End]**

**LLM Auto-Actions:**
```python
# 13. Save call summary (REQUIRED)
save_call_summary(
    customer_id="c_abc123",
    call_type="inbound",
    outcome="qualified",
    summary="First contact. Interested in enterprise plan but price sensitive. Demo scheduled for tomorrow 14h. Company: TechBrasil.",
    duration_seconds=180
)
```

---

## Example 2: Returning Customer (Follow-up)

### Conversation Flow

**[Call Start - System sends caller phone: +5511988887777]**

**LLM Auto-Actions:**
```python
# 1. Lookup customer immediately
get_customer_by_phone(phone="+5511988887777")
# Returns: {"success": True, "customer_id": "c_xyz789", "name": "Maria Santos", "company": "FastFood Corp"}

# 2. Get call history
get_call_history(customer_id="c_xyz789", limit=3)
# Returns: {"success": True, "calls": [
#   {"date": "2025-10-01", "outcome": "interested", "summary": "Asked about integrations"},
#   {"date": "2025-09-28", "outcome": "callback", "summary": "Needs approval from manager"}
# ]}

# 3. Get customer facts
get_customer_facts(customer_id="c_xyz789")
# Returns: {"success": True, "facts": [
#   {"type": "preference", "content": "Prefers communication via WhatsApp"},
#   {"type": "technical_requirement", "content": "Needs API integration with SAP"}
# ]}

# 4. Check pending tasks
get_pending_tasks(customer_id="c_xyz789")
# Returns: {"success": True, "tasks": [
#   {"description": "Send API integration documentation", "priority": "high", "due_date": "2025-10-07"}
# ]}
```

**Agent:** "Oi Maria! Tudo bem? Vi que você tinha pedido a documentação da integração com SAP."

**Customer:** "Sim! Conseguiu aprovação do meu gerente. Vamos fechar."

**LLM Auto-Actions:**
```python
# 5. Update customer status
update_customer_status(
    customer_id="c_xyz789",
    status="customer"
)

# 6. Save fact about approval
add_customer_fact(
    customer_id="c_xyz789",
    fact_type="business_context",
    content="Manager approved purchase - ready to close deal"
)

# 7. Complete the pending task
complete_task(task_id="t_pending_123")
```

**Agent:** "Que ótimo Maria! Vou te enviar o contrato por email agora. Confirma seu email?"

**Customer:** "maria.santos@fastfood.com.br"

**LLM Auto-Actions:**
```python
# 8. Send contract via email
send_email_template(
    to="maria.santos@fastfood.com.br",
    template_name="proposal",
    variables={
        "name": "Maria",
        "company": "FastFood Corp",
        "plan": "Enterprise",
        "price": "R$499/mês"
    }
)

# 9. Send WhatsApp (customer prefers WhatsApp per fact)
send_whatsapp_document(
    to="+5511988887777",
    document_url="https://storage.example.com/contracts/fastfood-enterprise.pdf",
    filename="Contrato_Enterprise_FastFood.pdf",
    caption="Oi Maria! Segue o contrato. Qualquer dúvida, me chama!"
)

# 10. Create onboarding task
create_task(
    customer_id="c_xyz789",
    description="Schedule onboarding call and SAP integration setup",
    task_type="follow_up",
    due_date="2025-10-09",
    priority="high"
)
```

**Agent:** "Pronto Maria! Enviei o contrato por email e WhatsApp. Depois de assinado, agendo a integração do SAP com você."

**[Call End]**

**LLM Auto-Actions:**
```python
# 11. Save call summary
save_call_summary(
    customer_id="c_xyz789",
    call_type="follow_up",
    outcome="qualified",
    summary="Customer approved by manager. Contract sent. Ready to close. Needs SAP integration setup after signing.",
    duration_seconds=120
)
```

---

## Example 3: Support Call with Scheduling

### Conversation Flow

**[Call Start - Caller: +5521999998888]**

**LLM Auto-Actions:**
```python
# 1. Lookup customer
get_customer_by_phone(phone="+5521999998888")
# Returns: {"success": True, "customer_id": "c_support123", "name": "Carlos Oliveira"}

# 2. Get context
get_call_history(customer_id="c_support123", limit=2)
get_customer_facts(customer_id="c_support123")
```

**Agent:** "Oi Carlos! Como posso ajudar?"

**Customer:** "Estou com dúvida sobre como resetar senha"

**LLM Auto-Actions:**
```python
# 3. Search knowledge base for solution
search_knowledge_base(
    query="resetar senha recuperar acesso",
    category="troubleshooting"
)
# Returns: {"success": True, "results": [
#   {"title": "Como resetar senha", "content": "Clique em 'Esqueci senha' no login..."}
# ]}
```

**Agent:** "Carlos, é simples. Na tela de login clique em 'Esqueci minha senha'. Você vai receber um email com o link. Consegue fazer agora?"

**Customer:** "Ah sim, encontrei. Mas aproveitando, queria agendar um treinamento para minha equipe"

**LLM Auto-Actions:**
```python
# 4. List available times
get_current_time(timezone="America/Sao_Paulo")
# Returns: "2025-10-07 10:30:00"

list_upcoming_calendar_events(max_results=10)
# Returns: Events showing availability gaps

# 5. Check specific slot
check_calendar_availability(
    start_time="2025-10-09T10:00:00",
    end_time="2025-10-09T12:00:00"
)
# Returns: {"available": True}
```

**Agent:** "Claro! Tenho disponível quinta-feira dia 9 de outubro às 10h. Serve para você?"

**Customer:** "Perfeito!"

**LLM Auto-Actions:**
```python
# 6. Create training event
create_calendar_event(
    summary="Treinamento - Equipe Carlos Oliveira",
    start_time="2025-10-09T10:00:00",
    end_time="2025-10-09T12:00:00",
    description="Team training session",
    attendees=[]
)

# 7. Send confirmation
send_email(
    to="carlos@empresa.com.br",
    subject="Treinamento Agendado - 09/10 às 10h",
    body="Olá Carlos,\n\nConfirmamos o treinamento para sua equipe em 09/10 às 10h (2 horas).\n\nEnviaremos o link da reunião 1 dia antes.\n\nAbraços!"
)

# 8. Create preparation task
create_task(
    customer_id="c_support123",
    description="Prepare training materials and send meeting link to Carlos",
    task_type="follow_up",
    due_date="2025-10-08",
    priority="medium"
)

# 9. Save fact about team size for future reference
add_customer_fact(
    customer_id="c_support123",
    fact_type="business_context",
    content="Requested team training - has team to onboard"
)
```

**Agent:** "Agendado Carlos! Dia 9 de outubro às 10h, 2 horas de treinamento. Te enviei confirmação por email."

**[Call End]**

**LLM Auto-Actions:**
```python
save_call_summary(
    customer_id="c_support123",
    call_type="inbound",
    outcome="qualified",
    summary="Support call - password reset question resolved. Scheduled team training for Oct 9 10am.",
    duration_seconds=150
)
```

---

## Example 4: Multi-Channel Communication Flow

### Conversation Flow

**[Call Start - Caller: +5548991234567]**

**LLM Auto-Actions:**
```python
get_customer_by_phone(phone="+5548991234567")
# Returns: {"success": True, "customer_id": "c_multi123", "name": "Ana Paula", "email": "ana@startup.io"}
```

**Agent:** "Oi Ana! Como vai?"

**Customer:** "Oi! Preciso da nossa proposta atualizada com os novos preços"

**LLM Auto-Actions:**
```python
# 1. Search for latest pricing
search_knowledge_base(query="pricing tabela preços 2025", category="pricing")
# Returns current pricing info

# 2. Get customer facts to personalize
get_customer_facts(customer_id="c_multi123")
# Returns: {"facts": [{"type": "business_context", "content": "Startup with 20 employees"}]}
```

**Agent:** "Claro Ana! Vou te enviar agora. Prefere por email ou WhatsApp?"

**Customer:** "Pode enviar nos dois por favor"

**LLM Auto-Actions:**
```python
# 3. Send via email (formal)
send_html_email(
    to="ana@startup.io",
    subject="Proposta Atualizada - Startup Plan",
    html_body="""
    <h2>Proposta Comercial - Startup Plan</h2>
    <p>Olá Ana,</p>
    <p>Conforme solicitado, segue nossa proposta atualizada:</p>
    <ul>
        <li>Plano Startup: R$199/mês (até 20 usuários)</li>
        <li>Suporte prioritário incluído</li>
        <li>Integração API ilimitada</li>
    </ul>
    <p>Validade: 30 dias</p>
    """
)

# 4. Send via WhatsApp (quick reference)
send_whatsapp_message(
    to="+5548991234567",
    body="Oi Ana! 📊 Proposta atualizada:\n\nPlano Startup\nR$199/mês (20 usuários)\n✅ Suporte prioritário\n✅ API ilimitada\n\nValidade: 30 dias\n\nEnviei versão completa por email!"
)

# 5. Optional: Send document via WhatsApp too
send_whatsapp_document(
    to="+5548991234567",
    document_url="https://storage.example.com/proposals/startup-2025.pdf",
    filename="Proposta_Startup_2025.pdf"
)

# 6. Save fact about communication preference
add_customer_fact(
    customer_id="c_multi123",
    fact_type="preference",
    content="Prefers receiving documents via both email and WhatsApp"
)

# 7. Create follow-up task
create_task(
    customer_id="c_multi123",
    description="Follow up on proposal in 3 days",
    task_type="follow_up",
    due_date="2025-10-10",
    priority="medium"
)
```

**Agent:** "Pronto Ana! Enviei por email a versão completa e por WhatsApp um resumo com o PDF. Qualquer dúvida me chama!"

---

## Example 5: Calendar + Location Sharing (Field Service)

### Conversation Flow

**[Call Start - Caller: +5585988776655]**

**LLM Auto-Actions:**
```python
get_customer_by_phone(phone="+5585988776655")
# Returns: {"success": True, "customer_id": "c_field123", "name": "Roberto Costa"}
```

**Agent:** "Oi Roberto!"

**Customer:** "Preciso de uma visita técnica urgente amanhã"

**LLM Auto-Actions:**
```python
# 1. Check tomorrow's availability
get_current_time(timezone="America/Sao_Paulo")
# Returns: "2025-10-07 11:00:00"

check_calendar_availability(
    start_time="2025-10-08T09:00:00",
    end_time="2025-10-08T11:00:00"
)
# Returns: {"available": True}
```

**Agent:** "Consigo amanhã às 9h da manhã. Qual o endereço?"

**Customer:** "Rua das Flores 123, Centro, Fortaleza"

**LLM Auto-Actions:**
```python
# 2. Create visit event
create_calendar_event(
    summary="Visita Técnica - Roberto Costa",
    start_time="2025-10-08T09:00:00",
    end_time="2025-10-08T11:00:00",
    location="Rua das Flores 123, Centro, Fortaleza - CE",
    description="Urgent technical visit"
)

# 3. Save address as fact
add_customer_fact(
    customer_id="c_field123",
    fact_type="business_context",
    content="Service address: Rua das Flores 123, Centro, Fortaleza - CE"
)

# 4. Send location pin via WhatsApp
send_whatsapp_location(
    to="+5585988776655",
    latitude=-3.7319,
    longitude=-38.5267,
    name="Local da Visita",
    address="Rua das Flores 123, Centro, Fortaleza"
)

# 5. Send confirmation
send_whatsapp_message(
    to="+5585988776655",
    body="Oi Roberto! ✅ Visita técnica agendada para amanhã 08/10 às 9h. Enviei o pin de localização acima. Até lá!"
)

# 6. Create task for technician
create_task(
    customer_id="c_field123",
    description="Technical visit - Rua das Flores 123 - urgent",
    task_type="follow_up",
    due_date="2025-10-08",
    priority="high",
    assigned_to="tecnico@empresa.com"
)
```

**Agent:** "Agendado Roberto! Amanhã 8 de outubro às 9h. Te enviei a localização por WhatsApp."

**[Call End]**

**LLM Auto-Actions:**
```python
save_call_summary(
    customer_id="c_field123",
    call_type="inbound",
    outcome="qualified",
    summary="Urgent technical visit scheduled for Oct 8 9am. Address: Rua das Flores 123, Centro, Fortaleza.",
    duration_seconds=90
)
```

---

## Example 6: DTMF Menu Navigation

### Conversation Flow

**Agent:** "Bem-vindo! Pressione 1 para vendas, 2 para suporte, ou 3 para falar com atendente."

**Customer:** *[Presses 1 on keypad]*

**[System sends: DtmfInput(digit="1")]*

**Agent receives:** `[User pressed 1 on phone keypad]`

**LLM Auto-Actions:**
```python
# No tool call needed - just process the menu choice
```

**Agent:** "Você escolheu vendas. Qual é o seu nome?"

**Customer:** "Pedro Santos"

**Agent:** "Olá Pedro! Quer saber sobre qual produto?"

**Customer:** *[Presses #]* (repeat last message)

**[System automatically repeats last TTS - no LLM response needed]**

**Customer:** "Plano básico"

**LLM Auto-Actions:**
```python
# Check if customer exists
get_customer_by_phone(phone="+55...")
# Create if not exists
# Then search knowledge base
search_knowledge_base(query="plano básico preço", category="pricing")
```

**Agent:** "O plano básico custa R$99/mês com até 10 usuários. Te interessou?"

---

## Key Patterns Demonstrated

### 1. **Automatic Tool Chaining**
- Customer request → Multiple tools called automatically
- Example: Schedule meeting → check_availability + create_event + send_email + send_whatsapp

### 2. **Proactive Information Gathering**
- Every call starts with `get_customer_by_phone()`
- Context loaded automatically (history, facts, tasks)
- No explicit user request needed

### 3. **Multi-Channel Communication**
- Same information sent via multiple channels
- Adapt format per channel (formal email, casual WhatsApp)
- Use customer preferences from facts

### 4. **Persistent Memory**
- Every interaction saved as facts
- Call summaries always recorded
- Tasks created for follow-ups

### 5. **Natural Language Abstraction**
- Tools called silently in background
- Responses feel human, not robotic
- Never mention "checking database" or "using tools"

### 6. **Error Resilience**
- If email missing, ask naturally
- If calendar unavailable, offer alternatives
- Graceful degradation

---

## Tool Usage Statistics (Per Example)

| Example | Tools Called | Categories Used | Response Time |
|---------|--------------|-----------------|---------------|
| 1. First-Time Caller | 13 | CRM, Email, WhatsApp, Calendar | ~8s |
| 2. Returning Customer | 11 | CRM, Email, WhatsApp | ~5s |
| 3. Support + Scheduling | 9 | CRM, Knowledge, Calendar, Email | ~6s |
| 4. Multi-Channel | 7 | CRM, Email, WhatsApp | ~4s |
| 5. Field Service | 8 | CRM, Calendar, WhatsApp, Location | ~5s |
| 6. DTMF Menu | 2 | CRM, Knowledge | ~3s |

---

## Best Practices from Examples

1. **Always start with customer lookup** - First tool call in every conversation
2. **Chain related tools** - Don't make customer wait for sequential actions
3. **Save everything as facts** - Preferences, pain points, context
4. **End every call with summary** - Required for CRM tracking
5. **Use customer preferences** - Email vs WhatsApp, communication style
6. **Be contextual** - Reference past calls, facts, pending tasks
7. **Create follow-up tasks** - Never let leads go cold

---

**Document Version:** 1.0  
**Last Updated:** Tue Oct 07 2025  
**Next Review:** After first production deployment
