-- Enable Extensions
create extension if not exists vector;
create extension if not exists "uuid-ossp";

-- 1. Organizations
create table organizations (
  id uuid primary key default uuid_generate_v4(),
  name text not null,
  slug text unique not null,
  plan_tier text check (plan_tier in ('starter', 'pro', 'enterprise')) default 'starter',
  settings jsonb default '{}'::jsonb,
  created_at timestamp with time zone default timezone('utc'::text, now()) not null
);

-- 2. Profiles (Users)
create table profiles (
  id uuid primary key references auth.users(id) on delete cascade,
  org_id uuid references organizations(id),
  role text check (role in ('admin', 'manager', 'viewer')) default 'viewer',
  full_name text,
  created_at timestamp with time zone default timezone('utc'::text, now()) not null
);

-- 3. Agents
create table agents (
  id uuid primary key default uuid_generate_v4(),
  org_id uuid references organizations(id) not null,
  name text not null,
  phone_number text unique not null,
  voice_id text not null,
  llm_config jsonb default '{}'::jsonb,
  tools_config jsonb default '{}'::jsonb,
  is_active boolean default true,
  created_at timestamp with time zone default timezone('utc'::text, now()) not null
);

-- 4. Customers
create table customers (
  id uuid primary key default uuid_generate_v4(),
  org_id uuid references organizations(id) not null,
  phone text not null,
  name text,
  email text,
  company text,
  role text,
  status text default 'new',
  lead_source text,
  assigned_to text,
  preferred_language text default 'pt-BR',
  timezone text default 'America/Sao_Paulo',
  has_whatsapp boolean,
  whatsapp_number text,
  google_calendar_email text,  -- Customer's calendar for dual booking
  notes text,
  created_at timestamp with time zone default timezone('utc'::text, now()) not null,
  updated_at timestamp with time zone default timezone('utc'::text, now()) not null,
  unique(org_id, phone)
);

-- 5. Calls
create table calls (
  id uuid primary key default uuid_generate_v4(),
  org_id uuid references organizations(id) not null,
  agent_id uuid references agents(id),
  customer_id uuid references customers(id),
  call_date timestamp with time zone default timezone('utc'::text, now()) not null,
  duration_seconds integer,
  call_type text,
  outcome text,
  transcript text,
  summary text,
  sentiment text,
  next_action text,
  next_action_date timestamp with time zone,
  metadata jsonb default '{}'::jsonb
);

-- 6. Customer Facts
create table customer_facts (
  id uuid primary key default uuid_generate_v4(),
  org_id uuid references organizations(id) not null,
  customer_id uuid references customers(id) not null,
  fact_type text,
  content text not null,
  confidence real default 1.0,
  learned_from_call uuid references calls(id),
  created_at timestamp with time zone default timezone('utc'::text, now()) not null
);

-- 7. Action Items
create table action_items (
  id uuid primary key default uuid_generate_v4(),
  org_id uuid references organizations(id) not null,
  customer_id uuid references customers(id) not null,
  created_in_call uuid references calls(id),
  task_type text,
  description text not null,
  due_date timestamp with time zone,
  status text default 'pending',
  priority text default 'medium',
  assigned_to text,
  completed_at timestamp with time zone,
  created_at timestamp with time zone default timezone('utc'::text, now()) not null
);

-- 8. Call Scripts
create table call_scripts (
  id uuid primary key default uuid_generate_v4(),
  org_id uuid references organizations(id) not null,
  name text not null,
  call_type text not null,
  industry text,
  greeting text,
  qualification_questions jsonb,
  objection_handlers jsonb,
  call_goals jsonb,
  closing_statements text,
  active boolean default true,
  created_at timestamp with time zone default timezone('utc'::text, now()) not null
);

-- 9. Knowledge Base
create table knowledge_base (
  id uuid primary key default uuid_generate_v4(),
  org_id uuid references organizations(id) not null,
  agent_id uuid references agents(id),
  category text not null,
  title text not null,
  content text not null,
  keywords text,
  source_document text,
  access_count integer default 0,
  created_at timestamp with time zone default timezone('utc'::text, now()) not null,
  updated_at timestamp with time zone default timezone('utc'::text, now()) not null,
  vector_embedding vector(1536)
);

-- 10. Communication Events
create table communication_events (
  id text primary key,
  correlation_id text not null,
  channel text not null,
  event_type text not null,
  status text not null,
  payload jsonb,
  created_at timestamp with time zone default timezone('utc'::text, now()) not null
);

-- 11. Organization Calendar Connections
create table org_calendar_connections (
  id uuid primary key default uuid_generate_v4(),
  org_id uuid references organizations(id) not null,
  attendee_name text not null,           -- Display name for agent (e.g., "Dr. João Silva")
  attendee_email text not null,          -- Google Calendar ID (email)
  calendar_id text,                       -- Optional: specific calendar ID (NULL = use email)
  is_default boolean default false,       -- Auto-select when only one or no preference
  calendar_type text default 'worker' check (calendar_type in ('worker', 'room', 'shared')),
  working_hours jsonb default '{
    "monday": {"start": "09:00", "end": "18:00"},
    "tuesday": {"start": "09:00", "end": "18:00"},
    "wednesday": {"start": "09:00", "end": "18:00"},
    "thursday": {"start": "09:00", "end": "18:00"},
    "friday": {"start": "09:00", "end": "18:00"}
  }'::jsonb,
  created_at timestamp with time zone default timezone('utc'::text, now()) not null
);

-- RLS Policies
alter table organizations enable row level security;
alter table profiles enable row level security;
alter table agents enable row level security;
alter table customers enable row level security;
alter table calls enable row level security;
alter table customer_facts enable row level security;
alter table action_items enable row level security;
alter table call_scripts enable row level security;
alter table knowledge_base enable row level security;
alter table communication_events enable row level security;
alter table org_calendar_connections enable row level security;

-- Helper Functions
create or replace function get_my_org_id()
returns uuid as $$
  select org_id from profiles where id = auth.uid() limit 1;
$$ language sql security definer;

create or replace function is_super_admin()
returns boolean as $$
  select exists (
    select 1 from profiles 
    where id = auth.uid() and role = 'admin'
  );
$$ language sql security definer;

-- Generic Org Isolation Policy
create policy "Org Isolation" on organizations
  for select using (id = get_my_org_id() OR is_super_admin());

create policy "Profile Visibility" on profiles
  for select using (id = auth.uid() OR is_super_admin());

-- Apply Org Isolation to all other tables
create policy "Agent Org Isolation" on agents for all using (org_id = get_my_org_id() OR is_super_admin());
create policy "Customer Org Isolation" on customers for all using (org_id = get_my_org_id() OR is_super_admin());
create policy "Call Org Isolation" on calls for all using (org_id = get_my_org_id() OR is_super_admin());
create policy "Fact Org Isolation" on customer_facts for all using (org_id = get_my_org_id() OR is_super_admin());
create policy "Task Org Isolation" on action_items for all using (org_id = get_my_org_id() OR is_super_admin());
create policy "Script Org Isolation" on call_scripts for all using (org_id = get_my_org_id() OR is_super_admin());
create policy "KB Org Isolation" on knowledge_base for all using (org_id = get_my_org_id() OR is_super_admin());
create policy "Calendar Connections Org Isolation" on org_calendar_connections for all using (org_id = get_my_org_id() OR is_super_admin());

-- Communication Events Policy (Admin only for now)
create policy "Communication Events Admin Access" on communication_events
  for all using (is_super_admin());

-- Indexes
create index idx_customers_org_phone on customers(org_id, phone);
create index idx_calls_org_date on calls(org_id, call_date desc);
create index idx_kb_embedding on knowledge_base using ivfflat (vector_embedding vector_cosine_ops);
create index idx_comm_events_correlation on communication_events(correlation_id);
create index idx_org_calendars_org on org_calendar_connections(org_id);
create index idx_org_calendars_name on org_calendar_connections(org_id, attendee_name);
create unique index idx_org_calendars_email on org_calendar_connections(org_id, attendee_email);
