# Database Modeling Case Study: Multi-Tenant Voice Agent Platform (Local Supabase Edition)

## 1. Executive Summary

This document outlines the database architecture evolution required to transition the Voice Agent V4 from a single-instance application to a scalable, multi-tenant platform using **Supabase**. The goal is to support multiple establishments (organizations), each with their own agents, customers, and data, while maintaining sub-second latency for voice interactions.

**Key Objectives:**

* **Multi-Tenancy:** Strict data isolation between organizations using Row Level Security (RLS).
* **Scalability:** Support for thousands of concurrent agents and calls via Supavisor connection pooling.
* **Low Latency:** <50ms database lookups for critical call path operations.
* **Flexibility:** Configurable agents and workflows per establishment.

## 2. Hosting Strategy: Local Self-Hosted (Docker)

For development and data sovereignty, we will use **Self-Hosted Supabase** running locally via Docker. This provides the full Supabase stack (Postgres, Auth, Realtime, Storage, Edge Functions, Studio) on your own infrastructure.

* **Infrastructure:** Docker & Docker Compose.
* **Control:** Full control over data, logs, and configuration.
* **Cost:** Free (runs on your hardware).
* **Parity:** Identical features to the cloud version, ensuring easy migration if needed.

## 3. Proposed Data Model

We recommend migrating to **Supabase (PostgreSQL)** as the primary relational store.

### 3.1. Core Entities

#### `organizations` (Tenants)

* `id`: UUID (PK)
* `name`: String
* `slug`: String (Unique)
* `plan_tier`: Enum (Starter, Pro, Enterprise)
* `created_at`: Timestamp

#### `profiles` (Users)

Extends Supabase's built-in `auth.users` table.

* `id`: UUID (PK, References `auth.users.id`)
* `org_id`: UUID (FK -> organizations.id)
* `role`: Enum (admin, manager, viewer)
* `full_name`: String

#### `agents`

* `id`: UUID (PK)
* `org_id`: UUID (FK -> organizations.id)
* `name`: String
* `phone_number`: String (Unique)
* `voice_id`: String
* `llm_config`: JSONB
* `is_active`: Boolean

### 3.2. CRM Entities (Scoped)

All CRM tables (`customers`, `calls`, `knowledge_base`) must include `org_id` to enable RLS policies.

## 4. Performance & Latency Strategy

* **Connection Pooling:** Use Supavisor (port 5432 or 6543) for the Python backend.
* **Caching:** Use Redis for hot agent configuration and active call state.
* **RLS:** Enable Row Level Security on all tables to enforce tenant isolation at the database layer.

## 5. Local Supabase Setup Guide (Docker)

Follow these steps to initialize the database environment locally.

### Prerequisites

* [Docker Desktop](https://www.docker.com/products/docker-desktop) (running)
* [Supabase CLI](https://supabase.com/docs/guides/cli) (`brew install supabase/tap/supabase` or `npm install -g supabase`)

### Step 1: Initialize Project

In the project root:

```bash
# Initialize Supabase configuration
supabase init

# Start the local stack (downloads images and starts containers)
supabase start
```

Once started, you will see the local endpoints:

* **Studio (Dashboard):** `http://localhost:54323`
* **API URL:** `http://localhost:54321`
* **DB URL:** `postgresql://postgres:postgres@localhost:54322/postgres`

### Step 2: Enable Extensions & Create Schema

Create a migration file to apply the schema:

```bash
supabase migration new init_schema
```

Edit the generated file in `supabase/migrations/` and paste the following SQL:

```sql
-- Enable Extensions
create extension if not exists vector;
create extension if not exists "uuid-ossp";

-- 1. Create Tables
create table organizations (
  id uuid primary key default uuid_generate_v4(),
  name text not null,
  slug text unique not null,
  created_at timestamp with time zone default timezone('utc'::text, now()) not null
);

create table profiles (
  id uuid primary key references auth.users(id) on delete cascade,
  org_id uuid references organizations(id),
  role text check (role in ('admin', 'manager', 'viewer')) default 'viewer',
  full_name text,
  created_at timestamp with time zone default timezone('utc'::text, now()) not null
);

create table agents (
  id uuid primary key default uuid_generate_v4(),
  org_id uuid references organizations(id) not null,
  name text not null,
  phone_number text unique not null,
  voice_id text not null,
  llm_config jsonb default '{}'::jsonb,
  is_active boolean default true,
  created_at timestamp with time zone default timezone('utc'::text, now()) not null
);

create table customers (
  id uuid primary key default uuid_generate_v4(),
  org_id uuid references organizations(id) not null,
  phone text not null,
  name text,
  email text,
  created_at timestamp with time zone default timezone('utc'::text, now()) not null,
  unique(org_id, phone)
);

-- 2. Enable RLS
alter table organizations enable row level security;
alter table profiles enable row level security;
alter table agents enable row level security;
alter table customers enable row level security;

-- 3. Define Access Policies
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

create policy "Org Isolation Policy" on customers
  for all using (
    org_id = get_my_org_id() OR is_super_admin()
  );

create policy "Agent Visibility" on agents
  for select using (org_id = get_my_org_id());

create policy "Profile Visibility" on profiles
  for select using (
    id = auth.uid() OR is_super_admin()
  );
```

Apply the migration:

```bash
supabase db reset
```

### Step 3: Create Admin Account (Local)

Since this is local, we can insert directly into `auth.users` via SQL or use the Studio.

1. Open **Supabase Studio** at `http://localhost:54323`.
2. Go to **Authentication** -> **Users** -> **Add User**.
    * Email: `admin@local.com`
    * Password: `password`
    * (Select "Auto Confirm Email")
3. Go to **SQL Editor** and run:

```sql
-- 1. Create Master Org
insert into organizations (id, name, slug)
values ('00000000-0000-0000-0000-000000000000', 'Local Admin', 'admin')
on conflict do nothing;

-- 2. Promote User (Replace UUID with the one from Auth -> Users)
insert into profiles (id, org_id, role, full_name)
values (
  'USER_UUID_FROM_DASHBOARD', 
  '00000000-0000-0000-0000-000000000000', 
  'admin', 
  'Local Administrator'
);
```

### Step 4: Configure Environment

Update your `.env` file with local credentials (default for `supabase start`):

```bash
# Local Supabase Credentials
SUPABASE_URL=http://127.0.0.1:54321
SUPABASE_KEY=your-anon-key-from-supabase-status-output
SUPABASE_SERVICE_ROLE_KEY=your-service-role-key-from-supabase-status-output

# Connection String (Direct)
DB_CONNECTION_STRING=postgresql://postgres:postgres@127.0.0.1:54322/postgres
```
