# Supabase Setup Guide (Local Docker)

This guide walks you through setting up a local Supabase instance using Docker and migrating your existing data from `crm.db`.

## Prerequisites

1. **Docker Desktop**: Ensure Docker is installed and running.
2. **Supabase CLI**:
    * **Recommended (No sudo required):** Use `npx` to run the CLI directly.

        ```bash
        npx supabase <command>
        ```

    * **Alternative (Linux/macOS via Brew):** `brew install supabase/tap/supabase`
    * **Alternative (Manual):** Download from [GitHub Releases](https://github.com/supabase/cli/releases).

## Step 1: Initialize Supabase

1. Run the following command in the project root:

    ```bash
    npx supabase init
    ```

2. Start the local Supabase stack:

    ```bash
    npx supabase start
    ```

    * This may take a few minutes to download images.
    * Once done, note the **API URL**, **DB URL**, and **Keys** printed in the terminal.

## Step 2: Apply Database Schema

1. Create a new migration:

    ```bash
    supabase migration new init_schema
    ```

2. Copy the contents of `database/schema.sql` into the generated file in `supabase/migrations/<timestamp>_init_schema.sql`.

3. Apply the migration:

    ```bash
    supabase db reset
    ```

## Step 3: Configure Environment

Update your `.env` file with the local credentials:

```bash
# Local Supabase Credentials
SUPABASE_URL=http://127.0.0.1:54321
SUPABASE_KEY=<your-anon-key>
SUPABASE_SERVICE_ROLE_KEY=<your-service-role-key>

# Connection String (Direct)
DB_CONNECTION_STRING=postgresql://postgres:postgres@127.0.0.1:54322/postgres
```

## Step 4: Migrate Data

1. Install the Supabase Python client:

    ```bash
    uv add supabase
    # OR
    pip install supabase
    ```

2. Run the migration script:

    ```bash
    python database/migrate_from_sqlite.py
    ```

3. This script will:
    * Create a default "Legacy Organization".
    * Import all customers, calls, facts, and tasks from `crm.db`.
    * Assign them to the legacy organization.

## Step 5: Verify

1. Open the **Supabase Studio** at `http://localhost:54323`.
2. Check the `customers` and `calls` tables to ensure data is present.
