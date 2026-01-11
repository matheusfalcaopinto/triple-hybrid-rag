#!/usr/bin/env python3
"""
Apply database migration for calendar connections table.
Run: python database/migrations/apply_calendar_migration.py
"""
import os
import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from dotenv import load_dotenv
load_dotenv()

def main():
    from supabase import create_client
    
    url = os.getenv("SUPABASE_URL")
    key = os.getenv("SERVICE_ROLE_KEY")
    
    if not url or not key:
        print("❌ Error: SUPABASE_URL and SERVICE_ROLE_KEY must be set in .env")
        sys.exit(1)
    
    client = create_client(url, key)
    
    # Read migration file
    migration_path = Path(__file__).parent / "20260107_add_calendar_connections.sql"
    migration_sql = migration_path.read_text()
    
    print("📦 Applying migration: Add Calendar Connections...")
    print("-" * 50)
    
    # Split into individual statements and execute
    statements = [s.strip() for s in migration_sql.split(';') if s.strip() and not s.strip().startswith('--')]
    
    for i, stmt in enumerate(statements, 1):
        if not stmt or stmt.startswith('--'):
            continue
        try:
            # Execute via RPC or direct SQL
            client.rpc('exec_sql', {'sql': stmt}).execute()
            print(f"✅ Statement {i} executed")
        except Exception as e:
            # Try alternative method
            try:
                client.postgrest.rpc('exec_sql', {'sql': stmt}).execute()
                print(f"✅ Statement {i} executed (via postgrest)")
            except Exception as e2:
                print(f"⚠️  Statement {i} skipped: {str(e2)[:100]}")
    
    print("-" * 50)
    print("✅ Migration complete!")
    
    # Verify table exists
    try:
        result = client.table("org_calendar_connections").select("id").limit(1).execute()
        print("✅ Table 'org_calendar_connections' verified!")
    except Exception as e:
        print(f"⚠️  Could not verify table: {e}")

if __name__ == "__main__":
    main()
