-- Migration: Add Google Calendar multi-attendee support
-- Date: 2026-01-07

-- Add google_calendar_email column to customers table
ALTER TABLE customers 
ADD COLUMN IF NOT EXISTS google_calendar_email TEXT;

-- Create org_calendar_connections table
CREATE TABLE IF NOT EXISTS org_calendar_connections (
  id uuid PRIMARY KEY DEFAULT uuid_generate_v4(),
  org_id uuid REFERENCES organizations(id) NOT NULL,
  attendee_name text NOT NULL,           -- Display name for agent (e.g., "Dr. João Silva")
  attendee_email text NOT NULL,          -- Google Calendar ID (email)
  calendar_id text,                       -- Optional: specific calendar ID (NULL = use email)
  is_default boolean DEFAULT false,       -- Auto-select when only one or no preference
  calendar_type text DEFAULT 'worker' CHECK (calendar_type IN ('worker', 'room', 'shared')),
  working_hours jsonb DEFAULT '{
    "monday": {"start": "09:00", "end": "18:00"},
    "tuesday": {"start": "09:00", "end": "18:00"},
    "wednesday": {"start": "09:00", "end": "18:00"},
    "thursday": {"start": "09:00", "end": "18:00"},
    "friday": {"start": "09:00", "end": "18:00"}
  }'::jsonb,
  created_at timestamp with time zone DEFAULT timezone('utc'::text, now()) NOT NULL
);

-- Enable RLS
ALTER TABLE org_calendar_connections ENABLE ROW LEVEL SECURITY;

-- Create RLS policy
DO $$ 
BEGIN
  IF NOT EXISTS (
    SELECT 1 FROM pg_policies 
    WHERE tablename = 'org_calendar_connections' 
    AND policyname = 'Calendar Connections Org Isolation'
  ) THEN
    CREATE POLICY "Calendar Connections Org Isolation" 
    ON org_calendar_connections 
    FOR ALL 
    USING (org_id = get_my_org_id() OR is_super_admin());
  END IF;
END $$;

-- Create indexes
CREATE INDEX IF NOT EXISTS idx_org_calendars_org ON org_calendar_connections(org_id);
CREATE INDEX IF NOT EXISTS idx_org_calendars_name ON org_calendar_connections(org_id, attendee_name);
CREATE UNIQUE INDEX IF NOT EXISTS idx_org_calendars_email ON org_calendar_connections(org_id, attendee_email);
