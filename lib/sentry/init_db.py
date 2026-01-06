#!/usr/bin/env python3
"""
Initialize Postgres schema for Literature Sentry.

Creates tables for:
- source_items: Discovery ledger
- download_jobs: Fetch tracking
- ingest_runs: Atomic commit tracking
- needs_user_tickets: Paywalled items queue
- eval_results: System performance

Usage:
    python3 -m lib.sentry.init_db
    python3 -m lib.sentry.init_db --check  # Verify schema
"""

import sys
from pathlib import Path

try:
    import psycopg2
    from psycopg2.extras import Json
except ImportError:
    print("Installing psycopg2...")
    import subprocess
    subprocess.run([sys.executable, "-m", "pip", "install", "psycopg2-binary", "-q"])
    import psycopg2
    from psycopg2.extras import Json


SCHEMA_SQL = """
-- ============================================================
-- LITERATURE SENTRY SCHEMA
-- Extends Polymath autopilot schema with enhanced features
-- ============================================================

-- Source items discovered from APIs (idempotent)
CREATE TABLE IF NOT EXISTS source_items (
    id SERIAL PRIMARY KEY,
    source TEXT NOT NULL,  -- europepmc, arxiv, biorxiv, github, youtube
    external_id TEXT NOT NULL,
    url TEXT,
    title TEXT,
    authors TEXT[],
    published_at TIMESTAMPTZ,
    doi TEXT,
    sha256 TEXT,
    status TEXT DEFAULT 'discovered',  -- discovered, queued, downloaded, ingested, failed, needs_user
    last_error TEXT,
    meta_json JSONB DEFAULT '{}'::jsonb,
    created_at TIMESTAMPTZ DEFAULT NOW(),
    updated_at TIMESTAMPTZ DEFAULT NOW(),
    UNIQUE(source, external_id)
);

-- Download jobs (idempotent, tracks fetch attempts)
CREATE TABLE IF NOT EXISTS download_jobs (
    id SERIAL PRIMARY KEY,
    source_item_id INTEGER REFERENCES source_items(id) ON DELETE CASCADE,
    attempt INTEGER DEFAULT 1,
    status TEXT DEFAULT 'pending',  -- pending, running, completed, failed, quarantined
    file_path TEXT,
    file_size BIGINT,
    sha256 TEXT,
    created_at TIMESTAMPTZ DEFAULT NOW(),
    updated_at TIMESTAMPTZ DEFAULT NOW()
);

-- Ingest runs (for crash recovery and atomic commits)
CREATE TABLE IF NOT EXISTS ingest_runs (
    run_id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    started_at TIMESTAMPTZ DEFAULT NOW(),
    finished_at TIMESTAMPTZ,
    status TEXT DEFAULT 'started',  -- started, committed, failed, rolled_back
    source_items_processed INTEGER DEFAULT 0,
    chunks_added INTEGER DEFAULT 0,
    concepts_linked INTEGER DEFAULT 0,
    git_sha TEXT,
    counts_json JSONB,
    error_log TEXT
);

-- Needs user tickets (paywalled content requiring manual download)
CREATE TABLE IF NOT EXISTS needs_user_tickets (
    id SERIAL PRIMARY KEY,
    source_item_id INTEGER REFERENCES source_items(id) ON DELETE CASCADE,
    doi TEXT,
    title TEXT,
    url TEXT,
    reason TEXT,  -- paywalled, requires_login, captcha, region_blocked
    resolved BOOLEAN DEFAULT FALSE,
    resolved_at TIMESTAMPTZ,
    created_at TIMESTAMPTZ DEFAULT NOW()
);

-- Eval results for tracking system performance
CREATE TABLE IF NOT EXISTS eval_results (
    id SERIAL PRIMARY KEY,
    run_id UUID REFERENCES ingest_runs(run_id) ON DELETE SET NULL,
    eval_suite TEXT,
    score FLOAT,
    details_json JSONB,
    created_at TIMESTAMPTZ DEFAULT NOW()
);

-- ============================================================
-- GENERATED COLUMNS for fast querying
-- ============================================================

-- Add priority_score as generated column (extracted from meta_json)
DO $$
BEGIN
    IF NOT EXISTS (
        SELECT 1 FROM information_schema.columns
        WHERE table_name = 'source_items' AND column_name = 'priority_score'
    ) THEN
        ALTER TABLE source_items ADD COLUMN priority_score FLOAT
            GENERATED ALWAYS AS (
                COALESCE((meta_json->>'priority_score')::float, 0.0)
            ) STORED;
    END IF;
END $$;

-- Add concept_domains as regular column (populated by trigger)
DO $$
BEGIN
    IF NOT EXISTS (
        SELECT 1 FROM information_schema.columns
        WHERE table_name = 'source_items' AND column_name = 'concept_domains'
    ) THEN
        ALTER TABLE source_items ADD COLUMN concept_domains TEXT[] DEFAULT '{}';
    END IF;
END $$;

-- Trigger to sync concept_domains from meta_json
CREATE OR REPLACE FUNCTION sync_concept_domains()
RETURNS TRIGGER AS $$
BEGIN
    NEW.concept_domains := ARRAY(
        SELECT jsonb_array_elements_text(
            COALESCE(NEW.meta_json->'concept_domains', '[]'::jsonb)
        )
    );
    RETURN NEW;
END;
$$ LANGUAGE plpgsql;

DROP TRIGGER IF EXISTS sync_concept_domains_trigger ON source_items;
CREATE TRIGGER sync_concept_domains_trigger
    BEFORE INSERT OR UPDATE ON source_items
    FOR EACH ROW EXECUTE FUNCTION sync_concept_domains();

-- Add is_hidden_gem flag
DO $$
BEGIN
    IF NOT EXISTS (
        SELECT 1 FROM information_schema.columns
        WHERE table_name = 'source_items' AND column_name = 'is_hidden_gem'
    ) THEN
        ALTER TABLE source_items ADD COLUMN is_hidden_gem BOOLEAN
            GENERATED ALWAYS AS (
                COALESCE((meta_json->>'is_hidden_gem')::boolean, false)
            ) STORED;
    END IF;
END $$;

-- ============================================================
-- INDEXES for fast lookups
-- ============================================================

CREATE INDEX IF NOT EXISTS idx_source_items_status ON source_items(status);
CREATE INDEX IF NOT EXISTS idx_source_items_source ON source_items(source);
CREATE INDEX IF NOT EXISTS idx_source_items_doi ON source_items(doi);
CREATE INDEX IF NOT EXISTS idx_source_items_priority ON source_items(priority_score DESC NULLS LAST);
CREATE INDEX IF NOT EXISTS idx_source_items_concepts ON source_items USING GIN(concept_domains);
CREATE INDEX IF NOT EXISTS idx_source_items_hidden_gem ON source_items(is_hidden_gem) WHERE is_hidden_gem = true;
CREATE INDEX IF NOT EXISTS idx_source_items_created ON source_items(created_at DESC);

CREATE INDEX IF NOT EXISTS idx_download_jobs_status ON download_jobs(status);
CREATE INDEX IF NOT EXISTS idx_download_jobs_source_item ON download_jobs(source_item_id);

CREATE INDEX IF NOT EXISTS idx_ingest_runs_status ON ingest_runs(status);
CREATE INDEX IF NOT EXISTS idx_ingest_runs_started ON ingest_runs(started_at DESC);

CREATE INDEX IF NOT EXISTS idx_needs_user_tickets_resolved ON needs_user_tickets(resolved);
CREATE INDEX IF NOT EXISTS idx_needs_user_tickets_created ON needs_user_tickets(created_at DESC);

-- ============================================================
-- TRIGGER for updated_at
-- ============================================================

CREATE OR REPLACE FUNCTION update_updated_at()
RETURNS TRIGGER AS $$
BEGIN
    NEW.updated_at = NOW();
    RETURN NEW;
END;
$$ LANGUAGE plpgsql;

DROP TRIGGER IF EXISTS source_items_updated_at ON source_items;
CREATE TRIGGER source_items_updated_at
    BEFORE UPDATE ON source_items
    FOR EACH ROW EXECUTE FUNCTION update_updated_at();

DROP TRIGGER IF EXISTS download_jobs_updated_at ON download_jobs;
CREATE TRIGGER download_jobs_updated_at
    BEFORE UPDATE ON download_jobs
    FOR EACH ROW EXECUTE FUNCTION update_updated_at();

-- ============================================================
-- LINK to existing artifacts table (if exists)
-- ============================================================

DO $$
BEGIN
    IF EXISTS (SELECT 1 FROM information_schema.tables WHERE table_name = 'artifacts') THEN
        IF NOT EXISTS (
            SELECT 1 FROM information_schema.columns
            WHERE table_name = 'artifacts' AND column_name = 'source_item_id'
        ) THEN
            ALTER TABLE artifacts ADD COLUMN source_item_id INTEGER
                REFERENCES source_items(id) ON DELETE SET NULL;
        END IF;

        IF NOT EXISTS (
            SELECT 1 FROM information_schema.columns
            WHERE table_name = 'artifacts' AND column_name = 'ingest_run_id'
        ) THEN
            ALTER TABLE artifacts ADD COLUMN ingest_run_id UUID
                REFERENCES ingest_runs(run_id) ON DELETE SET NULL;
        END IF;

        IF NOT EXISTS (
            SELECT 1 FROM information_schema.columns
            WHERE table_name = 'artifacts' AND column_name = 'ingestion_method'
        ) THEN
            ALTER TABLE artifacts ADD COLUMN ingestion_method TEXT;
        END IF;
    END IF;
END $$;

-- Grant permissions
GRANT ALL ON ALL TABLES IN SCHEMA public TO polymath;
GRANT ALL ON ALL SEQUENCES IN SCHEMA public TO polymath;
"""


def get_connection():
    """Get Postgres connection using peer auth."""
    return psycopg2.connect(
        dbname='polymath',
        user='polymath',
        host='/var/run/postgresql'
    )


def init_schema():
    """Initialize the database schema."""
    print("Initializing Literature Sentry schema...")

    conn = get_connection()
    cur = conn.cursor()

    try:
        cur.execute(SCHEMA_SQL)
        conn.commit()
        print("Schema initialized successfully!")

        # Verify
        cur.execute("""
            SELECT table_name FROM information_schema.tables
            WHERE table_schema = 'public'
            AND table_name IN ('source_items', 'download_jobs', 'ingest_runs',
                               'needs_user_tickets', 'eval_results')
            ORDER BY table_name
        """)
        tables = [row[0] for row in cur.fetchall()]
        print(f"Tables created: {', '.join(tables)}")

    except Exception as e:
        conn.rollback()
        print(f"Error initializing schema: {e}")
        raise
    finally:
        cur.close()
        conn.close()


def check_schema():
    """Verify schema is properly set up."""
    print("Checking Literature Sentry schema...")

    conn = get_connection()
    cur = conn.cursor()

    try:
        # Check tables
        cur.execute("""
            SELECT table_name FROM information_schema.tables
            WHERE table_schema = 'public'
            AND table_name IN ('source_items', 'download_jobs', 'ingest_runs',
                               'needs_user_tickets', 'eval_results')
        """)
        tables = {row[0] for row in cur.fetchall()}

        expected = {'source_items', 'download_jobs', 'ingest_runs',
                    'needs_user_tickets', 'eval_results'}

        if tables == expected:
            print("All tables present!")
        else:
            missing = expected - tables
            print(f"Missing tables: {missing}")
            return False

        # Check generated columns on source_items
        cur.execute("""
            SELECT column_name FROM information_schema.columns
            WHERE table_name = 'source_items'
            AND column_name IN ('priority_score', 'concept_domains', 'is_hidden_gem')
        """)
        columns = {row[0] for row in cur.fetchall()}

        if 'priority_score' in columns:
            print("Generated columns present!")
        else:
            print("Warning: Generated columns missing (run init to create)")

        # Count rows
        cur.execute("SELECT COUNT(*) FROM source_items")
        source_count = cur.fetchone()[0]

        cur.execute("SELECT COUNT(*) FROM needs_user_tickets WHERE resolved = false")
        ticket_count = cur.fetchone()[0]

        print(f"Source items: {source_count}")
        print(f"Pending tickets: {ticket_count}")

        return True

    finally:
        cur.close()
        conn.close()


def main():
    import argparse
    parser = argparse.ArgumentParser(description="Initialize Literature Sentry database")
    parser.add_argument("--check", action="store_true", help="Check schema without modifying")
    args = parser.parse_args()

    if args.check:
        check_schema()
    else:
        init_schema()
        check_schema()


if __name__ == "__main__":
    main()
