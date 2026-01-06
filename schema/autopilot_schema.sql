-- ============================================================
-- POLYMATH AUTOPILOT SCHEMA EXTENSION
-- Adds tables for: ingest tracking, source discovery, downloads
-- ============================================================

-- Source items discovered from APIs
CREATE TABLE IF NOT EXISTS source_items (
    id SERIAL PRIMARY KEY,
    source TEXT NOT NULL,  -- arxiv, biorxiv, europepmc, github, huggingface
    external_id TEXT NOT NULL,
    url TEXT,
    title TEXT,
    authors TEXT[],
    published_at TIMESTAMPTZ,
    doi TEXT,
    sha256 TEXT,
    status TEXT DEFAULT 'discovered',  -- discovered, queued, downloaded, ingested, failed, needs_user
    last_error TEXT,
    meta_json JSONB,
    created_at TIMESTAMPTZ DEFAULT NOW(),
    updated_at TIMESTAMPTZ DEFAULT NOW(),
    UNIQUE(source, external_id)
);

-- Download jobs (idempotent)
CREATE TABLE IF NOT EXISTS download_jobs (
    id SERIAL PRIMARY KEY,
    source_item_id INTEGER REFERENCES source_items(id),
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

-- Needs user tickets (paywalled content)
CREATE TABLE IF NOT EXISTS needs_user_tickets (
    id SERIAL PRIMARY KEY,
    source_item_id INTEGER REFERENCES source_items(id),
    doi TEXT,
    title TEXT,
    url TEXT,
    reason TEXT,  -- paywalled, requires_login, captcha
    resolved BOOLEAN DEFAULT FALSE,
    created_at TIMESTAMPTZ DEFAULT NOW()
);

-- Eval results for tracking system performance
CREATE TABLE IF NOT EXISTS eval_results (
    id SERIAL PRIMARY KEY,
    run_id UUID REFERENCES ingest_runs(run_id),
    eval_suite TEXT,
    score FLOAT,
    details_json JSONB,
    created_at TIMESTAMPTZ DEFAULT NOW()
);

-- Add columns to artifacts if not present
DO $$
BEGIN
    IF NOT EXISTS (SELECT 1 FROM information_schema.columns
                   WHERE table_name = 'artifacts' AND column_name = 'source_item_id') THEN
        ALTER TABLE artifacts ADD COLUMN source_item_id INTEGER REFERENCES source_items(id);
    END IF;

    IF NOT EXISTS (SELECT 1 FROM information_schema.columns
                   WHERE table_name = 'artifacts' AND column_name = 'ingest_run_id') THEN
        ALTER TABLE artifacts ADD COLUMN ingest_run_id UUID REFERENCES ingest_runs(run_id);
    END IF;

    IF NOT EXISTS (SELECT 1 FROM information_schema.columns
                   WHERE table_name = 'artifacts' AND column_name = 'ingestion_method') THEN
        ALTER TABLE artifacts ADD COLUMN ingestion_method TEXT;
    END IF;
END $$;

-- Indexes for fast lookups
CREATE INDEX IF NOT EXISTS idx_source_items_status ON source_items(status);
CREATE INDEX IF NOT EXISTS idx_source_items_source ON source_items(source);
CREATE INDEX IF NOT EXISTS idx_source_items_doi ON source_items(doi);
CREATE INDEX IF NOT EXISTS idx_download_jobs_status ON download_jobs(status);
CREATE INDEX IF NOT EXISTS idx_ingest_runs_status ON ingest_runs(status);
CREATE INDEX IF NOT EXISTS idx_needs_user_tickets_resolved ON needs_user_tickets(resolved);

-- Ensure FTS extension and chunk triggers exist
CREATE EXTENSION IF NOT EXISTS pg_trgm;

-- Recreate chunk FTS trigger (in case it was missing)
CREATE OR REPLACE FUNCTION update_chunk_tsv() RETURNS TRIGGER AS $$
BEGIN
    NEW.content_tsv := to_tsvector('english', COALESCE(NEW.content, ''));
    RETURN NEW;
END;
$$ LANGUAGE plpgsql;

DROP TRIGGER IF EXISTS chunk_tsv_update ON chunks;
CREATE TRIGGER chunk_tsv_update
    BEFORE INSERT OR UPDATE ON chunks
    FOR EACH ROW EXECUTE FUNCTION update_chunk_tsv();

-- Grant permissions
GRANT ALL ON ALL TABLES IN SCHEMA public TO polymath;
GRANT ALL ON ALL SEQUENCES IN SCHEMA public TO polymath;
