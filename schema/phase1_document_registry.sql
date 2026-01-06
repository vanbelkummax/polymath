-- Phase 1: Document Registry Schema
-- Creates canonical doc_id system with deterministic UUIDs
--
-- Version: HARDENING_2026-01-05
-- Prerequisite: Postgres database 'polymath' exists

-- 1. Create documents table (canonical source of truth)
CREATE TABLE IF NOT EXISTS documents (
    doc_id UUID PRIMARY KEY,              -- Deterministic UUIDv5 (NOT random)
    doi TEXT UNIQUE,                       -- DOI if known
    pmid TEXT UNIQUE,                      -- PubMed ID if known
    arxiv_id TEXT UNIQUE,                  -- arXiv ID (normalized, v1/v2 → base)
    title_hash TEXT UNIQUE,                -- SHA256 hash of normalized title (fallback dedup)
    title TEXT NOT NULL,                   -- Full title
    authors TEXT[],                        -- Author list
    year INTEGER,                          -- Publication year
    venue TEXT,                            -- Journal/conference name
    publication_type TEXT,                 -- 'primary', 'review', 'preprint', etc.
    parser_version TEXT,                   -- Track extraction method
    created_at TIMESTAMPTZ DEFAULT NOW(),
    updated_at TIMESTAMPTZ DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_documents_doi ON documents(doi) WHERE doi IS NOT NULL;
CREATE INDEX IF NOT EXISTS idx_documents_pmid ON documents(pmid) WHERE pmid IS NOT NULL;
CREATE INDEX IF NOT EXISTS idx_documents_arxiv ON documents(arxiv_id) WHERE arxiv_id IS NOT NULL;
CREATE INDEX IF NOT EXISTS idx_documents_year ON documents(year) WHERE year IS NOT NULL;
CREATE INDEX IF NOT EXISTS idx_documents_title_hash ON documents(title_hash);


-- 2. Create doc_aliases table (multi-identifier lookup)
CREATE TABLE IF NOT EXISTS doc_aliases (
    alias_id SERIAL PRIMARY KEY,
    doc_id UUID REFERENCES documents(doc_id) ON DELETE CASCADE NOT NULL,
    alias_type TEXT NOT NULL,              -- 'doi', 'pmid', 'arxiv', 'file_hash', 'legacy_id'
    alias_value TEXT NOT NULL,             -- The actual identifier value
    created_at TIMESTAMPTZ DEFAULT NOW(),
    UNIQUE(alias_type, alias_value)        -- Each identifier points to only one document
);

CREATE INDEX IF NOT EXISTS idx_doc_aliases_lookup ON doc_aliases(alias_type, alias_value);
CREATE INDEX IF NOT EXISTS idx_doc_aliases_doc ON doc_aliases(doc_id);


-- 3. Update artifacts table to link to documents
ALTER TABLE artifacts
    ADD COLUMN IF NOT EXISTS doc_id UUID REFERENCES documents(doc_id) ON DELETE SET NULL;

CREATE INDEX IF NOT EXISTS idx_artifacts_doc ON artifacts(doc_id);


-- 4. Create function to update updated_at timestamp
CREATE OR REPLACE FUNCTION update_updated_at_column()
RETURNS TRIGGER AS $$
BEGIN
    NEW.updated_at = NOW();
    RETURN NEW;
END;
$$ LANGUAGE plpgsql;

CREATE TRIGGER update_documents_updated_at
    BEFORE UPDATE ON documents
    FOR EACH ROW
    EXECUTE FUNCTION update_updated_at_column();


-- 5. Grant permissions
GRANT ALL ON documents TO polymath;
GRANT ALL ON doc_aliases TO polymath;
GRANT USAGE, SELECT ON SEQUENCE doc_aliases_alias_id_seq TO polymath;


-- Verification queries
\echo 'Document Registry Schema Created'
\echo ''
\echo 'Verification:'
SELECT
    'documents' as table_name,
    COUNT(*) as row_count
FROM documents
UNION ALL
SELECT
    'doc_aliases' as table_name,
    COUNT(*) as row_count
FROM doc_aliases;
