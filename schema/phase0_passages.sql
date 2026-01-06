-- Phase 0: Evidence Spans Schema (Citation-Eligible Passages)
-- Creates new passages table for evidence-bound citations
--
-- Version: HARDENING_2026-01-05
-- Prerequisite: Phase 1 (Document Registry) complete
--
-- CRITICAL DESIGN:
-- - Page-local char offsets (not global) → avoids overlap bugs
-- - Claim-agnostic evidence_spans → prevents duplication explosion
-- - ON DELETE CASCADE → prevents orphan passages/FK violations

-- 1. Create passages table (citation-eligible, provenance-first)
CREATE TABLE IF NOT EXISTS passages (
    passage_id UUID PRIMARY KEY,              -- Deterministic: uuid5(doc_id + page_num + char_start + char_end)
    doc_id UUID REFERENCES documents(doc_id) ON DELETE CASCADE NOT NULL,
    page_num INTEGER NOT NULL,
    page_char_start INTEGER NOT NULL,         -- Char offset WITHIN page (not global)
    page_char_end INTEGER NOT NULL,           -- Char offset WITHIN page
    section TEXT,                             -- 'abstract', 'methods', 'results', 'discussion', 'references'
    passage_text TEXT NOT NULL,
    quality_score FLOAT DEFAULT 1.0,          -- Section-based quality (1.0 = methods/results, 0.3 = references)
    parser_version TEXT,                      -- Track extraction method ('pdfplumber_v1', 'pymupdf_fallback', 'ocr')
    created_at TIMESTAMPTZ DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_passages_doc ON passages(doc_id);
CREATE INDEX IF NOT EXISTS idx_passages_page ON passages(doc_id, page_num);
CREATE INDEX IF NOT EXISTS idx_passages_quality ON passages(quality_score) WHERE quality_score >= 0.7;
CREATE INDEX IF NOT EXISTS idx_passages_section ON passages(section);


-- 2. Create evidence_spans table (CLAIM-AGNOSTIC, reusable)
CREATE TABLE IF NOT EXISTS evidence_spans (
    span_id SERIAL PRIMARY KEY,
    passage_id UUID REFERENCES passages(passage_id) ON DELETE CASCADE NOT NULL,
    doc_id UUID REFERENCES documents(doc_id) ON DELETE CASCADE NOT NULL,  -- Denormalized for fast lookup
    page_num INTEGER NOT NULL,
    sentence_idx INTEGER,                     -- Sentence index within passage (0-indexed)
    span_char_start INTEGER NOT NULL,         -- Char offset within PASSAGE (not page, not doc)
    span_char_end INTEGER NOT NULL,
    span_text TEXT NOT NULL,                  -- The actual sentence/span
    section TEXT,
    quality_score FLOAT,
    entailment_score FLOAT,                   -- NLI entailment score (for filtering/debugging)
    contradiction_score FLOAT,                -- NLI contradiction score (fail-closed if >0.7)
    created_at TIMESTAMPTZ DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_spans_passage ON evidence_spans(passage_id);
CREATE INDEX IF NOT EXISTS idx_spans_doc ON evidence_spans(doc_id);
CREATE UNIQUE INDEX IF NOT EXISTS idx_spans_unique ON evidence_spans(passage_id, sentence_idx);
CREATE INDEX IF NOT EXISTS idx_spans_entailment ON evidence_spans(entailment_score) WHERE entailment_score > 0.5;


-- 3. Create claim_evidence_links table (PQE responses reference these)
CREATE TABLE IF NOT EXISTS claim_evidence_links (
    link_id SERIAL PRIMARY KEY,
    response_id UUID,                         -- Links to stored PQE response (optional)
    claim_id TEXT NOT NULL,                   -- "P1", "P2", "mechanism_row_3"
    claim_text TEXT NOT NULL,
    span_id INTEGER REFERENCES evidence_spans(span_id) ON DELETE CASCADE NOT NULL,
    entailment_score FLOAT NOT NULL,          -- NLI score (0-1)
    support_type TEXT,                        -- 'strong' (≥0.8), 'partial' (0.6-0.8), 'weak' (0.5-0.6)
    created_at TIMESTAMPTZ DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_links_response ON claim_evidence_links(response_id);
CREATE INDEX IF NOT EXISTS idx_links_claim ON claim_evidence_links(claim_id);
CREATE INDEX IF NOT EXISTS idx_links_span ON claim_evidence_links(span_id);
CREATE INDEX IF NOT EXISTS idx_links_support ON claim_evidence_links(support_type);


-- 4. Mark legacy chunks (don't modify structure, just flag)
ALTER TABLE chunks ADD COLUMN IF NOT EXISTS is_citation_eligible BOOLEAN DEFAULT FALSE;

-- Set all existing chunks as NOT citation-eligible (legacy data)
UPDATE chunks SET is_citation_eligible = FALSE WHERE is_citation_eligible IS NULL;


-- 5. Add citation verification to artifacts (backward compat)
ALTER TABLE artifacts
    ADD COLUMN IF NOT EXISTS verified_doi BOOLEAN DEFAULT FALSE,
    ADD COLUMN IF NOT EXISTS publication_type TEXT;


-- 6. Grant permissions
GRANT ALL ON passages TO polymath;
GRANT ALL ON evidence_spans TO polymath;
GRANT ALL ON claim_evidence_links TO polymath;
GRANT USAGE, SELECT ON SEQUENCE evidence_spans_span_id_seq TO polymath;
GRANT USAGE, SELECT ON SEQUENCE claim_evidence_links_link_id_seq TO polymath;


-- Verification queries
\echo 'Evidence Spans Schema Created'
\echo ''
\echo 'Verification:'
SELECT
    'passages' as table_name,
    COUNT(*) as row_count
FROM passages
UNION ALL
SELECT
    'evidence_spans' as table_name,
    COUNT(*) as row_count
FROM evidence_spans
UNION ALL
SELECT
    'claim_evidence_links' as table_name,
    COUNT(*) as row_count
FROM claim_evidence_links;

\echo ''
\echo 'Next: Implement Enhanced PDF Parser (Phase 1)'
