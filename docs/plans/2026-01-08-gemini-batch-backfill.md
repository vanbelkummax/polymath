# Gemini Batch Backfill Plan

**Date**: 2026-01-08
**Status**: Implementation Complete, Awaiting Pilot Run
**Author**: Claude Code

## Overview

This plan addresses backfilling the `passage_concepts` table using Gemini Batch API to:
1. Process remaining ~538,865 passages efficiently
2. Populate the `evidence` JSONB field (currently always NULL)
3. Maintain typed concepts compatible with existing schema

## Problem Statement

Current state (from audit 2026-01-08):
- Total passages: 550,006
- With concepts (llm_v2): 11,141 (2.0%)
- With concepts (haiku_v1): 360
- **Remaining**: 538,865 passages need concept extraction
- Evidence field is NULL for ALL existing records

Previous approach (local Ollama) is too slow (~500/hour = 45 days).

## Solution: Gemini Batch API

### Why Gemini Batch?
- **Cost**: $0.01875/M input + $0.075/M output (batch pricing)
- **Speed**: Async batch processing, no rate limits
- **Quality**: Flash-Lite is comparable to GPT-4 class models
- **Structured Output**: Native JSON schema enforcement

### Cost Estimate
- ~538,865 passages × ~450 input tokens = 242M input tokens
- ~538,865 passages × ~230 output tokens = 124M output tokens
- **Estimated total: $10-15 USD**

## Implementation

### Files Created

| File | Purpose |
|------|---------|
| `lib/gemini_batch.py` | Batch API wrapper, cost estimation |
| `scripts/backfill_passage_concepts_gemini_batch.py` | Main backfill driver |
| `Makefile` | Make targets for easy execution |
| `docs/runlogs/gemini_backfill_env.md` | Environment audit log |

### Database Schema

Uses existing `passage_concepts` table:
```sql
passage_concepts (
    passage_id UUID,
    concept_name TEXT,
    concept_type TEXT,  -- method, domain, model, etc.
    aliases JSONB,
    confidence REAL,
    extractor_model TEXT,  -- gemini-2.0-flash-lite
    extractor_version TEXT,  -- gemini_batch_v1
    evidence JSONB,  -- NEW: populated with quotes/context
    created_at TIMESTAMPTZ
)
```

### Evidence JSONB Format
```json
{
  "surface": "concept_name",
  "support": [{"quote": "...", "start": 123, "end": 456}],
  "context": "surrounding text (<=300 chars)",
  "source_text": "truncated passage (<=1200 chars)",
  "quality": {"confidence": 0.85, "notes": "extracted by gemini-2.0-flash-lite"}
}
```

## Usage

### Prerequisites
```bash
# Get API key from https://aistudio.google.com/apikey
export GEMINI_API_KEY=your_api_key
```

### Run Pilot (200 passages)
```bash
cd /home/user/polymath-repo
make gemini_passage_pilot
```

This will:
1. Process 200 random remaining passages
2. Insert concepts with evidence populated
3. Generate QC report in `docs/runlogs/`

### Check Status
```bash
make gemini_backfill_status
```

### Run Full Backfill
```bash
make gemini_passage_backfill
```

Note: Prompts for confirmation before starting.

### Resume After Crash
The script automatically resumes from the last checkpoint:
```bash
# Just run again - it will resume
make gemini_passage_backfill
```

### Start Fresh (No Resume)
```bash
python scripts/backfill_passage_concepts_gemini_batch.py --full --no-resume
```

## QC Gates

After each run, the script generates a QC report checking:
1. **Evidence coverage**: % of concepts with non-null evidence (target: >99%)
2. **Concepts per passage**: Mean, P10, P90 distribution
3. **Type distribution**: Sanity check vs existing llm_v2
4. **Garbage detection**: Flags concept names >60 chars

Reports saved to: `docs/runlogs/gemini_batch_qc_<timestamp>.md`

## Rollback

If something goes wrong:

```sql
-- Check how many rows were inserted
SELECT COUNT(*) FROM passage_concepts WHERE extractor_version = 'gemini_batch_v1';

-- Delete all gemini_batch_v1 rows (safe - doesn't affect other versions)
DELETE FROM passage_concepts WHERE extractor_version = 'gemini_batch_v1';

-- Reset migration checkpoint
DELETE FROM kb_migrations WHERE job_name LIKE '%gemini_batch_v1%';
```

## Merge Policy

Multiple extractor versions coexist in `passage_concepts`:
- `llm_v2` (qwen2.5:7b-instruct) - 11,141 passages
- `haiku_v1` (claude-haiku) - 360 passages
- `gemini_batch_v1` (gemini-2.0-flash-lite) - NEW

Query-time preference order (not storage-time):
```sql
-- Get best concepts per passage (prefer gemini > llm > haiku)
WITH ranked AS (
  SELECT *,
    ROW_NUMBER() OVER (
      PARTITION BY passage_id, concept_name
      ORDER BY CASE extractor_version
        WHEN 'gemini_batch_v1' THEN 1
        WHEN 'llm_v2' THEN 2
        WHEN 'haiku_v1' THEN 3
      END
    ) as rn
  FROM passage_concepts
  WHERE confidence >= 0.7
)
SELECT * FROM ranked WHERE rn = 1;
```

## Acceptance Criteria

Pilot must pass before full run:
1. ✅ Evidence non-null for >= 99% of inserted concepts
2. ✅ Avg concepts per passage between 4 and 14
3. ✅ JSON parse failure rate <= 1%
4. ✅ Resume works after intentional SIGINT

## Timeline

1. **Pilot** (immediate): 200 passages, ~2 minutes
2. **QC Review**: Verify evidence quality, concept types
3. **Full Run** (after approval): ~538k passages, ~$10-15, ~4-8 hours

## Appendix: Gemini API Setup

1. Go to https://aistudio.google.com/apikey
2. Create a new API key (free tier has limits)
3. For batch API access, you may need to enable billing
4. Set environment variable:
   ```bash
   export GEMINI_API_KEY=AIza...
   ```
5. Test connection:
   ```bash
   python -c "from lib.gemini_batch import get_genai_client; get_genai_client()"
   ```
