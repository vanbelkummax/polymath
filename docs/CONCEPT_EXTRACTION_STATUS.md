# Concept Extraction Backfill Status

**Last Updated:** 2026-01-09

## Current Progress

| Metric | Value |
|--------|-------|
| Total Passages | 668,107 |
| Processed | 235,761 (35.3%) |
| Remaining | 432,346 |
| Concept Rows | 1,519,338 |

## Approach

**Model:** `gemini-2.0-flash` (winner of A/B test vs gemini-2.5-flash-lite)
- A/B test result: 8% failure vs 68% failure
- See `scripts/canary_model_test.py` for test methodology

**Configuration:**
- 16 parallel workers
- 0.1s delay between requests
- ~250 passages/min throughput
- Schema with tight constraints to prevent truncation

**Key Files:**
- `lib/gemini_batch.py` - Core extraction library with JSON schema
- `scripts/backfill_passage_concepts_gemini_batch.py` - Main backfill script
- `scripts/canary_model_test.py` - A/B model comparison tool

## Quality Audit (2026-01-09)

### Data Integrity: PASS
- No orphan concepts
- No null required fields
- 96.1% evidence field coverage

### Coverage Metrics
| Metric | Value |
|--------|-------|
| Avg concepts/passage | 6.3 |
| Min/Max | 1 / 27 |
| P50 / P90 | 6 / 8 |

### Type Distribution
| Type | Count | Percent |
|------|-------|---------|
| domain | 807,514 | 53.1% |
| method | 172,189 | 11.3% |
| metric | 138,188 | 9.1% |
| technique | 118,395 | 7.8% |
| model | 117,144 | 7.7% |
| field | 47,116 | 3.1% |
| math_object | 38,604 | 2.5% |
| dataset | 29,349 | 1.9% |
| objective | 26,113 | 1.7% |
| algorithm | 20,991 | 1.4% |

### Top Concepts
1. gene_expression (6,376)
2. optimal_transport (3,751)
3. machine_learning (3,606)
4. deep_learning (3,041)
5. artificial_intelligence (2,620)
6. spatial_transcriptomics (1,407)

### Confidence Analysis
- Average: 0.81
- Low confidence (<0.5): 4,064 (0.3%)

### Current Failure Rate
- New workers: ~11% JSON parse failure
- Improved from ~50% before schema tightening

## Resume Instructions

```bash
cd /home/user/polymath-repo
export $(grep -v '^#' .env | xargs)

# Start 16 workers with optimized settings
for i in $(seq 0 15); do
  nohup python3 -u scripts/backfill_passage_concepts_gemini_batch.py \
    --full --worker-id $i --num-workers 16 --delay 0.1 \
    >> logs/gemini_backfill_w$i.log 2>&1 &
done

# Check status
python3 scripts/backfill_passage_concepts_gemini_batch.py --status

# Monitor logs
tail -f logs/gemini_backfill_w0.log
```

## Pause Instructions

```bash
pkill -f "backfill_passage_concepts_gemini_batch"
```

## Schema

```sql
-- passage_concepts table
CREATE TABLE passage_concepts (
    passage_id UUID REFERENCES passages(passage_id),
    concept_name TEXT NOT NULL,
    concept_type TEXT,
    aliases TEXT[],
    confidence REAL,
    evidence JSONB,
    extractor_model TEXT NOT NULL,
    extractor_version TEXT NOT NULL,
    created_at TIMESTAMP DEFAULT NOW(),
    PRIMARY KEY (passage_id, concept_name, extractor_version)
);
```

## JSON Schema (Gemini)

```python
CONCEPT_SCHEMA = {
    "type": "object",
    "properties": {
        "concepts": {
            "type": "array",
            "maxItems": 8,
            "items": {
                "type": "object",
                "properties": {
                    "name": {"type": "string", "maxLength": 60},
                    "type": {"type": "string", "enum": [...]},
                    "aliases": {"type": "array", "maxItems": 3},
                    "confidence": {"type": "number", "minimum": 0, "maximum": 1},
                    "evidence": {
                        "type": "object",
                        "properties": {
                            "quote": {"type": "string", "maxLength": 120},
                            "context": {"type": "string", "maxLength": 240}
                        }
                    }
                },
                "required": ["name", "type", "confidence"]
            }
        }
    }
}
```

## Legacy (Removed)

The following scripts were removed in favor of the Gemini approach:
- `backfill_passage_concepts_haiku.py`
- `orchestrate_haiku_backfill.py`
- `queued_backfill.py`
- `backfill_simple.py`
- `ingest_with_haiku.py`
