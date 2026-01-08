# Passage Migration Safety Guide

## Current Status (2026-01-07)

**Migration Phase**: Code chunks → Passage extraction next
**Hardening Status**: ✅ Quality-gated evidence extraction ready
**Commits**:
- `cad36b7`: Harden passage evidence extraction with quality gating + support typing
- `00562b3`: Fix model names and add quality/support tracking

---

## Safety Protocol: Chunk→Passage Transition

### Why Stop and Restart?

When switching from code chunks to passages, you MUST stop the migrator and restart with `--resume` to ensure:
1. New quality-gated extraction code is loaded (no stale Python process)
2. Correct model names are used (`qwen2.5:7b-instruct`, not `qwen2.5:3b`)
3. Evidence JSONB contract is enforced

### Commands

#### 1. Stop Current Migrator (when chunks finish)

Find the process:
```bash
ps aux | grep backfill_chunk_concepts
# Or check for running python processes
ps aux | grep "python.*backfill"
```

Stop gracefully:
```bash
kill -SIGTERM <pid>
# Or if managed by nohup:
pkill -f backfill_chunk_concepts_llm
```

#### 2. Verify Checkpoint State

```bash
psql -U polymath -d polymath -c "
SELECT job_name, status, cursor_position, items_processed, updated_at
FROM kb_migrations
WHERE job_name LIKE '%chunk%' OR job_name LIKE '%passage%'
ORDER BY updated_at DESC
LIMIT 5;
"
```

Expected: Chunk job should show `status='completed'` or `status='running'` with recent timestamp.

#### 3. Set Environment Variables

```bash
export LOCAL_LLM_FAST="qwen2.5:7b-instruct"
export LOCAL_LLM_HEAVY="qwen2.5:14b-instruct"
export NEO4J_PASSWORD="polymathic2026"

# Verify Ollama models exist
ollama list | grep qwen2.5
```

Expected output:
```
qwen2.5:7b-instruct   <model_id>   4.7 GB   ...
qwen2.5:14b-instruct  <model_id>   9.0 GB   ...
```

#### 4. Restart with Passage-Only Mode

```bash
cd /home/user/polymath-repo

nohup python3 scripts/backfill_chunk_concepts_llm.py \
  --extractor-version llm_v3_quality_gated \
  --passages-only \
  --resume \
  > logs/backfill_passages_$(date +%Y%m%d_%H%M%S).log 2>&1 &

# Save PID
echo $! > /tmp/passage_backfill.pid
```

#### 5. Monitor Progress

**Tail logs:**
```bash
tail -f logs/backfill_passages_*.log
```

**Expected output every 100 passages:**
```
Processed 100 passages | Quality: {'clean': 45, 'glued': 30, 'marginal': 20, 'no_space': 5} |
Support: {'literal': 120, 'normalized': 80, 'none': 50, 'inferred': 10} |
Grant-grade (literal): 120/260 (46.2%)
```

**Check database progress:**
```bash
psql -U polymath -d polymath -c "
SELECT
  COUNT(*) AS total_concepts,
  COUNT(CASE WHEN evidence->>'support' = 'literal' THEN 1 END) AS literal,
  COUNT(CASE WHEN evidence->>'support' = 'normalized' THEN 1 END) AS normalized,
  COUNT(CASE WHEN evidence->>'support' = 'none' THEN 1 END) AS none,
  COUNT(CASE WHEN evidence->>'support' = 'inferred' THEN 1 END) AS inferred
FROM passage_concepts
WHERE extractor_version = 'llm_v3_quality_gated';
"
```

---

## Quality Acceptance Criteria

⚠️ **CRITICAL UNDERSTANDING**: Quality gating MEANS low overall literal support is expected and correct.

**Why**: If 40% of passages are `no_space` + 25% are `glued`, those route to `support="none"` by design. This protects audit integrity.

### What Actually Matters

1. **Corpus quality distribution** (passage-level):
   - How many passages are clean vs malformed?
   - Expected: 30-50% clean, 20-40% glued, 10-30% no_space

2. **Within clean passages, grant-grade yield**:
   - What % of clean passages produce ≥1 grant-grade concept?
   - Target: ≥40% of clean passages yield grant-grade concepts

3. **Total grant-grade concept count**:
   - Absolute number of citable concepts across all 532K passages
   - Even 5% of passages with 2 concepts each = 53K citable concepts

### Expected Metrics (REVISED)

| Metric | Target | Notes |
|--------|--------|-------|
| **Overall literal support** | 10-30% | LOW IS EXPECTED (quality gating working) |
| **Clean passage %** | 30-50% | From quality distribution |
| **Grant-grade yield (clean only)** | ≥40% | % of clean passages with ≥1 citable concept |
| **Total grant-grade concepts** | ≥20K | Absolute count matters more than % |

### Diagnostic Queries (Use These)

**A) Passage quality distribution:**
```sql
SELECT
  evidence->'quality'->>'label' AS quality,
  COUNT(DISTINCT passage_id) AS passages,
  ROUND(AVG((evidence->'quality'->>'score')::float)::numeric, 3) AS avg_score
FROM passage_concepts
WHERE extractor_version = 'llm_v3_quality_gated'
GROUP BY quality
ORDER BY passages DESC;
```

**B) Grant-grade yield by quality:**
```sql
WITH per_passage AS (
  SELECT
    passage_id,
    evidence->'quality'->>'label' AS quality,
    BOOL_OR(evidence->>'support'='literal' AND confidence>=0.8 AND (evidence->'quality'->>'score')::float>=0.5) AS has_grant_grade
  FROM passage_concepts
  WHERE extractor_version='llm_v3_quality_gated'
  GROUP BY passage_id, quality
)
SELECT
  quality,
  COUNT(*) AS passages,
  SUM(has_grant_grade::int) AS with_grant_grade,
  ROUND((SUM(has_grant_grade::int)::float / COUNT(*))::numeric, 3) AS yield_rate
FROM per_passage
GROUP BY quality
ORDER BY passages DESC;
```

**C) Total grant-grade concepts:**
```sql
SELECT COUNT(*) AS grant_grade_concepts
FROM passage_concepts
WHERE extractor_version='llm_v3_quality_gated'
  AND evidence->>'support'='literal'
  AND confidence >= 0.8
  AND (evidence->'quality'->>'score')::float >= 0.5;
```

### Red Flags (When to Stop)

- Clean passage % < 20% (suggests PDF extraction is completely broken)
- Grant-grade yield (clean only) < 20% (suggests LLM or validation broken)
- Total grant-grade concepts < 10K after full run (insufficient for grants)

---

## Hard Invariants (Verified in Tests)

These MUST be true in the code:

1. **Literal support ONLY if exact substring:**
   ```python
   if (surface in raw_text and context in raw_text and surface in context):
       return "literal"
   ```
   Location: `lib/local_extractor.py:664-669`

2. **Quality gating triggers fallback:**
   ```python
   if quality['score'] < 0.5 or quality['label'] in ('no_space', 'glued'):
       return self._extract_canonical_only(text, quality)
   ```
   Location: `lib/local_extractor.py:367-370`

3. **Canonical-only produces support="none":**
   ```python
   'evidence': {
       'surface': None,
       'context': None,
       'support': 'none',
       ...
   }
   ```
   Location: `lib/local_extractor.py:553-559`

---

## Emergency Stop

If grant-grade rate drops below 20% or errors spike:

```bash
# Stop immediately
pkill -f backfill_chunk_concepts_llm

# Check what went wrong
tail -100 logs/backfill_passages_*.log

# Run diagnostic sentinel
python3 scripts/passage_sentinel_v3.py --samples 200

# If models missing, pull them
ollama pull qwen2.5:7b-instruct
ollama pull qwen2.5:14b-instruct
```

---

## Citation Safety Reminders

**For grants/papers - Query ONLY literal support:**
```sql
SELECT pc.concept_name, pc.confidence, pc.evidence->>'surface'
FROM passage_concepts pc
WHERE
  pc.evidence->>'support' = 'literal'
  AND pc.confidence >= 0.8
  AND (pc.evidence->'quality'->>'score')::float >= 0.5
  AND pc.extractor_version = 'llm_v3_quality_gated'
ORDER BY pc.confidence DESC;
```

**For discovery - All concepts allowed:**
```sql
SELECT pc.concept_name, pc.evidence->>'support' AS support_type
FROM passage_concepts pc
WHERE pc.extractor_version = 'llm_v3_quality_gated'
ORDER BY pc.confidence DESC;
```

---

## Post-Migration Validation

After passages complete, run:

```bash
# Full quality report
python3 scripts/passage_sentinel_v3.py --samples 500

# Database stats
psql -U polymath -d polymath -c "
SELECT
  extractor_version,
  COUNT(*) AS total,
  AVG((evidence->'quality'->>'score')::float) AS avg_quality,
  COUNT(CASE WHEN evidence->>'support' = 'literal' THEN 1 END)::float / COUNT(*) AS literal_rate
FROM passage_concepts
GROUP BY extractor_version
ORDER BY extractor_version;
"
```

Expected: `literal_rate` between 0.30 and 0.60 (30-60%).

---

## Summary

1. ✅ **Oracles verified**: Code, tests, invariants all real
2. ✅ **Models fixed**: Defaults now `qwen2.5:7b-instruct` and `qwen2.5:14b-instruct`
3. ✅ **Tracking added**: Periodic quality/support reporting every 100 passages
4. ✅ **Safety protocol**: Stop → set env vars → restart with `--resume`
5. ✅ **Validation ready**: Sentinel v3 + SQL queries for monitoring

**Confidence**: 0.95 (all oracles passed, invariants verified)
