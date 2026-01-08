# Gemini Batch Backfill - Environment Discovery Log
**Generated**: 2026-01-08 12:15 CST

## System Environment

| Check | Result |
|-------|--------|
| Git status | Clean (some untracked files) |
| Python version | 3.12.11 |
| pip3 | /home/user/miniforge3/bin/pip3 |
| uv | /home/user/.local/bin/uv |
| DB connectivity | ✅ Connected |

## Database State

### kb_migrations Records
```
              job_name              |           cursor_position            | cursor_type |          updated_at           |  status   | items_processed
------------------------------------+--------------------------------------+-------------+-------------------------------+-----------+-----------------
 backfill_passage_concepts_haiku_v1 | 00363a99-b482-5d34-ab22-1659025f8a74 | passage_id  | 2026-01-08 12:08:12.771932-06 | running   |             385
 backfill_passage_concepts_llm_v2   | 05b6cf8a-9f80-53f3-8844-665cf2c4c589 | passage_id  | 2026-01-08 12:06:45.831838-06 | running   |           12272
 backfill_chunk_concepts_llm_v2     |                                      | chunk_id    | 2026-01-08 03:48:50.36542-06  | completed |           39584
```

### Passage Counts
| Metric | Count |
|--------|-------|
| Total passages | 550,006 |
| Passages with llm_v2 concepts | 11,141 |
| Passages with haiku_v1 concepts | 360 |
| **Passages WITHOUT llm_v2 (target)** | **538,865** |
| Existing gemini_batch_v1 | 0 |

### Code Chunk Counts
| Metric | Count |
|--------|-------|
| Total code_chunks | 416,397 |
| Chunks with llm_v2 concepts | 38,489 |
| Chunks remaining | 377,908 |

### Audit Findings

1. **Code chunks job shows "completed" but only 9.2% done (38,489/416,397)**
   - The job processed 39,584 items but many were likely updates/duplicates
   - Actual coverage: 9.2% - NOT complete despite status

2. **Passage backfill (llm_v2) is at 2.0% (11,141/550,006)**
   - Job status: "running" but process is paused

3. **Paused process PID 1314836**
   - Status: ALIVE but STOPPED (state 'T')
   - Command: python3 (backfill_chunk_concepts_llm.py)
   - NOT holding active DB locks
   - Some old hung queries from PIDs 1584308-1584310 (can be ignored)

## Scope Decision

**Primary target**: 538,865 passages without llm_v2 concepts
- Will write to new extractor_version: `gemini_batch_v1`
- Will NOT overwrite existing llm_v2 or haiku_v1 records

**Secondary target (optional)**: 377,908 code chunks without concepts
- Can be done with same pipeline if needed
- Lower priority than passages

## Cost Estimation Inputs

- Target passages: 538,865
- Avg passage length: ~600 chars (need to verify)
- Input tokens estimate: ~150 tokens/passage
- Output tokens: ~256 max (capped)
- Gemini Flash-Lite batch pricing: $0.01875/M input, $0.075/M output

**Rough estimate** (to be refined after pilot):
- Input: 538,865 × 150 = 80.8M tokens → ~$1.52
- Output: 538,865 × 200 = 107.8M tokens → ~$8.08
- **Total estimate: ~$10-15** for full passage backfill

## Next Steps

1. Implement Gemini Batch API wrapper
2. Create backfill script with evidence population
3. Run pilot (N=200)
4. Refine cost estimate
5. Full backfill after approval
