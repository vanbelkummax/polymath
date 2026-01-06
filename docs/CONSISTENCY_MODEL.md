# Polymath Consistency Model

**Date:** 2026-01-07  
**Version:** 1.2  
**Status:** Active

## 1. System of Record
**Postgres is the canonical system of record** for artifacts and chunks. ChromaDB (vectors) and Neo4j (concept graph) are **derived indexes** that can be rebuilt from Postgres or re-ingested from source files.

## 2. Ingest Run Tracking
Each ingestion call is tagged with an `ingest_run_id`:
- **Postgres**: `ingest_runs` table records run status and counts (when available).
- **Artifacts**: `artifacts.ingest_run_id` links artifacts to the run.
- **ChromaDB**: chunk metadata stores `ingest_run_id`.
- **Neo4j**: nodes store `ingest_run_id`.

This enables auditing, partial failure detection, and targeted re-ingestion.

## 3. Consistency Semantics
The system is **eventually consistent** across stores:
1. A run id is created (if Postgres is available).
2. Writes are attempted to **ChromaDB**, **Neo4j**, and **Postgres** (best-effort).
3. The run is marked:
   - `committed` if all writes succeed
   - `failed` if any store fails

There is no global transaction across stores; partial failures are expected and tracked.

## 4. Current Write Order
**Current behavior** in `lib/unified_ingest.py`:
1. ChromaDB (embeddings + chunk metadata)
2. Neo4j (nodes + relationships)
3. Postgres (artifacts + chunks)

Postgres remains the system of record, but ingestion can still proceed if Postgres is temporarily unavailable. In that case, the run is not recorded in Postgres.

## 5. Partial Failure Handling
If any store fails during ingestion:
- The run is marked `failed` in `ingest_runs` (when available).
- The error log captures the failure.
- The artifact is still retained in any store that succeeded.

### Recommended Reconciliation
1. Identify failed runs via `ingest_runs` in Postgres.
2. Re-run ingestion for those artifacts (optionally reusing the same `ingest_run_id`).
3. Rebuild indexes if drift is detected:
   - **ChromaDB** from source files or Postgres.
   - **Neo4j** from Postgres + concept extraction.

## 6. Consistency Checks
Run targeted checks to detect drift and orphaned entries:

```bash
python3 scripts/consistency_check.py --sample-size 200
```

Use `--strict` to return a non-zero exit code when mismatches are found.

## 7. Operational Notes
- If Postgres is unavailable, ingestion still proceeds for ChromaDB/Neo4j, but the run is untracked.
- There is **no automatic sweeper** yet; orphans must be cleaned manually.
- For production use, treat Postgres availability as required.
