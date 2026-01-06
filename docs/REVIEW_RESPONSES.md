# Senior Review Responses (Self-Answers)

**Date:** 2026-01-07
**Scope:** Architecture, Operations, Data Quality

---

## 1) Disaster recovery plan if ChromaDB corrupts?

- **Answer:** Restore from nightly backups (`scripts/backup.sh`).
- **RTO/RPO:** 3-5 minutes / 24 hours.
- **Procedure:** `docs/BACKUP_RESTORE_PROCEDURES.md` with `scripts/restore_test.sh` for validation.

## 2) How do you detect ingestion jobs failing silently?

- **Answer:** Ingest runs are tagged with `ingest_run_id` across stores. Failures are recorded in Postgres `ingest_runs` when available.
- **Detection:** Monitoring alerts + `scripts/consistency_check.py` to sample for drift and missing entries.

## 3) Consistency model across 4 databases?

- **Answer:** Eventual consistency, Postgres as canonical system of record.
- **Details:** `docs/CONSISTENCY_MODEL.md` documents write order, partial failure handling, and reconciliation.

## 4) How is search quality measured?

- **Answer:** Golden query suites and baseline overlap checks.
- **Tools:** `scripts/golden_query_eval.py`, `tests/golden_queries.json`, `tests/data/golden_queries.json`.

## 5) Performance budget for queries?

- **Answer:** Target p95 < 3s for hybrid search at 250k chunks; p95 < 5s at 500k chunks.
- **Action:** Benchmarks will be tracked in `POLYMATH_SCALE_TEST.md` and monitored via `monitoring/metrics_collector.py`.

## 6) How is concept extraction validated?

- **Answer:** Manual spot checks + concept coverage metrics, with targeted regression tests against the golden set.
- **Action:** Use Rosetta expansions to reduce vocabulary gaps; update `data/rosetta/term_mappings.json` when misses are identified.

## 7) Migration plan for 3,355 legacy file chunks?

- **Answer:** Keep legacy chunks tagged with `ingestion_method`, then re-ingest into dual-stream pipeline and deduplicate by hash.
- **Action:** Dedup tool is in place (`scripts/deduplicate_chunks.py`), and ingestion metadata now tracks method.

## 8) Load test at 500k/1M chunks?

- **Answer:** Not yet. Planned as part of scale testing.
- **Action:** Use `POLYMATH_SCALE_TEST.md` and benchmark scripts to capture latency vs. chunk count.

---

## Changes Completed (This Session)

- Added `scripts/consistency_check.py` for drift detection.
- Added Rosetta mapping and abstraction data files under `data/rosetta/`.
- Added caching + heuristic expansions in `lib/rosetta_query_expander.py`.
- Upgraded `abstract_problem` to use configurable abstraction patterns.

