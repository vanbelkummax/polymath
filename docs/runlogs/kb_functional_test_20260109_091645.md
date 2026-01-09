# KB Functional Test Report

**Generated:** 2026-01-09 09:16:45
**Extractor:** gemini_batch_v1

## Summary

| Metric | Value |
|--------|-------|
| Total Tests | 11 |
| Passed | 11 |
| Failed | 0 |
| Pass Rate | 100.0% |

## Results

### Invariant

| Test | Status | Details |
|------|--------|---------|
| invariant_no_duplicates | PASS | duplicates_found=0 |
| invariant_confidence_range | PASS | out_of_range_count=0 |
| invariant_valid_concept_types | PASS | invalid_types=[] |
| invariant_evidence_coverage | PASS | total=1460305, with_evidence=1460305, coverage_pct=100.0 |
| invariant_no_orphan_concepts | PASS | orphan_count=0 |

### Performance

| Test | Status | Details |
|------|--------|---------|
| performance_concept_lookup | PASS | execution_time_ms=0.561, uses_index=True |
| performance_passage_lookup | PASS | execution_time_ms=0.037 |

### Golden

| Test | Status | Details |
|------|--------|---------|
| golden_queries | PASS | total_queries=20, passed=20, failed=0 |

### Evidence

| Test | Status | Details |
|------|--------|---------|
| evidence_source_text_integrity | PASS | sample_size=5000, matched=4999, match_pct=99.98 |

### Coverage

| Test | Status | Details |
|------|--------|---------|
| coverage_concepts_per_passage | PASS | min=1, avg=6.1, max=14 |
| coverage_top_concepts | PASS | top_10=[{'name': 'machine_learning', 'freq': 3459}, {'nam... |
