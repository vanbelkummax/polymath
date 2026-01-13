# Polymath 2.0 Implementation Roadmap

**Author**: Max Van Belkum
**Last Updated**: 2026-01-13
**Status**: Planning

---

## Overview

This document details the 20-week implementation plan for Polymath 2.0, broken into 5 phases. Each phase has specific deliverables, success criteria, and validation procedures.

---

## Timeline Summary

```
Week  1  2  3  4  5  6  7  8  9 10 11 12 13 14 15 16 17 18 19 20
      ├──────────┼──────────┼───────────┼───────────┼───────────┤
      │ Phase 1  │ Phase 2  │  Phase 3  │  Phase 4  │  Phase 5  │
      │Foundation│Mechanism │  Agentic  │ Discovery │   Code    │
      │          │Extraction│   Layer   │  Engine   │Integration│
      └──────────┴──────────┴───────────┴───────────┴───────────┘
```

---

## Phase 1: Foundation (Weeks 1-4)

### Goal
Establish Zotero-first workflow and hierarchical passage storage.

### Week 1: Zotero Integration

| Task | Deliverable | Validation |
|------|-------------|------------|
| Design Zotero export schema | `docs/zotero_export_spec.md` | Schema review |
| Implement CSV parser | `scripts/ingest/parse_zotero_csv.py` | Unit tests |
| Create metadata validator | `scripts/ingest/validate_metadata.py` | Test on 100 papers |
| Build PDF path resolver | `scripts/ingest/resolve_pdf_paths.py` | Integration test |

**Success Criteria**:
- [ ] CSV parser handles all Zotero export fields
- [ ] Validator catches 95%+ of metadata issues
- [ ] PDF paths resolved for 100% of exports

### Week 2: Hierarchical Passage Schema

| Task | Deliverable | Validation |
|------|-------------|------------|
| Design `passages_v2` schema | `schemas/postgres_schema.sql` | DBA review |
| Implement hierarchy builder | `scripts/ingest/build_hierarchy.py` | Unit tests |
| Create section classifier | `scripts/extract/classify_sections.py` | Accuracy >90% |
| Build context expander | `scripts/search/expand_context.py` | Integration test |

**Success Criteria**:
- [ ] Schema supports 4 hierarchy levels
- [ ] Section classification >90% accuracy
- [ ] Context expansion <50ms latency

### Week 3: PDF Parser Enhancement

| Task | Deliverable | Validation |
|------|-------------|------------|
| Upgrade PDF parser | `scripts/ingest/parse_pdf_hierarchical.py` | Test on 50 PDFs |
| Add figure/table extraction | `scripts/ingest/extract_figures.py` | Manual review |
| Implement quality scoring | `scripts/ingest/score_quality.py` | Correlation check |
| Create fallback OCR | `scripts/ingest/ocr_fallback.py` | Compare to baseline |

**Success Criteria**:
- [ ] Parse 95% of PDFs without errors
- [ ] Extract figures with 80% recall
- [ ] Quality scores correlate with manual assessment

### Week 4: Migration and Integration

| Task | Deliverable | Validation |
|------|-------------|------------|
| Migrate existing passages | `scripts/maintenance/migrate_v1_to_v2.py` | Count verification |
| Update ChromaDB index | `scripts/maintenance/rebuild_chromadb.py` | Embedding check |
| Create Zotero sync automation | `scripts/maintenance/zotero_sync.sh` | Cron test |
| End-to-end integration test | `tests/integration/test_phase1.py` | All tests pass |

**Success Criteria**:
- [ ] All v1 passages migrated with parent_id
- [ ] ChromaDB re-indexed with new passages
- [ ] Nightly Zotero sync operational

### Phase 1 Deliverables Summary

```
scripts/
├── ingest/
│   ├── parse_zotero_csv.py
│   ├── validate_metadata.py
│   ├── resolve_pdf_paths.py
│   ├── parse_pdf_hierarchical.py
│   ├── build_hierarchy.py
│   ├── extract_figures.py
│   ├── score_quality.py
│   └── ocr_fallback.py
├── extract/
│   └── classify_sections.py
├── search/
│   └── expand_context.py
└── maintenance/
    ├── migrate_v1_to_v2.py
    ├── rebuild_chromadb.py
    └── zotero_sync.sh
```

---

## Phase 2: Mechanism Extraction (Weeks 5-8)

### Goal
Extract mechanisms, data structures, and objectives from papers.

### Week 5: Mechanism Schema Design

| Task | Deliverable | Validation |
|------|-------------|------------|
| Design mechanism node schema | `schemas/neo4j_schema.cypher` | Graph expert review |
| Create extraction prompt | `scripts/extract/mechanism_prompt.py` | LLM evaluation |
| Build validation rubric | `docs/mechanism_rubric.md` | Inter-rater reliability |
| Implement extractor | `scripts/extract/extract_mechanisms.py` | Test on 50 papers |

**Success Criteria**:
- [ ] Schema covers Method→Mechanism→DataStructure→Objective
- [ ] Extraction prompt achieves >80% precision
- [ ] Rubric IRR >0.8 (kappa)

### Week 6: Extraction Pipeline

| Task | Deliverable | Validation |
|------|-------------|------------|
| Build batch extraction | `scripts/extract/batch_mechanisms.py` | Process 1000 papers |
| Create graph writer | `scripts/extract/write_mechanism_graph.py` | Neo4j verification |
| Implement deduplication | `scripts/extract/dedupe_mechanisms.py` | Accuracy check |
| Add confidence scoring | `scripts/extract/score_confidence.py` | Calibration test |

**Success Criteria**:
- [ ] Process 1000 papers in <4 hours
- [ ] Graph contains valid relationships
- [ ] Confidence scores well-calibrated

### Week 7: Quality Validation

| Task | Deliverable | Validation |
|------|-------------|------------|
| Manual validation (100 samples) | `docs/validation_results.md` | >80% accuracy |
| Error analysis | `docs/extraction_errors.md` | Categorized issues |
| Prompt iteration | `scripts/extract/mechanism_prompt_v2.py` | Improved accuracy |
| Edge case handling | `scripts/extract/handle_edge_cases.py` | Coverage test |

**Success Criteria**:
- [ ] Manual validation >80% accuracy
- [ ] Major error categories addressed
- [ ] Improved prompt >85% accuracy

### Week 8: Full Corpus Processing

| Task | Deliverable | Validation |
|------|-------------|------------|
| Process all passages | Job completion | Count verification |
| Build indices | `scripts/maintenance/build_mechanism_indices.py` | Query performance |
| Create mechanism browser | `scripts/search/browse_mechanisms.py` | User test |
| Documentation | `docs/mechanism_extraction.md` | Review |

**Success Criteria**:
- [ ] All passages processed
- [ ] Graph queries <200ms
- [ ] Browser functional

### Phase 2 Deliverables Summary

```
scripts/
├── extract/
│   ├── mechanism_prompt.py
│   ├── mechanism_prompt_v2.py
│   ├── extract_mechanisms.py
│   ├── batch_mechanisms.py
│   ├── write_mechanism_graph.py
│   ├── dedupe_mechanisms.py
│   ├── score_confidence.py
│   └── handle_edge_cases.py
├── search/
│   └── browse_mechanisms.py
└── maintenance/
    └── build_mechanism_indices.py

docs/
├── mechanism_rubric.md
├── validation_results.md
├── extraction_errors.md
└── mechanism_extraction.md
```

---

## Phase 3: Agentic Layer (Weeks 9-12)

### Goal
Build grounded reasoning with citations and validation.

### Week 9: Query Decomposition

| Task | Deliverable | Validation |
|------|-------------|------------|
| Design decomposition schema | `docs/query_decomposition.md` | Review |
| Implement decomposer | `scripts/search/decompose_query.py` | Test suite |
| Build sub-query router | `scripts/search/route_subqueries.py` | Coverage test |
| Create query logger | `scripts/search/log_queries.py` | Audit trail |

### Week 10: Evidence Validation

| Task | Deliverable | Validation |
|------|-------------|------------|
| Implement NLI validator | `scripts/validate/nli_validator.py` | Accuracy test |
| Build citation linker | `scripts/validate/link_citations.py` | Precision/recall |
| Create contradiction detector | `scripts/validate/detect_contradictions.py` | Test cases |
| Add confidence scorer | `scripts/validate/score_evidence.py` | Calibration |

### Week 11: Response Synthesis

| Task | Deliverable | Validation |
|------|-------------|------------|
| Design response schema | `docs/response_schema.md` | Review |
| Implement synthesizer | `scripts/search/synthesize_response.py` | Quality test |
| Build citation formatter | `scripts/search/format_citations.py` | Style check |
| Add reasoning tracer | `scripts/search/trace_reasoning.py` | Audit test |

### Week 12: Integration Testing

| Task | Deliverable | Validation |
|------|-------------|------------|
| End-to-end tests | `tests/integration/test_phase3.py` | All pass |
| User acceptance testing | `docs/uat_results.md` | Feedback |
| Performance optimization | Profiling results | Latency <2s |
| Documentation | `docs/agentic_layer.md` | Review |

### Phase 3 Deliverables Summary

```
scripts/
├── search/
│   ├── decompose_query.py
│   ├── route_subqueries.py
│   ├── log_queries.py
│   ├── synthesize_response.py
│   ├── format_citations.py
│   └── trace_reasoning.py
└── validate/
    ├── nli_validator.py
    ├── link_citations.py
    ├── detect_contradictions.py
    └── score_evidence.py
```

---

## Phase 4: Discovery Engine (Weeks 13-16)

### Goal
Enable cross-domain transfer validation and hypothesis generation.

### Week 13: Gap Detection

| Task | Deliverable | Validation |
|------|-------------|------------|
| Design gap query | `scripts/discovery/gap_query.cypher` | Expert review |
| Implement gap finder | `scripts/discovery/find_gaps.py` | Test cases |
| Build result ranker | `scripts/discovery/rank_gaps.py` | Quality check |
| Create gap browser | `scripts/discovery/browse_gaps.py` | User test |

### Week 14: Transfer Validation

| Task | Deliverable | Validation |
|------|-------------|------------|
| Implement validator | `scripts/discovery/validate_transfer.py` | Test suite |
| Build novelty checker | `scripts/discovery/check_novelty.py` | API test |
| Create explanation generator | `scripts/discovery/explain_transfer.py` | Quality review |
| Add external API integration | `scripts/discovery/external_apis.py` | Integration test |

### Week 15: Hypothesis Generation

| Task | Deliverable | Validation |
|------|-------------|------------|
| Design hypothesis schema | `docs/hypothesis_schema.md` | Review |
| Implement generator | `scripts/discovery/generate_hypothesis.py` | Quality test |
| Build experiment specifier | `scripts/discovery/specify_experiment.py` | Feasibility check |
| Create priority scorer | `scripts/discovery/score_priority.py` | Ranking test |

### Week 16: Pipeline Integration

| Task | Deliverable | Validation |
|------|-------------|------------|
| End-to-end pipeline | `scripts/discovery/full_pipeline.py` | Integration test |
| Generate 20 hypotheses | `docs/generated_hypotheses.md` | Expert review |
| Performance optimization | Profiling results | <10s per hypothesis |
| Documentation | `docs/discovery_engine.md` | Review |

### Phase 4 Deliverables Summary

```
scripts/
└── discovery/
    ├── gap_query.cypher
    ├── find_gaps.py
    ├── rank_gaps.py
    ├── browse_gaps.py
    ├── validate_transfer.py
    ├── check_novelty.py
    ├── explain_transfer.py
    ├── external_apis.py
    ├── generate_hypothesis.py
    ├── specify_experiment.py
    ├── score_priority.py
    └── full_pipeline.py
```

---

## Phase 5: Code Integration (Weeks 17-20)

### Goal
Create unified code-literature semantic space.

### Week 17: Code Concept Extraction

| Task | Deliverable | Validation |
|------|-------------|------------|
| Design code concept schema | `schemas/code_concept_schema.sql` | Review |
| Implement extractor | `scripts/extract/extract_code_concepts.py` | Test suite |
| Build function analyzer | `scripts/extract/analyze_functions.py` | Accuracy test |
| Create docstring parser | `scripts/extract/parse_docstrings.py` | Coverage test |

### Week 18: Unified Embeddings

| Task | Deliverable | Validation |
|------|-------------|------------|
| Train code-text aligner | `scripts/extract/align_embeddings.py` | Similarity test |
| Build unified index | `scripts/maintenance/build_unified_index.py` | Query test |
| Implement cross-modal search | `scripts/search/cross_modal_search.py` | Recall test |
| Create link validator | `scripts/validate/validate_code_links.py` | Precision test |

### Week 19: Priority Repository Indexing

| Task | Deliverable | Validation |
|------|-------------|------------|
| Index Img2ST | Index complete | Manual review |
| Index HisToGene | Index complete | Manual review |
| Index foundation models | Index complete | Manual review |
| Index methods libraries | Index complete | Manual review |

### Week 20: Integration and Launch

| Task | Deliverable | Validation |
|------|-------------|------------|
| Full system integration | All components working | Integration test |
| Performance benchmarking | Benchmark results | Meet targets |
| Documentation complete | All docs finalized | Review |
| Launch preparation | Deployment ready | Checklist |

### Phase 5 Deliverables Summary

```
scripts/
├── extract/
│   ├── extract_code_concepts.py
│   ├── analyze_functions.py
│   ├── parse_docstrings.py
│   └── align_embeddings.py
├── search/
│   └── cross_modal_search.py
├── validate/
│   └── validate_code_links.py
└── maintenance/
    └── build_unified_index.py
```

---

## Risk Mitigation

| Risk | Probability | Impact | Mitigation |
|------|-------------|--------|------------|
| Zotero export limitations | Medium | High | Manual metadata supplement |
| LLM extraction accuracy | Medium | High | Iterative prompt engineering |
| Performance bottlenecks | Medium | Medium | Caching, parallelization |
| API rate limits | Low | Medium | Request throttling, caching |
| Data quality issues | High | Medium | Quality scoring, manual review |

---

## Success Criteria Summary

| Phase | Key Metric | Target |
|-------|------------|--------|
| 1 | DOI coverage | 100% |
| 1 | Passage length | 1500+ chars |
| 2 | Mechanism extraction accuracy | >80% |
| 3 | Citation accuracy | >95% |
| 4 | Actionable hypotheses | 20+ |
| 5 | Code-paper links | 1000+ |
