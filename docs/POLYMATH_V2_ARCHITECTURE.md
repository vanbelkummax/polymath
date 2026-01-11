# Polymath v2.0: Self-Improving Polymathic Research Agent

## Vision

Polymath v2.0 is an **institutional memory + idea engine** that:
1. **Compounds knowledge** across research, code, grants, and diligence
2. **Self-improves** by detecting and addressing its own gaps
3. **Produces artifacts** (memos, manuscripts, code) not just answers
4. **Validates its work** with built-in evaluation and regression testing

---

## Core Operating Modes

### Mode 1: Answer Pack (Default Output)

Every query produces a structured, auditable response:

```yaml
answer_pack:
  query: "original question"
  claims:
    - claim: "Main finding 1"
      confidence: 0.85
      evidence:
        supporting:
          - passage_id: uuid
            text: "quoted evidence"
            source: "Author et al., 2024"
            doi: "10.1234/..."
        refuting:
          - passage_id: uuid
            text: "contradicting evidence"
            source: "Other et al., 2023"
      reasoning_chain:
        - step: "Premise A from paper X"
        - step: "Combined with finding B from paper Y"
        - step: "Therefore, claim follows"
      uncertainty:
        what_would_change_my_mind: "If X were shown to be false"
        missing_evidence: "No data on edge case Y"

  next_actions:
    - type: "experiment"
      description: "Test hypothesis with..."
    - type: "code"
      description: "Implement using pattern from repo X"
    - type: "read"
      description: "Investigate papers on topic Z"

  counterfactuals:
    - "If assumption A is wrong, then..."
    - "Alternative interpretation: ..."

  metadata:
    retrieval_sources: ["chromadb", "postgres_fts", "neo4j"]
    passages_examined: 127
    time_ms: 2340
    model: "claude-opus-4-5"
```

### Mode 2: Implementation Archaeology

Find and compare implementations across codebases:

```yaml
implementation_report:
  query: "attention mechanism for spatial data"
  implementations:
    - repo: "mahmoodlab/HIPT"
      file: "models/attention.py"
      pattern: "Multi-head self-attention with position encoding"
      quality_signals:
        stars: 450
        citations: 89
        test_coverage: true
      footguns:
        - "Memory scales O(n²) with sequence length"
        - "Requires specific input normalization"
    - repo: "theislab/cell2location"
      file: "..."
      pattern: "..."

  comparison_matrix:
    | Aspect | HIPT | cell2location | squidpy |
    |--------|------|---------------|---------|
    | Speed  | ★★★  | ★★            | ★★★★    |
    | Memory | ★★   | ★★★           | ★★★★    |

  synthesis: "Use pattern X from HIPT with optimization Y from squidpy"
```

### Mode 3: Research Diligence

Kill bad ideas fast with comprehensive checks:

```yaml
diligence_report:
  idea: "Use transformers for 2μm spatial transcriptomics"

  prior_art:
    status: "partially_covered"
    key_papers:
      - "Img2ST (Huo et al.) - H&E to gene expression"
      - "BLEEP - contrastive learning approach"
    gap: "No work at 2μm resolution specifically"

  patent_landscape:
    blocking_patents: 0
    adjacent_patents: 3
    freedom_to_operate: "likely_clear"

  competitive_intelligence:
    active_labs:
      - name: "Mahmood Lab"
        recent_papers: 5
        funding: "$2.3M NIH R01"
      - name: "Lau Lab"
        recent_papers: 8
        funding: "$1.8M R21 + U01"
    companies:
      - name: "10x Genomics"
        activity: "Expanding Visium HD"

  funding_landscape:
    active_grants: 12
    total_funding: "$45M"
    trending_keywords: ["spatial", "multimodal", "foundation model"]

  recommendation: "PROCEED - clear differentiation at 2μm scale"
```

### Mode 4: Hypothesis Engine

Generate and validate research hypotheses:

```yaml
hypothesis_report:
  domain_bridge: "information_theory → spatial_biology"

  hypotheses:
    - id: "H001"
      statement: "Cell communication follows compressed sensing principles"
      novelty_score: 0.82
      validation:
        status: "partially_supported"
        supporting: 3 papers
        contradicting: 0 papers
        unknown: "No direct experimental test"
      experiment:
        design: "Measure mutual information between cell pairs"
        required_data: "Spatial transcriptomics with cell segmentation"
        estimated_effort: "2 weeks"

    - id: "H002"
      statement: "..."
```

### Mode 5: Field Intel Briefing

Weekly automated intelligence:

```yaml
intel_briefing:
  week: "2026-W02"

  new_papers:
    high_impact:
      - title: "..."
        why_matters: "First to show X"
    methodology:
      - title: "..."
        technique: "Novel approach to Y"

  new_code:
    - repo: "..."
      description: "State-of-the-art implementation of..."

  lab_movements:
    - lab: "Mahmood Lab"
      event: "New R01 funded on foundation models"
    - lab: "Theis Lab"
      event: "Released new single-cell method"

  funding_changes:
    new_rfas:
      - "PAR-26-XXX: Spatial Omics"
    recent_awards:
      - pi: "..."
        amount: "$2.1M"
        topic: "..."

  action_items:
    - "Read paper X - directly relevant to your work"
    - "Consider collaboration with lab Y"
```

### Mode 6: Manuscript Synthesis

Generate review manuscripts grounded in corpus:

```yaml
manuscript:
  type: "review"
  title: "Spatial Transcriptomics Prediction from H&E: A Systematic Review"

  sections:
    - name: "Introduction"
      grounded_claims:
        - claim: "ST enables spatial gene expression mapping"
          citations: ["10.1126/science.aaf2403", "..."]

    - name: "Methods Taxonomy"
      subsections:
        - "Patch-based approaches"
        - "Attention mechanisms"
        - "Graph neural networks"

    - name: "Benchmarks"
      tables:
        - comparison of methods
        - dataset characteristics

    - name: "Open Challenges"
      derived_from: "gap_analysis"

  bibliography:
    total: 127
    auto_generated: true
    format: "nature"
```

### Mode 7: Cross-Disciplinary Logic Analysis

Analyze how different fields approach problems:

```yaml
epistemology_analysis:
  comparison: ["physics", "biology", "ML"]

  reasoning_patterns:
    physics:
      dominant_mode: "first_principles + mathematical derivation"
      validation: "experimental confirmation of predictions"
      assumptions: "universal laws, reductionism"

    biology:
      dominant_mode: "empirical observation + statistical inference"
      validation: "replication, multiple lines of evidence"
      assumptions: "context-dependence, evolution"

    ML:
      dominant_mode: "benchmark performance + ablation"
      validation: "held-out test sets, sota claims"
      assumptions: "data sufficiency, iid"

  bridge_opportunities:
    - "Apply physics rigor to ML theory"
    - "Use biological robustness principles in model design"
    - "Import ML scalability to biological simulation"

  thinking_patterns:
    - pattern: "Conservation laws"
      origin: "physics"
      application_in_biology: "Mass balance in metabolism"
      application_in_ML: "Attention normalization"
```

---

## Self-Improvement System

### Gap Detection

The system continuously monitors for:

```python
class GapDetector:
    """Identifies gaps in Polymath's capabilities"""

    def detect_knowledge_gaps(self, query_log: List[Query]) -> List[Gap]:
        """Find queries where retrieval failed"""
        gaps = []
        for query in query_log:
            if query.retrieval_score < 0.5:
                gap = self.classify_gap(query)
                gaps.append(gap)
        return gaps

    def detect_capability_gaps(self, task_log: List[Task]) -> List[Gap]:
        """Find tasks that couldn't be completed"""
        # e.g., "User asked about patents but no patent MCP"
        pass

    def detect_data_gaps(self, citation_graph: Graph) -> List[Gap]:
        """Find missing keystone papers via citation analysis"""
        # Papers cited by many we have, but not in our corpus
        pass
```

### Self-Upgrade Execution

When gaps are detected:

```python
class SelfUpgrader:
    """Automatically addresses detected gaps"""

    def upgrade(self, gap: Gap) -> UpgradeResult:
        match gap.type:
            case "missing_papers":
                return self.ingest_papers(gap.paper_ids)

            case "missing_capability":
                if gap.requires_api_key:
                    return self.request_user_input(
                        f"Need API key for {gap.service}"
                    )
                else:
                    return self.create_new_mcp(gap.capability_spec)

            case "missing_skill":
                return self.generate_skill(gap.skill_spec)

            case "reasoning_failure":
                return self.add_reasoning_pattern(gap.pattern)

    def validate_upgrade(self, upgrade: UpgradeResult) -> bool:
        """Verify the upgrade actually fixed the gap"""
        # Re-run the original failing query/task
        # Check if it now succeeds
        pass
```

### CLAUDE.md Auto-Update

```python
class ClaudeMdManager:
    """Manages CLAUDE.md knowledge retention"""

    def should_update(self, learning: Learning) -> bool:
        """Determine if learning is important enough to persist"""
        criteria = [
            learning.reuse_potential > 0.7,
            learning.affects_future_tasks,
            not self.already_documented(learning),
        ]
        return all(criteria)

    def update(self, learning: Learning):
        """Add learning with minimal context"""
        entry = self.compress_to_essential(learning)
        self.append_to_section(entry, learning.category)

    def periodic_cleanup(self):
        """Reorganize and prune CLAUDE.md"""
        # Remove outdated info
        # Consolidate redundant entries
        # Ensure total size stays manageable
        pass
```

### Context Management

```python
class ContextManager:
    """Tracks and preserves context across compaction"""

    def __init__(self, max_context: int = 180000):
        self.max_context = max_context
        self.current_usage = 0
        self.mission_state = {}
        self.critical_files = []

    def pre_compact_hook(self):
        """Called before context compaction"""
        return {
            "mission": self.mission_state,
            "current_task": self.current_task,
            "progress": self.task_progress,
            "critical_context": self.extract_critical(),
            "resume_instructions": self.generate_resume_prompt(),
        }

    def extract_critical(self) -> Dict:
        """Extract minimum context needed to continue"""
        return {
            "files_being_edited": [...],
            "decisions_made": [...],
            "next_steps": [...],
        }
```

---

## Data Sources (Expanded)

### Current
- Papers (720K passages)
- Code (573K chunks)
- Vanderbilt professors (830 papers)

### To Add

| Source | Type | Integration | Priority |
|--------|------|-------------|----------|
| NIH Reporter | Grants | New MCP | HIGH |
| OpenAlex | Citations | Enrichment | HIGH |
| arXiv | Preprints | Daily scrape | MEDIUM |
| bioRxiv/medRxiv | Preprints | Daily scrape | MEDIUM |
| MICCAI/NeurIPS/CVPR | Proceedings | Annual batch | MEDIUM |
| GitHub Trending | Code | Weekly | LOW |
| ClinicalTrials.gov | Trials | Already have | DONE |

---

## Evaluation Framework

### Gold Query Suite

```yaml
gold_queries:
  prior_art:
    - query: "Has anyone predicted gene expression from H&E at 2μm?"
      expected_papers: ["Img2ST", "BLEEP", "HisToGene"]
      expected_answer_contains: ["resolution", "benchmark"]

    - query: "What loss functions work for sparse count data?"
      expected_papers: ["..."]

  implementation:
    - query: "How does Mahmood Lab implement attention pooling?"
      expected_code: ["HIPT/models/", "CLAM/models/"]

  synthesis:
    - query: "What's the connection between information theory and gene regulation?"
      expected_concepts: ["mutual information", "channel capacity"]
      min_cross_domain_links: 3
```

### Metrics

```python
class RAGMetrics:
    """Evaluation metrics for retrieval and generation"""

    def retrieval_hit_rate(self, queries: List[Query]) -> float:
        """Did top-k include expected evidence?"""
        pass

    def groundedness(self, answers: List[Answer]) -> float:
        """Did answer stay within retrieved evidence?"""
        pass

    def answer_relevance(self, answers: List[Answer]) -> float:
        """Did answer address the query?"""
        pass

    def time_to_artifact(self, tasks: List[Task]) -> float:
        """Minutes to usable output"""
        pass

    def citation_accuracy(self, answers: List[Answer]) -> float:
        """Are citations correct and complete?"""
        pass
```

### Regression Testing

```bash
# Nightly eval run
python3 scripts/eval_gold_queries.py \
  --queries data/gold_queries.yaml \
  --output reports/eval_$(date +%Y%m%d).json \
  --alert-on-regression
```

---

## Implementation Phases

### Phase 1: Foundation (Week 1-2)
- [ ] NIH Reporter MCP
- [ ] OpenAlex citation enrichment
- [ ] PDF metadata enrichment (931 remaining)
- [ ] Answer Pack output format
- [ ] Gold query suite (25 queries)

### Phase 2: Self-Improvement (Week 3-4)
- [ ] Gap detection system
- [ ] Self-upgrade execution
- [ ] CLAUDE.md auto-update
- [ ] Context management hooks
- [ ] Validation framework

### Phase 3: Advanced Reasoning (Week 5-6)
- [ ] Chain-of-thought integration
- [ ] Multi-hop reasoning
- [ ] Uncertainty quantification
- [ ] Counterfactual generation

### Phase 4: Production Polish (Week 7-8)
- [ ] Eval harness (RAGAS + custom)
- [ ] Field Intel Briefing automation
- [ ] Manuscript synthesis mode
- [ ] Cross-disciplinary logic analysis

---

## File Structure

```
/home/user/polymath-repo/
├── lib/
│   ├── answer_pack.py          # Answer Pack generation
│   ├── gap_detector.py         # Gap detection
│   ├── self_upgrader.py        # Self-upgrade execution
│   ├── context_manager.py      # Context preservation
│   ├── claude_md_manager.py    # CLAUDE.md updates
│   ├── reasoning/
│   │   ├── chain_of_thought.py
│   │   ├── multi_hop.py
│   │   ├── uncertainty.py
│   │   └── counterfactual.py
│   └── eval/
│       ├── gold_queries.py
│       ├── metrics.py
│       └── regression.py
├── mcp/
│   ├── nih_reporter_mcp.py     # NEW
│   └── ...
├── scripts/
│   ├── enrich_openalex.py      # Citation enrichment
│   ├── enrich_metadata.py      # PDF metadata
│   ├── eval_gold_queries.py    # Eval harness
│   └── field_intel_briefing.py # Weekly brief
├── data/
│   ├── gold_queries.yaml
│   └── eval_results/
└── docs/
    ├── POLYMATH_V2_ARCHITECTURE.md  # This file
    └── ...
```

---

## Success Criteria

After implementation, Polymath v2.0 should:

1. **Answer any research question** with auditable evidence chains
2. **Self-detect** when it's missing papers, capabilities, or skills
3. **Self-upgrade** to address gaps (or prompt user when needed)
4. **Produce artifacts** (memos, code, manuscripts) not just chat
5. **Never regress** due to eval harness catching regressions
6. **Compound knowledge** by ingesting its own outputs
7. **Stay current** via automated field intel briefings

---

## Appendix: Key Papers for Implementation

- RAGAS: Automated RAG evaluation (arXiv:2309.15217)
- GraphRAG: Graph-enhanced retrieval (arXiv:2404.16130)
- HyDE: Hypothetical document embeddings (ACL 2023)
- ColBERT: Late interaction retrieval (SIGIR 2020)
- Chain-of-Thought prompting (arXiv:2201.11903)
