---
name: startup-identifier
description: Generate technically-grounded startup proposals by mining the Polymath knowledge base for cross-domain opportunities, whitespace gaps, and "asset flip" pivots of existing code
---

# /startup-identifier - Polymath Venture Discovery

Transform the Polymath knowledge engine into a startup idea generator. This skill systematically mines 668K+ passages, 32K papers, and 563K code chunks to find:
- **Whitespace opportunities** (gaps in coverage)
- **Cross-domain bridges** (concepts connecting distant fields)
- **Asset flip pivots** (repurposing existing code for new markets)

## When to Use

- Preparing for investor pitches, mixers, or accelerator applications
- Exploring commercialization paths for research
- Identifying product opportunities from technical capabilities
- Finding "unfair advantages" in existing codebase

## The 5-Phase Workflow

### Phase 1: Seed Discovery (MCP Tools)

Run these polymath-v11 MCP tools to generate raw opportunities:

```
# 1. Serendipity - unexpected but useful connections
serendipity(seed="spatial transcriptomics")
serendipity(seed="foundation models pathology")
serendipity(seed="drug discovery AI")

# 2. Find gaps - orphan concepts and whitespace
find_gaps(topic="spatial biology foundation models")
find_gaps(topic="AI drug discovery")

# 3. Find analogies - cross-domain method transfer
find_analogy(problem="predicting gene expression from histology")

# 4. Generate hypothesis - novel research directions
generate_hypothesis(domains=["machine_learning", "biology"])
```

**Output**: List of raw opportunities with novelty scores

### Phase 2: Read the Overlap Atlas

The Overlap Atlas contains pre-computed cross-domain insights:

```bash
# Latest atlas report
cat /home/user/polymath-repo/docs/runlogs/polymath_overlap_atlas_latest.md
```

Extract from the atlas:
1. **Top field overlaps by PMI** - Strong co-occurrences (e.g., macrophages + T cells)
2. **Bridge concepts** - High betweenness centrality (connects many fields)
3. **Rare pairs** - Unexplored but plausible combinations
4. **Generated hypotheses** - Evidence-bound research directions

### Phase 3: Audit Technical Assets

Read these core modules to understand "asset flip" potential:

| Module | Location | Pivot Potential |
|--------|----------|-----------------|
| Evidence Extractor | `lib/evidence_extractor.py` | NLI verification for legal/compliance |
| Sentry Scoring | `lib/sentry/scoring.py` | Log-normalized scoring for any power-law data |
| Rosetta Expander | `lib/rosetta_query_expander.py` | Cross-domain vocabulary translation |
| Code Ingest | `lib/code_ingest.py` | Semantic code indexing for any language |
| PQE Generator | `lib/pqe_response_generator.py` | Hallucination-free report generation |
| Hybrid Search | `lib/hybrid_search_v2.py` | Tri-modal search (semantic + lexical + graph) |

For each module, identify:
- Current data source (papers, code, concepts)
- Target data source for pivot (financial, legal, medical)
- Unique technical moat (what competitors lack)

### Phase 4: Generate Proposals

For each opportunity, fill this template:

```markdown
## [STARTUP NAME]

**The Pitch:** [One sentence investor hook]

**The Polymath Engine:**
Powered by `[specific_file.py]`, swapping [current_data] for [new_data].
- [Specific feature 1 with line numbers]
- [Specific feature 2 with line numbers]
- [Specific feature 3 with line numbers]

**Why We Win:**
[Explain technical moat vs competitors]

**Market:** $[X]B [market_name]
```

### Phase 5: Rank and Prioritize

Score each proposal on:

| Criterion | Weight | Question |
|-----------|--------|----------|
| Technical moat depth | 30% | Is the code production-ready? |
| Market size | 25% | Is this a $1B+ market? |
| Speed to MVP | 25% | Can we demo in 2 weeks? |
| Founder credibility | 20% | Do we have domain expertise? |

## Vertical Templates

### Finance Pivots

Use these modules for financial applications:
- `sentry/scoring.py` → News/filing velocity scoring
- `code_ingest.py` → Trading strategy indexing
- `hybrid_search_v2.py` → Cross-asset pattern matching

### AI/SaaS Pivots

Use these modules for B2B applications:
- `evidence_extractor.py` → Hallucination firewall (legal, compliance)
- `rosetta_query_expander.py` → Cross-domain search (patents, IP)
- `pqe_response_generator.py` → Report automation (analyst, regulatory)

### Biotech Pivots

Use these knowledge sources:
- Overlap Atlas field overlaps → Platform opportunities
- Rare pairs → Novel therapeutic targets
- Bridge concepts → Translation platforms

## Example Output

```markdown
# VerifyAI — Hallucination Firewall for Legal

**Pitch:** "Every AI-generated clause gets NLI-verified against source
documents—or it doesn't ship."

**Engine:** `lib/evidence_extractor.py`
- 2-stage NLI pipeline (lines 85-100): Retrieve → verify with DeBERTa
- Fail-closed logic (lines 25-37): Contradictions auto-reject
- Sentence-level provenance (lines 31-35): Exact citation coordinates

**Moat:** Competitors generate plausible text. We verify every claim.
30 passages/sec on GPU.

**Market:** $15B legal tech
```

## Quick Commands

```bash
# Run full discovery workflow
python3 -c "
from lib.hybrid_search_v2 import HybridSearcherV2
hs = HybridSearcherV2()

# Find gaps
gaps = hs.search_papers('spatial biology foundation models', n=50)
print(f'Coverage: {len(gaps)} papers')

# Find analogies
analogs = hs.atlas_search('compressed sensing gene expression', n=20)
print(analogs['explain'])
"

# Check technical asset stats
python3 /home/user/polymath-repo/polymath_cli.py stats

# View overlap atlas
cat /home/user/polymath-repo/docs/runlogs/polymath_overlap_atlas_latest.md | head -200
```

## MCP Tools Reference

| Tool | Server | Use For |
|------|--------|---------|
| `serendipity` | polymath-v11 | Unexpected connections |
| `find_gaps` | polymath-v11 | Whitespace opportunities |
| `find_analogy` | polymath-v11 | Cross-domain method transfer |
| `generate_hypothesis` | polymath-v11 | Novel research directions |
| `deep_hunt` | polymath-v11 | Multi-source literature search |
| `trend_radar` | polymath-v11 | What's accelerating |
| `validate_hypothesis` | polymath-v11 | Is this actually novel? |
| `semantic_search` | polymath | Vector similarity search |
| `cross_reference` | polymath | Find related content |

## Killer Lines for Pitches

> "We're not building another AI wrapper. We built a 3-database knowledge
> engine with 668K passages, 32K papers, and 563K code chunks."

> "Our secret weapon is NLI verification. If the evidence says no, the
> answer is no. Fail-closed, not fail-open."

> "Competitors index documents. We index *knowledge*—concepts linked to
> concepts, code linked to papers, claims linked to evidence."

## Output Location

Save generated proposals to:
```
/home/user/work/polymax/STARTUP_PROPOSALS_[DATE].md
```

---

*Skill version: 1.0.0 | Created: 2026-01-10*
