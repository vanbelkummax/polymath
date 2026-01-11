---
name: patent-intelligence
description: Mine patent databases for IP whitespace, prior art, and competitive intelligence using Polymath's cross-domain search capabilities
---

# /patent-intelligence - IP Discovery & Analysis

Transform Polymath's research infrastructure into a patent intelligence engine. Uses the same tri-modal search (semantic + lexical + graph) that powers scientific discovery, now applied to 140M+ patents.

## When to Use

- **Freedom to operate** - Can we commercialize without infringement?
- **Prior art search** - Will our patent application survive?
- **IP whitespace** - Where are the unexploited opportunities?
- **Competitive landscape** - What are competitors protecting?
- **Licensing targets** - Who has IP we need?

## Data Sources

| Source | Coverage | Access | Rate Limit |
|--------|----------|--------|------------|
| [USPTO PatentsView](https://patentsview.org/apis) | US patents (12M+) | FREE | 45 req/min |
| [EPO OPS](https://www.epo.org/en/searching-for-patents/data/web-services/ops) | EU + INPADOC | FREE | 4GB/month |
| [WIPO PATENTSCOPE](https://patentscope.wipo.int/) | PCT (4M+) | FREE | 10K/export |
| [Lens.org](https://www.lens.org/) | Global (140M) | FREE (research) | API key needed |

## The 5-Phase Workflow

### Phase 1: Landscape Mapping

Query multiple patent databases to understand the terrain:

```python
# USPTO PatentsView - Quick landscape
import requests

def search_uspto(query: str, per_page: int = 100) -> dict:
    """Search USPTO PatentsView API."""
    url = "https://api.patentsview.org/patents/query"
    params = {
        "q": {"_text_any": {"patent_abstract": query}},
        "f": ["patent_id", "patent_title", "patent_date",
              "patent_abstract", "assignee_organization"],
        "o": {"per_page": per_page}
    }
    resp = requests.post(url, json=params)
    return resp.json()

# Example: Find spatial transcriptomics patents
results = search_uspto("spatial transcriptomics gene expression")
```

### Phase 2: Cross-Domain Expansion

Use Polymath's Rosetta expander to find patents using different terminology:

```python
from lib.rosetta_query_expander import expand_query_with_llm

# Expand query across domains
original = "predicting gene expression from histology images"
expanded_terms = expand_query_with_llm(
    original,
    source_domain="biology",
    target_domain="computer_science"
)
# Returns: "image classification", "convolutional neural network",
#          "feature extraction", "deep learning pathology"

# Now search patents with expanded vocabulary
for term in expanded_terms:
    patents = search_uspto(f"{original} OR {term}")
```

### Phase 3: Gap Detection

Adapt Polymath's `find_gaps` for IPC code analysis:

```python
from collections import Counter
from datetime import datetime

def find_patent_gaps(patents: list) -> dict:
    """Analyze patent landscape for whitespace."""

    # Extract IPC codes
    ipc_codes = [p.get("ipc_class") for p in patents if p.get("ipc_class")]
    ipc_counts = Counter(ipc_codes)

    # Extract filing years
    years = [int(p["patent_date"][:4]) for p in patents if p.get("patent_date")]
    year_counts = Counter(years)

    # Find gaps
    current_year = datetime.now().year
    recent_years = [current_year - i for i in range(3)]
    recent_count = sum(year_counts.get(y, 0) for y in recent_years)

    # Identify underexplored IPC classes
    rare_ipcs = [ipc for ipc, count in ipc_counts.items() if count == 1]

    return {
        "total_patents": len(patents),
        "temporal_gap": recent_count < 10,  # Less than 10 patents in 3 years
        "rare_ipc_codes": rare_ipcs[:10],
        "top_assignees": get_top_assignees(patents),
        "whitespace_opportunities": identify_missing_combinations(ipc_counts)
    }
```

### Phase 4: Prior Art Verification

Use Polymath's evidence extraction for claim-level analysis:

```python
from lib.evidence_extractor import EvidenceExtractor

extractor = EvidenceExtractor()

def verify_novelty(your_claim: str, prior_art_patents: list) -> dict:
    """Check if claim is novel against prior art."""

    conflicts = []
    for patent in prior_art_patents:
        # Check each claim in prior art
        for claim in patent.get("claims", []):
            spans = extractor.extract_spans_for_claim(your_claim)

            # High entailment = potential conflict
            for span in spans:
                if span.entailment_score > 0.7:
                    conflicts.append({
                        "patent_id": patent["patent_id"],
                        "claim": claim,
                        "overlap_score": span.entailment_score,
                        "matching_text": span.span_text
                    })

    return {
        "novel": len(conflicts) == 0,
        "conflicts": conflicts,
        "recommendation": "Modify claim language" if conflicts else "Proceed with filing"
    }
```

### Phase 5: Competitive Intelligence

Track assignee patent velocity (like `watch_competitor` for labs):

```python
from lib.sentry.scoring import log_normalize

def score_patent_portfolio(assignee: str, patents: list) -> dict:
    """Score a company's patent portfolio momentum."""

    # Filter to this assignee
    assignee_patents = [p for p in patents if assignee.lower() in
                        (p.get("assignee_organization", "") or "").lower()]

    # Calculate metrics
    total_patents = len(assignee_patents)
    recent_patents = len([p for p in assignee_patents
                          if int(p["patent_date"][:4]) >= 2023])

    # Log-normalize (handles power-law distribution)
    portfolio_score = log_normalize(total_patents, min_val=1, max_val=10000)
    velocity_score = log_normalize(recent_patents, min_val=1, max_val=100)

    return {
        "assignee": assignee,
        "portfolio_size": total_patents,
        "recent_filings": recent_patents,
        "portfolio_score": round(portfolio_score, 2),
        "velocity_score": round(velocity_score, 2),
        "combined_score": round((portfolio_score + velocity_score) / 2, 2)
    }
```

## Quick Commands

```bash
# Search USPTO for a technology area
curl -X POST "https://api.patentsview.org/patents/query" \
  -H "Content-Type: application/json" \
  -d '{"q":{"_text_any":{"patent_abstract":"spatial transcriptomics"}},"f":["patent_id","patent_title","patent_date"],"o":{"per_page":25}}'

# Search EPO OPS (requires registration)
curl "https://ops.epo.org/3.2/rest-services/published-data/search?q=spatial%20transcriptomics"

# Search Lens.org (requires API key)
curl -X POST "https://api.lens.org/patent/search" \
  -H "Authorization: Bearer $LENS_API_KEY" \
  -d '{"query": "spatial transcriptomics", "size": 25}'
```

## Integration with Polymath MCP

These patent searches feed back into the Polymath knowledge graph:

```python
# Link patents to papers via concepts
async def enrich_patent_with_papers(patent: dict) -> dict:
    """Find related academic papers for a patent."""

    from lib.hybrid_search_v2 import HybridSearcherV2
    hs = HybridSearcherV2()

    # Extract key concepts from patent
    concepts = extract_concepts(patent["patent_abstract"])

    # Find related papers
    related_papers = hs.search_papers(
        patent["patent_title"],
        concepts=concepts,
        n=10
    )

    return {
        **patent,
        "related_papers": [p.title for p in related_papers],
        "academic_citations": count_academic_refs(patent),
        "tech_readiness": estimate_trl(patent, related_papers)
    }
```

## Output Template

For each IP opportunity, generate:

```markdown
## [TECHNOLOGY] IP Landscape

**Total Patents:** X | **Top Assignees:** Y | **Whitespace Score:** Z/10

### Key Findings
1. [Finding with supporting patent IDs]
2. [Finding with supporting patent IDs]

### Whitespace Opportunities
| Gap Area | IPC Classes | Competition | Priority |
|----------|-------------|-------------|----------|
| [Area 1] | [G06N 3/08] | Low | High |

### Prior Art Concerns
| Your Claim | Conflicting Patent | Overlap | Mitigation |
|------------|-------------------|---------|------------|
| [Claim 1] | US12345678 | 72% | Narrow scope |

### Recommended Actions
1. File provisional on [specific area]
2. Monitor [competitor] filings
3. Consider licensing from [assignee]
```

## Example: Img2ST IP Analysis

```python
# Run full analysis for your research area
query = "predicting gene expression from histology deep learning"

# 1. Landscape
patents = search_uspto(query, per_page=500)
print(f"Found {len(patents)} patents")

# 2. Expand vocabulary
expanded = expand_query_with_llm(query, "biology", "engineering")
for term in expanded:
    more_patents = search_uspto(f"{query} OR {term}")
    patents.extend(more_patents)

# 3. Find gaps
gaps = find_patent_gaps(patents)
print(f"Whitespace opportunities: {gaps['rare_ipc_codes']}")

# 4. Check novelty of your approach
your_claims = [
    "A method for predicting spatially-resolved gene expression from H&E images using a transformer architecture",
    "A system for generating virtual spatial transcriptomics data from histology"
]
for claim in your_claims:
    novelty = verify_novelty(claim, patents)
    print(f"Claim: {claim[:50]}... Novel: {novelty['novel']}")

# 5. Track competitors
competitors = ["10x Genomics", "Nanostring", "Vizgen"]
for comp in competitors:
    score = score_patent_portfolio(comp, patents)
    print(f"{comp}: Portfolio={score['portfolio_score']}, Velocity={score['velocity_score']}")
```

## See Also

- `/startup-identifier` - Full startup ideation workflow
- `find_gaps` MCP tool - Research gap detection
- `rosetta_query_expander.py` - Cross-domain vocabulary
- `evidence_extractor.py` - Claim verification

---

*Skill version: 1.0.0 | Created: 2026-01-10*
