# Evidence-Bound Passage Extraction Plan

## Problem

Current extraction returns canonical concepts but loses:
- **Surface form** (what literally appears: "Wasserstein distance" vs `optimal_transport`)
- **Evidence snippet** (quote from text supporting the extraction)
- **Audit trail** (can't cite "mentioned on page 5" without literal proof)

**Result**: 49% literal support = can't use for grant citations

## Solution: Single-Pass Evidence-Bound Extraction

Extract BOTH polymathic canonical AND audit-grade evidence in ONE pass.

---

## Schema Changes

### 1. Postgres: Add evidence columns

```sql
-- Add to passage_concepts table
ALTER TABLE passage_concepts 
  ADD COLUMN IF NOT EXISTS surface_form TEXT,
  ADD COLUMN IF NOT EXISTS evidence_snippet TEXT,
  ADD COLUMN IF NOT EXISTS support_type TEXT DEFAULT 'literal';

-- Index for evidence queries
CREATE INDEX IF NOT EXISTS idx_passage_concepts_support 
  ON passage_concepts(support_type) 
  WHERE support_type = 'literal';
```

### 2. Neo4j: Add edge properties

```cypher
// MENTIONS edges will have:
// - surface: "Wasserstein distance" (literal text)
// - evidence: "...computed Wasserstein distance between..." (snippet)
// - support: "literal" or "inferred"
```

---

## Code Changes

### 1. New extractor function: `extract_concepts_with_evidence()`

**Location**: `lib/local_extractor.py`

**Signature**:
```python
def extract_concepts_with_evidence(
    self, 
    text: str, 
    require_literal: bool = True
) -> List[Dict[str, Any]]:
    """
    Extract concepts WITH evidence binding.
    
    Returns:
        [
            {
                "canonical": "optimal_transport",
                "type": "method",
                "surface": "Wasserstein distance",
                "evidence": "...computed Wasserstein distance between...",
                "aliases": ["OT", "earth mover's distance"],
                "confidence": 0.85
            }
        ]
    """
```

**LLM Prompt Changes**:
```
You are extracting scientific concepts from academic text.

CRITICAL RULES:
1. For each concept, you MUST quote the exact surface form as it appears in the text
2. Return the surface form EXACTLY as written (preserve case, spacing)
3. Include a short evidence snippet (10-30 words) containing the surface form
4. Normalize the concept to a canonical name (lowercase, underscores)
5. Only extract concepts if you can find literal text evidence

Output JSON format:
[
  {
    "canonical": "optimal_transport",
    "surface": "Wasserstein distance", 
    "evidence": "computed Wasserstein distance between distributions",
    "type": "method",
    "aliases": ["OT", "earth mover's distance"],
    "confidence": 0.85
  }
]

If the surface form differs from canonical, list surface as first alias.
```

### 2. Update backfill_passages() to use new extractor

**Location**: `scripts/backfill_chunk_concepts_llm.py`

**Change**:
```python
# OLD (for chunks):
concepts = extractor.extract_concepts(content)

# NEW (for passages):
concepts = extractor.extract_concepts_with_evidence(
    content, 
    require_literal=True
)

# Store both canonical and surface
for concept in concepts:
    upsert_passage_concept(
        conn,
        passage_id,
        concept_name=concept["canonical"],  # polymathic key
        concept_type=concept.get("type"),
        aliases=concept.get("aliases", []),
        confidence=concept.get("confidence", 0.0),
        extractor_model=extractor.fast_model,
        extractor_version=extractor_version,
        surface_form=concept.get("surface"),  # NEW
        evidence_snippet=concept.get("evidence"),  # NEW
        support_type="literal"  # NEW
    )
```

### 3. Update upsert_passage_concept()

**Location**: `lib/kb_derived.py`

**Add parameters**:
```python
def upsert_passage_concept(
    conn,
    passage_id: str,
    concept_name: str,
    concept_type: Optional[str],
    aliases: List[str],
    confidence: float,
    extractor_model: str,
    extractor_version: str,
    surface_form: Optional[str] = None,  # NEW
    evidence_snippet: Optional[str] = None,  # NEW
    support_type: str = "literal"  # NEW
):
    cursor = conn.cursor()
    cursor.execute("""
        INSERT INTO passage_concepts (
            passage_id, concept_name, concept_type, aliases, 
            confidence, extractor_model, extractor_version,
            surface_form, evidence_snippet, support_type
        )
        VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
        ON CONFLICT (passage_id, concept_name, extractor_version)
        DO UPDATE SET
            confidence = EXCLUDED.confidence,
            surface_form = EXCLUDED.surface_form,
            evidence_snippet = EXCLUDED.evidence_snippet,
            support_type = EXCLUDED.support_type
    """, (
        passage_id, concept_name, concept_type, 
        json.dumps(aliases) if aliases else None,
        confidence, extractor_model, extractor_version,
        surface_form, evidence_snippet, support_type
    ))
    conn.commit()
```

### 4. Update Neo4j rebuild script

**Location**: `scripts/rebuild_neo4j_concepts_v2.py`

**Add edge properties**:
```python
session.run("""
    MATCH (p:Passage {passage_id: $passage_id})
    MATCH (c:Concept {name: $name})
    MERGE (p)-[m:MENTIONS]->(c)
    SET m.confidence = $confidence,
        m.extractor_version = $extractor_version,
        m.edge_version = $edge_version,
        m.weight = $confidence,
        m.surface = $surface,  // NEW
        m.evidence = $evidence,  // NEW
        m.support = $support_type  // NEW
""", {
    "passage_id": passage_id,
    "name": concept_name,
    "confidence": confidence,
    "extractor_version": extractor_version,
    "edge_version": edge_version,
    "surface": surface_form,
    "evidence": evidence_snippet,
    "support_type": support_type
})
```

---

## Updated Sentinel Test

**Location**: `scripts/passage_sentinel_v2.py`

**New metric**:
```python
def check_evidence_support(surface, evidence, text):
    """Check if surface form appears in evidence snippet and text."""
    surface_lower = surface.lower()
    evidence_lower = evidence.lower() if evidence else ""
    text_lower = text.lower()
    
    # Both must be true for audit-grade citation
    in_evidence = surface_lower in evidence_lower
    in_text = surface_lower in text_lower
    
    return in_evidence and in_text

# Report
evidence_support_rate = ...  # should be ~95%+
```

---

## Testing Plan

### 1. Unit test new extractor (1 passage)

```bash
python3 << 'EOF'
from lib.local_extractor import LocalEntityExtractor

text = """
We compute the Wasserstein distance (also known as optimal transport)
between gene expression distributions using the Sinkhorn algorithm.
"""

ext = LocalEntityExtractor()
concepts = ext.extract_concepts_with_evidence(text)

# Should return:
# [
#   {
#     "canonical": "wasserstein_distance",
#     "surface": "Wasserstein distance",
#     "evidence": "compute the Wasserstein distance",
#     "type": "metric",
#     "aliases": ["optimal transport", "OT"],
#     "confidence": 0.9
#   },
#   ...
# ]
