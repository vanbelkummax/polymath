# Concept Linking Script - Quick Reference

## TL;DR

```bash
# Preview what will change (SAFE)
python3 /home/user/work/polymax/scripts/fix_concept_links.py --dry-run

# Actually create the links (USE WITH CARE)
python3 /home/user/work/polymax/scripts/fix_concept_links.py --live

# Process specific concepts
python3 /home/user/work/polymax/scripts/fix_concept_links.py --dry-run \
  --concepts maximum_entropy sparse_coding
```

## What It Does

Finds papers in ChromaDB that mention concepts but aren't linked in Neo4j.

**Problem:** Simple keyword matching misses sophisticated phrasing
- Misses: "variational free energy" → `free_energy_principle`
- Misses: "Turing pattern" → `reaction_diffusion`
- Misses: "MaxEnt" → `maximum_entropy`

**Solution:** Semantic search with concept-specific terms + relevance thresholds

## Usage Examples

### 1. Safe Preview (Recommended First)

```bash
# See what would change without modifying database
python3 fix_concept_links.py --dry-run

# Save report
python3 fix_concept_links.py --dry-run --output preview.json
```

### 2. Live Run (Actually Makes Changes)

```bash
# Create the links in Neo4j
python3 fix_concept_links.py --live

# Process only certain concepts
python3 fix_concept_links.py --live --concepts do_calculus maximum_entropy
```

### 3. Verify Results

```bash
# Check before
python3 /home/user/work/polymax/polymath_cli.py graph \
  "MATCH (c:CONCEPT {name: 'maximum_entropy'})
   RETURN size([(c)<-[:MENTIONS]-(p) | p]) as count"

# Run live
python3 fix_concept_links.py --live --concepts maximum_entropy

# Check after (should show 5 papers)
python3 /home/user/work/polymax/polymath_cli.py graph \
  "MATCH (c:CONCEPT {name: 'maximum_entropy'})<-[:MENTIONS]-(p:Paper)
   RETURN p.title"
```

## Current Results (2026-01-04)

**13 new linkages identified** across 10 concepts:

| Concept | Papers Found |
|---------|--------------|
| maximum_entropy | 5 |
| attractor_dynamics | 2 |
| do_calculus | 2 |
| sparse_coding | 2 |
| reaction_diffusion | 2 |

## Adding New Concepts

Edit the `CONCEPT_PATTERNS` list in the script:

```python
CONCEPT_PATTERNS = [
    # (concept_name, [search_terms], threshold)
    ("your_concept", [
        "canonical term",
        "synonym or abbreviation",
        "author name + concept",
        "domain-specific phrasing"
    ], 0.30),  # Threshold: 0.25-0.35
]
```

**Threshold guidelines:**
- 0.25: Common terms (broader matching)
- 0.30: Standard (balanced)
- 0.35: Rare terms (avoid false positives)

## Safety Features

1. **Dry-run by default**: Must explicitly use `--live`
2. **Idempotent**: Safe to rerun (uses MERGE in Cypher)
3. **Deduplication**: Won't create duplicate links
4. **Filters existing**: Only shows/creates new links

## Output Interpretation

```
Processing: maximum_entropy
  Currently linked: 0 papers          ← Existing state
  Searching with 5 terms...           ← Search terms used
  Found 5 candidate papers            ← Matches above threshold
  New papers to link: 5               ← Excluding already linked

  Top matches:
    [1] [0.469] Maximum Entropy...    ← [rank] [relevance] title
        via: 'entropy maximization'   ← Which search term found it
```

## Troubleshooting

### No matches found

1. **Lower threshold**: Try 0.20 instead of 0.30
2. **Add more search terms**: Try synonyms, author names
3. **Check if papers exist**: Search ChromaDB directly
   ```bash
   python3 /home/user/work/polymax/polymath_cli.py search "your concept terms"
   ```

### "Unknown" titles

- ChromaDB metadata is null for some chunks
- Script extracts from document text
- Check the actual paper with semantic search

### Title extraction fails

Some chunks are mid-paper, not start. Check manually:
```bash
python3 /home/user/work/polymax/polymath_cli.py search "relevant term" -n 10
```

## Files

- **Script:** `/home/user/work/polymax/scripts/fix_concept_links.py`
- **Report:** `/home/user/work/polymax/librarian_reports/interconnection_improvements.md`
- **Preview:** `/home/user/work/polymax/librarian_reports/concept_linking_preview.json`

## Integration with Polymath

After running, use improved concepts in:

```bash
# Cross-reference (now works!)
python3 /home/user/work/polymax/polymath_cli.py crossref \
  "maximum_entropy" "bayesian_inference"

# Hypothesis generation (richer concept pool)
python3 /home/user/work/polymax/polymath_cli.py hypothesis \
  --domain information_theory

# Search still works as before
python3 /home/user/work/polymax/polymath_cli.py search \
  "maximum entropy spatial biology"
```

## Maintenance

Run monthly as corpus grows:
```bash
# Add to cron or run manually
python3 fix_concept_links.py --dry-run --output monthly_report_$(date +%Y%m%d).json
# Review report, then run --live if looks good
```

---

**Author:** Librarian Agent
**Updated:** 2026-01-04
