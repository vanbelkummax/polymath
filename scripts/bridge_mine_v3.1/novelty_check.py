#!/usr/bin/env python3
"""
Novelty Check for Transfer Candidates

Checks PubMed/Semantic Scholar for prior art combining method + spatial transcriptomics.

v4.1 Changes:
- Loads candidates from A6_GAP_CANDIDATES.json (Neo4j output)
- Filters to transfer methods with < 500 mentions
- Dynamically generates search terms and hypotheses
"""

import os
import sys
import json
import time
import re
import requests
from typing import List, Dict, Tuple
import psycopg2

# Import RRF validator for multi-source evidence validation
try:
    from rrf_validator import RRFValidator
    RRF_AVAILABLE = True
except ImportError:
    print("WARNING: RRF validator not available. Run from bridge_mine_v3.1 directory.")
    RRF_AVAILABLE = False

# Database connection for dynamic penetration calculation
PG_CONN = "dbname=polymath user=polymath"


def load_gap_candidates() -> List[dict]:
    """Load gap candidates from Neo4j output and filter to actionable transfers."""
    gap_file = '/home/user/work/polymax/reports/bridge_mine_v4_a6_bucket/A6_GAP_CANDIDATES.json'

    if not os.path.exists(gap_file):
        print(f"ERROR: {gap_file} not found. Run a6_gap_detection.py first.")
        sys.exit(1)

    with open(gap_file) as f:
        raw_candidates = json.load(f)

    print(f"Loaded {len(raw_candidates)} raw gap candidates")

    # Filter to actionable candidates:
    # 1. is_spatial_target = True (target problem is spatial-relevant)
    # 2. is_transfer_method = True (method is in TRANSFER_CANDIDATE_METHODS list)
    # Note: No mention count filter needed - a6_gap_detection.py already filters by TRANSFER_CANDIDATE_METHODS
    filtered = []
    for c in raw_candidates:
        if c.get('is_spatial_target', False) and \
           c.get('is_transfer_method', False):
            filtered.append(c)

    print(f"Filtered to {len(filtered)} actionable candidates (spatial target, transfer method)")

    # Take top 20 by score
    filtered.sort(key=lambda x: -x.get('score', 0))
    return filtered[:20]


def calculate_spatial_penetration(method_name: str) -> float:
    """
    Calculate what % of this method's papers mention spatial transcriptomics.

    Returns percentage (0-100).
    """
    try:
        conn = psycopg2.connect(PG_CONN)
        cursor = conn.cursor()

        # Count papers where this method appears
        cursor.execute("""
            SELECT COUNT(DISTINCT p.doc_id) as total_docs
            FROM passage_concepts pc
            JOIN passages p ON pc.passage_id = p.passage_id
            WHERE pc.concept_name = %s
              AND pc.concept_type IN ('method', 'technique', 'algorithm', 'model')
        """, (method_name,))

        total_docs = cursor.fetchone()[0] or 0

        if total_docs == 0:
            return 0.0

        # Count papers where method AND spatial transcriptomics both appear
        cursor.execute("""
            SELECT COUNT(DISTINCT p.doc_id) as spatial_docs
            FROM passage_concepts pc
            JOIN passages p ON pc.passage_id = p.passage_id
            WHERE pc.concept_name = %s
              AND EXISTS (
                  SELECT 1 FROM passages p2
                  WHERE p2.doc_id = p.doc_id
                    AND (p2.passage_text ILIKE '%%spatial transcriptomics%%'
                         OR p2.passage_text ILIKE '%%spatial omics%%'
                         OR p2.passage_text ILIKE '%%visium%%'
                         OR p2.passage_text ILIKE '%%xenium%%')
              )
        """, (method_name,))

        spatial_docs = cursor.fetchone()[0] or 0

        conn.close()

        penetration = (spatial_docs / total_docs) * 100.0
        return round(penetration, 1)

    except Exception as e:
        print(f"  Error calculating penetration for {method_name}: {e}")
        return 0.0


def enrich_candidate(gap: dict) -> dict:
    """Enrich a gap candidate with search terms, hypothesis, and dynamic penetration."""
    method = gap['method']
    target_problem = gap['target_problem']
    source_problem = gap.get('source_problem', 'unknown')

    # Calculate spatial penetration dynamically
    spatial_pct = calculate_spatial_penetration(method)

    # Generate display name (capitalize, replace underscores)
    display = method.replace('_', ' ').title()

    # Generate transfer hypothesis
    hypothesis = (
        f"Apply {display} (currently used for {source_problem}) "
        f"to solve {target_problem} in spatial transcriptomics"
    )

    # Generate search terms
    search_terms = [
        f"{method.replace('_', ' ')} spatial transcriptomics",
        f"{method.replace('_', ' ')} gene expression",
        f"{method.replace('_', ' ')} spatial omics"
    ]

    return {
        "method": method,
        "display": display,
        "spatial_pct": spatial_pct,
        "total_papers": gap.get('method_docs', 0),
        "transfer_hypothesis": hypothesis,
        "search_terms": search_terms,
        "source_problem": source_problem,
        "target_problem": target_problem,
        "neo4j_score": gap.get('score', 0)
    }


# Load and enrich candidates dynamically
print("="*70)
print("LOADING TRANSFER CANDIDATES FROM NEO4J OUTPUT")
print("="*70)

TRANSFER_CANDIDATES = [enrich_candidate(gap) for gap in load_gap_candidates()]

print(f"\nEnriched {len(TRANSFER_CANDIDATES)} candidates:")
for i, c in enumerate(TRANSFER_CANDIDATES[:5], 1):
    print(f"  {i}. {c['display']} → {c['target_problem']} (penetration: {c['spatial_pct']}%)")

if not TRANSFER_CANDIDATES:
    print("\nERROR: No actionable transfer candidates found.")
    print("This likely means:")
    print("  1. A6_GAP_CANDIDATES.json has no candidates with is_transfer_method=True")
    print("  2. All methods have > 500 mentions (too generic)")
    print("\nRun a6_gap_detection.py with updated TRANSFER_CANDIDATE_METHODS list.")
    sys.exit(1)


def check_pubmed_with_years(query: str) -> Tuple[int, List[int]]:
    """
    Check PubMed for papers matching query and extract years.

    Returns: (total_count, list_of_years)
    """
    base_url_search = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esearch.fcgi"
    base_url_summary = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esummary.fcgi"

    params_search = {
        "db": "pubmed",
        "term": query,
        "retmax": 50,  # Increased from 5 to 50
        "retmode": "json"
    }

    try:
        # Get PMIDs
        resp = requests.get(base_url_search, params=params_search, timeout=10)
        data = resp.json()
        count = int(data.get("esearchresult", {}).get("count", 0))
        ids = data.get("esearchresult", {}).get("idlist", [])

        if not ids:
            return count, []

        # Get years for these PMIDs (batch request)
        params_summary = {
            "db": "pubmed",
            "id": ",".join(ids[:20]),  # Limit to first 20 for speed
            "retmode": "json"
        }
        resp_summary = requests.get(base_url_summary, params=params_summary, timeout=10)
        data_summary = resp_summary.json()

        years = []
        if "result" in data_summary:
            for pmid in ids[:20]:
                if pmid in data_summary["result"]:
                    pub_date = data_summary["result"][pmid].get("pubdate", "")
                    # Extract year (format: "2023 Nov 15" or "2023")
                    year_match = re.search(r'\b(19|20)\d{2}\b', pub_date)
                    if year_match:
                        years.append(int(year_match.group()))

        return count, years

    except Exception as e:
        print(f"  PubMed error: {e}")
        return -1, []


def check_pubmed(query: str) -> Tuple[int, List[str]]:
    """Legacy check_pubmed for backwards compatibility."""
    count, years = check_pubmed_with_years(query)
    return count, []  # Return empty list for IDs (not used)


def check_semantic_scholar(query: str) -> Tuple[int, List[Dict]]:
    """Check Semantic Scholar for papers matching query."""
    base_url = "https://api.semanticscholar.org/graph/v1/paper/search"
    params = {
        "query": query,
        "limit": 5,
        "fields": "title,year,citationCount"
    }

    try:
        resp = requests.get(base_url, params=params, timeout=10)
        if resp.status_code == 429:
            print("  S2 rate limited, waiting...")
            time.sleep(5)
            return -1, []
        data = resp.json()
        total = data.get("total", 0)
        papers = data.get("data", [])
        return total, papers
    except Exception as e:
        print(f"  S2 error: {e}")
        return -1, []


def run_rrf_validation(candidates: List[dict], top_n: int = 5) -> List[dict]:
    """
    Run RRF validation on top N candidates to check internal evidence support.

    Args:
        candidates: List of candidates sorted by novelty score
        top_n: Number of top candidates to validate

    Returns:
        Updated candidates with RRF evidence scores
    """
    if not RRF_AVAILABLE:
        print("\n⚠ RRF validation skipped (rrf_validator not available)")
        return candidates

    print("\n" + "="*70)
    print(f"RRF VALIDATION FOR TOP {top_n} CANDIDATES")
    print("="*70)

    validator = RRFValidator()

    for i, candidate in enumerate(candidates[:top_n]):
        print(f"\n{i+1}. Validating: {candidate['display']}")

        # Extract method and target problem
        method = candidate['method']
        target_problem = candidate['target_problem']

        # Run RRF validation
        fused = validator.validate_transfer(
            method=method,
            problem=target_problem.replace('_', ' '),
            use_postgres=True,
            use_chromadb=True,
            limit_per_source=5
        )

        # Update candidate with RRF evidence
        if fused:
            top_evidence = fused[0]
            candidate['rrf_score'] = top_evidence.rrf_score
            candidate['rrf_confidence'] = top_evidence.confidence
            candidate['rrf_sources'] = list(top_evidence.sources)
            candidate['rrf_evidence_count'] = len(fused)

            print(f"   ✓ RRF Score: {top_evidence.rrf_score:.4f} ({top_evidence.confidence} confidence)")
            print(f"   Sources: {', '.join(top_evidence.sources)}")
        else:
            candidate['rrf_score'] = 0.0
            candidate['rrf_confidence'] = 'none'
            candidate['rrf_sources'] = []
            candidate['rrf_evidence_count'] = 0
            print(f"   ✗ No internal evidence found")

    return candidates


def run_novelty_check(run_rrf: bool = True):
    """
    Run novelty check on all transfer candidates with year weighting.

    Args:
        run_rrf: Whether to run RRF validation on top candidates
    """
    print("="*70)
    print("NOVELTY CHECK FOR TRANSFER CANDIDATES (v4.1)")
    print("="*70)

    results = []

    for i, candidate in enumerate(TRANSFER_CANDIDATES):
        print(f"\n{i+1}. {candidate['display']}")
        print(f"   Spatial penetration: {candidate['spatial_pct']}%")
        print(f"   Hypothesis: {candidate['transfer_hypothesis'][:80]}...")

        novelty_score = 100  # Start with max novelty
        prior_art = []
        recent_count = 0
        older_count = 0

        for search_term in candidate['search_terms']:
            # Check PubMed with year extraction
            pm_count, pm_years = check_pubmed_with_years(search_term)
            if pm_count > 0:
                # Weight by recency
                recent = sum(1 for y in pm_years if y >= 2023)
                older = len(pm_years) - recent
                recent_count += recent
                older_count += older

                print(f"   PubMed '{search_term}': {pm_count} total ({recent} recent, {older} older)")

                # Recent work = lower novelty (10 points per paper)
                novelty_score -= recent * 10
                # Older work = less penalty (2 points per paper)
                novelty_score -= older * 2

                prior_art.append(f"PubMed: {pm_count} ({recent} since 2023)")

            # Check Semantic Scholar
            s2_count, s2_papers = check_semantic_scholar(search_term)
            if s2_count > 0:
                print(f"   S2 '{search_term}': {s2_count} results")
                # S2 penalty (less weight than PubMed)
                novelty_score -= min(30, s2_count * 2)
                if s2_papers:
                    for p in s2_papers[:2]:
                        year = p.get('year', 'Unknown')
                        prior_art.append(f"S2 ({year}): {p.get('title', 'Unknown')[:50]}...")

            time.sleep(0.5)  # Rate limiting

        novelty_score = max(0, novelty_score)

        result = {
            **candidate,
            "novelty_score": novelty_score,
            "prior_art_found": len(prior_art) > 0,
            "prior_art": prior_art,
            "recent_prior_art": recent_count,
            "older_prior_art": older_count,
            "recommendation": "INVESTIGATE" if novelty_score >= 50 else "EXISTING" if novelty_score >= 20 else "WELL-STUDIED"
        }
        results.append(result)

        if novelty_score >= 70:
            print(f"   ✓ HIGH NOVELTY ({novelty_score}/100) - Potential discovery!")
        elif novelty_score >= 40:
            print(f"   ~ MEDIUM NOVELTY ({novelty_score}/100) - Some prior work exists")
        else:
            print(f"   ✗ LOW NOVELTY ({novelty_score}/100) - Well-studied area")

    # Sort by novelty score
    results.sort(key=lambda x: -x['novelty_score'])

    # Run RRF validation on top candidates
    if run_rrf:
        results = run_rrf_validation(results, top_n=5)

    # Save results
    output_dir = '/home/user/work/polymax/reports/bridge_mine_v4_a6_bucket'
    os.makedirs(output_dir, exist_ok=True)

    output_file = os.path.join(output_dir, 'TOP10_GAPS_WITH_NOVELTY.json')
    with open(output_file, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\n\nSaved to {output_file}")

    # Generate summary
    print("\n" + "="*70)
    print("TOP 10 TRANSFER OPPORTUNITIES (by Novelty)")
    print("="*70)

    for i, r in enumerate(results):
        status = "🎯" if r['novelty_score'] >= 50 else "📚"
        print(f"\n{i+1}. {status} {r['display']}")
        print(f"   Novelty: {r['novelty_score']}/100 | Spatial: {r['spatial_pct']}%")
        print(f"   {r['transfer_hypothesis']}")
        print(f"   Recommendation: {r['recommendation']}")

    # Generate markdown report
    report_file = os.path.join(output_dir, 'TOP10_TRANSFER_OPPORTUNITIES.md')
    with open(report_file, 'w') as f:
        f.write("# Top 10 Transfer Opportunities for Spatial Transcriptomics\n\n")
        f.write(f"**Generated**: {time.strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        f.write("## Executive Summary\n\n")
        f.write("These methods have high usage in the literature but low penetration into spatial transcriptomics, ")
        f.write("representing potential transfer opportunities.\n\n")

        f.write("| Rank | Method | Novelty | Spatial % | Recommendation |\n")
        f.write("|------|--------|---------|-----------|----------------|\n")
        for i, r in enumerate(results):
            f.write(f"| {i+1} | {r['display']} | {r['novelty_score']}/100 | {r['spatial_pct']}% | {r['recommendation']} |\n")

        f.write("\n## Detailed Analysis\n\n")
        for i, r in enumerate(results):
            f.write(f"### {i+1}. {r['display']}\n\n")
            f.write(f"- **Novelty Score**: {r['novelty_score']}/100\n")
            f.write(f"- **Current Spatial Penetration**: {r['spatial_pct']}%\n")
            f.write(f"- **Total Papers in Corpus**: {r['total_papers']}\n")
            f.write(f"- **Recommendation**: {r['recommendation']}\n")

            # Add RRF evidence if available
            if 'rrf_score' in r:
                f.write(f"- **RRF Evidence Score**: {r['rrf_score']:.4f} ({r['rrf_confidence']} confidence)\n")
                if r['rrf_sources']:
                    f.write(f"- **Evidence Sources**: {', '.join(r['rrf_sources'])} ({r['rrf_evidence_count']} spans)\n")

            f.write(f"\n**Transfer Hypothesis**: {r['transfer_hypothesis']}\n\n")

            if r['prior_art']:
                f.write("**Prior Art Found**:\n")
                for pa in r['prior_art'][:3]:
                    f.write(f"- {pa}\n")
            f.write("\n---\n\n")

    print(f"\nSaved markdown report to {report_file}")

    return results


if __name__ == "__main__":
    run_novelty_check()
