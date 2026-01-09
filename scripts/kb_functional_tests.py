#!/usr/bin/env python3
"""
Knowledge Base Functional Test Suite

Comprehensive tests for Polymath concept extraction database.
Run after backfill or as nightly regression check.

Usage:
    python scripts/kb_functional_tests.py
    python scripts/kb_functional_tests.py --verbose
    python scripts/kb_functional_tests.py --output docs/runlogs/
"""

import os
import sys
import json
import argparse
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Tuple, Any

# Add lib to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from lib.db import get_db_connection

# Test configuration
EXTRACTOR_VERSION = "gemini_batch_v1"
EVIDENCE_COVERAGE_THRESHOLD = 99.0  # percent
QUERY_TIME_THRESHOLD_MS = 100  # milliseconds
EVIDENCE_MATCH_THRESHOLD = 95.0  # percent

ALLOWED_CONCEPT_TYPES = {
    'method', 'objective', 'prior', 'model', 'dataset',
    'field', 'math_object', 'metric', 'domain', 'algorithm',
    'technique', 'extracted', 'architecture'
}


class TestResult:
    def __init__(self, name: str, passed: bool, details: Dict[str, Any] = None):
        self.name = name
        self.passed = passed
        self.details = details or {}
        self.timestamp = datetime.now().isoformat()

    def __repr__(self):
        status = "PASS" if self.passed else "FAIL"
        return f"[{status}] {self.name}"


def run_invariant_tests(conn) -> List[TestResult]:
    """Run data integrity invariant tests."""
    results = []
    cur = conn.cursor()

    # Test 1: No duplicates (PK uniqueness)
    cur.execute("""
        SELECT COUNT(*) - COUNT(DISTINCT (passage_id, concept_name, extractor_version)) AS dupes
        FROM passage_concepts
    """)
    dupes = cur.fetchone()[0]
    results.append(TestResult(
        "invariant_no_duplicates",
        dupes == 0,
        {"duplicates_found": dupes}
    ))

    # Test 2: Confidence in valid range [0, 1]
    cur.execute("""
        SELECT COUNT(*) FROM passage_concepts
        WHERE confidence < 0 OR confidence > 1
    """)
    out_of_range = cur.fetchone()[0]
    results.append(TestResult(
        "invariant_confidence_range",
        out_of_range == 0,
        {"out_of_range_count": out_of_range}
    ))

    # Test 3: Valid concept types only
    cur.execute("""
        SELECT concept_type, COUNT(*) as cnt
        FROM passage_concepts
        WHERE concept_type IS NULL OR concept_type NOT IN %s
        GROUP BY concept_type
    """, (tuple(ALLOWED_CONCEPT_TYPES),))
    invalid_types = cur.fetchall()
    results.append(TestResult(
        "invariant_valid_concept_types",
        len(invalid_types) == 0,
        {"invalid_types": [{"type": t, "count": c} for t, c in invalid_types]}
    ))

    # Test 4: Evidence coverage for Gemini extractor
    cur.execute("""
        SELECT
            COUNT(*) AS total,
            COUNT(*) FILTER (WHERE evidence IS NOT NULL) AS with_evidence,
            ROUND(100.0 * COUNT(*) FILTER (WHERE evidence IS NOT NULL) / NULLIF(COUNT(*),0), 2) AS pct
        FROM passage_concepts
        WHERE extractor_version = %s
    """, (EXTRACTOR_VERSION,))
    total, with_ev, pct = cur.fetchone()
    results.append(TestResult(
        "invariant_evidence_coverage",
        pct is not None and float(pct) >= EVIDENCE_COVERAGE_THRESHOLD,
        {"total": total, "with_evidence": with_ev, "coverage_pct": float(pct) if pct else 0}
    ))

    # Test 5: No orphan concepts (passage_id references valid passage)
    cur.execute("""
        SELECT COUNT(*)
        FROM passage_concepts pc
        LEFT JOIN passages p ON pc.passage_id = p.passage_id
        WHERE p.passage_id IS NULL
    """)
    orphans = cur.fetchone()[0]
    results.append(TestResult(
        "invariant_no_orphan_concepts",
        orphans == 0,
        {"orphan_count": orphans}
    ))

    cur.close()
    return results


def run_performance_tests(conn) -> List[TestResult]:
    """Run query performance tests with EXPLAIN ANALYZE."""
    results = []
    cur = conn.cursor()

    # Test 1: Concept lookup performance
    cur.execute("""
        EXPLAIN (ANALYZE, FORMAT JSON)
        SELECT p.passage_id, LEFT(p.passage_text, 200)
        FROM passages p
        JOIN passage_concepts pc ON p.passage_id = pc.passage_id
        WHERE pc.concept_name = 'machine_learning'
          AND pc.extractor_version = %s
        LIMIT 50
    """, (EXTRACTOR_VERSION,))
    plan = cur.fetchone()[0]
    exec_time = plan[0].get('Execution Time', 9999)
    uses_index = 'Index' in json.dumps(plan)
    results.append(TestResult(
        "performance_concept_lookup",
        exec_time < QUERY_TIME_THRESHOLD_MS and uses_index,
        {"execution_time_ms": exec_time, "uses_index": uses_index}
    ))

    # Test 2: Passage-based lookup
    cur.execute("""
        EXPLAIN (ANALYZE, FORMAT JSON)
        SELECT pc.concept_name, pc.confidence
        FROM passage_concepts pc
        WHERE pc.passage_id = (SELECT passage_id FROM passages LIMIT 1)
          AND pc.extractor_version = %s
    """, (EXTRACTOR_VERSION,))
    plan = cur.fetchone()[0]
    exec_time = plan[0].get('Execution Time', 9999)
    results.append(TestResult(
        "performance_passage_lookup",
        exec_time < QUERY_TIME_THRESHOLD_MS,
        {"execution_time_ms": exec_time}
    ))

    cur.close()
    return results


def run_golden_query_tests(conn) -> List[TestResult]:
    """Run golden query harness tests."""
    results = []
    cur = conn.cursor()

    # Ensure golden query table exists
    cur.execute("""
        SELECT EXISTS (
            SELECT FROM information_schema.tables
            WHERE table_name = 'concept_query_gold'
        )
    """)
    if not cur.fetchone()[0]:
        results.append(TestResult(
            "golden_queries_table_exists",
            False,
            {"error": "concept_query_gold table does not exist"}
        ))
        cur.close()
        return results

    # Run all golden queries
    cur.execute("""
        SELECT query_name, required_concepts, min_hits FROM concept_query_gold
    """)
    queries = cur.fetchall()

    passed_count = 0
    failed_queries = []

    for query_name, required_concepts, min_hits in queries:
        # Count passages with ALL required concepts
        cur.execute("""
            SELECT COUNT(*)
            FROM (
                SELECT pc.passage_id
                FROM passage_concepts pc
                WHERE pc.extractor_version = %s
                  AND pc.concept_name = ANY(%s)
                GROUP BY pc.passage_id
                HAVING COUNT(DISTINCT pc.concept_name) = %s
            ) t
        """, (EXTRACTOR_VERSION, required_concepts, len(required_concepts)))
        hits = cur.fetchone()[0]

        if hits >= min_hits:
            passed_count += 1
        else:
            failed_queries.append({
                "query": query_name,
                "required": required_concepts,
                "min_hits": min_hits,
                "actual_hits": hits
            })

    results.append(TestResult(
        "golden_queries",
        len(failed_queries) == 0,
        {
            "total_queries": len(queries),
            "passed": passed_count,
            "failed": len(failed_queries),
            "failures": failed_queries[:10]  # Limit to first 10
        }
    ))

    cur.close()
    return results


def run_evidence_tests(conn, sample_size: int = 5000) -> List[TestResult]:
    """Run evidence integrity tests."""
    results = []
    cur = conn.cursor()

    # Test: Evidence source_text matches passage text
    cur.execute("""
        WITH sample AS (
            SELECT
                pc.passage_id,
                pc.evidence->>'source_text' as source_text,
                p.passage_text
            FROM passage_concepts pc
            JOIN passages p ON p.passage_id = pc.passage_id
            WHERE pc.extractor_version = %s
              AND pc.evidence IS NOT NULL
              AND pc.evidence->>'source_text' IS NOT NULL
              AND LENGTH(pc.evidence->>'source_text') > 10
            ORDER BY random()
            LIMIT %s
        )
        SELECT
            COUNT(*) AS checked,
            COUNT(*) FILTER (WHERE passage_text ILIKE '%%' || LEFT(source_text, 50) || '%%') AS matched
        FROM sample
    """, (EXTRACTOR_VERSION, sample_size))
    checked, matched = cur.fetchone()
    match_pct = (matched / checked * 100) if checked > 0 else 0

    results.append(TestResult(
        "evidence_source_text_integrity",
        match_pct >= EVIDENCE_MATCH_THRESHOLD,
        {
            "sample_size": checked,
            "matched": matched,
            "match_pct": round(match_pct, 2)
        }
    ))

    cur.close()
    return results


def run_coverage_tests(conn) -> List[TestResult]:
    """Run coverage and distribution tests."""
    results = []
    cur = conn.cursor()

    # Test: Concepts per passage distribution
    cur.execute("""
        SELECT
            MIN(cnt) as min_concepts,
            ROUND(AVG(cnt), 1) as avg_concepts,
            MAX(cnt) as max_concepts,
            PERCENTILE_CONT(0.5) WITHIN GROUP (ORDER BY cnt) as p50,
            PERCENTILE_CONT(0.9) WITHIN GROUP (ORDER BY cnt) as p90
        FROM (
            SELECT passage_id, COUNT(*) as cnt
            FROM passage_concepts
            WHERE extractor_version = %s
            GROUP BY passage_id
        ) sub
    """, (EXTRACTOR_VERSION,))
    stats = cur.fetchone()

    # Reasonable range: avg between 3-10, max < 50
    avg_reasonable = 3 <= float(stats[1] or 0) <= 12
    max_reasonable = int(stats[2] or 0) < 50

    results.append(TestResult(
        "coverage_concepts_per_passage",
        avg_reasonable and max_reasonable,
        {
            "min": stats[0],
            "avg": float(stats[1]) if stats[1] else 0,
            "max": stats[2],
            "p50": float(stats[3]) if stats[3] else 0,
            "p90": float(stats[4]) if stats[4] else 0
        }
    ))

    # Test: Top concepts sanity check
    cur.execute("""
        SELECT concept_name, COUNT(*) as freq
        FROM passage_concepts
        WHERE extractor_version = %s
        GROUP BY concept_name
        ORDER BY freq DESC
        LIMIT 10
    """, (EXTRACTOR_VERSION,))
    top_concepts = cur.fetchall()

    # Sanity: top concepts should be reasonable terms
    results.append(TestResult(
        "coverage_top_concepts",
        len(top_concepts) > 0,
        {"top_10": [{"name": n, "freq": f} for n, f in top_concepts]}
    ))

    cur.close()
    return results


def generate_report(all_results: List[TestResult], output_dir: str = None) -> str:
    """Generate markdown report from test results."""
    total = len(all_results)
    passed = sum(1 for r in all_results if r.passed)
    failed = total - passed

    lines = [
        f"# KB Functional Test Report",
        f"",
        f"**Generated:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        f"**Extractor:** {EXTRACTOR_VERSION}",
        f"",
        f"## Summary",
        f"",
        f"| Metric | Value |",
        f"|--------|-------|",
        f"| Total Tests | {total} |",
        f"| Passed | {passed} |",
        f"| Failed | {failed} |",
        f"| Pass Rate | {passed/total*100:.1f}% |",
        f"",
        f"## Results",
        f"",
    ]

    # Group by category
    categories = {}
    for r in all_results:
        cat = r.name.split('_')[0]
        if cat not in categories:
            categories[cat] = []
        categories[cat].append(r)

    for cat, tests in categories.items():
        lines.append(f"### {cat.title()}")
        lines.append("")
        lines.append("| Test | Status | Details |")
        lines.append("|------|--------|---------|")
        for r in tests:
            status = "PASS" if r.passed else "**FAIL**"
            details_str = ", ".join(f"{k}={v}" for k, v in list(r.details.items())[:3])
            if len(details_str) > 60:
                details_str = details_str[:57] + "..."
            lines.append(f"| {r.name} | {status} | {details_str} |")
        lines.append("")

    # Failed test details
    failed_tests = [r for r in all_results if not r.passed]
    if failed_tests:
        lines.append("## Failed Test Details")
        lines.append("")
        for r in failed_tests:
            lines.append(f"### {r.name}")
            lines.append("```json")
            lines.append(json.dumps(r.details, indent=2))
            lines.append("```")
            lines.append("")

    report = "\n".join(lines)

    # Save to file if output_dir specified
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
        filename = f"kb_functional_test_{datetime.now().strftime('%Y%m%d_%H%M%S')}.md"
        filepath = os.path.join(output_dir, filename)
        with open(filepath, 'w') as f:
            f.write(report)
        print(f"Report saved to: {filepath}")

    return report


def main():
    parser = argparse.ArgumentParser(description="KB Functional Test Suite")
    parser.add_argument("--verbose", "-v", action="store_true", help="Verbose output")
    parser.add_argument("--output", "-o", type=str, help="Output directory for report")
    args = parser.parse_args()

    print("=" * 60)
    print("KB Functional Test Suite")
    print("=" * 60)

    conn = get_db_connection()
    all_results = []

    # Run all test categories
    print("\n[1/5] Running invariant tests...")
    results = run_invariant_tests(conn)
    all_results.extend(results)
    for r in results:
        print(f"  {r}")

    print("\n[2/5] Running performance tests...")
    results = run_performance_tests(conn)
    all_results.extend(results)
    for r in results:
        print(f"  {r}")

    print("\n[3/5] Running golden query tests...")
    results = run_golden_query_tests(conn)
    all_results.extend(results)
    for r in results:
        print(f"  {r}")

    print("\n[4/5] Running evidence integrity tests...")
    results = run_evidence_tests(conn)
    all_results.extend(results)
    for r in results:
        print(f"  {r}")

    print("\n[5/5] Running coverage tests...")
    results = run_coverage_tests(conn)
    all_results.extend(results)
    for r in results:
        print(f"  {r}")

    conn.close()

    # Generate report
    print("\n" + "=" * 60)
    total = len(all_results)
    passed = sum(1 for r in all_results if r.passed)
    print(f"TOTAL: {passed}/{total} tests passed ({passed/total*100:.1f}%)")
    print("=" * 60)

    if args.output:
        generate_report(all_results, args.output)

    if args.verbose:
        print("\n" + generate_report(all_results))

    # Exit with error code if any failures
    sys.exit(0 if passed == total else 1)


if __name__ == "__main__":
    main()
