#!/usr/bin/env python3
"""
Golden Query Evaluation Harness

Runs a fixed query set and reports stability metrics. Supports:
- record mode: store baseline results
- compare mode: compare against baseline for regression detection
"""

import argparse
import json
import sys
from pathlib import Path
from typing import List, Dict, Any

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from lib.hybrid_search import HybridSearcher


def load_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def write_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)


def build_expected_terms(entry: Dict[str, Any]) -> List[str]:
    terms: List[str] = []
    for key in (
        "expected_terms",
        "expected_concepts",
        "expected_papers_keywords",
        "expected_repos",
        "expected_domains",
    ):
        values = entry.get(key, [])
        if values:
            terms.extend(values)
    # De-duplicate while preserving order
    seen = set()
    unique_terms = []
    for term in terms:
        if term not in seen:
            seen.add(term)
            unique_terms.append(term)
    return unique_terms


def collect_hits(results, expected_terms: List[str], rank_limit: int) -> List[str]:
    hits = set()
    for idx, result in enumerate(results):
        if idx >= rank_limit:
            break
        meta = json.dumps(result.metadata, default=str) if result.metadata else ""
        haystack = f"{result.title} {result.content} {meta}".lower()
        for term in expected_terms:
            term_lower = term.lower()
            variants = {
                term_lower,
                term_lower.replace("_", " "),
                term_lower.replace("_", "-"),
            }
            if any(variant in haystack for variant in variants):
                hits.add(term)
    return sorted(hits)


def run_queries(
    queries: List[Dict[str, Any]],
    k: int,
    use_rosetta: bool = False,
) -> List[Dict[str, Any]]:
    searcher = HybridSearcher()
    run_results = []

    for entry in queries:
        query = entry["query"]
        expected_terms = build_expected_terms(entry)
        min_expected_hits = entry.get("min_expected_hits")
        if min_expected_hits is None:
            min_expected_hits = 1 if expected_terms else 0

        try:
            rank_limit = entry.get("min_relevant_rank") or k
            search_k = max(k, rank_limit)
            results = searcher.hybrid_search(query, n_results=search_k, use_rosetta=use_rosetta)
            result_ids = [r.id for r in results]
            hits = collect_hits(results, expected_terms, rank_limit)
            run_results.append({
                "id": entry.get("id"),
                "query": query,
                "expected_terms": expected_terms,
                "min_relevant_rank": entry.get("min_relevant_rank"),
                "min_expected_hits": min_expected_hits,
                "hits": hits,
                "hit_count": len(hits),
                "result_ids": result_ids,
                "results": [
                    {
                        "id": r.id,
                        "title": r.title,
                        "score": r.score,
                        "source": r.source,
                    }
                    for r in results
                ],
            })
        except Exception as e:
            run_results.append({
                "id": entry.get("id"),
                "query": query,
                "expected_terms": expected_terms,
                "min_expected_hits": min_expected_hits,
                "hits": [],
                "hit_count": 0,
                "result_ids": [],
                "results": [],
                "error": str(e),
            })

    return run_results


def compare_baseline(baseline: List[Dict[str, Any]], current: List[Dict[str, Any]]) -> Dict[str, Any]:
    baseline_map = {entry["id"]: entry for entry in baseline}
    comparisons = []

    for entry in current:
        base_entry = baseline_map.get(entry["id"])
        if not base_entry:
            comparisons.append({
                "id": entry["id"],
                "overlap": 0.0,
                "note": "missing baseline",
            })
            continue

        base_ids = set(base_entry.get("result_ids", []))
        curr_ids = set(entry.get("result_ids", []))
        overlap = len(base_ids & curr_ids) / max(1, len(base_ids))
        comparisons.append({
            "id": entry["id"],
            "overlap": round(overlap, 3),
        })

    return {
        "comparisons": comparisons,
        "avg_overlap": round(
            sum(c["overlap"] for c in comparisons) / max(1, len(comparisons)), 3
        ),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Run golden query evaluation")
    parser.add_argument(
        "--queries",
        default=str(ROOT / "tests" / "golden_queries.json"),
        help="Path to golden queries JSON",
    )
    parser.add_argument(
        "--baseline",
        default=str(ROOT / "tests" / "golden_baseline.json"),
        help="Path to baseline JSON",
    )
    parser.add_argument(
        "--output",
        default=str(ROOT / "tests" / "golden_run.json"),
        help="Path to write current run output",
    )
    parser.add_argument("--k", type=int, default=5, help="Top-K results to record")
    parser.add_argument("--record", action="store_true", help="Record baseline results")
    parser.add_argument("--strict", action="store_true", help="Fail on unmet thresholds")
    parser.add_argument("--min-overlap", type=float, default=0.3, help="Minimum baseline overlap")
    parser.add_argument("--rosetta", action="store_true", help="Enable Rosetta Stone expansion")

    args = parser.parse_args()

    queries_path = Path(args.queries)
    baseline_path = Path(args.baseline)
    output_path = Path(args.output)

    queries = load_json(queries_path)
    run_results = run_queries(queries, k=args.k, use_rosetta=args.rosetta)
    write_json(output_path, run_results)

    if args.record:
        write_json(baseline_path, run_results)
        print(f"Baseline written to {baseline_path}")
        return 0

    summary = {
        "total_queries": len(run_results),
        "errors": sum(1 for r in run_results if r.get("error")),
        "zero_hits": sum(1 for r in run_results if r.get("hit_count", 0) == 0),
    }

    print(f"Queries: {summary['total_queries']}, Errors: {summary['errors']}, Zero hits: {summary['zero_hits']}")

    if baseline_path.exists():
        baseline = load_json(baseline_path)
        comparison = compare_baseline(baseline, run_results)
        write_json(output_path, {"results": run_results, "comparison": comparison})
        print(f"Baseline overlap avg: {comparison['avg_overlap']}")

        if args.strict:
            low_overlap = [c for c in comparison["comparisons"] if c["overlap"] < args.min_overlap]
            if low_overlap:
                print(f"FAIL: {len(low_overlap)} queries below overlap threshold")
                return 2
    elif args.strict:
        missing_hits = [
            r for r in run_results
            if r.get("hit_count", 0) < r.get("min_expected_hits", 0)
        ]
        if missing_hits:
            print(f"FAIL: {len(missing_hits)} queries below expected hit threshold")
            return 2

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
