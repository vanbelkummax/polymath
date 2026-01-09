#!/usr/bin/env python3
"""
A/B Canary Test for Gemini Model Comparison

Compares parse-failure rate, latency, and output quality between models.
"""

import os
import sys
import time
import json
import logging
import statistics
from typing import List, Dict, Tuple

# Add lib to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from lib.gemini_batch import (
    BatchRequest, run_sync_batch, parse_concept_response,
    get_genai_client, build_extraction_prompt, CONCEPT_SCHEMA, CONCEPT_SCHEMA_TIGHT
)

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def get_sample_passages(n: int = 200) -> List[Tuple[str, str]]:
    """Get random sample of passages for testing."""
    import psycopg2

    conn = psycopg2.connect(
        dbname="polymath",
        user="polymath",
        host="/var/run/postgresql"
    )
    cur = conn.cursor()

    # Get random passages (fast - skip the NOT IN which is slow)
    cur.execute("""
        SELECT passage_id::text, passage_text
        FROM passages
        WHERE LENGTH(passage_text) > 200
        ORDER BY RANDOM()
        LIMIT %s
    """, (n,))

    results = cur.fetchall()
    cur.close()
    conn.close()

    return results


def run_model_test(
    passages: List[Tuple[str, str]],
    model: str,
    max_output_tokens: int = 512
) -> Dict:
    """Run test with a specific model and collect metrics."""

    requests = [BatchRequest(custom_id=pid, text=text) for pid, text in passages]

    start_time = time.time()

    # Use lower-level API for more control
    client = get_genai_client()

    parse_failures = 0
    successes = 0
    latencies = []
    concepts_per_passage = []
    total_output_chars = 0
    raw_failures = []

    for i, req in enumerate(requests):
        req_start = time.time()

        try:
            prompt = build_extraction_prompt(req.text)

            response = client.models.generate_content(
                model=model,
                contents=prompt,
                config={
                    "response_mime_type": "application/json",
                    "response_schema": CONCEPT_SCHEMA,
                    "max_output_tokens": max_output_tokens,
                    "temperature": 0,
                }
            )

            req_latency = time.time() - req_start
            latencies.append(req_latency)

            # Extract response
            raw_text = ""
            if response.candidates:
                candidate = response.candidates[0]
                if candidate.content and candidate.content.parts:
                    raw_text = candidate.content.parts[0].text

            total_output_chars += len(raw_text)

            # Try to parse
            concepts = parse_concept_response(raw_text)

            if concepts:
                successes += 1
                concepts_per_passage.append(len(concepts))
            else:
                parse_failures += 1
                raw_failures.append({
                    "passage_id": req.custom_id,
                    "raw": raw_text[:300] if raw_text else None
                })

            # Rate limit
            time.sleep(0.3)

        except Exception as e:
            parse_failures += 1
            latencies.append(time.time() - req_start)
            raw_failures.append({
                "passage_id": req.custom_id,
                "error": str(e)
            })
            time.sleep(1)  # Back off on error

        if (i + 1) % 20 == 0:
            logger.info(f"  {model}: {i + 1}/{len(requests)} done, {parse_failures} failures so far")

    total_time = time.time() - start_time

    return {
        "model": model,
        "total_passages": len(passages),
        "successes": successes,
        "parse_failures": parse_failures,
        "failure_rate": round(parse_failures / len(passages) * 100, 1),
        "mean_latency_ms": round(statistics.mean(latencies) * 1000, 1) if latencies else 0,
        "p95_latency_ms": round(sorted(latencies)[int(len(latencies) * 0.95)] * 1000, 1) if latencies else 0,
        "mean_concepts": round(statistics.mean(concepts_per_passage), 1) if concepts_per_passage else 0,
        "total_output_chars": total_output_chars,
        "avg_output_chars": round(total_output_chars / len(passages), 1),
        "total_time_s": round(total_time, 1),
        "throughput_per_min": round(len(passages) / total_time * 60, 1),
        "sample_failures": raw_failures[:3]  # Keep first 3 for debugging
    }


def main():
    import argparse
    parser = argparse.ArgumentParser(description="A/B model comparison")
    parser.add_argument("--passages", type=int, default=200, help="Total passages to test")
    parser.add_argument("--baseline", default="gemini-2.0-flash", help="Baseline model")
    parser.add_argument("--test", default="gemini-2.5-flash-lite", help="Test model")
    args = parser.parse_args()

    # Check API key
    if not os.environ.get("GEMINI_API_KEY"):
        print("ERROR: GEMINI_API_KEY not set")
        sys.exit(1)

    logger.info(f"Fetching {args.passages} sample passages...")
    all_passages = get_sample_passages(args.passages)

    if len(all_passages) < args.passages:
        logger.warning(f"Only got {len(all_passages)} passages")

    # Split evenly
    mid = len(all_passages) // 2
    baseline_passages = all_passages[:mid]
    test_passages = all_passages[mid:]

    logger.info(f"\n{'='*60}")
    logger.info(f"BASELINE: {args.baseline} ({len(baseline_passages)} passages)")
    logger.info(f"{'='*60}")
    baseline_results = run_model_test(baseline_passages, args.baseline)

    logger.info(f"\n{'='*60}")
    logger.info(f"TEST: {args.test} ({len(test_passages)} passages)")
    logger.info(f"{'='*60}")
    test_results = run_model_test(test_passages, args.test)

    # Print comparison
    print("\n" + "="*70)
    print("RESULTS COMPARISON")
    print("="*70)

    metrics = [
        ("Model", "model"),
        ("Passages", "total_passages"),
        ("Successes", "successes"),
        ("Parse Failures", "parse_failures"),
        ("Failure Rate %", "failure_rate"),
        ("Mean Latency (ms)", "mean_latency_ms"),
        ("P95 Latency (ms)", "p95_latency_ms"),
        ("Mean Concepts", "mean_concepts"),
        ("Avg Output Chars", "avg_output_chars"),
        ("Throughput/min", "throughput_per_min"),
    ]

    print(f"{'Metric':<25} {'Baseline':<20} {'Test':<20} {'Winner':<10}")
    print("-"*70)

    for label, key in metrics:
        b_val = baseline_results[key]
        t_val = test_results[key]

        # Determine winner (lower is better for failures/latency, higher for others)
        winner = ""
        if key in ["parse_failures", "failure_rate", "mean_latency_ms", "p95_latency_ms"]:
            if isinstance(b_val, (int, float)) and isinstance(t_val, (int, float)):
                winner = "TEST" if t_val < b_val else ("BASELINE" if b_val < t_val else "TIE")
        elif key in ["successes", "mean_concepts", "throughput_per_min"]:
            if isinstance(b_val, (int, float)) and isinstance(t_val, (int, float)):
                winner = "TEST" if t_val > b_val else ("BASELINE" if b_val > t_val else "TIE")

        print(f"{label:<25} {str(b_val):<20} {str(t_val):<20} {winner:<10}")

    print("\n" + "="*70)

    # Recommendation
    baseline_fail = baseline_results["failure_rate"]
    test_fail = test_results["failure_rate"]

    if test_fail < baseline_fail * 0.7:  # 30%+ improvement
        print(f"✅ RECOMMENDATION: Switch to {args.test}")
        print(f"   Failure rate dropped from {baseline_fail}% to {test_fail}%")
    elif test_fail < baseline_fail:
        print(f"⚠️  MARGINAL: {args.test} is slightly better ({test_fail}% vs {baseline_fail}%)")
        print("   Consider running longer test")
    else:
        print(f"❌ KEEP BASELINE: {args.baseline} performs better or equal")

    # Save full results
    output_file = "/home/user/polymath-repo/logs/canary_test_results.json"
    with open(output_file, "w") as f:
        json.dump({
            "baseline": baseline_results,
            "test": test_results,
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S")
        }, f, indent=2)

    print(f"\nFull results saved to: {output_file}")

    # Show sample failures if any
    if baseline_results["sample_failures"]:
        print(f"\nBaseline sample failures:")
        for f in baseline_results["sample_failures"][:2]:
            print(f"  - {f}")

    if test_results["sample_failures"]:
        print(f"\nTest sample failures:")
        for f in test_results["sample_failures"][:2]:
            print(f"  - {f}")


if __name__ == "__main__":
    main()
