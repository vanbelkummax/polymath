#!/usr/bin/env python3
"""
Polymathic Exam Readiness Check

Validates system capabilities across all polymathic domains before exam.
"""

import sys
sys.path.insert(0, '/home/user/work/polymax/lib')
sys.path.insert(0, '/home/user/work/polymax/mcp')

from polymathic_search import polymathic_search, oblique_search, domain_bridge_search
import chromadb
from datetime import datetime


def test_polymathic_domains():
    """Test coverage across major polymathic domains"""

    print("=" * 80)
    print("POLYMATHIC EXAM READINESS CHECK")
    print("=" * 80)
    print(f"Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print()

    # Test queries across domains
    test_queries = [
        # Mathematics → Biology
        ("Topology in tissue morphology", "Math→Bio"),
        ("Graph theory for molecular networks", "Math→Bio"),
        ("Differential equations morphogenesis", "Math→Bio"),

        # Physics → Medicine
        ("Thermodynamics of disease", "Physics→Med"),
        ("Statistical mechanics cellular systems", "Physics→Bio"),
        ("Diffusion in tissue microenvironment", "Physics→Bio"),

        # Computer Science → Biology
        ("Graph neural networks protein folding", "CS→Bio"),
        ("Transformer architecture genomics", "CS→Bio"),
        ("Active inference neuroscience", "CS→Neuro"),

        # Cross-domain concepts
        ("Compressed sensing spatial transcriptomics", "Signal→Bio"),
        ("Information theory evolution", "Info→Bio"),
        ("Causal inference genomics", "Stats→Bio"),
        ("Bayesian optimization drug discovery", "Stats→Pharma"),
        ("Reaction-diffusion pattern formation", "Physics→Bio"),

        # Key implementations (code)
        ("Topological data analysis python", "Code"),
        ("Active inference implementation", "Code"),
        ("Spatial transcriptomics deconvolution", "Code"),
        ("Foundation model pathology", "Code"),
    ]

    scores = {"passed": 0, "failed": 0}
    failures = []

    print("Testing polymathic queries...")
    print()

    for query, domain_bridge in test_queries:
        try:
            results = polymathic_search(query, n_results=10, use_rosetta=True)

            # Success criteria: at least 5 relevant results
            success = results['total_found'] >= 5

            status = "✅ PASS" if success else "❌ FAIL"
            if success:
                scores["passed"] += 1
            else:
                scores["failed"] += 1
                failures.append((query, domain_bridge, results['total_found']))

            print(f"{status} [{domain_bridge:15s}] {query[:50]:50s} ({results['total_found']} results)")

        except Exception as e:
            print(f"❌ ERROR [{domain_bridge:15s}] {query[:50]:50s} - {str(e)[:30]}")
            scores["failed"] += 1
            failures.append((query, domain_bridge, f"Error: {e}"))

    print()
    print("=" * 80)
    print("SUMMARY")
    print("=" * 80)
    print(f"Total queries: {len(test_queries)}")
    print(f"Passed: {scores['passed']} ({scores['passed']/len(test_queries)*100:.1f}%)")
    print(f"Failed: {scores['failed']} ({scores['failed']/len(test_queries)*100:.1f}%)")
    print()

    if failures:
        print("FAILED QUERIES:")
        print("-" * 80)
        for query, domain, result in failures:
            print(f"  • {domain:15s} | {query}")
            print(f"    Result: {result}")
        print()

    # Overall assessment
    pass_rate = scores['passed'] / len(test_queries)
    if pass_rate >= 0.9:
        grade = "A"
        status = "✅ EXCELLENT - Ready for polymathic exam"
    elif pass_rate >= 0.8:
        grade = "B"
        status = "⚠️  GOOD - Minor gaps, mostly ready"
    elif pass_rate >= 0.7:
        grade = "C"
        status = "⚠️  FAIR - Significant gaps, needs work"
    else:
        grade = "F"
        status = "❌ POOR - Major gaps, not ready"

    print(f"EXAM READINESS: {grade} ({pass_rate*100:.1f}%)")
    print(f"Status: {status}")
    print()

    return scores, failures


def test_oblique_retrieval():
    """Test oblique retrieval capability"""

    print("=" * 80)
    print("OBLIQUE RETRIEVAL TEST")
    print("=" * 80)
    print()

    oblique_tests = [
        "Predict gene expression from tissue images",
        "Infer protein structure from sequence",
        "Reconstruct signal from sparse measurements",
        "Classify images with limited labels"
    ]

    scores = {"passed": 0, "failed": 0}

    for query in oblique_tests:
        try:
            results = oblique_search(query, n_results=10)

            # Success: found abstract patterns in other domains
            success = results['total_found'] >= 3

            status = "✅" if success else "❌"
            if success:
                scores["passed"] += 1
            else:
                scores["failed"] += 1

            print(f"{status} {query[:60]:60s} ({results['total_found']} results)")
            if results.get('abstraction', {}).get('abstract_terms'):
                print(f"   Abstract terms: {results['abstraction']['abstract_terms'][:5]}")

        except Exception as e:
            print(f"❌ {query[:60]:60s} - Error: {e}")
            scores["failed"] += 1

    print()
    print(f"Oblique retrieval: {scores['passed']}/{len(oblique_tests)} passed")
    print()

    return scores


def test_domain_bridges():
    """Test domain bridge search"""

    print("=" * 80)
    print("DOMAIN BRIDGE TEST")
    print("=" * 80)
    print()

    bridge_tests = [
        ("active inference", "cancer evolution"),
        ("topology", "pathology"),
        ("compressed sensing", "spatial transcriptomics"),
        ("graph neural networks", "protein folding"),
        ("thermodynamics", "cell signaling")
    ]

    scores = {"passed": 0, "failed": 0}

    for c1, c2 in bridge_tests:
        try:
            results = domain_bridge_search(c1, c2, n_results=10)

            # Success: found papers bridging both concepts
            success = results['total_bridges_found'] >= 2

            status = "✅" if success else "❌"
            if success:
                scores["passed"] += 1
            else:
                scores["failed"] += 1

            print(f"{status} {c1:25s} ↔ {c2:25s} ({results['total_bridges_found']} bridges)")

        except Exception as e:
            print(f"❌ {c1:25s} ↔ {c2:25s} - Error: {e}")
            scores["failed"] += 1

    print()
    print(f"Domain bridges: {scores['passed']}/{len(bridge_tests)} passed")
    print()

    return scores


def check_system_health():
    """Check system health"""

    print("=" * 80)
    print("SYSTEM HEALTH CHECK")
    print("=" * 80)
    print()

    # Check ChromaDB
    try:
        client = chromadb.PersistentClient(path="/home/user/work/polymax/chromadb/polymath_v2")
        collection = client.get_collection("polymath_corpus")
        chunk_count = collection.count()
        print(f"✅ ChromaDB: {chunk_count:,} chunks")
    except Exception as e:
        print(f"❌ ChromaDB: Error - {e}")

    # Check backups
    import os
    backup_dir = "/home/user/backups/polymax"
    if os.path.exists(backup_dir):
        backups = [f for f in os.listdir(backup_dir) if f.endswith('.gz')]
        print(f"✅ Backups: {len(backups)} backup files found")
    else:
        print(f"❌ Backups: No backup directory")

    # Check monitoring
    monitor_db = "/home/user/work/polymax/monitoring/metrics.db"
    if os.path.exists(monitor_db):
        print(f"✅ Monitoring: Active")
    else:
        print(f"⚠️  Monitoring: Not found")

    print()


def main():
    print("\n")
    print("╔" + "═" * 78 + "╗")
    print("║" + " " * 20 + "POLYMATH EXAM READINESS CHECK" + " " * 29 + "║")
    print("╚" + "═" * 78 + "╝")
    print()

    # System health
    check_system_health()

    # Polymathic domain coverage
    poly_scores, failures = test_polymathic_domains()

    # Oblique retrieval
    oblique_scores = test_oblique_retrieval()

    # Domain bridges
    bridge_scores = test_domain_bridges()

    # Final report
    print("=" * 80)
    print("FINAL ASSESSMENT")
    print("=" * 80)

    total_tests = (
        len([1 for _ in test_polymathic_domains.__code__.co_consts if isinstance(_, str)]) +
        4 + 5  # Approximate counts
    )

    total_passed = poly_scores["passed"] + oblique_scores["passed"] + bridge_scores["passed"]
    total_failed = poly_scores["failed"] + oblique_scores["failed"] + bridge_scores["failed"]

    overall_rate = total_passed / (total_passed + total_failed) if (total_passed + total_failed) > 0 else 0

    print(f"Overall pass rate: {overall_rate*100:.1f}%")
    print()

    if overall_rate >= 0.85:
        print("🎓 VERDICT: System is READY for polymathic examination")
        print("   The knowledge base demonstrates strong cross-domain coverage.")
    elif overall_rate >= 0.70:
        print("⚠️  VERDICT: System is MOSTLY READY")
        print("   Some knowledge gaps exist but core capabilities are solid.")
    else:
        print("❌ VERDICT: System NEEDS MORE PREPARATION")
        print("   Significant gaps in cross-domain knowledge.")
        print()
        print("Recommendations:")
        if failures:
            print("  • Ingest missing repos for failed domains")
            print("  • Use Literature Sentry to fill paper gaps")
            print("  • Focus on cross-domain bridge papers")

    print()
    print("=" * 80)
    print()


if __name__ == "__main__":
    main()
