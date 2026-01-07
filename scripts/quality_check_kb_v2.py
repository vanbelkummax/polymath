#!/usr/bin/env python3
"""
KB V2 Quality Spot Check - Non-Invasive
Analyzes extracted concepts without interrupting migration
"""

import psycopg2
import json
from collections import Counter, defaultdict

# Thresholds
THRESHOLDS = {
    "min_supported_rate": 0.90,  # 90% of chunks should have concepts
    "max_junk_in_top20": 0.20,   # Max 20% junk in top 20 concepts
    "min_avg_concepts": 4.0,     # Min 4 concepts per chunk
    "max_avg_concepts": 8.0,     # Max 8 concepts per chunk (too high = noise)
}

# Junk patterns
JUNK_PATTERNS = [
    "__init__", "__main__", "__call__", "__getitem__", "__setitem__",
    "forward", "forward_pass", "backward", "backward_pass",
    "self", "cls", "super",
    "torch", "nn", "nn_module", "tf", "keras",
    "pytest", "unittest", "test_",
    "import", "from", "def", "class",
    "logger", "logging", "print", "debug",
    "config", "settings", "params",
]

def is_junk(concept_name):
    """Check if concept is likely junk."""
    concept_lower = concept_name.lower()
    
    # Exact matches
    if concept_lower in JUNK_PATTERNS:
        return True
    
    # Patterns
    if concept_lower.startswith("test_"):
        return True
    if concept_lower.startswith("_"):
        return True
    if len(concept_name) <= 2:
        return True
    
    return False

def main():
    conn = psycopg2.connect("dbname=polymath user=polymath host=/var/run/postgresql")
    cursor = conn.cursor()
    
    print("=" * 80)
    print("KB V2 QUALITY SPOT CHECK")
    print("=" * 80)
    print()
    
    # 1. Get migration status
    cursor.execute("""
        SELECT items_processed, items_failed, updated_at
        FROM kb_migrations 
        WHERE job_name = 'backfill_chunk_concepts_llm_v2'
    """)
    items_processed, items_failed, updated_at = cursor.fetchone()
    print(f"Migration Status:")
    print(f"  Chunks processed: {items_processed:,}")
    print(f"  Failed: {items_failed}")
    print(f"  Last update: {updated_at}")
    print()
    
    # 2. Sample N=100 random chunks with their concepts
    cursor.execute("""
        WITH sampled_chunks AS (
            SELECT chunk_id, RANDOM() as r
            FROM chunk_concepts
            WHERE extractor_version = 'llm_v2'
            GROUP BY chunk_id
            ORDER BY r
            LIMIT 100
        )
        SELECT 
            c.chunk_id,
            c.concept_name,
            c.concept_type,
            c.confidence
        FROM chunk_concepts c
        JOIN sampled_chunks s ON c.chunk_id = s.chunk_id
        WHERE c.extractor_version = 'llm_v2'
        ORDER BY c.chunk_id
    """)
    
    samples = cursor.fetchall()
    
    # Group by chunk
    chunks_concepts = defaultdict(list)
    for chunk_id, concept_name, concept_type, confidence in samples:
        chunks_concepts[chunk_id].append({
            'name': concept_name,
            'type': concept_type,
            'confidence': confidence
        })
    
    # 3. Compute concepts/chunk distribution
    concept_counts = [len(concepts) for concepts in chunks_concepts.values()]
    avg_concepts = sum(concept_counts) / len(concept_counts) if concept_counts else 0
    min_concepts = min(concept_counts) if concept_counts else 0
    max_concepts = max(concept_counts) if concept_counts else 0
    
    print(f"Concepts/Chunk Distribution (N={len(chunks_concepts)} chunks):")
    print(f"  Average: {avg_concepts:.1f}")
    print(f"  Min: {min_concepts}")
    print(f"  Max: {max_concepts}")
    
    # Distribution buckets
    buckets = Counter()
    for count in concept_counts:
        if count == 0:
            buckets["0"] += 1
        elif count <= 3:
            buckets["1-3"] += 1
        elif count <= 5:
            buckets["4-5"] += 1
        elif count <= 8:
            buckets["6-8"] += 1
        else:
            buckets["9+"] += 1
    
    print(f"  Distribution:")
    for bucket in ["0", "1-3", "4-5", "6-8", "9+"]:
        pct = (buckets[bucket] / len(concept_counts) * 100) if concept_counts else 0
        print(f"    {bucket:>5}: {buckets[bucket]:3d} ({pct:5.1f}%)")
    print()
    
    # 4. Supported vs unsupported rate (from all processed chunks)
    cursor.execute("""
        WITH chunk_counts AS (
            SELECT chunk_id, COUNT(*) as concept_count
            FROM chunk_concepts
            WHERE extractor_version = 'llm_v2'
            GROUP BY chunk_id
        )
        SELECT 
            COUNT(*) as chunks_with_concepts,
            (SELECT items_processed FROM kb_migrations WHERE job_name = 'backfill_chunk_concepts_llm_v2') as total_processed
        FROM chunk_counts
    """)
    chunks_with_concepts, total_processed = cursor.fetchone()
    chunks_without_concepts = total_processed - chunks_with_concepts
    supported_rate = chunks_with_concepts / total_processed if total_processed > 0 else 0
    
    print(f"Support Rate (all {total_processed:,} processed chunks):")
    print(f"  Chunks with concepts: {chunks_with_concepts:,} ({supported_rate*100:.1f}%)")
    print(f"  Chunks with 0 concepts: {chunks_without_concepts:,} ({(1-supported_rate)*100:.1f}%)")
    print()
    
    # 5. Top 20 concepts + junk analysis
    cursor.execute("""
        SELECT 
            concept_name,
            COUNT(*) as frequency
        FROM chunk_concepts
        WHERE extractor_version = 'llm_v2'
        GROUP BY concept_name
        ORDER BY frequency DESC
        LIMIT 20
    """)
    
    top20 = cursor.fetchall()
    junk_count = sum(1 for name, _ in top20 if is_junk(name))
    junk_rate = junk_count / len(top20) if top20 else 0
    
    print(f"Top 20 Concepts:")
    for i, (name, freq) in enumerate(top20, 1):
        junk_marker = " [JUNK]" if is_junk(name) else ""
        print(f"  {i:2d}. {name:30s} {freq:5,}{junk_marker}")
    print()
    print(f"Junk in Top 20: {junk_count}/20 ({junk_rate*100:.1f}%)")
    print()
    
    # 6. Confidence distribution
    cursor.execute("""
        SELECT 
            CASE 
                WHEN confidence < 0.6 THEN '<0.6'
                WHEN confidence < 0.7 THEN '0.6-0.7'
                WHEN confidence < 0.8 THEN '0.7-0.8'
                WHEN confidence < 0.9 THEN '0.8-0.9'
                ELSE '0.9+'
            END as conf_bucket,
            COUNT(*) as count
        FROM chunk_concepts
        WHERE extractor_version = 'llm_v2'
        GROUP BY conf_bucket
        ORDER BY conf_bucket
    """)
    
    conf_dist = cursor.fetchall()
    total_concepts = sum(count for _, count in conf_dist)
    
    print(f"Confidence Distribution (N={total_concepts:,} concepts):")
    for bucket, count in conf_dist:
        pct = (count / total_concepts * 100) if total_concepts > 0 else 0
        print(f"  {bucket:>8}: {count:7,} ({pct:5.1f}%)")
    print()
    
    # 7. Type distribution
    cursor.execute("""
        SELECT 
            concept_type,
            COUNT(*) as count
        FROM chunk_concepts
        WHERE extractor_version = 'llm_v2'
        GROUP BY concept_type
        ORDER BY count DESC
    """)
    
    type_dist = cursor.fetchall()
    print(f"Concept Type Distribution:")
    for ctype, count in type_dist:
        pct = (count / total_concepts * 100) if total_concepts > 0 else 0
        print(f"  {ctype:15s}: {count:7,} ({pct:5.1f}%)")
    print()
    
    # 8. Thresholds + PASS/WARN
    print("=" * 80)
    print("QUALITY ASSESSMENT")
    print("=" * 80)
    
    results = []
    
    # Support rate
    if supported_rate >= THRESHOLDS["min_supported_rate"]:
        results.append(("✓ PASS", f"Support rate: {supported_rate*100:.1f}% >= {THRESHOLDS['min_supported_rate']*100:.0f}%"))
    else:
        results.append(("✗ WARN", f"Support rate: {supported_rate*100:.1f}% < {THRESHOLDS['min_supported_rate']*100:.0f}%"))
    
    # Junk rate
    if junk_rate <= THRESHOLDS["max_junk_in_top20"]:
        results.append(("✓ PASS", f"Junk in top 20: {junk_rate*100:.1f}% <= {THRESHOLDS['max_junk_in_top20']*100:.0f}%"))
    else:
        results.append(("✗ WARN", f"Junk in top 20: {junk_rate*100:.1f}% > {THRESHOLDS['max_junk_in_top20']*100:.0f}%"))
    
    # Avg concepts
    if THRESHOLDS["min_avg_concepts"] <= avg_concepts <= THRESHOLDS["max_avg_concepts"]:
        results.append(("✓ PASS", f"Avg concepts/chunk: {avg_concepts:.1f} in [{THRESHOLDS['min_avg_concepts']}, {THRESHOLDS['max_avg_concepts']}]"))
    else:
        results.append(("✗ WARN", f"Avg concepts/chunk: {avg_concepts:.1f} outside [{THRESHOLDS['min_avg_concepts']}, {THRESHOLDS['max_avg_concepts']}]"))
    
    # Zero failures
    if items_failed == 0:
        results.append(("✓ PASS", f"Failures: {items_failed}"))
    else:
        results.append(("✗ WARN", f"Failures: {items_failed} > 0"))
    
    for status, msg in results:
        print(f"{status}  {msg}")
    
    print()
    
    # Overall verdict
    fail_count = sum(1 for status, _ in results if status == "✗ WARN")
    if fail_count == 0:
        print("=" * 80)
        print("✓ OVERALL: QUALITY EXCELLENT - Continue migration")
        print("=" * 80)
    else:
        print("=" * 80)
        print(f"⚠ OVERALL: {fail_count} warnings - Review but likely OK to continue")
        print("=" * 80)
    
    cursor.close()
    conn.close()

if __name__ == "__main__":
    main()
