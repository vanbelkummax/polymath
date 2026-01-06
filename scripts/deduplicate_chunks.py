#!/usr/bin/env python3
"""
Polymath Chunk Deduplication Tool

Detects and removes duplicate chunks across 3 ingestion methods:
1. Legacy file-by-file ingestion (3,355 files)
2. Repo-level chunks (57 repos)
3. Dual-stream ingestion (Tier 1/2 repos)

Strategy:
- Use content hash (SHA256) to detect exact duplicates
- Use embedding similarity (cosine >0.99) to detect near-duplicates
- Add ingestion_method metadata to track provenance
- Prefer newer ingestion methods over legacy when choosing which to keep
"""

import chromadb
import hashlib
import json
from collections import defaultdict
from datetime import datetime
import numpy as np

# ChromaDB client
CHROMA_PATH = "/home/user/work/polymax/chromadb/polymath_v2"
COLLECTION_NAME = "polymath_corpus"

# Deduplication thresholds
EXACT_DUPLICATE_THRESHOLD = 1.0  # Hash match
NEAR_DUPLICATE_THRESHOLD = 0.99  # Cosine similarity

# Ingestion method priority (higher = keep this one)
PRIORITY = {
    "dual_stream": 3,    # Newest, most granular
    "repo_level": 2,     # Good, but less granular
    "file_by_file": 1,   # Legacy, being replaced
    "unknown": 0         # No metadata, lowest priority
}


def get_content_hash(text):
    """Generate SHA256 hash of text content for exact duplicate detection"""
    return hashlib.sha256(text.encode('utf-8')).hexdigest()


def cosine_similarity(a, b):
    """Calculate cosine similarity between two embedding vectors"""
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))


def detect_ingestion_method(metadata):
    """
    Detect ingestion method from metadata

    Heuristics:
    - dual_stream: ingestion_method starts with 'dual_stream'
    - repo_level: Has 'type=repo' in metadata
    - file_by_file: Has 'type=code' but no chunk_type
    - unknown: No clear markers
    """
    if not metadata:
        return 'unknown'

    method = metadata.get("ingestion_method") or metadata.get("ingest_method")
    if method:
        method_lower = str(method).lower()
        if method_lower.startswith("dual_stream"):
            return "dual_stream"
        if method_lower in {"repo_ingest", "smart_repo_ingest"}:
            return "repo_level"
        if any(key in method_lower for key in ("repo_file", "code_file", "batch", "sentry", "file", "doc")):
            return "file_by_file"

    if 'chunk_type' in metadata and metadata.get("chunk_type") == "repo_flattened":
        return 'repo_level'
    if 'chunk_type' in metadata and metadata.get("chunk_type") == "code_file":
        return 'file_by_file'
    if 'chunk_type' in metadata and metadata.get("chunk_type") == "paper":
        return 'file_by_file'
    elif metadata.get('type') == 'repo':
        return 'repo_level'
    elif metadata.get('type') == 'code':
        return 'file_by_file'
    else:
        return 'unknown'


def find_duplicates(dry_run=True):
    """
    Find duplicate chunks in ChromaDB collection

    Returns:
        dict: Statistics about duplicates found
    """
    print("🔍 Scanning ChromaDB for duplicates...")
    print(f"   Collection: {COLLECTION_NAME}")
    print(f"   Path: {CHROMA_PATH}")
    print()

    client = chromadb.PersistentClient(path=CHROMA_PATH)
    collection = client.get_collection(COLLECTION_NAME)

    # Get all documents with metadata and embeddings
    print("📥 Fetching all chunks...")
    total_count = collection.count()
    print(f"   Total chunks: {total_count:,}")

    # Fetch in batches to avoid memory issues
    BATCH_SIZE = 1000
    all_data = {"ids": [], "documents": [], "metadatas": [], "embeddings": []}

    offset = 0
    while offset < total_count:
        result = collection.get(
            limit=BATCH_SIZE,
            offset=offset,
            include=["documents", "metadatas", "embeddings"]
        )

        all_data["ids"].extend(result["ids"])
        all_data["documents"].extend(result["documents"])
        all_data["metadatas"].extend(result["metadatas"])
        all_data["embeddings"].extend(result["embeddings"])

        offset += BATCH_SIZE
        print(f"   Fetched {offset:,}/{total_count:,} chunks...", end='\r')

    print(f"\n   ✓ Fetched {len(all_data['ids']):,} chunks")
    print()

    # Phase 1: Exact duplicates (content hash)
    print("🔎 Phase 1: Finding exact duplicates (content hash)...")
    hash_to_ids = defaultdict(list)

    for i, doc in enumerate(all_data["documents"]):
        content_hash = get_content_hash(doc)
        hash_to_ids[content_hash].append(i)

    exact_duplicates = {h: ids for h, ids in hash_to_ids.items() if len(ids) > 1}
    print(f"   Found {len(exact_duplicates):,} sets of exact duplicates")
    print(f"   Total duplicate chunks: {sum(len(ids) - 1 for ids in exact_duplicates.values()):,}")
    print()

    # Phase 2: Near duplicates (embedding similarity)
    print("🔎 Phase 2: Finding near duplicates (embedding similarity >0.99)...")
    print("   (This may take a while for large collections)")

    # TODO: Implement efficient near-duplicate detection
    # For now, skip to avoid O(n^2) comparison
    # Future: Use locality-sensitive hashing (LSH) or FAISS
    print("   ⏸️  Skipped (requires LSH implementation for efficiency)")
    print()

    # Analyze duplicates
    print("📊 Analyzing duplicates by ingestion method...")
    stats = {
        "total_chunks": total_count,
        "exact_duplicates": len(exact_duplicates),
        "duplicate_chunks": sum(len(ids) - 1 for ids in exact_duplicates.values()),
        "by_ingestion_method": defaultdict(int),
        "to_delete": []
    }

    # For each duplicate set, keep highest priority and mark others for deletion
    for content_hash, indices in exact_duplicates.items():
        # Get metadata for all duplicates
        duplicates = []
        for idx in indices:
            chunk_id = all_data["ids"][idx]
            metadata = all_data["metadatas"][idx]
            ingestion_method = detect_ingestion_method(metadata)

            duplicates.append({
                "id": chunk_id,
                "index": idx,
                "method": ingestion_method,
                "priority": PRIORITY[ingestion_method],
                "metadata": metadata
            })

            stats["by_ingestion_method"][ingestion_method] += 1

        # Sort by priority (descending) - keep first, delete rest
        duplicates.sort(key=lambda x: x["priority"], reverse=True)

        # Mark all but first for deletion
        for dup in duplicates[1:]:
            stats["to_delete"].append({
                "id": dup["id"],
                "method": dup["method"],
                "reason": f"Duplicate of {duplicates[0]['id']} (kept {duplicates[0]['method']})"
            })

    # Print statistics
    print()
    print("=" * 80)
    print("DEDUPLICATION SUMMARY")
    print("=" * 80)
    print(f"Total chunks:           {stats['total_chunks']:>10,}")
    print(f"Unique content:         {stats['total_chunks'] - stats['duplicate_chunks']:>10,}")
    print(f"Duplicate chunks:       {stats['duplicate_chunks']:>10,}")
    print(f"Deduplication rate:     {stats['duplicate_chunks'] / stats['total_chunks'] * 100:>10.2f}%")
    print()
    print("Duplicates by ingestion method:")
    for method, count in sorted(stats["by_ingestion_method"].items(), key=lambda x: x[1], reverse=True):
        print(f"  {method:20s} {count:>8,} chunks")
    print()
    print(f"Chunks to delete:       {len(stats['to_delete']):>10,}")
    print()

    if dry_run:
        print("🔒 DRY RUN MODE - No changes made")
        print()
        print("To actually remove duplicates, run with --execute flag:")
        print("  python3 scripts/deduplicate_chunks.py --execute")
    else:
        print("⚠️  EXECUTE MODE - Deleting duplicates...")

        # Delete duplicates in batches
        ids_to_delete = [d["id"] for d in stats["to_delete"]]

        BATCH_SIZE = 100
        deleted = 0
        for i in range(0, len(ids_to_delete), BATCH_SIZE):
            batch = ids_to_delete[i:i + BATCH_SIZE]
            collection.delete(ids=batch)
            deleted += len(batch)
            print(f"   Deleted {deleted:,}/{len(ids_to_delete):,} duplicates...", end='\r')

        print(f"\n   ✓ Deleted {deleted:,} duplicate chunks")

        # Verify
        new_count = collection.count()
        print()
        print(f"Final chunk count: {new_count:,} (was {total_count:,})")
        print(f"Removed: {total_count - new_count:,} chunks")

    print()
    print("=" * 80)

    # Save deduplication report
    report_file = f"/home/user/work/polymax/logs/deduplication_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    with open(report_file, 'w') as f:
        # Convert defaultdict to dict for JSON serialization
        stats_copy = dict(stats)
        stats_copy["by_ingestion_method"] = dict(stats["by_ingestion_method"])
        json.dump(stats_copy, f, indent=2)

    print(f"📄 Full report saved: {report_file}")
    print()

    return stats


def main():
    import argparse

    parser = argparse.ArgumentParser(
        description="Deduplicate chunks in Polymath ChromaDB collection",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Dry run (default) - shows what would be deleted
  python3 scripts/deduplicate_chunks.py

  # Actually delete duplicates
  python3 scripts/deduplicate_chunks.py --execute

  # View specific duplicate set
  python3 scripts/deduplicate_chunks.py --hash abc123...
        """
    )

    parser.add_argument(
        '--execute',
        action='store_true',
        help='Actually delete duplicates (default is dry-run)'
    )

    parser.add_argument(
        '--hash',
        help='Show details for specific content hash'
    )

    args = parser.parse_args()

    if args.hash:
        print(f"TODO: Show details for hash {args.hash}")
        return

    # Run deduplication
    stats = find_duplicates(dry_run=not args.execute)

    if not args.execute and stats["duplicate_chunks"] > 0:
        print("💡 TIP: Review the report, then run with --execute to remove duplicates")


if __name__ == "__main__":
    main()
