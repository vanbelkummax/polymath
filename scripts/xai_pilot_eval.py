#!/usr/bin/env python3
"""
xAI Collections API Pilot Evaluation

Uploads a sample of PDFs to xAI Collections and compares search quality
against local ChromaDB (BGE-M3).

Usage:
    # Set API key
    export XAI_API_KEY="xai-..."

    # Upload pilot set
    python3 scripts/xai_pilot_eval.py --upload --limit 200

    # Run comparison
    python3 scripts/xai_pilot_eval.py --compare

    # Full evaluation
    python3 scripts/xai_pilot_eval.py --upload --compare --limit 200
"""

import argparse
import json
import os
import sys
import time
from pathlib import Path
from typing import List, Dict, Optional
from dataclasses import dataclass, asdict

# Add lib to path
sys.path.insert(0, str(Path(__file__).parent.parent))

# Check for xai-sdk
try:
    from xai_sdk import Client
    XAI_AVAILABLE = True
except ImportError:
    XAI_AVAILABLE = False
    print("Warning: xai-sdk not installed. Run: pip install xai-sdk")

from lib.hybrid_search_v2 import HybridSearcherV2

XAI_API_KEY = os.getenv("XAI_API_KEY")
COLLECTION_NAME = "polymath_pilot"
PDF_DIR = Path("/scratch/polymath_archive/pdfs")


@dataclass
class SearchResult:
    title: str
    score: float
    source: str  # 'xai' or 'local'
    snippet: str = ""


@dataclass
class ComparisonResult:
    query: str
    xai_results: List[SearchResult]
    local_results: List[SearchResult]
    overlap_at_5: float
    overlap_at_10: float
    xai_unique: List[str]
    local_unique: List[str]


XAI_MANAGEMENT_API_KEY = os.getenv("XAI_MANAGEMENT_API_KEY")


def get_xai_client() -> Optional["Client"]:
    """Initialize xAI client."""
    if not XAI_AVAILABLE:
        print("xai-sdk not installed. Run: pip install xai-sdk")
        return None

    if not XAI_API_KEY:
        print("XAI_API_KEY not set")
        return None

    if not XAI_MANAGEMENT_API_KEY:
        print("XAI_MANAGEMENT_API_KEY not set")
        print("Create one at: https://console.x.ai → Management Keys → AddFileToCollection permission")
        return None

    return Client(
        api_key=XAI_API_KEY,
        management_api_key=XAI_MANAGEMENT_API_KEY,
        timeout=3600
    )


def create_collection(client: "Client") -> str:
    """Create or get existing collection."""
    # Check if collection exists
    response = client.collections.list()
    collections = response.collections if hasattr(response, 'collections') else []
    for c in collections:
        cname = c.collection_name if hasattr(c, 'collection_name') else getattr(c, 'name', '')
        cid = c.collection_id if hasattr(c, 'collection_id') else getattr(c, 'id', '')
        if cname == COLLECTION_NAME:
            print(f"Using existing collection: {cid}")
            return cid

    # Create new collection
    collection = client.collections.create(name=COLLECTION_NAME)
    cid = collection.collection_id if hasattr(collection, 'collection_id') else collection.id
    print(f"Created collection: {cid}")
    return cid


def upload_pdfs(client: "Client", collection_id: str, limit: int = 200) -> int:
    """Upload PDFs to xAI collection."""
    pdfs = list(PDF_DIR.glob("*.pdf"))[:limit]
    print(f"Uploading {len(pdfs)} PDFs to collection {collection_id}")

    uploaded = 0
    failed = 0

    for i, pdf in enumerate(pdfs):
        # Retry logic for transient errors
        max_retries = 3
        for attempt in range(max_retries):
            try:
                # Extract year from filename if possible
                year = None
                if match := __import__('re').search(r'(19|20)\d{2}', pdf.name):
                    year = match.group(0)

                # Determine source type from filename
                source_type = "unknown"
                if pdf.name.startswith("10."):
                    source_type = "doi"
                elif __import__('re').match(r'\d{4}\.\d+', pdf.name):
                    source_type = "arxiv"
                elif pdf.name.startswith("1-s2.0"):
                    source_type = "sciencedirect"

                with open(pdf, "rb") as f:
                    # SDK auto-detects content type from extension
                    client.collections.upload_document(
                        collection_id=collection_id,
                        name=pdf.name,
                        data=f.read(),
                        fields={
                            "filename": pdf.name,
                            "year": year or "",
                            "source_type": source_type
                        }
                    )
                uploaded += 1
                break  # Success, exit retry loop

            except Exception as e:
                if attempt < max_retries - 1:
                    print(f"  Retry {attempt + 1}/{max_retries} for {pdf.name}: {e}")
                    time.sleep(2 ** attempt)  # Exponential backoff
                else:
                    print(f"  Failed to upload {pdf.name}: {e}")
                    failed += 1

        if (i + 1) % 20 == 0:
            print(f"  Uploaded {i + 1}/{len(pdfs)}...")

        # Rate limiting - longer delay to avoid SSL issues
        time.sleep(0.5)

    print(f"\nUpload complete: {uploaded} success, {failed} failed")
    return uploaded


def search_xai(client: "Client", collection_id: str, query: str, n: int = 10) -> List[SearchResult]:
    """Search xAI collection."""
    try:
        response = client.collections.search(
            query=query,
            collection_ids=[collection_id],
            retrieval_mode={"type": "hybrid"}
        )

        results = []
        for item in response.results[:n]:
            results.append(SearchResult(
                title=item.document.name if hasattr(item, 'document') else str(item),
                score=item.score if hasattr(item, 'score') else 0.0,
                source='xai',
                snippet=item.text[:200] if hasattr(item, 'text') else ""
            ))
        return results

    except Exception as e:
        print(f"xAI search error: {e}")
        return []


def search_local(query: str, n: int = 10) -> List[SearchResult]:
    """Search local ChromaDB."""
    try:
        hs = HybridSearcherV2()
        results = hs.hybrid_search(query, n=n)

        return [
            SearchResult(
                title=r.get('title', r.get('doc_title', 'Unknown')),
                score=r.get('score', 0.0),
                source='local',
                snippet=r.get('text', '')[:200]
            )
            for r in results
        ]
    except Exception as e:
        print(f"Local search error: {e}")
        return []


def compute_overlap(list1: List[str], list2: List[str], k: int) -> float:
    """Compute overlap at k between two result lists."""
    set1 = set(list1[:k])
    set2 = set(list2[:k])
    if not set1 or not set2:
        return 0.0
    return len(set1 & set2) / k


def run_comparison(client: "Client", collection_id: str, queries: List[str]) -> List[ComparisonResult]:
    """Run search comparison between xAI and local."""
    results = []

    for query in queries:
        print(f"\nQuery: {query[:50]}...")

        xai_results = search_xai(client, collection_id, query)
        local_results = search_local(query)

        # Extract titles for comparison
        xai_titles = [r.title.lower() for r in xai_results]
        local_titles = [r.title.lower() for r in local_results]

        # Compute metrics
        overlap_5 = compute_overlap(xai_titles, local_titles, 5)
        overlap_10 = compute_overlap(xai_titles, local_titles, 10)

        xai_unique = [t for t in xai_titles[:10] if t not in local_titles[:10]]
        local_unique = [t for t in local_titles[:10] if t not in xai_titles[:10]]

        comparison = ComparisonResult(
            query=query,
            xai_results=xai_results,
            local_results=local_results,
            overlap_at_5=overlap_5,
            overlap_at_10=overlap_10,
            xai_unique=xai_unique,
            local_unique=local_unique
        )
        results.append(comparison)

        print(f"  Overlap@5: {overlap_5:.0%}, Overlap@10: {overlap_10:.0%}")
        print(f"  xAI unique: {len(xai_unique)}, Local unique: {len(local_unique)}")

        # Rate limiting for xAI
        time.sleep(0.5)

    return results


# Test queries for evaluation
TEST_QUERIES = [
    "spatial transcriptomics gene expression prediction from histology",
    "transformer attention mechanism in medical imaging",
    "colibactin pks island colorectal cancer microbiome",
    "single cell RNA sequencing clustering analysis",
    "deep learning pathology whole slide image classification",
    "graph neural network molecular property prediction",
    "contrastive learning self-supervised representation",
    "vision transformer ViT image classification",
    "diffusion models image generation",
    "foundation model transfer learning biology",
    "cell segmentation instance detection",
    "multi-modal learning image text",
    "uncertainty quantification neural networks",
    "active learning sample selection",
    "federated learning privacy preserving",
    "causal inference treatment effect",
    "optimal transport computational biology",
    "topological data analysis persistence",
    "bayesian inference probabilistic modeling",
    "reinforcement learning decision making",
]


def main():
    parser = argparse.ArgumentParser(description="xAI Collections pilot evaluation")
    parser.add_argument("--upload", action="store_true", help="Upload PDFs to xAI")
    parser.add_argument("--compare", action="store_true", help="Run comparison")
    parser.add_argument("--limit", type=int, default=200, help="Number of PDFs to upload")
    parser.add_argument("--queries", type=int, default=20, help="Number of test queries")
    parser.add_argument("--output", default="xai_comparison.json", help="Output file")
    parser.add_argument("--collection-id", help="Existing collection ID")
    args = parser.parse_args()

    if not (args.upload or args.compare):
        parser.print_help()
        return

    client = get_xai_client()
    if not client:
        print("Failed to initialize xAI client")
        return

    # Get or create collection
    if args.collection_id:
        collection_id = args.collection_id
    else:
        collection_id = create_collection(client)

    # Upload phase
    if args.upload:
        print("\n" + "="*60)
        print("PHASE 1: Upload PDFs to xAI")
        print("="*60)
        upload_pdfs(client, collection_id, limit=args.limit)
        print("\nWaiting for indexing to complete (30 seconds)...")
        time.sleep(30)

    # Comparison phase
    if args.compare:
        print("\n" + "="*60)
        print("PHASE 2: Search Comparison")
        print("="*60)

        queries = TEST_QUERIES[:args.queries]
        results = run_comparison(client, collection_id, queries)

        # Summary stats
        print("\n" + "="*60)
        print("SUMMARY")
        print("="*60)

        avg_overlap_5 = sum(r.overlap_at_5 for r in results) / len(results)
        avg_overlap_10 = sum(r.overlap_at_10 for r in results) / len(results)

        print(f"Average Overlap@5: {avg_overlap_5:.1%}")
        print(f"Average Overlap@10: {avg_overlap_10:.1%}")

        # Count where xAI found unique results
        xai_better = sum(1 for r in results if len(r.xai_unique) > len(r.local_unique))
        local_better = sum(1 for r in results if len(r.local_unique) > len(r.xai_unique))
        tie = len(results) - xai_better - local_better

        print(f"\nWinner analysis:")
        print(f"  xAI found more unique results: {xai_better}")
        print(f"  Local found more unique results: {local_better}")
        print(f"  Tie: {tie}")

        # Save results
        output_path = Path(args.output)
        with open(output_path, 'w') as f:
            json.dump({
                'collection_id': collection_id,
                'num_queries': len(results),
                'avg_overlap_5': avg_overlap_5,
                'avg_overlap_10': avg_overlap_10,
                'results': [
                    {
                        'query': r.query,
                        'overlap_5': r.overlap_at_5,
                        'overlap_10': r.overlap_at_10,
                        'xai_unique_count': len(r.xai_unique),
                        'local_unique_count': len(r.local_unique)
                    }
                    for r in results
                ]
            }, f, indent=2)

        print(f"\nResults saved to {output_path}")

        # Cost estimate
        print(f"\nEstimated cost: ${len(results) * 0.0025:.4f} ({len(results)} searches @ $2.50/1000)")


if __name__ == "__main__":
    main()
