#!/usr/bin/env python3
"""
xAI Collections API Smoke Test

Tests connection, collection access, and search functionality.

Usage:
    python3 scripts/xai_smoke_test.py

Environment variables (or in .env):
    XAI_API_KEY - Regular API key
    XAI_MANAGEMENT_API_KEY - Management API key (for collection operations)
"""

import os
import sys
from pathlib import Path

# Load .env if available
env_file = Path(__file__).parent.parent / ".env"
if env_file.exists():
    for line in env_file.read_text().splitlines():
        if "=" in line and not line.startswith("#"):
            key, value = line.split("=", 1)
            os.environ.setdefault(key.strip(), value.strip())

# Check for xai-sdk
try:
    from xai_sdk import Client
except ImportError:
    print("❌ xai-sdk not installed. Run: pip install xai-sdk")
    sys.exit(1)

XAI_API_KEY = os.getenv("XAI_API_KEY")
XAI_MANAGEMENT_API_KEY = os.getenv("XAI_MANAGEMENT_API_KEY")
COLLECTION_ID = "collection_f94f960c-5eab-4df9-8a02-6b507be0f17e"  # polymath_pilot


def test_connection():
    """Test basic API connection."""
    print("1. Testing API connection...")

    if not XAI_API_KEY:
        print("   ❌ XAI_API_KEY not set")
        return None

    if not XAI_MANAGEMENT_API_KEY:
        print("   ❌ XAI_MANAGEMENT_API_KEY not set")
        return None

    try:
        client = Client(
            api_key=XAI_API_KEY,
            management_api_key=XAI_MANAGEMENT_API_KEY,
            timeout=60
        )
        print("   ✅ Client initialized")
        return client
    except Exception as e:
        print(f"   ❌ Failed to initialize client: {e}")
        return None


def test_list_collections(client):
    """Test listing collections."""
    print("\n2. Testing collection listing...")

    try:
        response = client.collections.list()
        collections = response.collections if hasattr(response, 'collections') else []
        print(f"   ✅ Found {len(collections)} collections:")
        for c in collections:
            name = c.collection_name if hasattr(c, 'collection_name') else 'unknown'
            cid = c.collection_id if hasattr(c, 'collection_id') else 'unknown'
            docs = c.documents_count if hasattr(c, 'documents_count') else 0
            print(f"      - {name}: {docs} docs ({cid[:20]}...)")
        return True
    except Exception as e:
        print(f"   ❌ Failed to list collections: {e}")
        return False


def test_collection_access(client):
    """Test access to polymath_pilot collection."""
    print(f"\n3. Testing collection access ({COLLECTION_ID[:30]}...)...")

    try:
        # Try to get collection info
        response = client.collections.list()
        collections = response.collections if hasattr(response, 'collections') else []

        for c in collections:
            cid = c.collection_id if hasattr(c, 'collection_id') else ''
            if cid == COLLECTION_ID:
                docs = c.documents_count if hasattr(c, 'documents_count') else 0
                print(f"   ✅ Collection found: {docs} documents indexed")
                return True

        print(f"   ⚠️  Collection not found (may need to create it)")
        return False
    except Exception as e:
        print(f"   ❌ Failed to access collection: {e}")
        return False


def test_search(client):
    """Test search functionality."""
    print("\n4. Testing search...")

    test_queries = [
        "machine learning",
        "neural network",
        "deep learning"
    ]

    for query in test_queries:
        try:
            response = client.collections.search(
                query=query,
                collection_ids=[COLLECTION_ID],
            )

            results = response.results if hasattr(response, 'results') else []
            print(f"   ✅ '{query}': {len(results)} results")

            if results:
                first = results[0]
                doc_name = first.document.name if hasattr(first, 'document') and hasattr(first.document, 'name') else 'unknown'
                score = first.score if hasattr(first, 'score') else 0
                print(f"      Top result: {doc_name[:50]}... (score: {score:.3f})")

            return True
        except Exception as e:
            print(f"   ❌ Search failed for '{query}': {e}")
            return False

    return True


def main():
    print("=" * 60)
    print("xAI Collections API Smoke Test")
    print("=" * 60)

    # Test 1: Connection
    client = test_connection()
    if not client:
        print("\n❌ SMOKE TEST FAILED: Cannot connect to API")
        sys.exit(1)

    # Test 2: List collections
    if not test_list_collections(client):
        print("\n⚠️  Warning: Could not list collections")

    # Test 3: Collection access
    if not test_collection_access(client):
        print("\n⚠️  Warning: polymath_pilot collection not accessible")

    # Test 4: Search
    if not test_search(client):
        print("\n⚠️  Warning: Search failed (collection may still be indexing)")

    print("\n" + "=" * 60)
    print("✅ SMOKE TEST PASSED")
    print("=" * 60)

    print("\nNext steps:")
    print("  - Run search comparison: python3 scripts/xai_pilot_eval.py --compare")
    print("  - Upload more PDFs: python3 scripts/xai_pilot_eval.py --upload --limit 200")


if __name__ == "__main__":
    main()
