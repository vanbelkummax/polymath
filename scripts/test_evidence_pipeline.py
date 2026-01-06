#!/usr/bin/env python3
"""
Evidence Pipeline Smoke Test

Tests the complete flow: Parser → Database → Extractor → Verification

This verifies:
1. Page-local coordinates are correct
2. NLI extraction works
3. Database insertion succeeds
4. Char offsets match actual text

Usage:
  python3 scripts/test_evidence_pipeline.py <pdf_path> <claim>

Example:
  python3 scripts/test_evidence_pipeline.py ./data/test.pdf "GraphST uses graph neural networks"
"""
import sys
import os
import uuid
import logging
from typing import List

# Setup paths so imports work
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from lib.doc_identity import compute_doc_id, upsert_document
from lib.enhanced_pdf_parser import EnhancedPDFParser
from lib.evidence_extractor import EvidenceExtractor
from lib.db import db

# Setup logging to see the "thought process"
logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
logger = logging.getLogger(__name__)

def run_smoke_test(pdf_path: str, claim: str):
    """
    Runs a single PDF through the entire 'Evidence-Bound' pipeline.
    """
    print(f"🔬 SMOKE TEST: {pdf_path}")
    print(f"❓ CLAIM: {claim}\n")

    # ========================================================
    # BLOCK 1: IDENTITY & INGESTION (The "Input Pipe")
    # ========================================================
    print("--- 1. PARSING & IDENTITY ---")

    # 1. Generate Deterministic ID
    # We fake the metadata here for the test, but in prod this comes from user/API
    doc_id = compute_doc_id(
        title="Test Document",
        authors=["Tester et al."],
        year=2024,
        doi="10.1000/test"
    )
    print(f"✅ Generated Doc ID: {doc_id}")

    # 2. Register Document (Ensures FK constraints pass)
    upsert_document(
        doc_id=doc_id,
        title="Test Document",
        authors=["Tester et al."],
        year=2024,
        doi="10.1000/test"
    )
    print("✅ Document Registered in DB")

    # 3. Parse with Page-Local Coordinates
    parser = EnhancedPDFParser()
    passages = parser.extract_with_provenance(pdf_path, doc_id)
    print(f"✅ Extracted {len(passages)} passages using {passages[0].parser_version if passages else 'N/A'}")

    if not passages:
        print("❌ FAIL: No passages extracted from PDF")
        return

    # ========================================================
    # BLOCK 2: STORAGE (The "Memory")
    # ========================================================
    print("\n--- 2. DATABASE STORAGE ---")

    # Clean up previous test runs (Idempotency)
    db.execute("DELETE FROM evidence_spans WHERE doc_id = %s", (str(doc_id),))
    db.execute("DELETE FROM passages WHERE doc_id = %s", (str(doc_id),))

    # Insert Passages
    # This populates the 'Searchable Unit' table
    for p in passages:
        db.execute(
            """
            INSERT INTO passages
            (passage_id, doc_id, page_num, page_char_start, page_char_end, section, passage_text, quality_score, parser_version)
            VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s)
            """,
            (str(p.passage_id), str(p.doc_id), p.page_num, p.page_char_start,
             p.page_char_end, p.section, p.passage_text, p.quality_score, p.parser_version)
        )
    print(f"✅ Stored {len(passages)} passages in Postgres")

    # IMPORTANT: In a real run, you'd also insert into ChromaDB here.
    # For this smoke test, we rely on Postgres full-text search

    # ========================================================
    # BLOCK 3: EXTRACTION (The "Brain")
    # ========================================================
    print("\n--- 3. EVIDENCE EXTRACTION (The Batched NLI) ---")

    extractor = EvidenceExtractor()

    # Try full flow first (may fail if ChromaDB not running)
    try:
        spans = extractor.extract_spans_for_claim(claim)
    except Exception as e:
        print(f"⚠️  Retrieval failed (expected if ChromaDB not synced): {e}")
        print("   -> Fallback: extracting from first 5 passages directly")

        # Direct injection bypass for testing logic
        test_candidates = []
        for p in passages[:5]:
            test_candidates.append({
                'passage_id': str(p.passage_id),
                'doc_id': str(p.doc_id),
                'page_num': p.page_num,
                'page_char_start': p.page_char_start,
                'page_char_end': p.page_char_end,
                'section': p.section,
                'passage_text': p.passage_text,
                'quality_score': p.quality_score
            })

        # Override retrieval for smoke test
        original_retrieve = extractor._retrieve_passages
        extractor._retrieve_passages = lambda q, limit: test_candidates

        spans = extractor.extract_spans_for_claim(claim)

        # Restore original method
        extractor._retrieve_passages = original_retrieve

    # ========================================================
    # BLOCK 4: VERIFICATION (The "Truth")
    # ========================================================
    print("\n--- 4. RESULTS ---")

    if not spans:
        print("❌ No evidence found. (Check NLI threshold or Claim relevance)")
        print("\nDEBUG INFO:")
        print(f"  - Passages in DB: {len(passages)}")
        print(f"  - First passage preview: {passages[0].passage_text[:100]}...")
        return

    print(f"✅ Found {len(spans)} evidence spans\n")

    for i, span in enumerate(spans, 1):
        status = "✅ STRONG" if span.entailment_score > 0.8 else "⚠️  WEAK"
        print(f"[Result {i}] {status} Evidence (Score: {span.entailment_score:.3f})")
        print(f"   📍 Location: Page {span.page_num}")
        print(f"   📜 Text: \"{span.span_text[:80]}...\"")
        print(f"   🧠 Contradiction Score: {span.contradiction_score:.3f}")

        # VERIFICATION: Check coordinates
        # We verify that the span text actually exists in the passage at those coordinates
        try:
            passage = next(p for p in passages if str(p.passage_id) == str(span.passage_id))
            sliced_text = passage.passage_text[span.span_char_start:span.span_char_end]

            if sliced_text.strip() == span.span_text.strip():
                print(f"   🕵️  Offset Verification: PASS (Exact Match)")
            else:
                print(f"   🛑 Offset Verification: FAIL")
                print(f"      Expected: {span.span_text[:50]}...")
                print(f"      Got     : {sliced_text[:50]}...")
        except StopIteration:
            print(f"   ⚠️  Could not find passage {span.passage_id} in parsed passages")
        except Exception as e:
            print(f"   ⚠️  Verification error: {e}")

        print()

    print("\n" + "="*60)
    print("SMOKE TEST COMPLETE")
    print("="*60)

if __name__ == "__main__":
    if len(sys.argv) < 3:
        print("Usage: python3 scripts/test_evidence_pipeline.py <pdf_path> <claim>")
        print("\nExample:")
        print("  python3 scripts/test_evidence_pipeline.py ./data/test.pdf 'GraphST uses graph neural networks'")
        sys.exit(1)

    run_smoke_test(sys.argv[1], sys.argv[2])
