#!/usr/bin/env python3
"""
Batch concept extraction using Gemini API with GCS storage.

Usage:
    python scripts/batch_concept_extraction.py --limit 500
    python scripts/batch_concept_extraction.py --limit 100 --dry-run
"""

import os
import sys
import json
import re
import time
import argparse
from datetime import datetime, timezone
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed

sys.path.insert(0, str(Path(__file__).parent.parent))

from lib.config import (
    GCP_PROJECT_ID, GCP_BUCKET, GCP_SERVICE_ACCOUNT,
    GEMINI_API_KEY, POSTGRES_DSN
)

# Set credentials for GCP
os.environ['GOOGLE_APPLICATION_CREDENTIALS'] = GCP_SERVICE_ACCOUNT

import psycopg2
from google.cloud import storage
from google import genai

# Initialize client with Vertex AI (global location for latest models)
client = genai.Client(
    vertexai=True,
    project=GCP_PROJECT_ID,
    location="global"
)

# Model choice - gemini-2.5-flash-lite is best for batch (fast, cheap, 65k output)
DEFAULT_MODEL = "gemini-2.5-flash-lite"


CONCEPT_PROMPT = """Extract key scientific concepts from this research text. Return ONLY valid JSON:
{{"methods": [], "problems": [], "domains": [], "entities": []}}

Rules:
- methods: techniques, algorithms, tools (e.g., "spatial transcriptomics", "gradient descent", "CRISPR")
- problems: challenges being addressed (e.g., "cell type deconvolution", "batch effects")
- domains: research fields (e.g., "computational pathology", "drug discovery")
- entities: specific genes, proteins, datasets, diseases (e.g., "TP53", "Alzheimer's disease")
- Be specific, not generic
- Only include explicitly mentioned concepts

Text:
{text}"""


def extract_concepts_gemini(text: str, model: str = DEFAULT_MODEL, max_retries: int = 3) -> dict:
    """Extract concepts using Gemini via google-genai SDK with retry logic."""
    for attempt in range(max_retries):
        try:
            response = client.models.generate_content(
                model=model,
                contents=CONCEPT_PROMPT.format(text=text[:2000])
            )

            # Parse JSON from response
            raw = response.text
            cleaned = re.sub(r'```json\s*', '', raw)
            cleaned = re.sub(r'```', '', cleaned).strip()
            return json.loads(cleaned)

        except json.JSONDecodeError:
            if attempt < max_retries - 1:
                time.sleep(1)
                continue
            return None
        except Exception as e:
            if "429" in str(e) or "quota" in str(e).lower() or "RESOURCE_EXHAUSTED" in str(e):
                time.sleep(5 * (attempt + 1))
                continue
            if attempt < max_retries - 1:
                time.sleep(1)
                continue
            return None
    return None


def get_passages_needing_concepts(conn, limit: int) -> list:
    """Get passages that don't have Gemini-extracted concepts yet."""
    cur = conn.cursor()

    # Get passages without recent concept extraction
    cur.execute("""
        SELECT p.passage_id, p.passage_text, d.title, d.doc_id
        FROM passages p
        JOIN documents d ON p.doc_id = d.doc_id
        WHERE LENGTH(p.passage_text) BETWEEN 300 AND 3000
        AND p.passage_text !~ '^[0-9\\s\\.\\,]+$'
        AND NOT EXISTS (
            SELECT 1 FROM passage_concepts pc
            WHERE pc.passage_id = p.passage_id
            AND pc.extractor_version = 'gemini-2.0-flash'
        )
        ORDER BY RANDOM()
        LIMIT %s
    """, (limit,))

    passages = cur.fetchall()
    cur.close()
    return passages


def store_concepts(conn, passage_id: str, concepts: dict):
    """Store extracted concepts in database."""
    cur = conn.cursor()

    timestamp = datetime.now(timezone.utc)
    extractor = 'gemini-2.0-flash'

    for concept_type, concept_list in concepts.items():
        for concept_name in concept_list:
            if not concept_name or len(concept_name) < 2:
                continue
            try:
                cur.execute("""
                    INSERT INTO passage_concepts
                    (passage_id, concept_name, concept_type, confidence, extractor_version, created_at)
                    VALUES (%s, %s, %s, %s, %s, %s)
                    ON CONFLICT (passage_id, concept_name, concept_type) DO UPDATE
                    SET confidence = EXCLUDED.confidence,
                        extractor_version = EXCLUDED.extractor_version,
                        created_at = EXCLUDED.created_at
                """, (passage_id, concept_name.lower()[:200], concept_type, 0.9, extractor, timestamp))
            except Exception as e:
                conn.rollback()
                continue

    conn.commit()
    cur.close()


def run_batch_extraction(limit: int = 500, dry_run: bool = False, workers: int = 4, model: str = DEFAULT_MODEL):
    """Run batch concept extraction."""
    print(f"\n{'='*60}")
    print(f"BATCH CONCEPT EXTRACTION")
    print(f"Model: {model} | Limit: {limit} | Dry run: {dry_run}")
    print(f"{'='*60}\n")

    conn = psycopg2.connect(POSTGRES_DSN)
    storage_client = storage.Client(project=GCP_PROJECT_ID)
    bucket = storage_client.bucket(GCP_BUCKET)

    # Get passages
    print("Fetching passages...")
    passages = get_passages_needing_concepts(conn, limit)
    print(f"Found {len(passages)} passages needing concept extraction\n")

    if not passages:
        print("No passages need processing!")
        return

    # Process
    results = []
    errors = []
    stats = {'methods': 0, 'problems': 0, 'domains': 0, 'entities': 0}

    start_time = time.time()

    for i, (passage_id, text, title, doc_id) in enumerate(passages):
        if i % 50 == 0:
            elapsed = time.time() - start_time
            rate = i / elapsed if elapsed > 0 else 0
            print(f"[{i}/{len(passages)}] {rate:.1f} passages/sec - {title[:40]}...")

        concepts = extract_concepts_gemini(text, model=model)

        if concepts:
            results.append({
                'passage_id': str(passage_id),
                'doc_id': str(doc_id),
                'title': title,
                'concepts': concepts
            })

            # Update stats
            for key in stats:
                stats[key] += len(concepts.get(key, []))

            # Store in database
            if not dry_run:
                store_concepts(conn, passage_id, concepts)

            # Rate limiting
            time.sleep(0.3)
        else:
            errors.append({'passage_id': str(passage_id), 'title': title})

        # Progress save every 100
        if i > 0 and i % 100 == 0 and not dry_run:
            checkpoint = {
                'checkpoint': i,
                'timestamp': datetime.now(timezone.utc).isoformat(),
                'stats': stats,
                'results_count': len(results)
            }
            blob = bucket.blob(f'batch_runs/checkpoint_{i}.json')
            blob.upload_from_string(json.dumps(checkpoint))

    elapsed = time.time() - start_time

    # Final summary
    print(f"\n{'='*60}")
    print(f"BATCH COMPLETE")
    print(f"{'='*60}")
    print(f"Processed: {len(results)}/{len(passages)} passages")
    print(f"Errors: {len(errors)}")
    print(f"Time: {elapsed:.1f}s ({len(results)/elapsed:.1f} passages/sec)")
    print(f"\nConcepts extracted:")
    for key, count in stats.items():
        print(f"  {key}: {count}")
    print(f"  TOTAL: {sum(stats.values())}")

    # Save final results to GCS
    if not dry_run:
        timestamp = datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S')
        final_output = {
            'run_id': timestamp,
            'passages_processed': len(passages),
            'successful': len(results),
            'errors': len(errors),
            'elapsed_seconds': elapsed,
            'stats': stats,
            'results': results,
            'error_list': errors
        }

        blob = bucket.blob(f'batch_runs/batch_{timestamp}.json')
        blob.upload_from_string(json.dumps(final_output, indent=2))
        print(f"\n✓ Results saved to gs://{GCP_BUCKET}/batch_runs/batch_{timestamp}.json")

    conn.close()
    return results, stats


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Batch concept extraction with Gemini")
    parser.add_argument("--limit", type=int, default=500, help="Number of passages to process")
    parser.add_argument("--dry-run", action="store_true", help="Don't save to database")
    parser.add_argument("--workers", type=int, default=4, help="Parallel workers (future)")
    parser.add_argument("--model", type=str, default=DEFAULT_MODEL,
                        choices=["gemini-2.5-flash-lite", "gemini-2.5-flash", "gemini-2.0-flash",
                                 "gemini-2.0-flash-lite", "gemini-3-flash-preview"],
                        help="Gemini model to use (default: gemini-2.5-flash-lite)")
    args = parser.parse_args()

    run_batch_extraction(limit=args.limit, dry_run=args.dry_run, workers=args.workers, model=args.model)
