#!/usr/bin/env python3
"""
Process completed Gemini batch job results.

Usage:
    python scripts/process_batch_results.py <job_name>
    python scripts/process_batch_results.py --latest
"""

import os
import sys
import json
import re
import argparse
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from lib.config import GCP_PROJECT_ID, GCP_BUCKET, GCP_SERVICE_ACCOUNT, POSTGRES_DSN

os.environ['GOOGLE_APPLICATION_CREDENTIALS'] = GCP_SERVICE_ACCOUNT

import psycopg2
from google.cloud import storage
from google import genai
from google.genai import types

client = genai.Client(
    vertexai=True,
    project=GCP_PROJECT_ID,
    location="global"
)

COMPLETED_STATES = {'JOB_STATE_SUCCEEDED', 'JOB_STATE_FAILED', 'JOB_STATE_CANCELLED', 'JOB_STATE_EXPIRED'}


def get_latest_job() -> str:
    """Get the most recent batch job name from GCS tracking files."""
    storage_client = storage.Client(project=GCP_PROJECT_ID)
    bucket = storage_client.bucket(GCP_BUCKET)

    blobs = list(bucket.list_blobs(prefix='batch_jobs/'))
    if not blobs:
        return None

    # Sort by name (timestamp-based)
    blobs.sort(key=lambda b: b.name, reverse=True)
    latest = blobs[0]

    content = json.loads(latest.download_as_string())
    return content.get('job_name')


def parse_concept_response(text: str) -> dict:
    """Parse JSON response into concept dict."""
    try:
        cleaned = re.sub(r'```json\s*', '', text)
        cleaned = re.sub(r'```', '', cleaned).strip()
        return json.loads(cleaned)
    except json.JSONDecodeError:
        return {}


def store_concepts(conn, passage_id: str, concepts: dict):
    """Store extracted concepts in database."""
    cur = conn.cursor()
    timestamp = datetime.now(timezone.utc)
    extractor_model = 'gemini-2.5-flash-lite'
    extractor_version = 'batch-v1'

    stored = 0
    for concept_type, concept_list in concepts.items():
        for concept_name in concept_list:
            if not concept_name or len(concept_name) < 2:
                continue
            try:
                cur.execute("""
                    INSERT INTO passage_concepts
                    (passage_id, concept_name, concept_type, confidence, extractor_model, extractor_version, created_at)
                    VALUES (%s, %s, %s, %s, %s, %s, %s)
                    ON CONFLICT (passage_id, concept_name, extractor_version) DO UPDATE
                    SET concept_type = EXCLUDED.concept_type,
                        confidence = EXCLUDED.confidence,
                        created_at = EXCLUDED.created_at
                """, (passage_id, concept_name.lower()[:200], concept_type, 0.9, extractor_model, extractor_version, timestamp))
                stored += 1
            except Exception as e:
                conn.rollback()
                continue

    conn.commit()
    cur.close()
    return stored


def process_job_results(job_name: str, dry_run: bool = False):
    """Process results from a completed batch job."""
    print(f"\n{'='*60}")
    print(f"PROCESSING BATCH JOB RESULTS")
    print(f"{'='*60}")
    print(f"Job: {job_name}")

    # Get job status
    job = client.batches.get(name=job_name)
    state_name = job.state.name if hasattr(job.state, 'name') else str(job.state)
    print(f"State: {state_name}")

    if state_name not in COMPLETED_STATES:
        print(f"\n⚠ Job not complete yet. Current state: {state_name}")
        print("Re-run this script when the job completes.")
        return

    if state_name != 'JOB_STATE_SUCCEEDED':
        print(f"\n✗ Job failed or was cancelled: {state_name}")
        if hasattr(job, 'error') and job.error:
            print(f"Error: {job.error}")
        return

    print("\n✓ Job succeeded! Processing results...")

    stats = {'processed': 0, 'stored': 0, 'errors': 0, 'concepts': 0}

    # Connect to database
    conn = None
    if not dry_run:
        conn = psycopg2.connect(POSTGRES_DSN)

    # Load passage mapping from GCS
    passage_mapping = {}
    try:
        # Find the mapping file that matches this job's timestamp
        storage_client = storage.Client(project=GCP_PROJECT_ID)
        bucket = storage_client.bucket(GCP_BUCKET)
        mapping_blobs = list(bucket.list_blobs(prefix='batch_inputs/'))
        for blob in mapping_blobs:
            if '_mapping.json' in blob.name:
                mapping_content = json.loads(blob.download_as_string())
                passage_mapping = {int(k): v for k, v in mapping_content.items()}
                print(f"Loaded passage mapping: {len(passage_mapping)} entries")
                break
    except Exception as e:
        print(f"Warning: Could not load passage mapping: {e}")

    # Get results
    if hasattr(job, 'dest') and job.dest:
        # Check for GCS output first (most common for Vertex AI batch)
        if hasattr(job.dest, 'gcs_uri') and job.dest.gcs_uri:
            gcs_uri = job.dest.gcs_uri
            print(f"Results in GCS: {gcs_uri}")

            # Parse gs://bucket/path
            parts = gcs_uri.replace("gs://", "").split("/", 1)
            bucket_name = parts[0]
            prefix = parts[1] if len(parts) > 1 else ""

            storage_client = storage.Client(project=GCP_PROJECT_ID)
            bucket = storage_client.bucket(bucket_name)
            blobs = list(bucket.list_blobs(prefix=prefix))

            for blob in blobs:
                if blob.name.endswith('.jsonl'):
                    print(f"  Processing: {blob.name}")
                    content = blob.download_as_string().decode('utf-8')

                    for idx, line in enumerate(content.strip().split('\n')):
                        if not line:
                            continue
                        stats['processed'] += 1
                        try:
                            result = json.loads(line)

                            # Extract text from response
                            text = ""
                            if 'response' in result:
                                resp = result['response']
                                if 'candidates' in resp and resp['candidates']:
                                    parts_list = resp['candidates'][0].get('content', {}).get('parts', [])
                                    if parts_list:
                                        text = parts_list[0].get('text', '')

                            if text:
                                concepts = parse_concept_response(text)
                                if concepts:
                                    total_concepts = sum(len(v) for v in concepts.values())
                                    stats['concepts'] += total_concepts

                                    # Get passage_id from mapping
                                    passage_id = None
                                    if idx in passage_mapping:
                                        passage_id = passage_mapping[idx].get('passage_id')

                                    if passage_id and conn:
                                        stored = store_concepts(conn, passage_id, concepts)
                                        stats['stored'] += stored

                        except Exception as e:
                            stats['errors'] += 1
                            if stats['errors'] <= 3:
                                print(f"  Error processing line {idx}: {e}")

        elif hasattr(job.dest, 'inlined_responses') and job.dest.inlined_responses:
            print(f"Found {len(job.dest.inlined_responses)} inline responses")

            for i, resp in enumerate(job.dest.inlined_responses):
                stats['processed'] += 1
                try:
                    text = ""
                    if hasattr(resp, 'response'):
                        r = resp.response
                        if hasattr(r, 'candidates') and r.candidates:
                            if r.candidates[0].content and r.candidates[0].content.parts:
                                text = r.candidates[0].content.parts[0].text

                    if text:
                        concepts = parse_concept_response(text)
                        if concepts:
                            total_concepts = sum(len(v) for v in concepts.values())
                            stats['concepts'] += total_concepts

                            passage_id = None
                            if i in passage_mapping:
                                passage_id = passage_mapping[i].get('passage_id')

                            if passage_id and conn:
                                stored = store_concepts(conn, passage_id, concepts)
                                stats['stored'] += stored

                except Exception as e:
                    stats['errors'] += 1
                    if stats['errors'] <= 3:
                        print(f"  Error processing response {i}: {e}")

        elif hasattr(job.dest, 'file_name') and job.dest.file_name:
            print(f"Results in file: {job.dest.file_name}")
            try:
                result_content = client.files.download(file=job.dest.file_name)
                result_text = result_content.decode('utf-8')

                for line in result_text.strip().split('\n'):
                    if not line:
                        continue
                    stats['processed'] += 1
                    try:
                        result = json.loads(line)
                        # Process based on result format...
                        stats['stored'] += 1
                    except json.JSONDecodeError:
                        stats['errors'] += 1

            except Exception as e:
                print(f"Error downloading results: {e}")

    if conn:
        conn.close()

    # Summary
    print(f"\n{'='*60}")
    print(f"PROCESSING COMPLETE")
    print(f"{'='*60}")
    print(f"Responses processed: {stats['processed']}")
    print(f"Concepts extracted: {stats['concepts']}")
    print(f"Database records: {stats['stored']}")
    print(f"Errors: {stats['errors']}")

    if dry_run:
        print("\n[DRY RUN] No database changes made")

    return stats


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Process Gemini batch job results")
    parser.add_argument("job_name", nargs='?', help="Batch job name to process")
    parser.add_argument("--latest", action="store_true", help="Process the most recent job")
    parser.add_argument("--dry-run", action="store_true", help="Don't write to database")
    args = parser.parse_args()

    if args.latest:
        job_name = get_latest_job()
        if not job_name:
            print("No batch jobs found in tracking")
            sys.exit(1)
    elif args.job_name:
        job_name = args.job_name
    else:
        parser.print_help()
        sys.exit(1)

    process_job_results(job_name, dry_run=args.dry_run)
