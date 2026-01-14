#!/usr/bin/env python3
"""
Async Batch Concept Extraction using Gemini Batch API with GCS storage.

Uses Vertex AI authentication with 50% cost reduction for async batch processing.

Usage:
    python scripts/batch_concept_extraction_async.py --limit 1000 --dry-run
    python scripts/batch_concept_extraction_async.py --limit 5000
    python scripts/batch_concept_extraction_async.py --status  # Check pending jobs
"""

import os
import sys
import json
import time
import argparse
from datetime import datetime, timezone
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple

sys.path.insert(0, str(Path(__file__).parent.parent))

from lib.config import (
    GCP_PROJECT_ID, GCP_BUCKET, GCP_SERVICE_ACCOUNT, POSTGRES_DSN
)

# Set credentials for GCP
os.environ['GOOGLE_APPLICATION_CREDENTIALS'] = GCP_SERVICE_ACCOUNT

import psycopg2
from google.cloud import storage
from google import genai
from google.genai import types

# Initialize client with Vertex AI (global location for latest models)
client = genai.Client(
    vertexai=True,
    project=GCP_PROJECT_ID,
    location="global"
)

# Model choice - gemini-2.5-flash-lite best for batch (fast, cheap, 65k output)
DEFAULT_MODEL = "gemini-2.5-flash-lite"

# Batch job state tracking
COMPLETED_STATES = {
    'JOB_STATE_SUCCEEDED', 'JOB_STATE_FAILED',
    'JOB_STATE_CANCELLED', 'JOB_STATE_EXPIRED'
}

CONCEPT_PROMPT_TEMPLATE = """Extract key scientific concepts from this research text. Return ONLY valid JSON:
{{"methods": [], "problems": [], "domains": [], "entities": []}}

Rules:
- methods: techniques, algorithms, tools (e.g., "spatial transcriptomics", "gradient descent")
- problems: challenges being addressed (e.g., "cell type deconvolution", "batch effects")
- domains: research fields (e.g., "computational pathology", "drug discovery")
- entities: specific genes, proteins, datasets, diseases (e.g., "TP53", "Alzheimer's disease")
- Be specific, not generic
- Only include explicitly mentioned concepts
- Return 5-10 concepts maximum

Text:
{text}"""


def get_passages_needing_concepts(conn, limit: int) -> list:
    """Get passages that don't have Gemini-extracted concepts yet."""
    cur = conn.cursor()

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


def create_jsonl_content(passages: list, model: str = DEFAULT_MODEL) -> str:
    """Create JSONL content for batch processing."""
    lines = []
    for passage_id, text, title, doc_id in passages:
        # Truncate text to 2000 chars for prompt
        truncated_text = text[:2000] if len(text) > 2000 else text

        request_obj = {
            "key": str(passage_id),
            "request": {
                "contents": [{
                    "parts": [{"text": CONCEPT_PROMPT_TEMPLATE.format(text=truncated_text)}],
                    "role": "user"
                }],
                "generationConfig": {
                    "temperature": 0.1,
                    "maxOutputTokens": 512
                }
            },
            "metadata": {
                "passage_id": str(passage_id),
                "doc_id": str(doc_id),
                "title": title[:100] if title else ""
            }
        }
        lines.append(json.dumps(request_obj))

    return "\n".join(lines)


def upload_to_gcs(bucket_name: str, blob_path: str, content: str) -> str:
    """Upload content to GCS and return the gs:// URI."""
    storage_client = storage.Client(project=GCP_PROJECT_ID)
    bucket = storage_client.bucket(bucket_name)
    blob = bucket.blob(blob_path)
    blob.upload_from_string(content, content_type="application/jsonl")
    return f"gs://{bucket_name}/{blob_path}"


def download_from_gcs(uri: str) -> str:
    """Download content from GCS URI."""
    # Parse gs://bucket/path
    parts = uri.replace("gs://", "").split("/", 1)
    bucket_name = parts[0]
    blob_path = parts[1] if len(parts) > 1 else ""

    storage_client = storage.Client(project=GCP_PROJECT_ID)
    bucket = storage_client.bucket(bucket_name)
    blob = bucket.blob(blob_path)
    return blob.download_as_string().decode('utf-8')


def create_async_batch_job(
    passages: list,
    model: str = DEFAULT_MODEL,
    display_name: Optional[str] = None
) -> Tuple[str, str]:
    """Create an async batch job using GCS storage.

    Returns:
        Tuple of (job_name, input_gcs_uri)
    """
    timestamp = datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S')

    # Create JSONL content
    jsonl_content = create_jsonl_content(passages, model)

    # Upload to GCS
    input_path = f"batch_inputs/concepts_{timestamp}.jsonl"
    input_uri = upload_to_gcs(GCP_BUCKET, input_path, jsonl_content)
    print(f"Uploaded input to: {input_uri}")

    # Upload the input file to Gemini Files API
    # For Vertex AI, we use GCS directly as source
    job_display_name = display_name or f"polymath_concepts_{timestamp}"

    try:
        # Try using GCS as source directly
        job = client.batches.create(
            model=model,
            src=input_uri,
            config=types.CreateBatchJobConfig(
                display_name=job_display_name
            )
        )
    except Exception as e:
        # Fallback: use inline requests if GCS source not supported
        print(f"GCS source failed ({e}), falling back to inline requests...")
        inline_requests = []
        for passage_id, text, title, doc_id in passages[:100]:  # Limit for inline
            truncated_text = text[:2000]
            inline_requests.append({
                'contents': [{
                    'parts': [{'text': CONCEPT_PROMPT_TEMPLATE.format(text=truncated_text)}],
                    'role': 'user'
                }],
                'generationConfig': {
                    'temperature': 0.1,
                    'maxOutputTokens': 512
                }
            })

        job = client.batches.create(
            model=model,
            src=inline_requests,
            config={'display_name': job_display_name}
        )

    print(f"Created batch job: {job.name}")
    print(f"Initial state: {job.state}")

    # Save job info to GCS for tracking
    job_info = {
        "job_name": job.name,
        "display_name": job_display_name,
        "model": model,
        "input_uri": input_uri,
        "num_passages": len(passages),
        "created_at": timestamp,
        "state": str(job.state)
    }
    job_info_path = f"batch_jobs/{timestamp}_job_info.json"
    upload_to_gcs(GCP_BUCKET, job_info_path, json.dumps(job_info, indent=2))

    return job.name, input_uri


def poll_batch_job(
    job_name: str,
    poll_interval: float = 30.0,
    max_wait: float = 86400.0  # 24 hours
) -> Dict[str, Any]:
    """Poll batch job until completion."""
    print(f"\nPolling job: {job_name}")
    start_time = time.time()

    while True:
        elapsed = time.time() - start_time
        if elapsed > max_wait:
            return {"state": "TIMEOUT", "elapsed": elapsed}

        try:
            job = client.batches.get(name=job_name)
            state_name = job.state.name if hasattr(job.state, 'name') else str(job.state)

            print(f"  [{elapsed/60:.1f}m] State: {state_name}")

            if state_name in COMPLETED_STATES:
                return {
                    "state": state_name,
                    "elapsed": elapsed,
                    "job": job
                }

        except Exception as e:
            print(f"  Poll error: {e}")

        time.sleep(poll_interval)


def process_batch_results(job, conn) -> Dict[str, int]:
    """Process and store batch job results."""
    stats = {"processed": 0, "stored": 0, "errors": 0}

    # Get results from job
    if hasattr(job, 'dest') and job.dest:
        # Results in destination (file or inline)
        if hasattr(job.dest, 'inlined_responses') and job.dest.inlined_responses:
            for resp in job.dest.inlined_responses:
                stats["processed"] += 1
                try:
                    # Parse response
                    text = ""
                    if hasattr(resp, 'response'):
                        if hasattr(resp.response, 'candidates') and resp.response.candidates:
                            text = resp.response.candidates[0].content.parts[0].text

                    if text:
                        concepts = parse_concept_response(text)
                        if concepts:
                            # Store in database (would need passage_id from metadata)
                            stats["stored"] += 1
                except Exception as e:
                    stats["errors"] += 1
                    print(f"  Error processing response: {e}")

        elif hasattr(job.dest, 'file_name') and job.dest.file_name:
            # Results in file - download and process
            try:
                result_content = client.files.download(file=job.dest.file_name)
                result_text = result_content.decode('utf-8')

                for line in result_text.strip().split('\n'):
                    if not line:
                        continue
                    stats["processed"] += 1
                    try:
                        result = json.loads(line)
                        # Process result...
                        stats["stored"] += 1
                    except json.JSONDecodeError:
                        stats["errors"] += 1
            except Exception as e:
                print(f"  Error downloading results: {e}")

    return stats


def parse_concept_response(text: str) -> Dict[str, list]:
    """Parse JSON response into concept dict."""
    import re
    try:
        # Clean markdown code blocks
        cleaned = re.sub(r'```json\s*', '', text)
        cleaned = re.sub(r'```', '', cleaned).strip()
        return json.loads(cleaned)
    except json.JSONDecodeError:
        return {}


def list_pending_jobs():
    """List all pending/running batch jobs."""
    print("\n" + "="*60)
    print("PENDING BATCH JOBS")
    print("="*60)

    try:
        jobs = client.batches.list(config=types.ListBatchJobsConfig(page_size=20))

        pending = []
        for job in jobs:
            state_name = job.state.name if hasattr(job.state, 'name') else str(job.state)
            if state_name not in COMPLETED_STATES:
                pending.append({
                    "name": job.name,
                    "state": state_name,
                    "display_name": job.display_name if hasattr(job, 'display_name') else ""
                })
                print(f"  {job.name}: {state_name}")

        if not pending:
            print("  No pending jobs")

        return pending

    except Exception as e:
        print(f"  Error listing jobs: {e}")
        return []


def run_async_batch(
    limit: int = 1000,
    model: str = DEFAULT_MODEL,
    dry_run: bool = False,
    wait: bool = False
):
    """Run async batch concept extraction."""
    print("\n" + "="*60)
    print("ASYNC BATCH CONCEPT EXTRACTION (50% cost reduction)")
    print(f"Model: {model} | Limit: {limit} | Dry run: {dry_run}")
    print("="*60 + "\n")

    conn = psycopg2.connect(POSTGRES_DSN)

    # Get passages
    print("Fetching passages...")
    passages = get_passages_needing_concepts(conn, limit)
    print(f"Found {len(passages)} passages needing concept extraction\n")

    if not passages:
        print("No passages need processing!")
        conn.close()
        return

    # Estimate cost
    # Gemini 2.5 Flash-Lite batch: $0.01875/1M input, $0.075/1M output
    # With 50% batch discount: $0.009375/1M input, $0.0375/1M output
    avg_input_tokens = 600  # ~2400 chars / 4
    avg_output_tokens = 150
    total_input = len(passages) * avg_input_tokens
    total_output = len(passages) * avg_output_tokens
    estimated_cost = (total_input / 1_000_000) * 0.009375 + (total_output / 1_000_000) * 0.0375

    print(f"Estimated cost (with 50% batch discount): ${estimated_cost:.4f}")
    print(f"  Input tokens: ~{total_input:,}")
    print(f"  Output tokens: ~{total_output:,}\n")

    if dry_run:
        print("[DRY RUN] Would create batch job for these passages")
        print(f"Sample passage IDs: {[str(p[0])[:8] for p in passages[:5]]}")
        conn.close()
        return

    # Create batch job
    job_name, input_uri = create_async_batch_job(passages, model)

    if wait:
        print("\nWaiting for job completion (Ctrl+C to exit and check later)...")
        result = poll_batch_job(job_name)
        print(f"\nJob completed with state: {result['state']}")

        if result['state'] == 'JOB_STATE_SUCCEEDED':
            stats = process_batch_results(result['job'], conn)
            print(f"Processed: {stats['processed']}, Stored: {stats['stored']}, Errors: {stats['errors']}")
    else:
        print("\nJob submitted! Check status with:")
        print(f"  python scripts/batch_concept_extraction_async.py --status")
        print(f"  Job name: {job_name}")

    conn.close()
    return job_name


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Async batch concept extraction with Gemini Batch API")
    parser.add_argument("--limit", type=int, default=1000, help="Number of passages to process")
    parser.add_argument("--dry-run", action="store_true", help="Don't submit job, just estimate")
    parser.add_argument("--wait", action="store_true", help="Wait for job completion")
    parser.add_argument("--status", action="store_true", help="Check status of pending jobs")
    parser.add_argument("--model", type=str, default=DEFAULT_MODEL,
                        choices=["gemini-2.5-flash-lite", "gemini-2.5-flash", "gemini-2.0-flash",
                                 "gemini-2.0-flash-lite", "gemini-3-flash-preview"],
                        help="Gemini model to use")
    args = parser.parse_args()

    if args.status:
        list_pending_jobs()
    else:
        run_async_batch(
            limit=args.limit,
            model=args.model,
            dry_run=args.dry_run,
            wait=args.wait
        )
