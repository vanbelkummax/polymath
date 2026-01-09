#!/usr/bin/env python3
"""
Gemini Batch API Wrapper for Polymath Concept Extraction

Provides:
- InlinedRequest batching for small-medium batches
- Cost estimation utilities
- Structured JSON output enforcement
- Retry logic with exponential backoff
"""

import os
import json
import time
import logging
import hashlib
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass, asdict
from pathlib import Path

logger = logging.getLogger(__name__)

# Lazy import to allow module to load even if google-genai not installed
_genai_client = None


def get_genai_client():
    """Get or create the Gemini API client."""
    global _genai_client
    if _genai_client is None:
        api_key = os.environ.get("GEMINI_API_KEY")
        if not api_key:
            raise ValueError(
                "GEMINI_API_KEY environment variable not set. "
                "Get your API key from https://aistudio.google.com/apikey"
            )
        from google import genai
        _genai_client = genai.Client(api_key=api_key)
    return _genai_client


# Allowed concept types (must match existing schema)
ALLOWED_TYPES = {
    "method", "objective", "prior", "model", "dataset",
    "field", "math_object", "metric", "domain", "algorithm", "technique"
}

# Default model for batch operations
DEFAULT_MODEL = "gemini-2.0-flash"  # Winner of A/B test (8% vs 68% failure rate)

# JSON schema for structured output - tight constraints to prevent truncation
CONCEPT_SCHEMA = {
    "type": "object",
    "properties": {
        "concepts": {
            "type": "array",
            "maxItems": 8,  # Hard cap at 8 concepts
            "items": {
                "type": "object",
                "properties": {
                    "name": {"type": "string", "maxLength": 60},
                    "type": {"type": "string", "enum": list(ALLOWED_TYPES)},
                    "aliases": {"type": "array", "items": {"type": "string", "maxLength": 30}, "maxItems": 3},
                    "confidence": {"type": "number", "minimum": 0, "maximum": 1},
                    "evidence": {
                        "type": "object",
                        "properties": {
                            "quote": {"type": "string", "maxLength": 120},
                            "context": {"type": "string", "maxLength": 240}
                        },
                        "required": ["quote"]
                    }
                },
                "required": ["name", "type", "confidence"]
            }
        }
    },
    "required": ["concepts"]
}

# Tighter schema for retry attempts
CONCEPT_SCHEMA_TIGHT = {
    "type": "object",
    "properties": {
        "concepts": {
            "type": "array",
            "maxItems": 6,
            "items": {
                "type": "object",
                "properties": {
                    "name": {"type": "string", "maxLength": 50},
                    "type": {"type": "string", "enum": list(ALLOWED_TYPES)},
                    "confidence": {"type": "number"}
                },
                "required": ["name", "type", "confidence"]
            }
        }
    },
    "required": ["concepts"]
}


def build_extraction_prompt(text: str, max_chars: int = 1200) -> str:
    """Build the concept extraction prompt.

    Args:
        text: Source text to extract concepts from
        max_chars: Maximum characters to include (truncates if longer)

    Returns:
        Formatted prompt string
    """
    # Truncate if needed, keeping start (usually most informative)
    if len(text) > max_chars:
        text = text[:max_chars] + "..."

    # Minimal prompt to reduce output tokens
    return f"""Extract 5-8 key scientific concepts from this text as JSON.
Format: {{"concepts":[{{"name":"snake_case","type":"domain|method|model|metric|technique","confidence":0.8,"quote":"short quote"}}]}}
Types: domain, method, model, metric, technique, dataset, algorithm, field, objective, prior, math_object
Text: {text}"""


@dataclass
class BatchRequest:
    """A single request in a batch."""
    custom_id: str  # passage_id as string
    text: str

    def to_inlined_request(self, model: str = DEFAULT_MODEL, max_output_tokens: int = 384) -> Dict:
        """Convert to google-genai InlinedRequest format."""
        prompt = build_extraction_prompt(self.text)
        return {
            "model": model,
            "contents": prompt,
            "metadata": {"passage_id": self.custom_id},
            "config": {
                "response_mime_type": "application/json",
                "response_schema": CONCEPT_SCHEMA,
                "max_output_tokens": max_output_tokens,
                "temperature": 0.1,  # Low for consistency
            }
        }


@dataclass
class BatchResult:
    """Result from processing a batch."""
    job_name: str
    state: str
    total_requests: int
    completed: int
    failed: int
    results: List[Dict[str, Any]]  # List of {custom_id, response, error}


def create_batch_job(
    requests: List[BatchRequest],
    model: str = DEFAULT_MODEL,
    max_output_tokens: int = 384,
    display_name: Optional[str] = None
) -> str:
    """Create a batch job with inlined requests.

    Args:
        requests: List of BatchRequest objects
        model: Model ID to use
        max_output_tokens: Max tokens per response
        display_name: Optional job name

    Returns:
        Batch job name/ID
    """
    client = get_genai_client()

    # Convert to inlined requests
    inlined = [r.to_inlined_request(model, max_output_tokens) for r in requests]

    # Create the batch job
    from google.genai import types

    job = client.batches.create(
        model=model,
        src=types.BatchJobSource(inlined_requests=[
            types.InlinedRequest(**req) for req in inlined
        ]),
        config=types.CreateBatchJobConfig(
            display_name=display_name or f"polymath_concepts_{int(time.time())}"
        )
    )

    logger.info(f"Created batch job: {job.name}, state: {job.state}")
    return job.name


def poll_batch_job(
    job_name: str,
    poll_interval: float = 10.0,
    max_wait: float = 3600.0,
    callback: Optional[callable] = None
) -> Tuple[str, Optional[Dict]]:
    """Poll a batch job until completion.

    Args:
        job_name: The batch job name/ID
        poll_interval: Seconds between polls
        max_wait: Maximum seconds to wait
        callback: Optional function called on each poll with (job, elapsed)

    Returns:
        Tuple of (final_state, completion_stats)
    """
    client = get_genai_client()
    start_time = time.time()

    while True:
        elapsed = time.time() - start_time
        if elapsed > max_wait:
            logger.warning(f"Job {job_name} timed out after {max_wait}s")
            return "TIMEOUT", None

        job = client.batches.get(name=job_name)

        if callback:
            callback(job, elapsed)

        # Check terminal states
        state_str = str(job.state) if job.state else "UNKNOWN"
        if "SUCCEEDED" in state_str or "COMPLETED" in state_str:
            logger.info(f"Job {job_name} completed successfully")
            return "SUCCEEDED", job.completion_stats
        elif "FAILED" in state_str:
            logger.error(f"Job {job_name} failed: {job.error}")
            return "FAILED", None
        elif "CANCELLED" in state_str:
            logger.warning(f"Job {job_name} was cancelled")
            return "CANCELLED", None

        logger.debug(f"Job {job_name} state: {state_str}, elapsed: {elapsed:.1f}s")
        time.sleep(poll_interval)


def get_batch_results(job_name: str) -> List[Dict[str, Any]]:
    """Get results from a completed batch job.

    Args:
        job_name: The batch job name/ID

    Returns:
        List of result dicts with custom_id, response, and any errors
    """
    client = get_genai_client()
    job = client.batches.get(name=job_name)

    results = []

    # Results are in the destination (if specified) or inline
    # For inline jobs, we need to iterate through responses
    if hasattr(job, 'dest') and job.dest:
        # Results are in GCS or BigQuery - need to download
        logger.warning("External destination not yet implemented, use inline batches")
        return results

    # For inline jobs, results should be accessible through job object
    # The exact API depends on google-genai version
    # Try to access inline results
    if hasattr(job, 'responses'):
        for resp in job.responses:
            results.append({
                "custom_id": resp.metadata.get("passage_id") if resp.metadata else None,
                "response": resp.candidates[0].content.parts[0].text if resp.candidates else None,
                "error": None
            })

    return results


def parse_concept_response(response_text: str) -> List[Dict[str, Any]]:
    """Parse a JSON response into concept dicts.

    Args:
        response_text: JSON string from model

    Returns:
        List of concept dicts, or empty list on parse failure
    """
    if not response_text:
        return []

    try:
        data = json.loads(response_text)

        # Handle different response formats
        if isinstance(data, list):
            concepts = data  # Direct list of concepts
        elif isinstance(data, dict):
            concepts = data.get("concepts", [])
        else:
            return []

        # Validate and normalize each concept
        valid_concepts = []
        for c in concepts:
            name = c.get("name", "").strip()
            ctype = c.get("type", "domain")

            # Normalize name to snake_case
            name = normalize_snake_case(name)
            if not name or len(name) < 2:
                continue

            # Validate type
            if ctype not in ALLOWED_TYPES:
                ctype = "domain"  # Default fallback

            # Build normalized concept
            valid_concepts.append({
                "name": name[:100],  # Cap length
                "type": ctype,
                "aliases": [a[:40] for a in c.get("aliases", [])[:6]],  # Cap aliases
                "confidence": max(0.0, min(1.0, float(c.get("confidence", 0.7)))),
                "evidence": {
                    "quote": c.get("evidence", {}).get("quote", "")[:160],
                    "context": c.get("evidence", {}).get("context", "")[:300]
                }
            })

        return valid_concepts

    except (json.JSONDecodeError, KeyError, TypeError) as e:
        logger.warning(f"Failed to parse response: {e}")
        return []


def normalize_snake_case(s: str) -> str:
    """Normalize a string to snake_case.

    Args:
        s: Input string

    Returns:
        snake_case normalized string
    """
    import re
    # Remove special chars, convert spaces/hyphens to underscores
    s = re.sub(r'[^\w\s-]', '', s.lower())
    s = re.sub(r'[-\s]+', '_', s)
    s = re.sub(r'_+', '_', s)  # Collapse multiple underscores
    return s.strip('_')


def estimate_cost(
    num_requests: int,
    avg_input_chars: int = 1500,
    max_output_tokens: int = 384,
    model: str = DEFAULT_MODEL
) -> Dict[str, float]:
    """Estimate batch job cost.

    Args:
        num_requests: Number of passages to process
        avg_input_chars: Average input text length
        max_output_tokens: Maximum output tokens per request
        model: Model to use

    Returns:
        Dict with input_tokens, output_tokens, and estimated_cost_usd
    """
    # Rough token estimation: ~4 chars per token
    input_tokens_per_request = (avg_input_chars + 300) // 4  # +300 for prompt
    output_tokens_per_request = max_output_tokens * 0.6  # Assume 60% utilization

    total_input_tokens = num_requests * input_tokens_per_request
    total_output_tokens = num_requests * output_tokens_per_request

    # Gemini 2.0 Flash-Lite batch pricing (as of 2026-01)
    # Input: $0.01875 per 1M tokens
    # Output: $0.075 per 1M tokens
    input_cost = (total_input_tokens / 1_000_000) * 0.01875
    output_cost = (total_output_tokens / 1_000_000) * 0.075

    return {
        "num_requests": num_requests,
        "input_tokens": int(total_input_tokens),
        "output_tokens": int(total_output_tokens),
        "input_cost_usd": round(input_cost, 4),
        "output_cost_usd": round(output_cost, 4),
        "total_cost_usd": round(input_cost + output_cost, 4),
        "model": model
    }


def build_evidence_jsonb(
    concept: Dict[str, Any],
    source_text: str,
    max_source_chars: int = 1200
) -> Dict[str, Any]:
    """Build the evidence JSONB structure for passage_concepts.

    Args:
        concept: Parsed concept dict with evidence
        source_text: Original passage text
        max_source_chars: Max chars for source_text field

    Returns:
        JSONB-ready dict
    """
    evidence = concept.get("evidence", {})
    quote = evidence.get("quote", "")
    context = evidence.get("context", "")

    # Try to find quote position in source text
    start_pos = None
    end_pos = None
    if quote and len(quote) > 10:
        pos = source_text.lower().find(quote.lower()[:50])
        if pos >= 0:
            start_pos = pos
            end_pos = pos + len(quote)

    return {
        "surface": concept.get("name", ""),
        "support": [{
            "quote": quote,
            "start": start_pos,
            "end": end_pos
        }] if quote else [],
        "context": context,
        "source_text": source_text[:max_source_chars],
        "quality": {
            "confidence": concept.get("confidence", 0.0),
            "notes": f"extracted by {DEFAULT_MODEL}"
        }
    }


# Simple synchronous batch runner for pilot/small batches
def run_sync_batch(
    requests: List[BatchRequest],
    model: str = DEFAULT_MODEL,
    max_output_tokens: int = 512,
    delay_between_requests: float = 0.3,
    max_retries: int = 3,
    parse_retries: int = 2
) -> List[Dict[str, Any]]:
    """Run a batch synchronously (blocking) with rate limiting and parse-retry.

    For small batches (< 100 requests), this is simpler than async polling.
    Uses individual requests with delays to avoid rate limits.

    Args:
        requests: List of BatchRequest objects
        model: Model ID
        max_output_tokens: Max tokens per response (512 is safe for 8 concepts)
        delay_between_requests: Seconds to wait between requests
        max_retries: Max retries per request on 429 errors
        parse_retries: Max retries on JSON parse failures

    Returns:
        List of {custom_id, concepts, error} dicts
    """
    client = get_genai_client()
    results = []

    for i, req in enumerate(requests):
        # Add delay between requests to avoid rate limits
        if i > 0:
            time.sleep(delay_between_requests)

        result = _process_single_request(
            client, req, model, max_output_tokens, max_retries, parse_retries
        )
        results.append(result)

    return results


def _process_single_request(
    client,
    req: BatchRequest,
    model: str,
    max_output_tokens: int,
    max_retries: int,
    parse_retries: int
) -> Dict[str, Any]:
    """Process a single request with rate-limit and parse-fail retries."""

    retries = 0
    while retries <= max_retries:
        try:
            prompt = build_extraction_prompt(req.text)

            # First attempt with standard schema
            concepts, raw_text, parse_ok = _call_with_retry_on_parse(
                client, model, prompt, CONCEPT_SCHEMA, max_output_tokens, parse_retries
            )

            if concepts or parse_ok:
                return {
                    "custom_id": req.custom_id,
                    "concepts": concepts,
                    "raw_response": raw_text,
                    "error": None
                }
            else:
                # All parse retries failed - log for later
                logger.warning(f"All parse retries failed for {req.custom_id}, raw: {raw_text[:200] if raw_text else 'None'}...")
                return {
                    "custom_id": req.custom_id,
                    "concepts": [],
                    "raw_response": raw_text,
                    "error": "JSON_PARSE_FAILED"
                }

        except Exception as e:
            error_str = str(e)
            if "429" in error_str or "RESOURCE_EXHAUSTED" in error_str:
                retries += 1
                if retries <= max_retries:
                    wait_time = 2 ** retries * 10
                    logger.warning(f"Rate limited, waiting {wait_time}s (retry {retries}/{max_retries})")
                    time.sleep(wait_time)
                    continue

            logger.error(f"Request {req.custom_id} failed: {e}")
            return {
                "custom_id": req.custom_id,
                "concepts": [],
                "raw_response": None,
                "error": error_str
            }

    return {
        "custom_id": req.custom_id,
        "concepts": [],
        "raw_response": None,
        "error": "MAX_RETRIES_EXCEEDED"
    }


def _call_with_retry_on_parse(
    client,
    model: str,
    prompt: str,
    schema: Dict,
    max_output_tokens: int,
    parse_retries: int
) -> Tuple[List[Dict], str, bool]:
    """Call API with retry on JSON parse failure.

    Returns:
        Tuple of (concepts, raw_text, parse_succeeded)
    """
    raw_text = None

    for attempt in range(parse_retries + 1):
        try:
            # Use tighter schema and repair prompt on retries
            use_schema = CONCEPT_SCHEMA_TIGHT if attempt > 0 else schema
            use_tokens = max(256, max_output_tokens - (attempt * 100))  # Reduce tokens on retry

            current_prompt = prompt
            if attempt > 0:
                current_prompt = f"Return ONLY valid JSON matching schema, no extra text.\n\n{prompt}"

            response = client.models.generate_content(
                model=model,
                contents=current_prompt,
                config={
                    "response_mime_type": "application/json",
                    "response_schema": use_schema,
                    "max_output_tokens": use_tokens,
                    "temperature": 0,
                }
            )

            # Try to get parsed response from SDK first (structured output)
            parsed_data = None
            if hasattr(response, 'parsed') and response.parsed:
                parsed_data = response.parsed

            # Fallback to text extraction
            raw_text = ""
            if response.candidates:
                candidate = response.candidates[0]
                if candidate.content and candidate.content.parts:
                    raw_text = candidate.content.parts[0].text

            # Use parsed data if available, otherwise parse text
            if parsed_data:
                if isinstance(parsed_data, dict):
                    concepts = parsed_data.get("concepts", [])
                else:
                    concepts = parse_concept_response(json.dumps(parsed_data))
                return concepts, raw_text, True

            concepts = parse_concept_response(raw_text)
            if concepts:
                return concepts, raw_text, True

            # Parse failed, will retry if attempts remain
            if attempt < parse_retries:
                logger.debug(f"Parse attempt {attempt + 1} failed, retrying with tighter schema")
                time.sleep(0.2)

        except json.JSONDecodeError:
            if attempt < parse_retries:
                logger.debug(f"JSON decode failed on attempt {attempt + 1}, retrying")
                time.sleep(0.2)

    return [], raw_text, False


if __name__ == "__main__":
    # Quick test
    logging.basicConfig(level=logging.INFO)

    # Test cost estimation
    cost = estimate_cost(538865, avg_input_chars=1500)
    print(f"Estimated cost for 538,865 passages: ${cost['total_cost_usd']:.2f}")
    print(f"  Input: {cost['input_tokens']:,} tokens (${cost['input_cost_usd']:.2f})")
    print(f"  Output: {cost['output_tokens']:,} tokens (${cost['output_cost_usd']:.2f})")
