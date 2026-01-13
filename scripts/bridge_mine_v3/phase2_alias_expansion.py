#!/usr/bin/env python3
"""
Phase 2: Gemini Alias Expansion for Bridge Mine v3
For each bridge, generate 15-30 aliases per concept + 10 combined query templates.
Uses Gemini 2.0 Flash for alias generation.
"""

import json
import os
import sys
import time
from pathlib import Path

sys.path.insert(0, '/home/user/polymath-repo')

import google.generativeai as genai

OUTPUT_DIR = os.environ.get('OUTPUT_DIR', '/home/user/work/polymax/reports/bridge_mine_v3_20260112_2126')

# Load API key
API_KEY = os.environ.get('GEMINI_API_KEY')
if not API_KEY:
    with open('/home/user/polymath-repo/.env', 'r') as f:
        for line in f:
            if line.startswith('GEMINI_API_KEY='):
                API_KEY = line.strip().split('=', 1)[1]
                break

genai.configure(api_key=API_KEY)
# Use stable gemini-2.0-flash model
model = genai.GenerativeModel('gemini-2.0-flash')


ALIAS_PROMPT = """You are an expert in biomedical informatics and spatial transcriptomics.

Given a concept from computational biology / machine learning, generate diverse aliases and phrasings that researchers might use when writing about this concept in papers.

CONCEPT: {concept}
CONTEXT: This concept is being used in the context of {context}

Generate a JSON object with:
1. "aliases": A list of 15-25 alternative phrases/names for this concept. Include:
   - Formal method names (e.g., "variational autoencoder" → "VAE", "ELBO optimization")
   - How authors describe it in methods sections
   - Related synonyms and near-synonyms
   - Acronyms and abbreviations
   - Application-specific names (e.g., "scVI" for single-cell VAE)
   - Descriptive phrases (e.g., "latent variable model")

2. "search_terms": A list of 8-12 specific search terms that would find papers discussing this concept in the spatial/pathology context

Output ONLY valid JSON, no markdown:
{{"aliases": [...], "search_terms": [...]}}"""


COMBINED_QUERY_PROMPT = """You are an expert in cross-domain research discovery.

Given a bridge connecting two concepts, generate search queries that would find papers discussing how these concepts relate or could be combined.

SOURCE CONCEPT: {source}
SOURCE ALIASES: {source_aliases}

TARGET CONCEPT: {target}
TARGET ALIASES: {target_aliases}

BRIDGE CONTEXT: {context}

Generate a JSON object with:
1. "combined_queries": A list of 10-12 search queries that would find evidence connecting these concepts. Include:
   - Direct combinations: "<source_alias> for <target_alias>"
   - Analogical: "<source_alias> approach to <target_alias>"
   - Methodological: "using <source_alias> to achieve <target_alias>"
   - Problem-focused: "<target_problem> solved with <source_method>"

2. "hyde_abstract": A 2-3 sentence hypothetical abstract that describes a paper combining these concepts (for HyDE search)

Output ONLY valid JSON, no markdown:
{{"combined_queries": [...], "hyde_abstract": "..."}}"""


def generate_aliases(concept: str, context: str, max_retries: int = 5) -> dict:
    """Generate aliases for a single concept with exponential backoff."""
    prompt = ALIAS_PROMPT.format(concept=concept, context=context)

    for attempt in range(max_retries):
        try:
            response = model.generate_content(
                prompt,
                generation_config={
                    'temperature': 0.7,
                    'max_output_tokens': 1024,
                }
            )
            text = response.text.strip()
            # Clean up JSON if wrapped in markdown
            if text.startswith('```'):
                text = text.split('```')[1]
                if text.startswith('json'):
                    text = text[4:]
            return json.loads(text)
        except Exception as e:
            error_str = str(e)
            if '429' in error_str or 'quota' in error_str.lower():
                wait_time = 10 * (2 ** attempt)  # Exponential backoff: 10, 20, 40, 80, 160 seconds
                print(f"    Rate limited, waiting {wait_time}s...")
                time.sleep(wait_time)
            elif attempt < max_retries - 1:
                time.sleep(2 * (attempt + 1))
            else:
                print(f"  ⚠ Failed to generate aliases for {concept}: {e}")
                return {"aliases": [concept], "search_terms": [concept]}
    return {"aliases": [concept], "search_terms": [concept]}


def generate_combined_queries(source: str, target: str, source_aliases: list, target_aliases: list, context: str, max_retries: int = 5) -> dict:
    """Generate combined queries for a bridge with exponential backoff."""
    prompt = COMBINED_QUERY_PROMPT.format(
        source=source,
        target=target,
        source_aliases=json.dumps(source_aliases[:10]),
        target_aliases=json.dumps(target_aliases[:10]),
        context=context
    )

    for attempt in range(max_retries):
        try:
            response = model.generate_content(
                prompt,
                generation_config={
                    'temperature': 0.7,
                    'max_output_tokens': 1024,
                }
            )
            text = response.text.strip()
            if text.startswith('```'):
                text = text.split('```')[1]
                if text.startswith('json'):
                    text = text[4:]
            return json.loads(text)
        except Exception as e:
            error_str = str(e)
            if '429' in error_str or 'quota' in error_str.lower():
                wait_time = 10 * (2 ** attempt)
                print(f"    Rate limited, waiting {wait_time}s...")
                time.sleep(wait_time)
            elif attempt < max_retries - 1:
                time.sleep(2 * (attempt + 1))
            else:
                print(f"  ⚠ Failed to generate combined queries: {e}")
                return {"combined_queries": [f"{source} {target}"], "hyde_abstract": ""}
    return {"combined_queries": [f"{source} {target}"], "hyde_abstract": ""}


def main():
    # Load candidates
    candidates_file = Path(OUTPUT_DIR) / "INTERMEDIATE" / "BRIDGE_CANDIDATES_SEED.json"
    with open(candidates_file, 'r') as f:
        data = json.load(f)

    candidates = data["candidates"]
    output_file = Path(OUTPUT_DIR) / "INTERMEDIATE" / "ALIASES.jsonl"

    # Check for existing progress (resume support)
    processed_ids = set()
    if output_file.exists():
        with open(output_file, 'r') as f:
            for line in f:
                try:
                    existing = json.loads(line)
                    processed_ids.add(existing.get("bridge_id"))
                except:
                    pass
        print(f"Resuming from {len(processed_ids)} already processed bridges")

    # Track progress
    processed = 0
    total = len(candidates)
    skipped = 0

    with open(output_file, 'a') as f:  # Append mode for resume
        for candidate in candidates:
            processed += 1
            bridge_id = candidate["bridge_id"]

            # Skip already processed
            if bridge_id in processed_ids:
                skipped += 1
                continue

            source = candidate["source_concept"]
            target = candidate["target_concept"]
            context = candidate["bucket_description"]

            print(f"[{processed}/{total}] {bridge_id}: {source} → {target}")

            # Generate aliases for source
            source_data = generate_aliases(source, context)
            time.sleep(3)  # Rate limiting - 3s between calls

            # Generate aliases for target
            target_data = generate_aliases(target, context)
            time.sleep(3)

            # Generate combined queries
            combined_data = generate_combined_queries(
                source, target,
                source_data.get("aliases", []),
                target_data.get("aliases", []),
                context
            )
            time.sleep(3)

            # Combine all data
            result = {
                **candidate,
                "source_aliases": source_data.get("aliases", []),
                "source_search_terms": source_data.get("search_terms", []),
                "target_aliases": target_data.get("aliases", []),
                "target_search_terms": target_data.get("search_terms", []),
                "combined_queries": combined_data.get("combined_queries", []),
                "hyde_abstract": combined_data.get("hyde_abstract", ""),
            }

            f.write(json.dumps(result) + '\n')
            f.flush()

            # Progress indicator
            if processed % 10 == 0:
                print(f"  ✓ Processed {processed}/{total} bridges")

    print(f"\n✓ Saved aliases to {output_file}")
    print(f"  Total bridges processed: {processed}")

    # Quick stats
    total_aliases = 0
    with open(output_file, 'r') as f:
        for line in f:
            data = json.loads(line)
            total_aliases += len(data.get("source_aliases", [])) + len(data.get("target_aliases", []))

    print(f"  Total aliases generated: {total_aliases}")
    print(f"  Average aliases per bridge: {total_aliases / processed:.1f}")


if __name__ == "__main__":
    main()
