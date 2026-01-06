#!/usr/bin/env python3
"""
Ontologist Agent: Extract semantic triples from papers and build Knowledge Graph
Uses Claude Haiku for cost efficiency (~$0.25 per million input tokens)
"""

import anthropic
import chromadb
from neo4j import GraphDatabase
import json
from tqdm import tqdm
import time
import os
from typing import List, Dict, Any
import argparse

# Configuration
NEO4J_URI = "bolt://localhost:7687"
NEO4J_USER = "neo4j"
NEO4J_PASSWORD = "polymathic2026"
CHROMADB_PATH = "/mnt/z/chromadb_critical_papers"
COLLECTION_NAME = "critical_papers_2026"
OUTPUT_PATH = "/mnt/z/kg_extractions/"

# Create output directory
os.makedirs(OUTPUT_PATH, exist_ok=True)

# Initialize clients
# Try to get API key from environment, or let the SDK find it in default locations
api_key = os.environ.get('ANTHROPIC_API_KEY')
if api_key:
    client_anthropic = anthropic.Anthropic(api_key=api_key)
else:
    # Let the SDK try to find it in default locations (~/.anthropic/api_key, etc.)
    try:
        client_anthropic = anthropic.Anthropic()
    except Exception as e:
        print(f"ERROR: Could not initialize Anthropic client: {e}")
        print("Please set ANTHROPIC_API_KEY environment variable or create ~/.anthropic/api_key")
        raise
client_db = chromadb.PersistentClient(path=CHROMADB_PATH)
driver_neo4j = GraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USER, NEO4J_PASSWORD))

ONTOLOGIST_PROMPT = """You are an expert Ontologist building a knowledge graph for polymathic research.

Extract semantic triples from this paper in the format: (Source, Relationship, Target)

Focus on:
1. **Methodology**: Methods, algorithms, techniques, tools
2. **Theory**: Concepts, principles, frameworks, models
3. **Physical Constraints**: Biological entities, genes, proteins, tissues, diseases

Important:
- Disambiguate entities (e.g., "Cell2location" not "cell2location")
- Use standard relationship types: USES, CITES, PRODUCES, CAUSES, COMPARED_WITH, REQUIRES, IMPROVES
- Include confidence score (0.0-1.0) based on explicitness in text
- Only extract explicitly mentioned relationships

Paper Title: {title}

Text:
{text}

Return ONLY valid JSON:
{{
  "entities": [
    {{"name": "Entity Name", "type": "method|concept|gene|institution|author", "aliases": ["alt1", "alt2"]}}
  ],
  "relationships": [
    {{"source": "Entity1", "target": "Entity2", "type": "USES", "confidence": 0.95, "evidence": "quote from text"}}
  ]
}}
"""

def extract_triples_haiku(text: str, title: str, max_retries: int = 3) -> Dict[str, Any]:
    """
    Use Claude Haiku to extract semantic triples
    Cost: ~$0.25 per 1M input tokens, ~$1.25 per 1M output tokens
    """
    for attempt in range(max_retries):
        try:
            response = client_anthropic.messages.create(
                model="claude-haiku-4.5",  # Cost-efficient model
                max_tokens=2000,
                temperature=0.0,  # Deterministic for extraction
                messages=[{
                    "role": "user",
                    "content": ONTOLOGIST_PROMPT.format(
                        title=title,
                        text=text[:8000]  # First 8K chars to stay within context
                    )
                }]
            )

            # Parse JSON response
            content = response.content[0].text

            # Handle markdown code blocks if present
            if "```json" in content:
                content = content.split("```json")[1].split("```")[0].strip()
            elif "```" in content:
                content = content.split("```")[1].split("```")[0].strip()

            result = json.loads(content)

            # Add metadata
            result['paper_title'] = title
            result['extraction_model'] = 'claude-haiku-4.5'
            result['extraction_timestamp'] = time.time()

            return result

        except json.JSONDecodeError as e:
            print(f"JSON parse error (attempt {attempt+1}/{max_retries}): {e}")
            if attempt == max_retries - 1:
                return {
                    "entities": [],
                    "relationships": [],
                    "error": str(e),
                    "paper_title": title
                }
            time.sleep(1)
        except Exception as e:
            print(f"Extraction error (attempt {attempt+1}/{max_retries}): {e}")
            if attempt == max_retries - 1:
                return {
                    "entities": [],
                    "relationships": [],
                    "error": str(e),
                    "paper_title": title
                }
            time.sleep(2)

    return {"entities": [], "relationships": [], "paper_title": title}

def write_to_neo4j(tx, extraction: Dict[str, Any]):
    """Write extracted entities and relationships to Neo4j"""

    paper_title = extraction.get('paper_title', 'Unknown')
    paper_id = extraction.get('paper_id', '')

    # Create paper node
    tx.run("""
        MERGE (p:Paper {title: $title})
        SET p.id = $id,
            p.extraction_timestamp = $timestamp
    """, title=paper_title, id=paper_id, timestamp=extraction.get('extraction_timestamp', 0))

    # Create entity nodes
    for entity in extraction.get('entities', []):
        entity_name = entity['name']
        entity_type = entity['type'].upper()

        # Create typed entity node
        tx.run(f"""
            MERGE (e:{entity_type} {{name: $name}})
            SET e.aliases = $aliases
            WITH e
            MATCH (p:Paper {{title: $paper_title}})
            MERGE (p)-[:MENTIONS]->(e)
        """, name=entity_name, aliases=entity.get('aliases', []), paper_title=paper_title)

    # Create relationships
    for rel in extraction.get('relationships', []):
        source = rel['source']
        target = rel['target']
        rel_type = rel['type'].upper().replace('-', '_')
        confidence = rel.get('confidence', 0.5)
        evidence = rel.get('evidence', '')

        tx.run(f"""
            MATCH (s {{name: $source}})
            MATCH (t {{name: $target}})
            MERGE (s)-[r:{rel_type}]->(t)
            SET r.confidence = $confidence,
                r.evidence = $evidence,
                r.paper = $paper_title
        """, source=source, target=target, confidence=confidence, evidence=evidence, paper_title=paper_title)

def process_papers(limit: int = None, test_mode: bool = False):
    """
    Process papers from ChromaDB and build knowledge graph

    Args:
        limit: Number of papers to process (None = all)
        test_mode: If True, only process first 10 papers for validation
    """
    # Load papers from ChromaDB
    coll = client_db.get_collection(COLLECTION_NAME)
    papers = coll.get(include=['documents', 'metadatas'])

    total_papers = len(papers['ids'])
    if test_mode:
        limit = 10
    if limit:
        total_papers = min(total_papers, limit)

    print(f"\n{'='*60}")
    print(f"ONTOLOGIST AGENT - Knowledge Graph Builder")
    print(f"{'='*60}")
    print(f"Processing {total_papers} papers...")
    print(f"Model: Claude Haiku 4.5 (cost-efficient)")
    print(f"ChromaDB: {CHROMADB_PATH}")
    print(f"Neo4j: {NEO4J_URI}")
    print(f"{'='*60}\n")

    results = []
    total_entities = 0
    total_relationships = 0

    with driver_neo4j.session() as session:
        for i, (doc, meta, pid) in enumerate(tqdm(
            zip(papers['documents'][:total_papers],
                papers['metadatas'][:total_papers],
                papers['ids'][:total_papers]),
            total=total_papers,
            desc="Extracting triples"
        )):
            title = meta.get('title', '')[:200]

            # Extract triples using Claude Haiku
            extraction = extract_triples_haiku(doc, title)
            extraction['paper_id'] = pid

            # Write to Neo4j
            try:
                session.execute_write(write_to_neo4j, extraction)
            except Exception as e:
                print(f"\nNeo4j write error for '{title}': {e}")
                extraction['neo4j_error'] = str(e)

            # Save extraction to file (incremental backup)
            output_file = os.path.join(OUTPUT_PATH, f"extraction_{i:04d}.json")
            with open(output_file, 'w') as f:
                json.dump(extraction, f, indent=2)

            # Track stats
            num_entities = len(extraction.get('entities', []))
            num_rels = len(extraction.get('relationships', []))
            total_entities += num_entities
            total_relationships += num_rels

            results.append({
                'paper_id': pid,
                'title': title,
                'entities': num_entities,
                'relationships': num_rels
            })

            # Rate limiting (be nice to Claude API)
            time.sleep(0.5)

    # Summary
    print(f"\n{'='*60}")
    print(f"EXTRACTION COMPLETE")
    print(f"{'='*60}")
    print(f"Papers processed: {len(results)}")
    print(f"Total entities: {total_entities}")
    print(f"Total relationships: {total_relationships}")
    print(f"Avg entities/paper: {total_entities/len(results):.1f}")
    print(f"Avg relationships/paper: {total_relationships/len(results):.1f}")
    print(f"{'='*60}\n")

    # Save summary
    summary = {
        'total_papers': len(results),
        'total_entities': total_entities,
        'total_relationships': total_relationships,
        'papers': results
    }

    with open(os.path.join(OUTPUT_PATH, 'summary.json'), 'w') as f:
        json.dump(summary, f, indent=2)

    print(f"Extractions saved to: {OUTPUT_PATH}")
    print(f"Neo4j Browser: http://localhost:7474")
    print(f"\nExample Cypher queries:")
    print(f"  MATCH (n) RETURN n LIMIT 25")
    print(f"  MATCH (p:Paper)-[r:MENTIONS]->(e) RETURN p.title, type(r), e.name LIMIT 50")
    print(f"  MATCH (m:METHOD)-[r]->(c:CONCEPT) RETURN m.name, type(r), c.name")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Build Knowledge Graph from papers")
    parser.add_argument('--test', action='store_true', help='Test mode: process only 10 papers')
    parser.add_argument('--limit', type=int, help='Limit number of papers to process')

    args = parser.parse_args()

    try:
        process_papers(limit=args.limit, test_mode=args.test)
    except KeyboardInterrupt:
        print("\n\nInterrupted by user. Partial results saved.")
    finally:
        driver_neo4j.close()
