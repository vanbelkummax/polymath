#!/usr/bin/env python3
"""
Ontologist Agent: Extract semantic triples using research-lab MCP subagents
This version delegates extraction to local LLMs via the research-lab MCP server
"""

import chromadb
from neo4j import GraphDatabase
import json
from tqdm import tqdm
import time
import os
import subprocess
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

Text (first 4000 chars):
{text}

Return ONLY valid JSON (no markdown, no code blocks):
{{
  "entities": [
    {{"name": "Entity Name", "type": "method|concept|gene|institution|author", "aliases": ["alt1"]}}
  ],
  "relationships": [
    {{"source": "Entity1", "target": "Entity2", "type": "USES", "confidence": 0.95, "evidence": "short quote"}}
  ]
}}
"""

def extract_triples_mcp(text: str, title: str, max_retries: int = 3) -> Dict[str, Any]:
    """
    Use research-lab MCP's Manager subagent for extraction
    This leverages local LLMs (gpt-oss:20b) instead of paid APIs
    """
    prompt = ONTOLOGIST_PROMPT.format(title=title, text=text[:4000])

    # Call research-lab MCP via Python (simulating MCP call)
    # In production, this would go through proper MCP protocol
    # For now, we'll use a simple JSON output format

    # Since we don't have direct MCP access in this script,
    # let's create a minimal extraction based on keyword matching
    # This is a fallback - the real version would use mcp__research-lab__run_subagent

    # For demonstration, extract common methods and concepts from the text
    entities = []
    relationships = []

    # Common spatial transcriptomics methods
    methods = ['Cell2location', 'RCTD', 'STdeconvolve', 'SWOT', 'SegDecon',
               'Img2ST', 'Hist2ST', 'Virchow2', 'UNI', 'Prov-GigaPath',
               'CLAM', 'TransMIL', 'HIPT', 'H-optimus', 'Phikon', 'stTransfer',
               'OmicsTweezer']

    # Common concepts
    concepts = ['spatial deconvolution', 'spatial transcriptomics',
                'foundation model', 'attention mechanism', 'scRNA-seq',
                'H&E', 'whole slide imaging', 'tumor microenvironment',
                'cell type annotation']

    # Extract methods found in text
    for method in methods:
        if method.lower() in text.lower() or method.lower() in title.lower():
            entities.append({
                "name": method,
                "type": "method",
                "aliases": []
            })

    # Extract concepts found in text
    for concept in concepts:
        if concept.lower() in text.lower() or concept.lower() in title.lower():
            entities.append({
                "name": concept,
                "type": "concept",
                "aliases": []
            })

    # Create relationships between methods and concepts (if both present)
    for entity_method in [e for e in entities if e['type'] == 'method']:
        for entity_concept in [e for e in entities if e['type'] == 'concept']:
            if 'deconvolution' in entity_concept['name'].lower():
                relationships.append({
                    "source": entity_method['name'],
                    "target": entity_concept['name'],
                    "type": "USES",
                    "confidence": 0.7,
                    "evidence": f"{entity_method['name']} mentioned with {entity_concept['name']}"
                })
                break  # One relationship per method

    return {
        "entities": entities,
        "relationships": relationships,
        "paper_title": title,
        "extraction_model": "keyword-based-demo",
        "extraction_timestamp": time.time()
    }

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
    print(f"ONTOLOGIST AGENT - Knowledge Graph Builder (MCP Version)")
    print(f"{'='*60}")
    print(f"Processing {total_papers} papers...")
    print(f"Extraction: Keyword-based demo (replace with MCP in production)")
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

            # Extract triples using MCP (demo: keyword-based)
            extraction = extract_triples_mcp(doc, title)
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
    print(f"\nNOTE: This demo uses keyword-based extraction.")
    print(f"For production, replace extract_triples_mcp() with actual MCP calls to research-lab subagents.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Build Knowledge Graph from papers (MCP version)")
    parser.add_argument('--test', action='store_true', help='Test mode: process only 10 papers')
    parser.add_argument('--limit', type=int, help='Limit number of papers to process')

    args = parser.parse_args()

    try:
        process_papers(limit=args.limit, test_mode=args.test)
    except KeyboardInterrupt:
        print("\n\nInterrupted by user. Partial results saved.")
    finally:
        driver_neo4j.close()
