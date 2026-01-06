#!/usr/bin/env python3
"""
Enrich Neo4j Knowledge Graph with papers from polymax-synthesizer papers.db

This script:
1. Reads all papers with abstracts from papers.db
2. Extracts key concepts from titles/abstracts
3. Adds papers and concepts to Neo4j
4. Creates MENTIONS relationships
"""

import sqlite3
import re
from neo4j import GraphDatabase
from tqdm import tqdm

# Configuration
PAPERS_DB = "/home/user/mcp_servers/polymax-synthesizer/papers.db"
NEO4J_URI = "bolt://localhost:7687"
NEO4J_USER = "neo4j"
NEO4J_PASSWORD = "polymathic2026"

# Key concepts to extract (domain-specific)
CONCEPT_PATTERNS = [
    # Methods
    r'\b(transformer|attention|CNN|ResNet|ViT|UNET|GAN|VAE|diffusion model)\b',
    r'\b(foundation model|pretrained|fine-tun\w+|transfer learning)\b',
    r'\b(segmentation|detection|classification|regression|clustering)\b',
    r'\b(self-supervised|contrastive|masked|autoencoder)\b',

    # Spatial biology
    r'\b(spatial transcriptomics?|Visium|10x Genomics|ST data)\b',
    r'\b(single[- ]cell|scRNA-seq|RNA-seq)\b',
    r'\b(deconvolution|cell type|cell-type)\b',
    r'\b(histology|H&E|pathology|whole[- ]slide)\b',
    r'\b(tumor microenvironment|TME|microbiome)\b',

    # Diseases
    r'\b(cancer|carcinoma|adenoma|tumor|neoplasm)\b',
    r'\b(colorectal|CRC|colon|rectal)\b',
    r'\b(inflammatory|IBD|colitis|Crohn)\b',

    # General ML
    r'\b(deep learning|neural network|machine learning)\b',
    r'\b(graph neural|GNN|message passing)\b',
    r'\b(embedding|latent|representation)\b',
]


def extract_concepts(text: str) -> list:
    """Extract concepts from text using patterns."""
    if not text:
        return []

    concepts = set()
    text_lower = text.lower()

    for pattern in CONCEPT_PATTERNS:
        matches = re.findall(pattern, text, re.IGNORECASE)
        for match in matches:
            # Normalize
            concept = match.lower().strip()
            concept = re.sub(r'[- ]+', '_', concept)
            concepts.add(concept)

    return list(concepts)


def get_papers_from_db():
    """Load all papers with abstracts from SQLite."""
    conn = sqlite3.connect(PAPERS_DB)
    cursor = conn.cursor()

    cursor.execute("""
        SELECT pmid, title, abstract, authors, journal, year, doi
        FROM papers
        WHERE title IS NOT NULL AND title != ''
    """)

    papers = []
    for row in cursor.fetchall():
        papers.append({
            'pmid': row[0],
            'title': row[1],
            'abstract': row[2] or '',
            'authors': row[3],
            'journal': row[4],
            'year': row[5],
            'doi': row[6]
        })

    conn.close()
    return papers


def enrich_neo4j(papers: list):
    """Add papers and concepts to Neo4j."""
    driver = GraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USER, NEO4J_PASSWORD))

    stats = {
        'papers_added': 0,
        'papers_updated': 0,
        'concepts_added': 0,
        'relationships_added': 0
    }

    with driver.session() as session:
        for paper in tqdm(papers, desc="Enriching Neo4j"):
            # Extract concepts from title + abstract
            text = f"{paper['title']} {paper['abstract']}"
            concepts = extract_concepts(text)

            # Add or update paper
            result = session.run("""
                MERGE (p:Paper {title: $title})
                ON CREATE SET
                    p.pmid = $pmid,
                    p.abstract = $abstract,
                    p.authors = $authors,
                    p.journal = $journal,
                    p.year = $year,
                    p.doi = $doi,
                    p.created = datetime()
                ON MATCH SET
                    p.pmid = COALESCE(p.pmid, $pmid),
                    p.abstract = COALESCE(p.abstract, $abstract),
                    p.authors = COALESCE(p.authors, $authors),
                    p.journal = COALESCE(p.journal, $journal),
                    p.year = COALESCE(p.year, $year),
                    p.doi = COALESCE(p.doi, $doi),
                    p.updated = datetime()
                RETURN p.created IS NOT NULL as created
            """, {
                'title': paper['title'],
                'pmid': paper['pmid'],
                'abstract': paper['abstract'][:2000] if paper['abstract'] else None,
                'authors': paper['authors'],
                'journal': paper['journal'],
                'year': paper['year'],
                'doi': paper['doi']
            })

            record = result.single()
            if record and record['created']:
                stats['papers_added'] += 1
            else:
                stats['papers_updated'] += 1

            # Add concepts and relationships
            for concept in concepts:
                # Create concept if not exists
                session.run("""
                    MERGE (c:CONCEPT {name: $name})
                    ON CREATE SET c.created = datetime()
                """, {'name': concept})

                # Create relationship
                result = session.run("""
                    MATCH (p:Paper {title: $title})
                    MATCH (c:CONCEPT {name: $concept})
                    MERGE (p)-[r:MENTIONS]->(c)
                    RETURN r
                """, {'title': paper['title'], 'concept': concept})

                if result.single():
                    stats['relationships_added'] += 1

        # Count final concepts
        result = session.run("MATCH (c:CONCEPT) RETURN count(c) as count")
        stats['total_concepts'] = result.single()['count']

        # Count final papers
        result = session.run("MATCH (p:Paper) RETURN count(p) as count")
        stats['total_papers'] = result.single()['count']

    driver.close()
    return stats


def main():
    print("=" * 60)
    print("NEO4J KNOWLEDGE GRAPH ENRICHMENT")
    print("=" * 60)

    # Load papers
    print("\n1. Loading papers from papers.db...")
    papers = get_papers_from_db()
    print(f"   Found {len(papers)} papers")

    # Sample concepts
    print("\n2. Sample concept extraction:")
    for paper in papers[:3]:
        concepts = extract_concepts(f"{paper['title']} {paper['abstract']}")
        print(f"   [{paper['year']}] {paper['title'][:50]}...")
        print(f"   Concepts: {concepts[:5]}")

    # Enrich
    print("\n3. Enriching Neo4j...")
    stats = enrich_neo4j(papers)

    print("\n" + "=" * 60)
    print("ENRICHMENT COMPLETE")
    print("=" * 60)
    print(f"Papers added:        {stats['papers_added']}")
    print(f"Papers updated:      {stats['papers_updated']}")
    print(f"Relationships added: {stats['relationships_added']}")
    print(f"Total papers:        {stats['total_papers']}")
    print(f"Total concepts:      {stats['total_concepts']}")


if __name__ == "__main__":
    main()
