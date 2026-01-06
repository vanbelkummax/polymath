#!/usr/bin/env python3
"""
Librarian Script: Fix Concept Linkages in Neo4j Knowledge Graph

This script improves knowledge graph interconnections by finding papers
in ChromaDB that mention sophisticated concepts but aren't linked in Neo4j.

The original ingestion used simple keyword matching (line 68-87 in ingest_batch.py)
which missed papers with sophisticated phrasing like "variational free energy"
instead of just "free energy".

This script performs semantic search on ChromaDB to find relevant papers,
then creates MENTIONS relationships in Neo4j.

Author: Librarian Agent
Date: 2026-01-04
"""

import sys
import chromadb
from sentence_transformers import SentenceTransformer
from neo4j import GraphDatabase
from typing import List, Dict, Tuple
import json
from datetime import datetime

# Configuration
CHROMADB_PATH = "/home/user/work/polymax/chromadb/polymath_v2"
NEO4J_URI = "bolt://localhost:7687"
NEO4J_USER = "neo4j"
NEO4J_PASSWORD = "polymathic2026"

# Concept search patterns
# Each entry: (concept_name, search_terms, relevance_threshold)
CONCEPT_PATTERNS = [
    # Free Energy Principle (Karl Friston, active inference, predictive coding)
    ("free_energy_principle", [
        "free energy principle",
        "variational free energy",
        "active inference",
        "predictive processing",
        "Friston free energy"
    ], 0.25),

    # Counterfactual Reasoning (causal inference, Pearl)
    ("counterfactual_reasoning", [
        "counterfactual reasoning",
        "counterfactual inference",
        "potential outcomes",
        "Pearl causality counterfactual",
        "do-calculus counterfactual"
    ], 0.30),

    # Sparse Coding (compressed sensing, sparse representation)
    ("sparse_coding", [
        "sparse coding",
        "sparse representation learning",
        "dictionary learning sparse",
        "sparse basis functions",
        "L1 regularization sparsity"
    ], 0.25),

    # Maximum Entropy (MaxEnt, Jaynes)
    ("maximum_entropy", [
        "maximum entropy principle",
        "maxent modeling",
        "Jaynes maximum entropy",
        "entropy maximization",
        "maximum entropy distribution"
    ], 0.30),

    # Information Bottleneck (Tishby, compression)
    ("information_bottleneck", [
        "information bottleneck",
        "Tishby bottleneck",
        "information compression bottleneck",
        "rate-distortion bottleneck",
        "minimal sufficient statistic"
    ], 0.35),

    # Reaction-Diffusion (Turing patterns, morphogenesis)
    ("reaction_diffusion", [
        "reaction-diffusion system",
        "Turing pattern",
        "Turing instability",
        "activator-inhibitor model",
        "pattern formation reaction diffusion"
    ], 0.25),

    # Additional high-value concepts with 0 papers
    ("autopoiesis", [
        "autopoiesis",
        "autopoietic system",
        "Maturana Varela autopoiesis",
        "self-production biological",
        "organizational closure"
    ], 0.30),

    ("attractor_dynamics", [
        "attractor dynamics",
        "basin of attraction",
        "fixed point attractor",
        "dynamical attractor",
        "limit cycle attractor"
    ], 0.25),

    ("do_calculus", [
        "do-calculus",
        "do calculus Pearl",
        "causal intervention calculus",
        "do-operator",
        "backdoor criterion frontdoor"
    ], 0.35),

    ("instrumental_variables", [
        "instrumental variables",
        "instrumental variable estimation",
        "IV regression",
        "two-stage least squares",
        "exogenous instrument"
    ], 0.30),
]


class ConceptLinker:
    """Links papers to concepts using semantic search."""

    def __init__(self, dry_run=True, verbose=True):
        self.dry_run = dry_run
        self.verbose = verbose
        self.model = None
        self.chroma_client = None
        self.chroma_collection = None
        self.neo4j_driver = None

    def load_resources(self):
        """Load ChromaDB, embedding model, and Neo4j."""
        print(f"[{datetime.now():%H:%M:%S}] Loading embedding model...")
        self.model = SentenceTransformer('all-mpnet-base-v2')

        print(f"[{datetime.now():%H:%M:%S}] Connecting to ChromaDB...")
        self.chroma_client = chromadb.PersistentClient(path=CHROMADB_PATH)
        self.chroma_collection = self.chroma_client.get_collection("polymath_corpus")
        print(f"  Found {self.chroma_collection.count()} chunks")

        print(f"[{datetime.now():%H:%M:%S}] Connecting to Neo4j...")
        self.neo4j_driver = GraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USER, NEO4J_PASSWORD))

        # Verify Neo4j connection
        with self.neo4j_driver.session() as session:
            result = session.run("MATCH (p:Paper) RETURN count(p) as count")
            paper_count = result.single()['count']
            print(f"  Found {paper_count} papers in Neo4j")

    def search_papers_for_concept(self, search_terms: List[str], threshold: float,
                                   n_results: int = 20) -> List[Dict]:
        """
        Search ChromaDB for papers matching the concept.

        Returns list of {chunk_id, document, metadata, relevance} for chunks
        above the threshold.
        """
        all_matches = []

        for term in search_terms:
            query_embedding = self.model.encode([term]).tolist()

            results = self.chroma_collection.query(
                query_embeddings=query_embedding,
                n_results=n_results,
                include=["documents", "metadatas", "distances"]
            )

            # Filter by threshold
            for i, (doc, meta, dist) in enumerate(zip(
                results["documents"][0],
                results["metadatas"][0],
                results["distances"][0]
            )):
                relevance = 1 - dist
                if relevance >= threshold:
                    # Extract title from metadata or document text
                    if meta is None:
                        meta = {}

                    title = meta.get("title", meta.get("source", None))

                    # If no metadata title, try to extract from document
                    if not title or title == "Unknown":
                        # Look for "Title: " at start of document
                        if doc.startswith("Title: "):
                            title_end = doc.find("\n")
                            if title_end > 0:
                                title = doc[7:title_end].strip()
                        else:
                            # Use first non-empty line
                            first_line = doc.split('\n')[0].strip()
                            if 10 < len(first_line) < 200:
                                title = first_line

                    if not title:
                        title = "Unknown"

                    all_matches.append({
                        "search_term": term,
                        "document": doc,
                        "metadata": meta,
                        "relevance": round(relevance, 4),
                        "title": title
                    })

        # Deduplicate by title and return highest relevance matches
        seen_titles = {}
        for match in sorted(all_matches, key=lambda x: x["relevance"], reverse=True):
            title = match["title"]
            if title not in seen_titles or match["relevance"] > seen_titles[title]["relevance"]:
                seen_titles[title] = match

        return list(seen_titles.values())

    def get_existing_links(self, concept_name: str) -> set:
        """Get paper titles already linked to this concept."""
        with self.neo4j_driver.session() as session:
            result = session.run("""
                MATCH (p:Paper)-[:MENTIONS]->(c:CONCEPT {name: $concept})
                RETURN p.title as title
            """, {"concept": concept_name})
            return {record["title"] for record in result}

    def link_paper_to_concept(self, paper_title: str, concept_name: str):
        """Create a MENTIONS relationship in Neo4j."""
        with self.neo4j_driver.session() as session:
            session.run("""
                MERGE (p:Paper {title: $title})
                MERGE (c:CONCEPT {name: $concept})
                MERGE (p)-[:MENTIONS]->(c)
            """, {"title": paper_title, "concept": concept_name})

    def process_concept(self, concept_name: str, search_terms: List[str],
                       threshold: float) -> Dict:
        """Process one concept: search, filter, and link."""
        print(f"\n{'='*70}")
        print(f"Processing: {concept_name}")
        print(f"{'='*70}")

        # Get existing links
        existing = self.get_existing_links(concept_name)
        print(f"  Currently linked: {len(existing)} papers")

        # Search for matches
        print(f"  Searching with {len(search_terms)} terms (threshold={threshold})...")
        matches = self.search_papers_for_concept(search_terms, threshold)
        print(f"  Found {len(matches)} candidate papers")

        # Filter out existing links
        new_matches = [m for m in matches if m["title"] not in existing]
        print(f"  New papers to link: {len(new_matches)}")

        # Create links
        linked_count = 0
        if new_matches:
            print(f"\n  Top matches:")
            for i, match in enumerate(new_matches[:10], 1):
                print(f"    [{i}] [{match['relevance']:.3f}] {match['title'][:70]}...")
                print(f"        via: '{match['search_term']}'")
                print(f"        excerpt: {match['document'][:100]}...")

                if not self.dry_run:
                    self.link_paper_to_concept(match["title"], concept_name)
                    linked_count += 1

            if self.dry_run:
                print(f"\n  [DRY RUN] Would link {len(new_matches)} papers")
            else:
                print(f"\n  [LIVE] Linked {linked_count} papers")

        return {
            "concept": concept_name,
            "existing_links": len(existing),
            "new_matches": len(new_matches),
            "linked": linked_count if not self.dry_run else 0,
            "top_matches": [
                {
                    "title": m["title"],
                    "relevance": m["relevance"],
                    "search_term": m["search_term"]
                } for m in new_matches[:10]
            ]
        }

    def run(self, concepts_to_process=None) -> Dict:
        """
        Run the linking process for specified concepts.

        Args:
            concepts_to_process: List of concept names, or None for all

        Returns:
            Summary statistics
        """
        self.load_resources()

        # Filter concepts
        patterns = CONCEPT_PATTERNS
        if concepts_to_process:
            patterns = [p for p in CONCEPT_PATTERNS if p[0] in concepts_to_process]

        print(f"\n{'='*70}")
        print(f"CONCEPT LINKING - {'DRY RUN' if self.dry_run else 'LIVE MODE'}")
        print(f"Processing {len(patterns)} concepts")
        print(f"{'='*70}")

        results = []
        for concept_name, search_terms, threshold in patterns:
            result = self.process_concept(concept_name, search_terms, threshold)
            results.append(result)

        # Summary
        total_existing = sum(r["existing_links"] for r in results)
        total_new = sum(r["new_matches"] for r in results)
        total_linked = sum(r["linked"] for r in results)

        print(f"\n{'='*70}")
        print(f"SUMMARY")
        print(f"{'='*70}")
        print(f"  Concepts processed: {len(results)}")
        print(f"  Existing links: {total_existing}")
        print(f"  New matches found: {total_new}")
        if self.dry_run:
            print(f"  [DRY RUN] Would create {total_new} new links")
        else:
            print(f"  [LIVE] Created {total_linked} new links")

        return {
            "timestamp": datetime.now().isoformat(),
            "dry_run": self.dry_run,
            "concepts_processed": len(results),
            "total_existing_links": total_existing,
            "total_new_matches": total_new,
            "total_linked": total_linked,
            "details": results
        }

    def close(self):
        """Clean up resources."""
        if self.neo4j_driver:
            self.neo4j_driver.close()


def main():
    """CLI entry point."""
    import argparse

    parser = argparse.ArgumentParser(
        description="Fix concept linkages in Polymath knowledge graph",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Dry run (preview only):
  python3 fix_concept_links.py --dry-run

  # Process specific concepts:
  python3 fix_concept_links.py --dry-run --concepts free_energy_principle sparse_coding

  # Live run (actually create links):
  python3 fix_concept_links.py

  # Save report to file:
  python3 fix_concept_links.py --dry-run --output report.json
        """
    )

    parser.add_argument('--dry-run', action='store_true', default=True,
                       help='Preview changes without modifying Neo4j (default: True)')
    parser.add_argument('--live', action='store_true',
                       help='Actually create links in Neo4j')
    parser.add_argument('--concepts', nargs='+',
                       help='Specific concepts to process (default: all)')
    parser.add_argument('--output', '-o',
                       help='Save JSON report to file')
    parser.add_argument('--quiet', '-q', action='store_true',
                       help='Minimal output')

    args = parser.parse_args()

    # Handle dry-run vs live
    dry_run = not args.live

    # Create linker
    linker = ConceptLinker(dry_run=dry_run, verbose=not args.quiet)

    try:
        # Run linking
        report = linker.run(concepts_to_process=args.concepts)

        # Save report if requested
        if args.output:
            with open(args.output, 'w') as f:
                json.dump(report, f, indent=2)
            print(f"\nReport saved to: {args.output}")

        return 0

    except Exception as e:
        print(f"\nERROR: {e}", file=sys.stderr)
        import traceback
        traceback.print_exc()
        return 1

    finally:
        linker.close()


if __name__ == "__main__":
    sys.exit(main())
