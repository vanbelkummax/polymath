#!/usr/bin/env python3
"""
Comprehensive retrieval test suite for Polymath knowledge base.

Tests:
1. Semantic search (ChromaDB)
2. Hybrid search (Postgres + ChromaDB)
3. Graph queries (Neo4j)
4. Cross-domain discovery
5. Code search
6. Concept-based retrieval
"""

import os
import sys
import json
import time
from pathlib import Path
from datetime import datetime

sys.path.insert(0, str(Path(__file__).parent.parent))

from lib.config import POSTGRES_DSN, NEO4J_URI, NEO4J_USER, NEO4J_PASSWORD, CHROMADB_PATH

import psycopg2
import chromadb
from neo4j import GraphDatabase


class RetrievalTester:
    def __init__(self):
        self.conn = psycopg2.connect(POSTGRES_DSN)
        self.chroma = chromadb.PersistentClient(path=str(CHROMADB_PATH))
        self.neo4j = GraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USER, NEO4J_PASSWORD))
        self.results = []

    def log_test(self, name: str, query: str, results: list, elapsed: float, success: bool = True):
        """Log test result."""
        self.results.append({
            'test': name,
            'query': query,
            'num_results': len(results),
            'elapsed_ms': round(elapsed * 1000, 2),
            'success': success,
            'sample': results[:3] if results else []
        })
        status = "✓" if success and results else "✗"
        print(f"  {status} {name}: {len(results)} results in {elapsed*1000:.0f}ms")

    # =========================================================================
    # TEST 1: Semantic Search (ChromaDB)
    # =========================================================================
    def test_semantic_search(self):
        print("\n" + "="*60)
        print("TEST 1: SEMANTIC SEARCH (ChromaDB)")
        print("="*60)

        collection = self.chroma.get_collection("polymath_bge_m3")

        queries = [
            ("Spatial transcriptomics methods", "spatial transcriptomics single cell tissue"),
            ("Deep learning pathology", "deep learning computational pathology histopathology"),
            ("Gene expression prediction", "gene expression prediction from histology images"),
            ("Cancer biomarkers", "cancer biomarkers tumor microenvironment"),
            ("CRISPR gene editing", "CRISPR Cas9 gene editing therapeutic"),
        ]

        for name, query in queries:
            start = time.time()
            try:
                results = collection.query(query_texts=[query], n_results=10)
                docs = results.get('documents', [[]])[0]
                elapsed = time.time() - start
                self.log_test(f"Semantic: {name}", query, docs, elapsed)
            except Exception as e:
                self.log_test(f"Semantic: {name}", query, [], 0, success=False)

    # =========================================================================
    # TEST 2: Full-Text Search (Postgres)
    # =========================================================================
    def test_fulltext_search(self):
        print("\n" + "="*60)
        print("TEST 2: FULL-TEXT SEARCH (Postgres)")
        print("="*60)

        cur = self.conn.cursor()

        queries = [
            ("Exact term: Visium", "Visium"),
            ("Phrase: spatial transcriptomics", "spatial transcriptomics"),
            ("Gene symbol: TP53", "TP53"),
            ("Method: attention mechanism", "attention mechanism"),
            ("Disease: colorectal cancer", "colorectal cancer"),
        ]

        for name, query in queries:
            start = time.time()
            try:
                cur.execute("""
                    SELECT p.passage_id, d.title,
                           ts_rank(p.search_vector, plainto_tsquery('english', %s)) as rank
                    FROM passages p
                    JOIN documents d ON p.doc_id = d.doc_id
                    WHERE p.search_vector @@ plainto_tsquery('english', %s)
                    ORDER BY rank DESC
                    LIMIT 10
                """, (query, query))
                results = cur.fetchall()
                elapsed = time.time() - start
                self.log_test(f"FTS: {name}", query, results, elapsed)
            except Exception as e:
                self.log_test(f"FTS: {name}", query, [], 0, success=False)

        cur.close()

    # =========================================================================
    # TEST 3: Hybrid Search
    # =========================================================================
    def test_hybrid_search(self):
        print("\n" + "="*60)
        print("TEST 3: HYBRID SEARCH (Postgres + ChromaDB)")
        print("="*60)

        try:
            from lib.hybrid_search_v2 import HybridSearcherV2
            searcher = HybridSearcherV2()

            queries = [
                "spatial transcriptomics resolution improvement",
                "transformer architecture medical imaging",
                "single cell RNA sequencing analysis",
                "graph neural networks protein structure",
                "contrastive learning histopathology",
            ]

            for query in queries:
                start = time.time()
                try:
                    results = searcher.hybrid_search(query, n=10, rerank=False)
                    elapsed = time.time() - start
                    self.log_test(f"Hybrid: {query[:35]}...", query, results, elapsed)
                except Exception as e:
                    self.log_test(f"Hybrid: {query[:35]}...", query, [], 0, success=False)

        except ImportError:
            print("  ⚠ HybridSearcherV2 not available, skipping")

    # =========================================================================
    # TEST 4: Graph Queries (Neo4j)
    # =========================================================================
    def test_graph_queries(self):
        print("\n" + "="*60)
        print("TEST 4: GRAPH QUERIES (Neo4j)")
        print("="*60)

        queries = [
            ("Count concepts by type", """
                MATCH (c:CONCEPT)
                RETURN c.type as type, count(*) as count
                ORDER BY count DESC LIMIT 10
            """),
            ("Methods in spatial domain", """
                MATCH (m:METHOD)-[:APPEARS_IN]->(p:PASSAGE)
                WHERE m.name CONTAINS 'spatial'
                RETURN m.name, count(p) as mentions
                ORDER BY mentions DESC LIMIT 10
            """),
            ("Connected concepts", """
                MATCH (c1:CONCEPT)-[r:SIMILAR_TO]-(c2:CONCEPT)
                RETURN c1.name, type(r), c2.name
                LIMIT 10
            """),
            ("Most mentioned entities", """
                MATCH (e:ENTITY)-[:APPEARS_IN]->(p:PASSAGE)
                RETURN e.name, count(p) as mentions
                ORDER BY mentions DESC LIMIT 10
            """),
            ("Cross-domain methods", """
                MATCH (m:METHOD)-[:APPEARS_IN]->(p1:PASSAGE),
                      (m)-[:APPEARS_IN]->(p2:PASSAGE)
                WHERE p1 <> p2
                WITH m, count(DISTINCT p1) as paper_count
                WHERE paper_count > 5
                RETURN m.name, paper_count
                ORDER BY paper_count DESC LIMIT 10
            """),
        ]

        with self.neo4j.session() as session:
            for name, query in queries:
                start = time.time()
                try:
                    result = session.run(query)
                    records = [dict(r) for r in result]
                    elapsed = time.time() - start
                    self.log_test(f"Graph: {name}", query[:50], records, elapsed)
                except Exception as e:
                    self.log_test(f"Graph: {name}", query[:50], [], 0, success=False)

    # =========================================================================
    # TEST 5: Code Search
    # =========================================================================
    def test_code_search(self):
        print("\n" + "="*60)
        print("TEST 5: CODE SEARCH")
        print("="*60)

        cur = self.conn.cursor()

        queries = [
            ("Function: attention", "attention"),
            ("Class: DataLoader", "DataLoader"),
            ("Method: forward pass", "def forward"),
            ("Import: torch", "import torch"),
            ("Pattern: transformer", "transformer"),
        ]

        for name, query in queries:
            start = time.time()
            try:
                cur.execute("""
                    SELECT c.chunk_id, c.repo_name, c.file_path, c.chunk_type
                    FROM code_chunks c
                    WHERE c.chunk_text ILIKE %s
                    LIMIT 10
                """, (f'%{query}%',))
                results = cur.fetchall()
                elapsed = time.time() - start
                self.log_test(f"Code: {name}", query, results, elapsed)
            except Exception as e:
                self.log_test(f"Code: {name}", query, [], 0, success=False)

        cur.close()

    # =========================================================================
    # TEST 6: Concept-Based Retrieval
    # =========================================================================
    def test_concept_retrieval(self):
        print("\n" + "="*60)
        print("TEST 6: CONCEPT-BASED RETRIEVAL")
        print("="*60)

        cur = self.conn.cursor()

        queries = [
            ("Concept: deep learning methods", "methods", "deep%learning%"),
            ("Concept: cancer entities", "entities", "%cancer%"),
            ("Concept: imaging domains", "domains", "%imaging%"),
            ("Concept: prediction problems", "problems", "%prediction%"),
            ("Recent Gemini extractions", None, None),
        ]

        for name, concept_type, pattern in queries:
            start = time.time()
            try:
                if concept_type and pattern:
                    cur.execute("""
                        SELECT pc.concept_name, pc.concept_type, count(*) as mentions
                        FROM passage_concepts pc
                        WHERE pc.concept_type = %s
                        AND pc.concept_name ILIKE %s
                        GROUP BY pc.concept_name, pc.concept_type
                        ORDER BY mentions DESC
                        LIMIT 10
                    """, (concept_type, pattern))
                else:
                    # Get recent Gemini extractions
                    cur.execute("""
                        SELECT pc.concept_name, pc.concept_type, pc.extractor_version
                        FROM passage_concepts pc
                        WHERE pc.extractor_version = 'gemini-2.0-flash'
                        ORDER BY pc.created_at DESC
                        LIMIT 20
                    """)
                results = cur.fetchall()
                elapsed = time.time() - start
                self.log_test(f"Concept: {name}", pattern or "gemini", results, elapsed)
            except Exception as e:
                self.log_test(f"Concept: {name}", pattern or "gemini", [], 0, success=False)

        cur.close()

    # =========================================================================
    # TEST 7: Cross-Domain Discovery
    # =========================================================================
    def test_cross_domain(self):
        print("\n" + "="*60)
        print("TEST 7: CROSS-DOMAIN DISCOVERY")
        print("="*60)

        cur = self.conn.cursor()

        # Find methods that appear in multiple domains
        start = time.time()
        try:
            cur.execute("""
                WITH method_domains AS (
                    SELECT
                        pc1.concept_name as method,
                        pc2.concept_name as domain,
                        count(*) as co_occurrences
                    FROM passage_concepts pc1
                    JOIN passage_concepts pc2 ON pc1.passage_id = pc2.passage_id
                    WHERE pc1.concept_type = 'methods'
                    AND pc2.concept_type = 'domains'
                    GROUP BY pc1.concept_name, pc2.concept_name
                    HAVING count(*) >= 3
                )
                SELECT method, array_agg(DISTINCT domain) as domains, sum(co_occurrences) as total
                FROM method_domains
                GROUP BY method
                HAVING count(DISTINCT domain) >= 2
                ORDER BY count(DISTINCT domain) DESC, total DESC
                LIMIT 15
            """)
            results = cur.fetchall()
            elapsed = time.time() - start
            self.log_test("Cross-domain methods", "methods across domains", results, elapsed)

            # Show sample
            if results:
                print("\n  Sample cross-domain methods:")
                for method, domains, total in results[:5]:
                    print(f"    • {method}: {domains[:3]}")

        except Exception as e:
            self.log_test("Cross-domain methods", "", [], 0, success=False)

        cur.close()

    def run_all_tests(self):
        """Run all retrieval tests."""
        print("\n" + "="*60)
        print("POLYMATH RETRIEVAL TEST SUITE")
        print(f"Started: {datetime.now().isoformat()}")
        print("="*60)

        self.test_semantic_search()
        self.test_fulltext_search()
        self.test_hybrid_search()
        self.test_graph_queries()
        self.test_code_search()
        self.test_concept_retrieval()
        self.test_cross_domain()

        # Summary
        print("\n" + "="*60)
        print("SUMMARY")
        print("="*60)

        successful = sum(1 for r in self.results if r['success'] and r['num_results'] > 0)
        total = len(self.results)

        print(f"Tests passed: {successful}/{total}")
        print(f"Total queries: {total}")
        avg_time = sum(r['elapsed_ms'] for r in self.results) / len(self.results)
        print(f"Average latency: {avg_time:.0f}ms")

        # Failed tests
        failed = [r for r in self.results if not r['success'] or r['num_results'] == 0]
        if failed:
            print(f"\nFailed/empty tests ({len(failed)}):")
            for r in failed:
                print(f"  ✗ {r['test']}")

        return self.results

    def close(self):
        self.conn.close()
        self.neo4j.close()


if __name__ == "__main__":
    tester = RetrievalTester()
    try:
        results = tester.run_all_tests()

        # Save results
        output_file = f"/home/user/work/polymax/reports/retrieval_test_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        Path(output_file).parent.mkdir(parents=True, exist_ok=True)
        with open(output_file, 'w') as f:
            json.dump(results, f, indent=2, default=str)
        print(f"\n✓ Results saved to {output_file}")

    finally:
        tester.close()
