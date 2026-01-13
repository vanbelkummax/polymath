#!/usr/bin/env python3
"""
Phase 3: Evidence Harvest from 3 Stores
For each bridge, query Postgres FTS, ChromaDB semantic, and Neo4j graph
to gather supporting evidence spans.
"""

import json
import os
import sys
import time
from pathlib import Path
from typing import List, Dict, Any

sys.path.insert(0, '/home/user/polymath-repo')

import psycopg2
import chromadb
from neo4j import GraphDatabase

OUTPUT_DIR = os.environ.get('OUTPUT_DIR', '/home/user/work/polymax/reports/bridge_mine_v3_20260112_2126')


class EvidenceHarvester:
    def __init__(self):
        # Initialize connections (Unix socket auth)
        self.pg_conn = psycopg2.connect(
            dbname='polymath',
            user='polymath',
            host='/var/run/postgresql'
        )
        self.chroma = chromadb.PersistentClient('/home/user/polymath-repo/chromadb')
        self.collection = self.chroma.get_collection('polymath_bge_m3')
        self.neo4j = GraphDatabase.driver('bolt://localhost:7687', auth=('neo4j', 'polymathic2026'))

        # Load embedder for ChromaDB queries
        from FlagEmbedding import BGEM3FlagModel
        print("Loading BGE-M3 model...")
        self.embedder = BGEM3FlagModel('BAAI/bge-m3', use_fp16=True)
        print("✓ BGE-M3 loaded")

    def close(self):
        self.pg_conn.close()
        self.neo4j.close()

    def embed_query(self, query: str) -> List[float]:
        """Embed a query string using BGE-M3."""
        result = self.embedder.encode([query], return_dense=True)
        return result['dense_vecs'][0].tolist()

    def search_postgres_fts(self, search_terms: List[str], limit: int = 20) -> List[Dict]:
        """Full-text search in Postgres passages."""
        results = []
        cur = self.pg_conn.cursor()

        for term in search_terms[:5]:  # Top 5 terms
            try:
                query = """
                SELECT p.passage_id, p.doc_id, p.passage_text, d.title,
                       ts_rank(to_tsvector('english', p.passage_text), plainto_tsquery('english', %s)) as rank
                FROM passages p
                JOIN documents d ON p.doc_id = d.doc_id
                WHERE to_tsvector('english', p.passage_text) @@ plainto_tsquery('english', %s)
                ORDER BY rank DESC
                LIMIT %s
                """
                cur.execute(query, (term, term, limit // len(search_terms[:5])))
                for row in cur.fetchall():
                    results.append({
                        'passage_id': str(row[0]),
                        'doc_id': str(row[1]),
                        'snippet': row[2][:500] if row[2] else '',
                        'title': row[3] or 'Unknown',
                        'retrieval_source': 'postgres_fts',
                        'query_used': term,
                        'score': float(row[4]) if row[4] else 0.0
                    })
            except Exception as e:
                print(f"    PG FTS error for '{term}': {e}")
                continue

        cur.close()
        return results

    def search_chromadb_semantic(self, queries: List[str], hyde_text: str = None, limit: int = 20) -> List[Dict]:
        """Semantic search in ChromaDB."""
        results = []

        # Use HyDE text if available, otherwise use first query
        search_text = hyde_text if hyde_text else queries[0] if queries else ""
        if not search_text:
            return results

        try:
            embedding = self.embed_query(search_text)
            chroma_results = self.collection.query(
                query_embeddings=[embedding],
                n_results=limit
            )

            for i, (doc_id, distance) in enumerate(zip(
                chroma_results['ids'][0],
                chroma_results['distances'][0]
            )):
                metadata = chroma_results['metadatas'][0][i] if chroma_results['metadatas'] else {}
                content = chroma_results['documents'][0][i] if chroma_results['documents'] else ''
                results.append({
                    'passage_id': doc_id,
                    'doc_id': metadata.get('doc_id', ''),
                    'snippet': content[:500] if content else '',
                    'title': metadata.get('title', 'Unknown'),
                    'retrieval_source': 'chromadb_semantic',
                    'query_used': search_text[:100] + '...',
                    'score': 1.0 - float(distance)  # Convert distance to similarity
                })
        except Exception as e:
            print(f"    ChromaDB error: {e}")

        return results

    def search_neo4j_graph(self, source_aliases: List[str], target_aliases: List[str], limit: int = 20) -> List[Dict]:
        """Graph-based search in Neo4j - find co-mentioned concepts."""
        results = []

        # Search for concepts matching aliases and find connecting passages
        source_pattern = '|'.join(source_aliases[:5])
        target_pattern = '|'.join(target_aliases[:5])

        query = """
        MATCH (c1:Concept)-[:MENTIONS]-(p:Passage)-[:MENTIONS]-(c2:Concept)
        WHERE c1.name =~ $source_pattern
          AND c2.name =~ $target_pattern
          AND c1 <> c2
        RETURN DISTINCT p.passage_id as passage_id, p.doc_id as doc_id,
               p.content as snippet, c1.name as source_concept, c2.name as target_concept
        LIMIT $limit
        """

        try:
            result = self.neo4j.execute_query(
                query,
                source_pattern=f"(?i).*({source_pattern}).*",
                target_pattern=f"(?i).*({target_pattern}).*",
                limit=limit
            )
            for record in result[0]:
                results.append({
                    'passage_id': str(record['passage_id']) if record['passage_id'] else '',
                    'doc_id': str(record['doc_id']) if record['doc_id'] else '',
                    'snippet': record['snippet'][:500] if record['snippet'] else '',
                    'title': f"Graph: {record['source_concept']} <-> {record['target_concept']}",
                    'retrieval_source': 'neo4j_graph',
                    'query_used': f"{record['source_concept']} → {record['target_concept']}",
                    'score': 1.0  # Graph matches are binary
                })
        except Exception as e:
            print(f"    Neo4j error: {e}")

        return results

    def dedupe_results(self, results: List[Dict]) -> List[Dict]:
        """Remove duplicate passages, keeping highest score."""
        seen = {}
        for r in results:
            pid = r.get('passage_id', '')
            if not pid:
                continue
            if pid not in seen or r.get('score', 0) > seen[pid].get('score', 0):
                seen[pid] = r
        return list(seen.values())


def main():
    # Load aliases
    aliases_file = Path(OUTPUT_DIR) / "INTERMEDIATE" / "ALIASES.jsonl"
    if not aliases_file.exists():
        print(f"ERROR: {aliases_file} not found. Run Phase 2 first.")
        return

    bridges = []
    with open(aliases_file, 'r') as f:
        for line in f:
            bridges.append(json.loads(line))

    print(f"Loaded {len(bridges)} bridges with aliases")

    # Initialize harvester
    harvester = EvidenceHarvester()

    output_file = Path(OUTPUT_DIR) / "EVIDENCE" / "RAW_EVIDENCE.jsonl"
    total_evidence = 0

    with open(output_file, 'w') as f:
        for i, bridge in enumerate(bridges):
            bridge_id = bridge['bridge_id']
            bucket = bridge['bucket']

            print(f"[{i+1}/{len(bridges)}] {bridge_id}: Harvesting evidence...")

            # Build search terms
            source_aliases = bridge.get('source_aliases', [bridge['source_concept']])
            target_aliases = bridge.get('target_aliases', [bridge['target_concept']])
            combined_queries = bridge.get('combined_queries', [])
            hyde_text = bridge.get('hyde_abstract', '')

            # 1. Postgres FTS
            pg_results = harvester.search_postgres_fts(
                source_aliases[:3] + target_aliases[:3] + combined_queries[:2],
                limit=20
            )
            print(f"    PG FTS: {len(pg_results)} results")

            # 2. ChromaDB semantic
            chroma_results = harvester.search_chromadb_semantic(
                combined_queries,
                hyde_text=hyde_text,
                limit=20
            )
            print(f"    ChromaDB: {len(chroma_results)} results")

            # 3. Neo4j graph
            neo4j_results = harvester.search_neo4j_graph(
                source_aliases[:5],
                target_aliases[:5],
                limit=20
            )
            print(f"    Neo4j: {len(neo4j_results)} results")

            # Merge and dedupe
            all_results = pg_results + chroma_results + neo4j_results
            deduped = harvester.dedupe_results(all_results)
            print(f"    After dedup: {len(deduped)} unique passages")

            # Write to output
            for result in deduped:
                record = {
                    'bridge_id': bridge_id,
                    'bucket': bucket,
                    **result
                }
                f.write(json.dumps(record) + '\n')
                total_evidence += 1

            f.flush()

    harvester.close()

    print(f"\n✓ Harvested {total_evidence} evidence spans")
    print(f"  Output: {output_file}")

    # Quick stats
    with open(output_file, 'r') as f:
        evidence_by_bridge = {}
        evidence_by_source = {'postgres_fts': 0, 'chromadb_semantic': 0, 'neo4j_graph': 0}
        for line in f:
            record = json.loads(line)
            bid = record['bridge_id']
            evidence_by_bridge[bid] = evidence_by_bridge.get(bid, 0) + 1
            evidence_by_source[record['retrieval_source']] = evidence_by_source.get(record['retrieval_source'], 0) + 1

    print(f"\n  Bridges with evidence: {len(evidence_by_bridge)}")
    print(f"  Evidence by source:")
    for src, cnt in evidence_by_source.items():
        print(f"    {src}: {cnt}")


if __name__ == "__main__":
    main()
