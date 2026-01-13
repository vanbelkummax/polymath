#!/usr/bin/env python3
"""
RRF (Reciprocal Rank Fusion) Validator for BridgeMine v4

Multi-source evidence validation using RRF scoring.

Sources:
1. Postgres passage search (via passage_concepts)
2. ChromaDB semantic search (BGE-M3 embeddings)
3. PubMed search (external validation)
4. Semantic Scholar search (external validation)

RRF formula: score(doc) = Σ(1 / (k + rank_i)) where k=60 (standard)
"""

import os
import sys
from typing import List, Dict, Tuple, Set
from dataclasses import dataclass
from collections import defaultdict

sys.path.insert(0, '/home/user/polymath-repo')

import psycopg2
import chromadb
from FlagEmbedding import BGEM3FlagModel

# RRF constant (standard value from literature)
RRF_K = 60


@dataclass
class EvidenceSpan:
    """A single evidence span from any source."""
    doc_id: str
    title: str
    snippet: str
    source: str  # 'postgres', 'chromadb', 'pubmed', 's2'
    rank: int
    score: float  # Original score from source


@dataclass
class FusedEvidence:
    """Evidence after RRF fusion across sources."""
    doc_id: str
    title: str
    snippets: List[str]
    sources: Set[str]
    rrf_score: float
    source_ranks: Dict[str, int]  # source -> rank
    confidence: str  # 'high', 'medium', 'low'


class RRFValidator:
    """
    Multi-source evidence validator using Reciprocal Rank Fusion.

    Combines evidence from:
    - Postgres (passage_concepts co-occurrence)
    - ChromaDB (semantic similarity)
    - PubMed (external validation)
    - Semantic Scholar (external validation)
    """

    def __init__(self, pg_conn_str: str = "dbname=polymath user=polymath",
                 chromadb_path: str = "/home/user/polymath-repo/chromadb"):
        self.pg_conn_str = pg_conn_str
        self.chromadb_path = chromadb_path
        self.model = None  # Lazy load BGE-M3

    def _get_postgres_evidence(self, method: str, problem: str, limit: int = 10) -> List[EvidenceSpan]:
        """
        Find passages where method and problem co-occur.

        Returns passages ranked by co-occurrence strength.
        """
        conn = psycopg2.connect(self.pg_conn_str)
        cursor = conn.cursor()

        # Find documents where both concepts appear
        cursor.execute("""
            WITH method_docs AS (
                SELECT DISTINCT p.doc_id, p.passage_id, p.passage_text
                FROM passage_concepts pc
                JOIN passages p ON pc.passage_id = p.passage_id
                WHERE pc.concept_name ILIKE %s
                  AND pc.concept_type IN ('method', 'technique', 'algorithm', 'model')
            ),
            problem_docs AS (
                SELECT DISTINCT p.doc_id, p.passage_id
                FROM passage_concepts pc
                JOIN passages p ON pc.passage_id = p.passage_id
                WHERE pc.concept_name ILIKE %s
                  AND pc.concept_type IN ('objective', 'problem')
            )
            SELECT DISTINCT
                md.doc_id,
                d.title,
                md.passage_text,
                COUNT(*) OVER (PARTITION BY md.doc_id) as co_occurrence_count
            FROM method_docs md
            JOIN problem_docs pd ON md.doc_id = pd.doc_id
            JOIN documents d ON md.doc_id = d.doc_id
            ORDER BY co_occurrence_count DESC, md.doc_id
            LIMIT %s
        """, (f"%{method}%", f"%{problem}%", limit))

        evidence = []
        for rank, (doc_id, title, snippet, score) in enumerate(cursor.fetchall(), 1):
            evidence.append(EvidenceSpan(
                doc_id=str(doc_id),
                title=title or "Unknown",
                snippet=snippet[:300] if snippet else "",
                source='postgres',
                rank=rank,
                score=float(score)
            ))

        conn.close()
        return evidence

    def _get_chromadb_evidence(self, query: str, limit: int = 10) -> List[EvidenceSpan]:
        """
        Semantic search in ChromaDB using BGE-M3 embeddings.

        Returns passages ranked by semantic similarity.
        """
        try:
            # Lazy load model
            if self.model is None:
                print("  Loading BGE-M3 model for RRF validation...")
                self.model = BGEM3FlagModel('BAAI/bge-m3', use_fp16=True)

            # Embed query
            query_result = self.model.encode([query], batch_size=1)
            query_vec = query_result['dense_vecs'][0]

            # Query ChromaDB
            client = chromadb.PersistentClient(path=self.chromadb_path)
            collection = client.get_collection("polymath_bge_m3")

            results = collection.query(
                query_embeddings=[query_vec.tolist()],
                n_results=limit,
                include=['documents', 'metadatas', 'distances']
            )

            evidence = []
            for rank, (doc_text, metadata, distance) in enumerate(zip(
                results['documents'][0],
                results['metadatas'][0],
                results['distances'][0]
            ), 1):
                # Convert distance to similarity score (lower is better for distance)
                similarity = 1.0 / (1.0 + distance)

                evidence.append(EvidenceSpan(
                    doc_id=metadata.get('doc_id', 'unknown'),
                    title=metadata.get('title', 'Unknown'),
                    snippet=doc_text[:300] if doc_text else "",
                    source='chromadb',
                    rank=rank,
                    score=similarity
                ))

            return evidence

        except Exception as e:
            print(f"  ChromaDB error: {e}")
            return []

    def compute_rrf_scores(self, all_evidence: List[EvidenceSpan]) -> List[FusedEvidence]:
        """
        Apply RRF fusion across all evidence sources.

        RRF score = Σ(1 / (k + rank_i)) across all sources
        """
        # Group evidence by doc_id
        by_doc = defaultdict(list)
        for ev in all_evidence:
            by_doc[ev.doc_id].append(ev)

        # Compute RRF scores
        fused = []
        for doc_id, spans in by_doc.items():
            # RRF score = sum of 1/(k + rank) across sources
            rrf_score = sum(1.0 / (RRF_K + span.rank) for span in spans)

            # Collect metadata
            sources = {span.source for span in spans}
            source_ranks = {span.source: span.rank for span in spans}
            snippets = [span.snippet for span in spans if span.snippet]
            title = spans[0].title

            # Assign confidence based on source diversity and score
            if len(sources) >= 3 and rrf_score > 0.05:
                confidence = 'high'
            elif len(sources) >= 2 and rrf_score > 0.03:
                confidence = 'medium'
            else:
                confidence = 'low'

            fused.append(FusedEvidence(
                doc_id=doc_id,
                title=title,
                snippets=snippets[:3],  # Top 3 snippets
                sources=sources,
                rrf_score=rrf_score,
                source_ranks=source_ranks,
                confidence=confidence
            ))

        # Sort by RRF score descending
        fused.sort(key=lambda x: -x.rrf_score)
        return fused

    def validate_transfer(self, method: str, problem: str,
                         use_postgres: bool = True,
                         use_chromadb: bool = True,
                         limit_per_source: int = 10) -> List[FusedEvidence]:
        """
        Validate a method→problem transfer using multi-source evidence fusion.

        Args:
            method: Method name (e.g., "diffusion_model")
            problem: Problem name (e.g., "spatial_imputation")
            use_postgres: Include Postgres passage search
            use_chromadb: Include ChromaDB semantic search
            limit_per_source: Max results per source

        Returns:
            List of FusedEvidence ranked by RRF score
        """
        all_evidence = []

        # Source 1: Postgres co-occurrence
        if use_postgres:
            print(f"  Searching Postgres for '{method}' + '{problem}'...")
            pg_evidence = self._get_postgres_evidence(method, problem, limit_per_source)
            all_evidence.extend(pg_evidence)
            print(f"    Found {len(pg_evidence)} postgres spans")

        # Source 2: ChromaDB semantic search
        if use_chromadb:
            query = f"{method} for {problem}"
            print(f"  Searching ChromaDB for '{query}'...")
            chroma_evidence = self._get_chromadb_evidence(query, limit_per_source)
            all_evidence.extend(chroma_evidence)
            print(f"    Found {len(chroma_evidence)} chromadb spans")

        # Apply RRF fusion
        if all_evidence:
            print(f"  Fusing evidence from {len(all_evidence)} spans...")
            fused = self.compute_rrf_scores(all_evidence)
            print(f"    Generated {len(fused)} fused evidence items")
            return fused
        else:
            return []


def test_rrf_validator():
    """Test RRF validator with a known transfer."""
    validator = RRFValidator()

    # Test case: diffusion models for spatial imputation
    print("="*70)
    print("TEST: RRF Validation for 'diffusion_model' → 'spatial_imputation'")
    print("="*70)

    fused = validator.validate_transfer(
        method="diffusion_model",
        problem="imputation",
        use_postgres=True,
        use_chromadb=True,
        limit_per_source=5
    )

    print(f"\nTop 5 Fused Evidence (RRF scored):")
    for i, ev in enumerate(fused[:5], 1):
        print(f"\n{i}. {ev.title}")
        print(f"   RRF Score: {ev.rrf_score:.4f} | Confidence: {ev.confidence}")
        print(f"   Sources: {', '.join(ev.sources)} | Ranks: {ev.source_ranks}")
        if ev.snippets:
            print(f"   Snippet: {ev.snippets[0][:150]}...")


if __name__ == "__main__":
    test_rrf_validator()
