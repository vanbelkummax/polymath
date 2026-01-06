#!/usr/bin/env python3
"""
HyDE (Hypothetical Document Embeddings) Bridge Search

Fixes the 0/5 domain bridge problem by using LLM to hallucinate
a bridge document, then searching for real papers matching that semantic space.

Core Insight: The bridge exists in LOGIC, not VOCABULARY.
"""

import sys
sys.path.insert(0, '/home/user/work/polymax/lib')
sys.path.insert(0, '/home/user/work/polymax/mcp')

import chromadb
from sentence_transformers import SentenceTransformer
from typing import List, Dict, Tuple
import os


class HyDEBridgeSearcher:
    """
    Searches for domain bridges using Hypothetical Document Embeddings.

    Strategy:
    1. User asks: "Connect optimal transport to Waddington landscape"
    2. LLM hallucinates: "Optimal transport minimizes cost of moving
       probability distributions... this formalizes Waddington's landscape
       by treating differentiation as flow on manifold..."
    3. Embed hallucination and search for real papers matching that vector
    4. Result: Find papers on "manifold learning in single-cell" and
       "differentiation trajectories" that bridge the concepts
    """

    def __init__(self):
        self._chroma = None
        self._embedder = None

    def _get_chroma(self):
        if self._chroma is None:
            client = chromadb.PersistentClient(
                path=os.environ.get("CHROMA_PATH", "/home/user/work/polymax/chromadb/polymath_v2")
            )
            self._chroma = client.get_collection("polymath_corpus")
        return self._chroma

    def _get_embedder(self):
        if self._embedder is None:
            self._embedder = SentenceTransformer("all-mpnet-base-v2")
        return self._embedder

    def generate_hypothetical_bridge(
        self,
        concept_a: str,
        concept_b: str,
        mode: str = "structural"
    ) -> str:
        """
        Generate hypothetical bridge document using domain-specific templates.

        Three modes optimized for MD/PhD polymath context:
        1. structural: Math/Physics → Biology (isomorphisms)
        2. mechanistic: CS/ML → Biology (algorithms as mechanisms)
        3. methodological: Tools/Methods → Biology (applications)
        """

        if mode == "structural":
            # Best for: Math/Physics → Biology
            template = f"""Abstract:

We demonstrate a rigorous mathematical framework connecting {concept_a} and {concept_b}.
The key insight is that {concept_a} provides a formal language for describing the
structural properties observed in {concept_b}. Specifically, we show that the
mathematical formalism of {concept_a} can be used to model the dynamics, constraints,
and optimization principles underlying {concept_b}.

Our approach reveals a structural isomorphism between the abstract mathematical space
of {concept_a} and the biological state space of {concept_b}. This connection enables
us to apply computational methods from {concept_a} to solve open problems in {concept_b},
including trajectory inference, optimal pathways, and constraint satisfaction.

We validate this framework on experimental data and show that {concept_a}-based models
capture essential features of {concept_b} that were previously described only qualitatively.
This work bridges mathematical theory and biological mechanism, providing a foundation
for quantitative systems biology.

Keywords: {concept_a}, {concept_b}, mathematical biology, systems biology, dynamical systems,
optimization, state space, trajectory inference, computational modeling"""

        elif mode == "mechanistic":
            # Best for: CS/ML → Biology
            template = f"""Abstract:

We present a novel computational framework that applies {concept_a} to understand the
mechanisms underlying {concept_b}. Traditional biological approaches to {concept_b}
have been largely descriptive; here we show that {concept_a} provides both a predictive
model and mechanistic insight.

Our key contribution is recognizing that biological systems implementing {concept_b}
are performing computations analogous to {concept_a} algorithms. We formalize this
connection by mapping biological components (cells, proteins, regulatory networks) to
computational primitives (nodes, layers, update rules) in {concept_a}.

Using this framework, we derive testable predictions about {concept_b} and validate
them experimentally. The {concept_a} perspective reveals hidden structure in {concept_b}
data, including latent representations, information bottlenecks, and optimization objectives
that cells implicitly solve. This computational lens transforms our understanding of
{concept_b} from description to prediction.

Keywords: {concept_a}, {concept_b}, computational biology, machine learning, biological
computation, neural networks, information processing, deep learning, representation learning"""

        elif mode == "methodological":
            # Best for: Methods/Tools → Biology applications
            template = f"""Abstract:

We apply {concept_a} methodology to address fundamental questions in {concept_b}.
While {concept_a} was originally developed for [physics/engineering/computer science],
its core principles - including [sparsity/optimization/inference/geometry] - are
directly applicable to biological systems.

Our approach adapts {concept_a} techniques to handle the unique challenges of {concept_b}
data: high dimensionality, noise, missing values, and heterogeneity. We develop a
pipeline that integrates {concept_a} with standard biological analysis, enabling
researchers to leverage powerful computational methods without requiring deep mathematical
expertise.

We demonstrate the utility of this approach on [single-cell data/imaging data/omics data],
showing that {concept_a} reveals patterns invisible to conventional methods. Specifically,
we identify [cell states/spatial organization/regulatory networks/disease signatures]
that provide new biological insights into {concept_b}. This work establishes {concept_a}
as a practical tool for modern experimental biology.

Keywords: {concept_a}, {concept_b}, methodology, data analysis, computational tools,
high-dimensional data, single-cell, spatial transcriptomics, image analysis"""

        else:
            # Fallback: Generic bridge
            template = f"""Abstract:

This paper establishes a connection between {concept_a} and {concept_b}. We show that
principles from {concept_a} can be applied to understand {concept_b}, revealing new
insights and enabling novel predictions. Our framework bridges theoretical concepts
and biological mechanisms, providing a foundation for future research.

Keywords: {concept_a}, {concept_b}, interdisciplinary, systems biology"""

        return template

    def bridge_search(
        self,
        concept_a: str,
        concept_b: str,
        n_results: int = 10,
        auto_mode: bool = True
    ) -> List[Dict]:
        """
        HyDE-based domain bridge search.

        Args:
            concept_a: First concept (e.g., "optimal transport")
            concept_b: Second concept (e.g., "Waddington landscape")
            n_results: Number of results to return
            auto_mode: If True, automatically select best template mode

        Returns:
            List of search results with relevance scores
        """

        # Step 1: Detect which mode to use
        if auto_mode:
            mode = self._detect_mode(concept_a, concept_b)
        else:
            mode = "structural"

        # Step 2: Generate hypothetical bridge documents (try all 3 modes)
        modes_to_try = [mode, "structural", "mechanistic", "methodological"]
        all_results = []

        for m in modes_to_try[:2]:  # Try top 2 modes
            # Generate hallucination
            hypothetical_doc = self.generate_hypothetical_bridge(concept_a, concept_b, mode=m)

            # Step 3: Embed and search
            embedder = self._get_embedder()
            query_embedding = embedder.encode([hypothetical_doc]).tolist()[0]

            collection = self._get_chroma()
            results = collection.query(
                query_embeddings=[query_embedding],
                n_results=n_results
            )

            # Convert to standard format
            for i, doc_id in enumerate(results['ids'][0]):
                all_results.append({
                    'id': doc_id,
                    'text': results['documents'][0][i],
                    'metadata': results['metadatas'][0][i],
                    'distance': results['distances'][0][i] if 'distances' in results else 0.0,
                    'relevance': 1 - results['distances'][0][i] if 'distances' in results else 0.9,
                    'hyde_mode': m,
                    'is_bridge': True
                })

        # Step 4: Deduplicate and rank
        seen_ids = set()
        unique_results = []
        for r in all_results:
            if r['id'] not in seen_ids:
                seen_ids.add(r['id'])
                unique_results.append(r)

        # Sort by relevance
        unique_results.sort(key=lambda x: x['relevance'], reverse=True)

        return unique_results[:n_results]

    def _detect_mode(self, concept_a: str, concept_b: str) -> str:
        """
        Auto-detect which HyDE mode to use based on concept keywords.

        Returns: "structural", "mechanistic", or "methodological"
        """
        c_lower = (concept_a + " " + concept_b).lower()

        # Structural: Math/Physics concepts
        structural_keywords = [
            'topology', 'manifold', 'geometry', 'algebra', 'category theory',
            'differential', 'optimization', 'thermodynamics', 'entropy',
            'graph theory', 'dynamics', 'control theory', 'information theory'
        ]

        # Mechanistic: CS/ML/AI concepts
        mechanistic_keywords = [
            'neural network', 'transformer', 'attention', 'deep learning',
            'machine learning', 'reinforcement learning', 'active inference',
            'bayesian', 'inference', 'model', 'algorithm', 'computation'
        ]

        # Methodological: Tools/techniques
        methodological_keywords = [
            'compressed sensing', 'sparse', 'dimensionality reduction',
            'clustering', 'classification', 'regression', 'tda', 'topological data',
            'causal inference', 'statistical', 'analysis', 'method'
        ]

        # Count matches
        structural_score = sum(1 for kw in structural_keywords if kw in c_lower)
        mechanistic_score = sum(1 for kw in mechanistic_keywords if kw in c_lower)
        methodological_score = sum(1 for kw in methodological_keywords if kw in c_lower)

        # Return highest scoring mode
        scores = {
            'structural': structural_score,
            'mechanistic': mechanistic_score,
            'methodological': methodological_score
        }

        best_mode = max(scores, key=scores.get)

        # Default to structural if all zero
        if scores[best_mode] == 0:
            return 'structural'

        return best_mode


# CLI for testing
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="HyDE Bridge Search")
    parser.add_argument("concept_a", help="First concept")
    parser.add_argument("concept_b", help="Second concept")
    parser.add_argument("-n", "--num", type=int, default=10, help="Number of results")
    parser.add_argument("--mode", choices=["structural", "mechanistic", "methodological", "auto"],
                       default="auto", help="HyDE template mode")

    args = parser.parse_args()

    searcher = HyDEBridgeSearcher()

    print(f"\n=== HyDE Bridge Search: '{args.concept_a}' ↔ '{args.concept_b}' ===")

    # Detect mode if auto
    if args.mode == "auto":
        detected_mode = searcher._detect_mode(args.concept_a, args.concept_b)
        print(f"Auto-detected mode: {detected_mode}")
        print()

    results = searcher.bridge_search(
        args.concept_a,
        args.concept_b,
        n_results=args.num,
        auto_mode=(args.mode == "auto")
    )

    print(f"Found: {len(results)} bridge papers\n")

    for i, r in enumerate(results, 1):
        print(f"{i}. [HyDE:{r['hyde_mode']}] [{r['relevance']:.3f}]")
        print(f"   {r['text'][:150]}...")
        print()
