#!/usr/bin/env python3
"""
Rosetta Stone Query Expander

Solves the "Vocabulary Gap" problem by auto-expanding queries with cross-domain synonyms.

Example:
    Input:  "persistent homology tissue analysis"
    Expanded: "persistent homology OR betti numbers OR topological features OR
               tissue topology OR vascular loops OR glandular architecture"

Uses LLM to generate domain-specific terminology on-the-fly, rather than
fixed dictionaries.
"""

import os
import re
from typing import List, Dict, Optional
import logging

logger = logging.getLogger(__name__)

# Domain pairs that commonly need translation
DOMAIN_PAIRS = {
    ("mathematics", "biology"),
    ("mathematics", "medicine"),
    ("physics", "biology"),
    ("computer_science", "biology"),
    ("engineering", "medicine"),
}

# Keywords that trigger domain detection
DOMAIN_KEYWORDS = {
    "mathematics": [
        "topology", "homology", "manifold", "algebra", "geometry", "betti",
        "sparse", "matrix", "dimension", "signal", "sampling", "reconstruction",
        "inverse", "latent", "hidden", "variable"
    ],
    "biology": [
        "cell", "tissue", "gene", "protein", "organism", "evolution",
        "morphogenesis", "genomics", "transcriptomics",
    ],
    "medicine": ["pathology", "diagnosis", "treatment", "clinical", "patient", "disease"],
    "physics": ["thermodynamics", "entropy", "energy", "dynamics", "force", "field", "diffusion", "fluctuation"],
    "computer_science": [
        "algorithm", "neural network", "optimization", "learning", "model",
        "infer", "latent", "variable", "hidden", "sampling", "reconstruction", "imputation", "measurements"
    ],
}


def detect_domains(query: str) -> List[str]:
    """
    Detect which domains are present in the query.
    """
    query_lower = query.lower()
    detected = []

    for domain, keywords in DOMAIN_KEYWORDS.items():
        for keyword in keywords:
            if keyword in query_lower:
                detected.append(domain)
                break

    return detected


def generate_expansion_prompt(query: str, source_domain: str, target_domain: str) -> str:
    """
    Generate a prompt for LLM to expand query vocabulary.
    """
    return f"""You are a cross-domain terminology translator for scientific search.

Given a search query with terms from {source_domain}, provide equivalent terms used in {target_domain}.

Original query: "{query}"

Task: List 5-8 {target_domain} terms or phrases that refer to the same underlying concepts, structures, or phenomena as in the original query.

Rules:
- Focus on technical terms, not generic words
- Include both formal and colloquial {target_domain} terminology
- Prioritize terms that would actually appear in {target_domain} research papers/code
- Format as comma-separated list only, no explanations

Example:
Query: "persistent homology tissue analysis"
Domains: mathematics -> biology
Output: betti numbers, topological features, tissue topology, vascular loops, glandular architecture, structural holes, cavity detection, geometric patterns

Now translate:
Query: "{query}"
Domains: {source_domain} -> {target_domain}
Output:"""


def expand_query_with_llm(
    query: str,
    source_domain: str = None,
    target_domain: str = None,
    use_claude: bool = True
) -> List[str]:
    """
    Expand query using LLM to generate cross-domain synonyms.
    """
    # Auto-detect domains if not specified
    if source_domain is None or target_domain is None:
        detected = detect_domains(query)
        
        if len(detected) < 1:
            logger.debug("No clear domain detected, skipping expansion")
            return []

        # Use first detected as source
        source_domain = detected[0] if source_domain is None else source_domain

        # Find target domain from common pairs
        if target_domain is None:
            for d1, d2 in DOMAIN_PAIRS:
                if d1 == source_domain:
                    target_domain = d2
                    break
                elif d2 == source_domain:
                    target_domain = d1
                    break
            
            # If no pair found, default to biology for CS/Math queries (common in this system)
            if target_domain is None and source_domain in ["computer_science", "mathematics", "physics"]:
                 target_domain = "biology"

        if target_domain is None:
            logger.debug("No target domain found for expansion")
            return []

    # Generate expansion
    prompt = generate_expansion_prompt(query, source_domain, target_domain)

    try:
        if use_claude:
            try:
                expansion_text = _call_claude_api(prompt)
            except Exception as e:
                expansion_text = _fallback_expansion(query, source_domain, target_domain)
        else:
            # Fallback: Use local LLM or fixed dictionary
            expansion_text = _fallback_expansion(query, source_domain, target_domain)

        # Parse expansion
        terms = [t.strip() for t in expansion_text.split(',')]
        terms = [t for t in terms if t]  # Remove empty

        return terms

    except Exception as e:
        logger.error(f"Query expansion failed completely: {e}")
        return []


def _call_claude_api(prompt: str, model: str = "claude-3-5-haiku-20241022") -> str:
    """
    Call Claude API for query expansion.
    """
    import anthropic

    api_key = os.environ.get("ANTHROPIC_API_KEY")
    if not api_key:
        raise ValueError("ANTHROPIC_API_KEY not set")

    client = anthropic.Anthropic(api_key=api_key)

    response = client.messages.create(
        model=model,
        max_tokens=200,
        temperature=0.3,  # Low temp for consistent terminology
        messages=[{"role": "user", "content": prompt}]
    )

    return response.content[0].text.strip()


def _fallback_expansion(query: str, source_domain: str, target_domain: str) -> str:
    """
    Fallback expansion using fixed dictionary (when LLM unavailable).
    """
    # Simple fixed mappings
    FALLBACK_MAPPINGS = {
                ("mathematics", "biology"): {
                    "topology": "spatial structure, tissue architecture, geometric patterns",
                    "homology": "structural similarity, conserved features, analogous structures",
                    "manifold": "curved space, tissue surface, membrane structure",
                    "persistent homology": "topological features, structural persistence, betti numbers, vascular loops, tissue holes",
                    "betti": "cycles, loops, voids, holes, cavities",
                    "sparse": "compressed sensing, sparse coding, undersampled, dropout, missing data, low coverage, spot dropout",
                    "sampling": "spot sampling, sparse measurements, low coverage",
                    "reconstruction": "imputation, recovery, deconvolution, reconstruction",
                    "inverse": "active inference, deconvolution, latent state inference, parameter estimation, variational inference, bayesian inference",
                    "latent": "active inference, hidden state, unobserved factor, latent variable, state inference",
                    "signal": "gene expression, molecular abundance, spatial profile",
                    "hidden": "active inference, latent, unobserved, underlying",
                    "variable": "gene, protein, feature",
                    "infer": "active inference, predict, estimate, deconvolve",
                },
        ("mathematics", "medicine"):
            {
                "topology": "spatial structure, tissue architecture, geometric patterns",
                "homology": "structural similarity, conserved features, analogous structures",
                "persistent homology": "topological features, structural persistence, vascular loops, glandular architecture",
                "sparse": "undersampled, missing data, low coverage",
                "reconstruction": "imputation, recovery, deconvolution",
                "inverse": "deconvolution, latent state inference",
            },
        ("biology", "mathematics"):
            {
                "tissue": "structured space, geometric domain, spatial network",
                "cell": "discrete unit, lattice point, node",
                "network": "graph, connectivity, topology",
                "morphogenesis": "pattern formation, self-organization, reaction diffusion",
            },
        ("physics", "biology"):
            {
                "entropy": "disorder, information content, randomness, variability, diversity",
                "free energy": "potential, driving force, thermodynamic favorability, fitness cost",
                "diffusion": "spread, distribution, gradient flow, chemical gradient, migration",
            },
        ("computer_science", "biology"):
            {
                "infer": "predict, estimate, deconvolve, reconstruct, bayesian inference, variational inference, active inference",
                "latent": "hidden state, underlying phenotype, cell type, regulatory program, state inference, active inference",
                "hidden": "unobserved, latent, underlying",
                "variable": "gene, feature, protein, factor",
                "sampling": "spot sampling, sparse measurements, low coverage",
                "reconstruction": "imputation, recovery, deconvolution",
                "imputation": "spot imputation, gene recovery, deconvolution",
                "neural network": "brain circuit, connectionist model, biological network",
                "optimization": "evolution, adaptation, fitness maximization",
                "measurements": "gene expression, observations, reads, counts",
            },
    }

    query_lower = query.lower()
    mapping = FALLBACK_MAPPINGS.get((source_domain, target_domain), {})
    
    # If no exact pair, try to use biology as generic target if applicable
    if not mapping and target_domain == "biology":
         # Combine mappings from math->bio and cs->bio as fallback
         mapping = {}
         mapping.update(FALLBACK_MAPPINGS.get(("mathematics", "biology"), {}))
         mapping.update(FALLBACK_MAPPINGS.get(("computer_science", "biology"), {}))

    expansions = []
    # Check for multi-word phrases first
    for term, expansion in mapping.items():
        if term in query_lower:
            expansions.append(expansion)

    return ", ".join(expansions) if expansions else ""


def expand_query(query: str, max_expansions: int = 15) -> str:
    """
    Main entry point: Expand query with cross-domain terminology.
    """
    # Get expansion terms
    expansion_terms = expand_query_with_llm(query)

    if not expansion_terms:
        return query  # No expansion

    # Limit expansions
    expansion_terms = expansion_terms[:max_expansions]

    # Build expanded query
    # Format: "original query OR term1 OR term2 OR ..."
    expanded = f"{query} OR " + " OR ".join(expansion_terms)

    logger.info(f"Expanded query: {query} -> {len(expansion_terms)} terms")
    logger.debug(f"Full expansion: {expanded}")

    return expanded


# CLI for testing
def main():
    import argparse

    parser = argparse.ArgumentParser(description="Test Rosetta Stone query expander")
    parser.add_argument("query", help="Query to expand")
    parser.add_argument("--source", help="Source domain")
    parser.add_argument("--target", help="Target domain")
    parser.add_argument("--no-llm", action="store_true", help="Use fallback (no API call)")

    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO)

    # Detect domains
    if args.source is None or args.target is None:
        detected = detect_domains(args.query)
        print(f"Detected domains: {detected}")

    # Expand
    use_llm = not args.no_llm
    expansion_terms = expand_query_with_llm(
        args.query,
        source_domain=args.source,
        target_domain=args.target,
        use_claude=use_llm
    )

    print(f"\nOriginal query: {args.query}")
    print(f"\nExpansion terms ({len(expansion_terms)}):")
    for i, term in enumerate(expansion_terms, 1):
        print(f"  {i}. {term}")

    print(f"\nExpanded query:")
    expanded = expand_query(args.query)
    print(f"  {expanded}")


if __name__ == "__main__":
    main()