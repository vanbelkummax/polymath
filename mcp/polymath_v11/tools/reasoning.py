"""
Polymath v11 Reasoning Tools

Tools 9-14: Hypothesis generation, analogy finding, serendipity

Includes "slow thinking" mode for hypothesis generation that:
- Decomposes problems step-by-step
- Validates each reasoning step against corpus
- Produces explicit reasoning chains
"""

import sys
import os
import random
from pathlib import Path
from typing import Optional, Set, List, Dict
from collections import Counter
import math
import logging

# Add repo root to path so `lib.*` imports resolve correctly.
REPO_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO_ROOT))

logger = logging.getLogger(__name__)

from lib.hybrid_search_v2 import HybridSearcherV2
from lib.db import get_db_connection


# Domain categories for novelty scoring
CONCEPT_DOMAINS = {
    "signal_processing": {"compressed_sensing", "sparse_coding", "wavelet", "fourier", "denoising"},
    "physics": {"entropy", "free_energy", "thermodynamics", "diffusion", "reaction_diffusion"},
    "causality": {"causal_inference", "counterfactual", "do_calculus", "instrumental_variable"},
    "systems": {"feedback", "control_theory", "autopoiesis", "emergence", "cybernetics"},
    "cognitive": {"predictive_coding", "bayesian_brain", "active_inference", "affordance"},
    "ml_ai": {"neural_network", "transformer", "attention", "contrastive_learning", "foundation_model"},
    "geometry": {"graph_neural_network", "topology", "manifold", "geometric_deep_learning", "optimal_transport"},
    "biology": {"spatial_transcriptomics", "single_cell", "gene_expression", "tissue_structure", "cell_type"},
}


def _get_concept_domain(concept: str) -> Optional[str]:
    """Get the domain a concept belongs to."""
    concept_lower = concept.lower().replace(" ", "_")
    for domain, concepts in CONCEPT_DOMAINS.items():
        if concept_lower in concepts or any(c in concept_lower for c in concepts):
            return domain
    return None


def _compute_novelty_score(
    source_concept: str,
    target_domain: str,
    searcher: 'HybridSearcherV2'
) -> float:
    """
    Compute novelty score based on:
    1. Domain distance: Cross-domain bridges are more novel
    2. Corpus coverage: Fewer existing papers = higher novelty
    3. Combination rarity: How rare is this specific pairing

    Returns score between 0.0 (well-established) and 1.0 (highly novel)
    """
    # Get source domain
    source_domain = _get_concept_domain(source_concept)

    # Get target domain keywords
    target_domain_cat = _get_concept_domain(target_domain)

    # Component 1: Domain distance (0.0-0.4)
    # Same domain = 0.0, cross-domain = 0.2, very distant = 0.4
    if source_domain == target_domain_cat:
        domain_distance = 0.0
    elif source_domain is None or target_domain_cat is None:
        domain_distance = 0.2  # Unknown domain, moderate novelty
    elif source_domain in ("ml_ai", "geometry") and target_domain_cat == "biology":
        domain_distance = 0.3  # Common ML→bio bridge
    elif source_domain in ("physics", "signal_processing") and target_domain_cat == "biology":
        domain_distance = 0.4  # Less common bridge
    else:
        domain_distance = 0.35  # General cross-domain

    # Component 2: Corpus coverage (0.0-0.4)
    # Search for papers combining both concepts
    combined_query = f"{source_concept} {target_domain}"
    combined_results = searcher.search_papers(combined_query, n=20)

    if len(combined_results) == 0:
        coverage_novelty = 0.4  # Not in corpus - highly novel
    elif len(combined_results) < 5:
        coverage_novelty = 0.3  # Few papers - quite novel
    elif len(combined_results) < 10:
        coverage_novelty = 0.2  # Some papers - moderately novel
    else:
        # Many papers - use average score to gauge relevance
        avg_score = sum(r.score for r in combined_results) / len(combined_results)
        if avg_score > 0.5:
            coverage_novelty = 0.05  # Well-covered in corpus
        else:
            coverage_novelty = 0.15  # Related but not direct

    # Component 3: Combination specificity (0.0-0.2)
    # Search for source concept alone - if rare itself, adds novelty
    source_results = searcher.search_papers(source_concept, n=10)
    if len(source_results) < 3:
        combo_novelty = 0.2  # Source concept is rare
    elif source_results[0].score < 0.4:
        combo_novelty = 0.15  # Low relevance matches
    else:
        combo_novelty = 0.1  # Normal

    # Combine components
    total_novelty = domain_distance + coverage_novelty + combo_novelty

    # Clamp to [0.3, 0.95] range - nothing is truly 0% or 100% novel
    return min(0.95, max(0.3, total_novelty))


class ReasoningTools:
    """Reasoning and hypothesis tools for Polymath v11."""

    def __init__(self):
        self.searcher = None

    def _get_searcher(self):
        if self.searcher is None:
            self.searcher = HybridSearcherV2()
        return self.searcher

    def _has_llm_api(self) -> bool:
        """Check if LLM API is available and enabled."""
        from lib.config import ENABLE_LLM_API, ANTHROPIC_API_KEY
        return ENABLE_LLM_API and bool(ANTHROPIC_API_KEY)

    async def _slow_think_step(
        self,
        step_name: str,
        question: str,
        searcher: 'HybridSearcherV2'
    ) -> Dict:
        """Execute one step of slow thinking with corpus validation.

        Args:
            step_name: Name of this reasoning step
            question: Question to answer in this step
            searcher: HybridSearcher for corpus queries

        Returns:
            Dict with step results and supporting evidence
        """
        # Search corpus for relevant evidence
        results = searcher.search_papers(question, n=10)

        evidence = []
        for r in results[:3]:
            evidence.append({
                "title": r.title,
                "snippet": r.content[:150],
                "score": round(r.score, 3)
            })

        return {
            "step": step_name,
            "question": question,
            "evidence_found": len(results),
            "top_evidence": evidence,
            "confidence": min(0.9, len(results) / 10.0) if results else 0.1
        }

    async def _slow_think_hypothesis(
        self,
        research_area: str,
        source_concept: str,
        target_property: str,
        searcher: 'HybridSearcherV2'
    ) -> Dict:
        """Generate hypothesis using step-by-step slow thinking.

        Chain of thought:
        1. What problem does {source_concept} solve in its native domain?
        2. What analogous problem exists in {research_area}?
        3. How would {source_concept} methods transfer?
        4. What adaptations would be needed?
        5. What evidence supports this transfer?
        6. What could go wrong?
        """
        reasoning_chain = []

        # Step 1: Source domain analysis
        step1 = await self._slow_think_step(
            "source_analysis",
            f"{source_concept} methods applications",
            searcher
        )
        step1["finding"] = f"Found {step1['evidence_found']} papers on {source_concept}"
        reasoning_chain.append(step1)

        # Step 2: Target domain gap analysis
        step2 = await self._slow_think_step(
            "gap_analysis",
            f"{research_area} challenges {target_property} problems",
            searcher
        )
        step2["finding"] = f"Found {step2['evidence_found']} papers on challenges in {research_area}"
        reasoning_chain.append(step2)

        # Step 3: Transfer feasibility
        step3 = await self._slow_think_step(
            "transfer_analysis",
            f"{source_concept} applied to {research_area}",
            searcher
        )
        existing_bridges = step3['evidence_found']
        step3["finding"] = f"Found {existing_bridges} papers bridging these concepts"
        reasoning_chain.append(step3)

        # Step 4: Risk analysis
        step4 = await self._slow_think_step(
            "risk_analysis",
            f"{source_concept} limitations failures",
            searcher
        )
        step4["finding"] = f"Identified {step4['evidence_found']} papers discussing limitations"
        reasoning_chain.append(step4)

        # Compute composite scores
        transfer_feasibility = "high" if existing_bridges < 5 else "medium" if existing_bridges < 15 else "well-established"
        overall_confidence = sum(s["confidence"] for s in reasoning_chain) / len(reasoning_chain)

        # Generate hypothesis with full reasoning trace
        return {
            "source_domain": source_concept.replace("_", " "),
            "target_domain": research_area,
            "hypothesis": f"Applying {source_concept.replace('_', ' ')} methods to {research_area} could improve {target_property.replace('_', ' ')}",
            "reasoning_chain": reasoning_chain,
            "reasoning_mode": "slow_thinking",
            "novelty_score": _compute_novelty_score(source_concept, research_area, searcher),
            "transfer_feasibility": transfer_feasibility,
            "overall_confidence": round(overall_confidence, 3),
            "testability": "high" if existing_bridges < 5 else "medium",
            "potential_risks": [
                f"May require adaptation of {source_concept} for {research_area} data structures",
                f"Existing {existing_bridges} papers may have already explored this direction"
            ] if existing_bridges > 0 else [
                "Novel combination - limited prior work to guide implementation",
                "May face unforeseen domain-specific challenges"
            ]
        }

    async def _llm_slow_think(
        self,
        research_area: str,
        source: str,
        target: str,
        context: List[Dict]
    ) -> Optional[Dict]:
        """Use LLM for deep reasoning about hypothesis.

        Only called when ENABLE_LLM_API=true.
        Primary mode uses Claude subagents via Task tool in Claude Code sessions.
        """
        if not self._has_llm_api():
            # Return None to signal caller should use rule-based fallback
            return None

        try:
            from anthropic import Anthropic
            from lib.config import LLM_MODEL

            client = Anthropic()

            # Build context from corpus
            context_text = "\n".join([
                f"- {p.get('title', 'Unknown')}: {p.get('snippet', '')[:200]}"
                for p in context[:10]
            ])

            prompt = f"""You are a research scientist generating novel hypotheses.

Research Area: {research_area}
Source Concept: {source}
Target Property: {target}

Relevant Papers:
{context_text}

Think step-by-step through this potential cross-domain hypothesis:

1. PROBLEM: What specific problem in {research_area} might benefit from {source} methods?
2. ANALOGY: What structural similarity exists between {source} and the target problem?
3. TRANSFER: What specific methods from {source} could transfer?
4. ADAPTATION: What modifications would be needed for {research_area}?
5. VALIDATION: How could this hypothesis be tested?
6. RISKS: What could prevent this from working?

Output a JSON object with:
- hypothesis: Clear, testable statement
- reasoning_chain: Array of step-by-step reasoning
- novelty_assessment: Why novel (or not)
- testability: "high", "medium", or "low"
- confidence: 0.0 to 1.0
"""

            response = client.messages.create(
                model=LLM_MODEL,
                max_tokens=2000,
                messages=[{"role": "user", "content": prompt}]
            )

            # Parse LLM response
            import json
            result_text = response.content[0].text

            # Extract JSON from response (may be wrapped in markdown)
            if "```json" in result_text:
                result_text = result_text.split("```json")[1].split("```")[0]
            elif "```" in result_text:
                result_text = result_text.split("```")[1].split("```")[0]

            result = json.loads(result_text)
            result["source_domain"] = source
            result["target_domain"] = research_area
            result["reasoning_mode"] = "llm_slow_think"

            return result

        except Exception as e:
            logger.warning(f"LLM slow thinking failed: {e}, falling back to rule-based")
            return None

    async def generate_hypothesis(
        self,
        research_area: str,
        num_hypotheses: int = 5,
        use_slow_thinking: bool = True
    ) -> dict:
        """
        Tool 9: Generate cross-domain hypotheses.

        Uses structure mapping to find analogies between domains
        and generate testable hypotheses.

        Args:
            research_area: The target research area
            num_hypotheses: Number of hypotheses to generate
            use_slow_thinking: If True, use step-by-step reasoning with corpus validation

        Returns:
            Dict with hypotheses and reasoning methodology
        """
        hs = self._get_searcher()

        # Get concepts in research area
        results = hs.search_papers(research_area, n=50)

        # Extract concepts (SearchResult objects)
        area_concepts = []
        for r in results:
            concepts_raw = r.metadata.get("concepts", "")
            if isinstance(concepts_raw, str) and concepts_raw:
                area_concepts.extend(concepts_raw.split(","))
            elif concepts_raw:
                area_concepts.extend(concepts_raw)

        concept_counts = Counter(area_concepts)
        top_concepts = [c for c, _ in concept_counts.most_common(15)]

        # Cross-domain concept pairs to explore
        cross_domain_seeds = [
            ("compressed_sensing", "imputation"),
            ("attention_mechanism", "spatial_patterns"),
            ("diffusion_model", "generation"),
            ("graph_neural_network", "tissue_structure"),
            ("active_inference", "exploration"),
            ("causal_inference", "perturbation"),
            ("optimal_transport", "alignment"),
            ("topological_data_analysis", "shape"),
        ]

        hypotheses = []
        reasoning_mode = "fast" if not use_slow_thinking else "slow_thinking"

        # Find bridges between research area and cross-domain concepts
        for source_concept, target_property in cross_domain_seeds[:num_hypotheses]:
            # Search for source concept
            source_results = hs.search_papers(source_concept, n=10)

            if source_results:
                if use_slow_thinking:
                    # Use slow thinking with step-by-step reasoning
                    # First try LLM if available
                    context = [{"title": r.title, "snippet": r.content[:200]} for r in source_results[:5]]
                    llm_result = await self._llm_slow_think(research_area, source_concept, target_property, context)

                    if llm_result:
                        hypotheses.append(llm_result)
                    else:
                        # Fall back to rule-based slow thinking
                        hypothesis = await self._slow_think_hypothesis(
                            research_area, source_concept, target_property, hs
                        )
                        hypotheses.append(hypothesis)
                else:
                    # Fast mode: simple hypothesis without deep reasoning
                    source_doc = source_results[0].content[:200]

                    hypothesis = {
                        "source_domain": source_concept.replace("_", " "),
                        "target_domain": research_area,
                        "hypothesis": f"Methods from {source_concept.replace('_', ' ')} can be applied to {research_area} for {target_property.replace('_', ' ')}",
                        "rationale": f"Both involve {target_property.replace('_', ' ')} problems with similar mathematical structure",
                        "testability": "high" if source_concept in ["compressed_sensing", "attention_mechanism"] else "medium",
                        "novelty_score": _compute_novelty_score(source_concept, research_area, hs),
                        "reasoning_mode": "fast",
                        "supporting_evidence": source_doc,
                        "next_steps": [
                            f"Search for existing work combining {source_concept} and {research_area}",
                            f"Identify specific {target_property} problem in {research_area}",
                            f"Design proof-of-concept experiment"
                        ]
                    }
                    hypotheses.append(hypothesis)

        return {
            "research_area": research_area,
            "concepts_found": top_concepts[:10],
            "hypotheses": hypotheses,
            "reasoning_mode": reasoning_mode,
            "methodology": "Structure mapping + cross-domain analogy" + (" with slow thinking" if use_slow_thinking else ""),
            "disclaimer": "Hypotheses require validation - search for existing work first"
        }

    async def validate_hypothesis(self, hypothesis: str) -> dict:
        """
        Tool 10: Validate a hypothesis against the corpus.
        """
        hs = self._get_searcher()

        # Search for supporting evidence
        supporting_results = hs.search_papers(hypothesis, n=20)

        # Search for contradicting evidence (negate key terms)
        # Simple heuristic: search for "not" + keywords
        keywords = [w for w in hypothesis.lower().split() if len(w) > 4][:5]

        # Score evidence (SearchResult objects)
        support_score = 0
        contradict_score = 0
        evidence = []

        for r in supporting_results:
            doc_lower = r.content.lower()

            # Check if supports or contradicts
            if any(neg in doc_lower for neg in ["not ", "failed", "unable", "incorrect"]):
                contradict_score += r.score
                evidence.append({
                    "type": "potential_contradiction",
                    "title": r.title,
                    "doi": r.metadata.get("doi"),
                    "relevance": round(r.score, 3),
                    "snippet": r.content[:200]
                })
            else:
                support_score += r.score
                evidence.append({
                    "type": "potential_support",
                    "title": r.title,
                    "doi": r.metadata.get("doi"),
                    "relevance": round(r.score, 3),
                    "snippet": r.content[:200]
                })

        # Calculate confidence
        total_score = support_score + contradict_score
        confidence = support_score / total_score if total_score > 0 else 0.5

        # Identify gaps
        gaps = []
        if len(supporting_results) < 5:
            gaps.append("Limited prior work found - may be novel or poorly phrased")
        if contradict_score > support_score:
            gaps.append("More contradicting than supporting evidence found")

        return {
            "hypothesis": hypothesis,
            "validation_status": "supported" if confidence > 0.6 else "needs_investigation" if confidence > 0.4 else "potentially_contradicted",
            "confidence": round(confidence, 3),
            "evidence_count": len(evidence),
            "supporting_evidence": [e for e in evidence if e["type"] == "potential_support"][:5],
            "contradicting_evidence": [e for e in evidence if e["type"] == "potential_contradiction"][:3],
            "knowledge_gaps": gaps,
            "next_steps": [
                "Read top supporting papers in detail",
                "Check methods sections for experimental validation",
                "Design experiment to test hypothesis directly"
            ]
        }

    async def find_analogy(self, problem: str) -> dict:
        """
        Tool 11: Find analogous solutions from unexpected domains.
        """
        hs = self._get_searcher()

        # Abstract the problem to mathematical/structural terms
        abstractions = {
            "predict": ["regression", "inference", "estimation"],
            "classify": ["discrimination", "categorization", "clustering"],
            "impute": ["reconstruction", "completion", "interpolation"],
            "align": ["registration", "matching", "correspondence"],
            "detect": ["anomaly", "outlier", "novelty"],
            "segment": ["partition", "decomposition", "clustering"],
            "generate": ["synthesis", "sampling", "creation"],
            "compress": ["dimensionality_reduction", "encoding", "summarization"],
        }

        # Find abstract terms in problem
        problem_lower = problem.lower()
        abstract_terms = []
        for verb, abstractions_list in abstractions.items():
            if verb in problem_lower:
                abstract_terms.extend(abstractions_list)

        if not abstract_terms:
            abstract_terms = ["optimization", "inference", "learning"]

        # Search across domains with abstract terms
        analogies = []

        for abstract_term in abstract_terms[:3]:
            # Search in different domains
            for domain in ["physics", "economics", "neuroscience", "ecology", "signal processing"]:
                query = f"{abstract_term} {domain}"
                results = hs.search_papers(query, n=5)

                if results:
                    r = results[0]
                    analogies.append({
                        "domain": domain,
                        "abstract_concept": abstract_term,
                        "paper_title": r.title,
                        "doi": r.metadata.get("doi"),
                        "relevance": round(r.score, 3),
                        "potential_insight": f"{abstract_term.capitalize()} methods from {domain} may apply",
                        "snippet": r.content[:150]
                    })

        # Sort by relevance and diversity
        analogies = sorted(analogies, key=lambda x: x["relevance"], reverse=True)

        # Remove duplicates by domain
        seen_domains = set()
        unique_analogies = []
        for a in analogies:
            if a["domain"] not in seen_domains:
                unique_analogies.append(a)
                seen_domains.add(a["domain"])

        return {
            "original_problem": problem,
            "abstract_structure": abstract_terms,
            "analogies_found": len(unique_analogies),
            "analogies": unique_analogies[:5],
            "recommendation": "Check if techniques from top analogies have been tried in your domain"
        }

    async def serendipity(self, seed_concept: Optional[str] = None) -> dict:
        """
        Tool 12: Surface unexpected but potentially useful connections.

        Returns surprising concept bridges you might not think to look for.
        """
        hs = self._get_searcher()

        # Interesting concept pairs that rarely appear together but have structural similarity
        serendipity_pairs = [
            ("morphogenesis", "neural_network"),
            ("thermodynamics", "information_theory"),
            ("game_theory", "evolution"),
            ("compressed_sensing", "gene_expression"),
            ("origami", "protein_folding"),
            ("swarm_intelligence", "immune_system"),
            ("phase_transition", "learning"),
            ("reaction_diffusion", "pattern_recognition"),
            ("percolation", "epidemiology"),
            ("renormalization", "deep_learning"),
            ("autopoiesis", "metabolism"),
            ("cybernetics", "homeostasis"),
            ("topology", "neural_connectivity"),
            ("optimal_transport", "cell_differentiation"),
        ]

        if seed_concept:
            # Find pairs involving the seed
            relevant_pairs = [(a, b) for a, b in serendipity_pairs if seed_concept.lower() in a.lower() or seed_concept.lower() in b.lower()]
            if not relevant_pairs:
                # Search for papers with seed and find their concepts
                results = hs.search_papers(seed_concept, n=20)
                concepts = []
                for r in results:
                    concepts_raw = r.metadata.get("concepts", "")
                    if isinstance(concepts_raw, str) and concepts_raw:
                        concepts.extend(concepts_raw.split(","))
                    elif concepts_raw:
                        concepts.extend(concepts_raw)

                # Create new pairs
                concept_counts = Counter(concepts)
                top_concepts = [c for c, _ in concept_counts.most_common(5)]
                relevant_pairs = [(seed_concept, c) for c in top_concepts if c != seed_concept]

            selected_pair = relevant_pairs[0] if relevant_pairs else random.choice(serendipity_pairs)
        else:
            selected_pair = random.choice(serendipity_pairs)

        concept_a, concept_b = selected_pair

        # Search for each concept
        results_a = hs.search_papers(concept_a, n=5)
        results_b = hs.search_papers(concept_b, n=5)

        # Search for intersection
        intersection_query = f"{concept_a} {concept_b}"
        intersection_results = hs.search_papers(intersection_query, n=5)

        bridge_strength = "unexplored" if len(intersection_results) < 2 else "emerging" if len(intersection_results) < 5 else "established"

        return {
            "serendipity_bridge": {
                "concept_a": concept_a.replace("_", " "),
                "concept_b": concept_b.replace("_", " "),
                "bridge_strength": bridge_strength,
                "potential_insight": f"What if we viewed {concept_a.replace('_', ' ')} through the lens of {concept_b.replace('_', ' ')}?"
            },
            "concept_a_context": {
                "papers": len(results_a),
                "sample_title": results_a[0].title if results_a else None
            },
            "concept_b_context": {
                "papers": len(results_b),
                "sample_title": results_b[0].title if results_b else None
            },
            "existing_bridges": len(intersection_results),
            "bridge_papers": [
                {"title": r.title, "doi": r.metadata.get("doi")}
                for r in intersection_results[:3]
            ],
            "prompt_for_thinking": [
                f"What mathematical structure is shared by {concept_a} and {concept_b}?",
                f"Has anyone applied {concept_a} methods to {concept_b} problems?",
                f"What would a unified theory of {concept_a} and {concept_b} look like?"
            ],
            "action": "Use find_analogy tool to explore this bridge further" if bridge_strength == "unexplored" else "Read the existing bridge papers"
        }
