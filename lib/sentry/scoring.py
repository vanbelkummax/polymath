#!/usr/bin/env python3
"""
Literature Sentry Scoring Module

Log-normalized cross-domain scoring with:
- Power-law aware normalization (handles citations, stars, views)
- Freshness decay based on source velocity
- Cross-domain bridge detection
- Hidden gem identification
- Trusted lab whitelist
"""

import math
from datetime import date, datetime
from typing import Dict, List, Optional, Set, Any
from dataclasses import dataclass

# ============================================================
# THRESHOLDS (Tuned for niche scientific content)
# ============================================================

SOURCE_THRESHOLDS = {
    "europepmc": {
        "citations": (1, 500),
        "halflife_days": 730,  # Papers decay slowly
    },
    "pubmed": {
        "citations": (1, 500),
        "halflife_days": 730,
    },
    "arxiv": {
        "citations": (1, 50),
        "halflife_days": 180,  # Preprints decay faster
    },
    "biorxiv": {
        "citations": (1, 100),
        "halflife_days": 180,
    },
    "github": {
        "stars": (1, 1000),  # 1000 stars is excellent for niche bio tools
        "forks": (1, 200),
        "halflife_days": 365,
    },
    "youtube": {
        "views": (500, 50000),  # 50k is "viral" for academic lectures
        "halflife_days": 365,
    },
}

# ============================================================
# TRUSTED LAB WHITELIST (Auto high score)
# ============================================================

TRUSTED_GITHUB_ORGS = {
    "mahmoodlab",       # Harvard Comp Path
    "ken-lau-lab",      # Vanderbilt Spatial
    "theislab",         # Helmholtz single-cell
    "hrlblab",          # Huo Lab
    "masilab",          # MASI neuroimaging
    "satijalab",        # Seurat
    "scverse",          # scanpy ecosystem
    "owkin",            # Owkin
    "deepchem",         # DeepChem
    "stanfordnlp",      # Stanford NLP (DSPy)
    "pyg-team",         # PyTorch Geometric
    "huggingface",      # HuggingFace
    "anthropics",       # Anthropic
    "openai",           # OpenAI
}

TRUSTED_AUTHORS = {
    # Comp Path
    "faisal mahmood",
    "richard chen",
    # Spatial Biology
    "ken lau",
    "bob coffey",
    "fabian theis",
    "oliver stegle",
    # Your Vanderbilt targets
    "yuankai huo",
    "tae hyun hwang",
    "bennett landman",
    # Foundational
    "yoshua bengio",
    "geoffrey hinton",
    "yann lecun",
}

# ============================================================
# CROSS-DOMAIN CONCEPT SETS
# ============================================================

CONCEPT_DOMAINS = {
    "signal_processing": {
        "compressed_sensing", "sparse_coding", "wavelet", "fourier",
        "information_bottleneck", "denoising", "reconstruction",
    },
    "physics": {
        "entropy", "free_energy", "thermodynamics", "diffusion",
        "reaction_diffusion", "statistical_mechanics", "boltzmann",
    },
    "causality": {
        "causal_inference", "counterfactual", "do_calculus",
        "instrumental_variable", "confounding", "mediation",
    },
    "systems": {
        "feedback", "control_theory", "homeostasis", "autopoiesis",
        "emergence", "self_organization", "cybernetics",
    },
    "cognitive": {
        "predictive_coding", "bayesian_brain", "active_inference",
        "embodied_cognition", "affordance", "enactivism",
    },
    "geometry": {
        "graph_neural_network", "topology", "manifold",
        "geometric_deep_learning", "topological_data_analysis",
    },
    "ml_core": {
        "neural_network", "transformer", "attention", "embedding",
        "contrastive_learning", "self_supervised", "foundation_model",
    },
    "biology": {
        "spatial_transcriptomics", "single_cell", "gene_expression",
        "morphogenesis", "gene_regulatory_network", "epigenetics",
    },
}


@dataclass
class ScoredItem:
    """A discovered item with computed scores."""
    external_id: str
    source: str
    title: str
    url: str
    authors: List[str]
    published_date: Optional[date]
    raw_metrics: Dict[str, Any]

    # Computed scores
    priority_score: float = 0.0
    popularity_score: float = 0.0
    freshness_score: float = 0.0
    bridge_score: float = 0.0
    gap_fill_score: float = 0.0
    author_trust_score: float = 0.0

    # Flags
    is_hidden_gem: bool = False
    is_trusted_lab: bool = False
    concept_domains: List[str] = None

    def __post_init__(self):
        if self.concept_domains is None:
            self.concept_domains = []


class Scorer:
    """
    Cross-domain scorer for Literature Sentry.

    Uses log-normalization to handle power-law distributions
    and weights multiple signals for polymathic discovery.

    Includes Expected Information Gain (EIG) for Active Inference:
    Papers that resolve uncertainty about open questions get boosted.
    """

    # Score weights (must sum to 1.0)
    WEIGHTS = {
        "popularity": 0.15,
        "freshness": 0.10,
        "bridge_score": 0.25,  # Cross-domain is key
        "gap_fill": 0.15,
        "author_trust": 0.15,
        "eig_score": 0.20,  # NEW: Expected Information Gain
    }

    # Open questions for EIG scoring (imported from config or defined here)
    OPEN_QUESTIONS = {
        "h_and_e_signal_sufficiency": {
            "question": "Does H&E contain sufficient high-frequency signal for RNA inference?",
            "related_concepts": {"spatial_transcriptomics", "compressed_sensing", "information_bottleneck", "wavelet"},
            "uncertainty": 0.7,
        },
        "optimal_encoder_architecture": {
            "question": "What encoder architecture best captures morphological-transcriptomic mapping?",
            "related_concepts": {"transformer", "graph_neural_network", "attention", "foundation_model", "contrastive_learning"},
            "uncertainty": 0.5,
        },
        "colibactin_detection_sensitivity": {
            "question": "What is the detection limit for pks+ E. coli in metagenomes?",
            "related_concepts": {"gene_regulatory_network", "metagenomics", "sparse_coding"},
            "uncertainty": 0.6,
        },
        "cell_type_deconvolution": {
            "question": "How to accurately deconvolve cell types from 2um spots?",
            "related_concepts": {"single_cell", "spatial_transcriptomics", "sparse_coding", "bayesian_brain"},
            "uncertainty": 0.8,
        },
        "active_inference_biology": {
            "question": "Can cells be modeled as agents minimizing free energy?",
            "related_concepts": {"active_inference", "free_energy", "autopoiesis", "cybernetics", "predictive_coding"},
            "uncertainty": 0.9,
        },
    }

    def __init__(self, existing_concepts: Set[str] = None, concept_counts: Dict[str, int] = None):
        """
        Initialize scorer.

        Args:
            existing_concepts: Concepts already in the knowledge graph
            concept_counts: Count of papers per concept (for gap detection)
        """
        self.existing_concepts = existing_concepts or set()
        self.concept_counts = concept_counts or {}

        # Build reverse lookup: concept -> domain
        self.concept_to_domain = {}
        for domain, concepts in CONCEPT_DOMAINS.items():
            for concept in concepts:
                self.concept_to_domain[concept] = domain

    def score(self, item: Dict, query: str = "") -> ScoredItem:
        """
        Compute priority score for a discovered item.

        Args:
            item: Raw item from source API
            query: Original search query (for relevance scoring)

        Returns:
            ScoredItem with computed scores
        """
        source = item.get("source", "unknown")

        # Create scored item
        scored = ScoredItem(
            external_id=item.get("external_id", ""),
            source=source,
            title=item.get("title", ""),
            url=item.get("url", ""),
            authors=item.get("authors", []),
            published_date=self._parse_date(item.get("published_at")),
            raw_metrics=item,
        )

        # Check trusted lab first
        scored.is_trusted_lab = self._is_trusted_lab(item)
        if scored.is_trusted_lab:
            # Trusted labs get auto high score
            scored.priority_score = 0.95
            scored.author_trust_score = 1.0
            return scored

        # Extract concepts from title/abstract
        text = f"{item.get('title', '')} {item.get('abstract', '')}"
        detected_concepts = self._extract_concepts(text.lower())
        scored.concept_domains = list(detected_concepts)

        # Compute individual scores
        scored.popularity_score = self._compute_popularity(item, source)
        scored.freshness_score = self._compute_freshness(scored.published_date, source)
        scored.bridge_score = self._compute_bridge_score(detected_concepts)
        scored.gap_fill_score = self._compute_gap_fill(detected_concepts)
        scored.author_trust_score = self._compute_author_trust(item.get("authors", []))

        # NEW: Expected Information Gain (Active Inference)
        eig_score = self._compute_eig(detected_concepts, item)

        # Weighted combination
        scored.priority_score = (
            self.WEIGHTS["popularity"] * scored.popularity_score +
            self.WEIGHTS["freshness"] * scored.freshness_score +
            self.WEIGHTS["bridge_score"] * scored.bridge_score +
            self.WEIGHTS["gap_fill"] * scored.gap_fill_score +
            self.WEIGHTS["author_trust"] * scored.author_trust_score +
            self.WEIGHTS["eig_score"] * eig_score
        )

        # Hidden gem detection
        scored.is_hidden_gem = self._is_hidden_gem(scored, item)

        return scored

    def _normalize_log(self, value: float, source: str, metric: str) -> float:
        """
        Logarithmic normalization for power-law distributions.

        This compresses outliers and spreads out the "good but not viral" tier.

        Examples (for GitHub stars, max=1000):
            10 stars -> 0.33
            100 stars -> 0.67
            1000 stars -> 1.0
        """
        if source not in SOURCE_THRESHOLDS:
            return 0.0
        if metric not in SOURCE_THRESHOLDS[source]:
            return 0.0

        low, high = SOURCE_THRESHOLDS[source][metric]

        # Avoid log(0)
        val_safe = max(value, 1)
        low_safe = max(low, 1)

        # Log transform
        log_val = math.log10(val_safe)
        log_low = math.log10(low_safe)
        log_high = math.log10(high)

        if log_high <= log_low:
            return 0.0

        norm = (log_val - log_low) / (log_high - log_low)
        return min(1.0, max(0.0, norm))

    def _compute_popularity(self, item: Dict, source: str) -> float:
        """Compute normalized popularity score."""
        if source in ("europepmc", "pubmed", "arxiv", "biorxiv"):
            citations = item.get("citations", item.get("cited_by_count", 0)) or 0
            return self._normalize_log(citations, source, "citations")

        elif source == "github":
            stars = item.get("stars", item.get("stargazers_count", 0)) or 0
            return self._normalize_log(stars, source, "stars")

        elif source == "youtube":
            views = item.get("views", item.get("view_count", 0)) or 0
            return self._normalize_log(views, source, "views")

        return 0.0

    def _compute_freshness(self, pub_date: Optional[date], source: str) -> float:
        """
        Exponential decay based on source velocity.

        Uses radioactive decay formula: score = 2^(-days/halflife)
        """
        if pub_date is None:
            return 0.5  # Default for unknown dates

        if source not in SOURCE_THRESHOLDS:
            return 0.5

        halflife = SOURCE_THRESHOLDS[source].get("halflife_days", 365)

        today = date.today()
        if isinstance(pub_date, datetime):
            pub_date = pub_date.date()

        days_old = (today - pub_date).days

        if days_old < 0:
            return 1.0  # Future date = fresh

        return 2 ** (-days_old / halflife)

    def _compute_bridge_score(self, concepts: Set[str]) -> float:
        """
        Score based on spanning multiple concept domains.

        Papers that bridge domains (e.g., physics + ML) score higher.
        """
        if not concepts:
            return 0.0

        # Find which domains are covered
        domains_covered = set()
        for concept in concepts:
            if concept in self.concept_to_domain:
                domains_covered.add(self.concept_to_domain[concept])

        # Score based on number of domains
        num_domains = len(domains_covered)

        if num_domains == 0:
            return 0.0
        elif num_domains == 1:
            return 0.2  # Single domain
        elif num_domains == 2:
            return 0.6  # Bridge!
        else:
            return 1.0  # Multi-domain polymathic gold

    def _compute_gap_fill(self, concepts: Set[str]) -> float:
        """
        Score based on filling gaps in existing knowledge graph.

        Concepts with low representation get higher scores.
        """
        if not concepts or not self.concept_counts:
            return 0.5  # Neutral if no data

        gap_scores = []
        for concept in concepts:
            count = self.concept_counts.get(concept, 0)
            # Low count = high gap score
            if count == 0:
                gap_scores.append(1.0)
            elif count < 5:
                gap_scores.append(0.8)
            elif count < 20:
                gap_scores.append(0.4)
            else:
                gap_scores.append(0.1)

        return sum(gap_scores) / len(gap_scores) if gap_scores else 0.5

    def _compute_author_trust(self, authors: List[str]) -> float:
        """Score based on trusted author list."""
        if not authors:
            return 0.0

        for author in authors:
            if author.lower() in TRUSTED_AUTHORS:
                return 1.0

        return 0.0

    def _compute_eig(self, concepts: Set[str], item: Dict) -> float:
        """
        Compute Expected Information Gain (Active Inference).

        A paper that addresses open questions with high uncertainty
        gets a massive score boost - this is the "polymathic edge".

        EIG = sum over questions of: P(resolves_question) * uncertainty

        Where P(resolves_question) is estimated by concept overlap.
        """
        if not concepts:
            return 0.0

        eig_scores = []

        for question_id, question_data in self.OPEN_QUESTIONS.items():
            related_concepts = question_data["related_concepts"]
            uncertainty = question_data["uncertainty"]

            # Calculate concept overlap
            overlap = concepts & related_concepts
            if not overlap:
                continue

            # P(resolves) proportional to overlap ratio
            overlap_ratio = len(overlap) / len(related_concepts)

            # EIG for this question
            question_eig = overlap_ratio * uncertainty

            eig_scores.append(question_eig)

        if not eig_scores:
            return 0.0

        # Return max EIG (the question this paper most likely resolves)
        # Alternatively could sum, but max prevents dilution
        return max(eig_scores)

    def _is_trusted_lab(self, item: Dict) -> bool:
        """Check if item is from a trusted lab/org."""
        source = item.get("source", "")

        if source == "github":
            # Check org/owner
            owner = item.get("owner", item.get("repo_owner", "")).lower()
            full_name = item.get("full_name", "").lower()

            for org in TRUSTED_GITHUB_ORGS:
                if org in owner or org in full_name:
                    return True

        # Check authors for trusted affiliations
        authors = item.get("authors", [])
        for author in authors:
            if isinstance(author, str) and author.lower() in TRUSTED_AUTHORS:
                return True

        return False

    def _is_hidden_gem(self, scored: ScoredItem, item: Dict) -> bool:
        """
        Detect hidden gems: low popularity but high relevance.

        These are papers/repos that haven't gotten attention yet
        but are semantically relevant to the query.
        """
        # Low popularity threshold
        if scored.popularity_score > 0.2:
            return False

        # High bridge or gap-fill score
        high_relevance = (
            scored.bridge_score > 0.5 or
            scored.gap_fill_score > 0.6
        )

        # Or: recently published (< 90 days) with any detected concepts
        is_fresh = scored.freshness_score > 0.7
        has_concepts = len(scored.concept_domains) > 0

        return high_relevance or (is_fresh and has_concepts)

    def _extract_concepts(self, text: str) -> Set[str]:
        """Extract known concepts from text."""
        found = set()
        text_lower = text.lower()

        # Check all known concepts
        all_concepts = set()
        for concepts in CONCEPT_DOMAINS.values():
            all_concepts.update(concepts)

        for concept in all_concepts:
            # Handle underscore vs space
            variants = [
                concept,
                concept.replace("_", " "),
                concept.replace("_", "-"),
            ]
            for variant in variants:
                if variant in text_lower:
                    found.add(concept)
                    break

        return found

    def _parse_date(self, date_val) -> Optional[date]:
        """Parse various date formats."""
        if date_val is None:
            return None

        if isinstance(date_val, date):
            return date_val

        if isinstance(date_val, datetime):
            return date_val.date()

        if isinstance(date_val, str):
            # Try common formats
            for fmt in ("%Y-%m-%d", "%Y-%m-%dT%H:%M:%S", "%Y-%m-%d %H:%M:%S", "%Y"):
                try:
                    return datetime.strptime(date_val[:len(fmt.replace("%", ""))], fmt).date()
                except ValueError:
                    continue

        return None


def normalize_concept_name(name: str) -> str:
    """
    Normalize concept names to snake_case.

    Prevents "tag soup" where PubMed gives "Spatial Transcriptomics"
    and GitHub gives "spatial-transcriptomics".
    """
    # Lowercase
    name = name.lower()
    # Replace common separators with underscore
    for sep in [" ", "-", ".", "/"]:
        name = name.replace(sep, "_")
    # Remove non-alphanumeric except underscore
    name = "".join(c for c in name if c.isalnum() or c == "_")
    # Collapse multiple underscores
    while "__" in name:
        name = name.replace("__", "_")
    # Strip leading/trailing
    return name.strip("_")
