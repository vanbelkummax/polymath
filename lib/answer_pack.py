#!/usr/bin/env python3
"""
Answer Pack Generator

Produces structured, auditable responses with:
- Claims with confidence scores
- Supporting and refuting evidence
- Reasoning chains (chain-of-thought)
- Uncertainty quantification
- Counterfactuals
- Next action recommendations

This is the core output format for Polymath v2.0.
"""

import json
import logging
import os
from dataclasses import dataclass, field, asdict
from datetime import datetime
from typing import List, Dict, Optional, Any
from enum import Enum
import uuid

# Local imports
from lib.hybrid_search_v2 import HybridSearcherV2
from lib.db import get_db_connection

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ActionType(Enum):
    EXPERIMENT = "experiment"
    CODE = "code"
    READ = "read"
    COLLABORATE = "collaborate"
    INGEST = "ingest"
    VALIDATE = "validate"


@dataclass
class Evidence:
    """A piece of supporting or refuting evidence"""
    passage_id: str
    text: str
    source: str  # "Author et al., YYYY"
    doi: Optional[str] = None
    relevance_score: float = 0.0
    stance: str = "neutral"  # supporting, refuting, neutral


@dataclass
class ReasoningStep:
    """A step in a chain-of-thought"""
    step_num: int
    statement: str
    grounded_in: Optional[str] = None  # passage_id or None if inference


@dataclass
class Claim:
    """A claim with evidence and reasoning"""
    claim_id: str
    statement: str
    confidence: float  # 0-1
    supporting_evidence: List[Evidence] = field(default_factory=list)
    refuting_evidence: List[Evidence] = field(default_factory=list)
    reasoning_chain: List[ReasoningStep] = field(default_factory=list)
    uncertainty: Dict[str, Any] = field(default_factory=dict)


@dataclass
class NextAction:
    """A recommended next action"""
    action_type: ActionType
    description: str
    priority: str = "medium"  # high, medium, low
    details: Dict[str, Any] = field(default_factory=dict)


@dataclass
class Counterfactual:
    """A counterfactual analysis"""
    assumption: str
    if_wrong: str
    alternative_interpretation: Optional[str] = None


@dataclass
class AnswerPack:
    """Complete structured answer"""
    pack_id: str
    query: str
    created_at: str
    claims: List[Claim] = field(default_factory=list)
    next_actions: List[NextAction] = field(default_factory=list)
    counterfactuals: List[Counterfactual] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict:
        """Convert to dictionary for serialization"""
        return {
            "pack_id": self.pack_id,
            "query": self.query,
            "created_at": self.created_at,
            "claims": [
                {
                    "claim_id": c.claim_id,
                    "statement": c.statement,
                    "confidence": c.confidence,
                    "supporting_evidence": [
                        {
                            "passage_id": e.passage_id,
                            "text": e.text,
                            "source": e.source,
                            "doi": e.doi,
                            "relevance_score": e.relevance_score
                        }
                        for e in c.supporting_evidence
                    ],
                    "refuting_evidence": [
                        {
                            "passage_id": e.passage_id,
                            "text": e.text,
                            "source": e.source,
                            "doi": e.doi,
                            "relevance_score": e.relevance_score
                        }
                        for e in c.refuting_evidence
                    ],
                    "reasoning_chain": [
                        {"step": s.step_num, "statement": s.statement, "grounded_in": s.grounded_in}
                        for s in c.reasoning_chain
                    ],
                    "uncertainty": c.uncertainty
                }
                for c in self.claims
            ],
            "next_actions": [
                {
                    "type": a.action_type.value,
                    "description": a.description,
                    "priority": a.priority,
                    "details": a.details
                }
                for a in self.next_actions
            ],
            "counterfactuals": [
                {
                    "assumption": cf.assumption,
                    "if_wrong": cf.if_wrong,
                    "alternative": cf.alternative_interpretation
                }
                for cf in self.counterfactuals
            ],
            "metadata": self.metadata
        }

    def to_json(self) -> str:
        """Convert to JSON string"""
        return json.dumps(self.to_dict(), indent=2)

    def to_markdown(self) -> str:
        """Convert to readable markdown"""
        lines = []
        lines.append(f"# Answer Pack: {self.pack_id[:8]}")
        lines.append(f"\n**Query:** {self.query}")
        lines.append(f"**Generated:** {self.created_at}")

        lines.append("\n---\n")
        lines.append("## Claims\n")

        for i, claim in enumerate(self.claims, 1):
            conf_bar = "█" * int(claim.confidence * 10) + "░" * (10 - int(claim.confidence * 10))
            lines.append(f"### Claim {i}: {claim.statement}")
            lines.append(f"**Confidence:** {conf_bar} {claim.confidence:.0%}")

            if claim.supporting_evidence:
                lines.append("\n**Supporting Evidence:**")
                for e in claim.supporting_evidence[:3]:
                    lines.append(f"- [{e.source}] \"{e.text[:200]}...\"")

            if claim.refuting_evidence:
                lines.append("\n**Refuting Evidence:**")
                for e in claim.refuting_evidence[:2]:
                    lines.append(f"- [{e.source}] \"{e.text[:200]}...\"")

            if claim.reasoning_chain:
                lines.append("\n**Reasoning Chain:**")
                for step in claim.reasoning_chain:
                    grounding = f" (from {step.grounded_in[:8]}...)" if step.grounded_in else " (inference)"
                    lines.append(f"{step.step_num}. {step.statement}{grounding}")

            if claim.uncertainty:
                lines.append("\n**Uncertainty:**")
                if "what_would_change_my_mind" in claim.uncertainty:
                    lines.append(f"- Would change if: {claim.uncertainty['what_would_change_my_mind']}")
                if "missing_evidence" in claim.uncertainty:
                    lines.append(f"- Missing: {claim.uncertainty['missing_evidence']}")

            lines.append("")

        if self.counterfactuals:
            lines.append("## Counterfactuals\n")
            for cf in self.counterfactuals:
                lines.append(f"**If** {cf.assumption} **is wrong:**")
                lines.append(f"- {cf.if_wrong}")
                if cf.alternative_interpretation:
                    lines.append(f"- Alternative: {cf.alternative_interpretation}")
                lines.append("")

        if self.next_actions:
            lines.append("## Recommended Actions\n")
            for action in self.next_actions:
                icon = {"experiment": "🧪", "code": "💻", "read": "📖", "collaborate": "🤝", "ingest": "📥", "validate": "✅"}.get(action.action_type.value, "•")
                lines.append(f"{icon} **[{action.priority.upper()}]** {action.description}")
            lines.append("")

        if self.metadata:
            lines.append("---")
            lines.append(f"*Retrieved {self.metadata.get('passages_examined', 0)} passages in {self.metadata.get('time_ms', 0)}ms*")

        return "\n".join(lines)


class AnswerPackGenerator:
    """Generates Answer Packs from queries"""

    def __init__(self, llm_client=None):
        """
        Initialize the generator.

        Args:
            llm_client: Optional LLM client (Anthropic, OpenAI, etc.)
                       If None, uses rule-based extraction only.
        """
        self.searcher = HybridSearcherV2()
        self.llm = llm_client
        self.conn = get_db_connection()

    def generate(
        self,
        query: str,
        n_passages: int = 20,
        include_code: bool = False,
        use_llm: bool = True
    ) -> AnswerPack:
        """
        Generate an Answer Pack for a query.

        Args:
            query: The user's question
            n_passages: Number of passages to retrieve
            include_code: Whether to include code search
            use_llm: Whether to use LLM for reasoning (requires llm_client)

        Returns:
            AnswerPack with claims, evidence, reasoning, etc.
        """
        import time
        start_time = time.time()

        # Create pack
        pack = AnswerPack(
            pack_id=str(uuid.uuid4()),
            query=query,
            created_at=datetime.now().isoformat()
        )

        # Retrieve passages
        if include_code:
            results = self.searcher.hybrid_search(query, n=n_passages, rerank=True)
        else:
            results = self.searcher.search_papers(query, n=n_passages, rerank=True)

        passages_examined = len(results)

        # Convert to evidence
        evidence_list = []
        for r in results:
            source = f"{r.get('authors', 'Unknown')[:30]}, {r.get('year', 'N/A')}"
            evidence = Evidence(
                passage_id=r.get("id", ""),
                text=r.get("passage_text", r.get("text", ""))[:500],
                source=source,
                doi=r.get("doi"),
                relevance_score=r.get("score", 0.0)
            )
            evidence_list.append(evidence)

        # Use LLM for sophisticated reasoning, or fall back to heuristics
        if use_llm and self.llm:
            pack = self._generate_with_llm(pack, query, evidence_list)
        else:
            pack = self._generate_heuristic(pack, query, evidence_list)

        # Add metadata
        elapsed_ms = int((time.time() - start_time) * 1000)
        pack.metadata = {
            "passages_examined": passages_examined,
            "time_ms": elapsed_ms,
            "retrieval_sources": ["chromadb", "postgres_fts"],
            "model": "heuristic" if not use_llm else (self.llm.__class__.__name__ if self.llm else "none")
        }

        return pack

    def _generate_heuristic(
        self,
        pack: AnswerPack,
        query: str,
        evidence_list: List[Evidence]
    ) -> AnswerPack:
        """Generate answer pack using heuristic rules (no LLM)"""

        if not evidence_list:
            claim = Claim(
                claim_id=str(uuid.uuid4()),
                statement="No relevant evidence found for this query.",
                confidence=0.0,
                uncertainty={"missing_evidence": "Need more papers in corpus on this topic"}
            )
            pack.claims.append(claim)
            pack.next_actions.append(NextAction(
                action_type=ActionType.INGEST,
                description="Search for and ingest papers on this topic",
                priority="high"
            ))
            return pack

        # Create a main claim based on evidence coverage
        top_evidence = evidence_list[:5]
        avg_score = sum(e.relevance_score for e in top_evidence) / len(top_evidence)

        claim = Claim(
            claim_id=str(uuid.uuid4()),
            statement=f"Based on {len(evidence_list)} retrieved passages about: {query[:50]}...",
            confidence=min(0.9, avg_score * 1.2),  # Scale relevance to confidence
            supporting_evidence=top_evidence,
            refuting_evidence=[],  # Would need NLI to detect
            reasoning_chain=[
                ReasoningStep(1, f"Retrieved {len(evidence_list)} relevant passages"),
                ReasoningStep(2, f"Top passage relevance score: {top_evidence[0].relevance_score:.2f}"),
                ReasoningStep(3, f"Evidence spans years: {self._extract_year_range(evidence_list)}")
            ],
            uncertainty={
                "what_would_change_my_mind": "Finding contradictory evidence in literature",
                "missing_evidence": "May need more recent papers (check preprints)"
            }
        )
        pack.claims.append(claim)

        # Add actions
        pack.next_actions = [
            NextAction(
                action_type=ActionType.READ,
                description=f"Deep read top paper: {top_evidence[0].source}",
                priority="high"
            ),
            NextAction(
                action_type=ActionType.VALIDATE,
                description="Verify claims against original papers",
                priority="medium"
            )
        ]

        # Add counterfactual
        pack.counterfactuals = [
            Counterfactual(
                assumption="Retrieved passages are representative of the field",
                if_wrong="May be missing important perspectives not in corpus",
                alternative_interpretation="Consider searching preprint servers and conference proceedings"
            )
        ]

        return pack

    def _generate_with_llm(
        self,
        pack: AnswerPack,
        query: str,
        evidence_list: List[Evidence]
    ) -> AnswerPack:
        """Generate answer pack using LLM for reasoning"""

        # Build context from evidence
        context = "\n\n".join([
            f"[{i+1}] {e.source}: {e.text}"
            for i, e in enumerate(evidence_list[:10])
        ])

        prompt = f"""Based on the following evidence, answer the query with structured reasoning.

QUERY: {query}

EVIDENCE:
{context}

Provide your response as JSON with this structure:
{{
    "claims": [
        {{
            "statement": "Your claim",
            "confidence": 0.0-1.0,
            "supporting_evidence_indices": [1, 2],  // Which evidence supports this
            "refuting_evidence_indices": [],  // Which evidence contradicts this
            "reasoning": ["Step 1...", "Step 2..."],
            "what_would_change_mind": "...",
            "missing_evidence": "..."
        }}
    ],
    "counterfactuals": [
        {{
            "assumption": "Key assumption made",
            "if_wrong": "What changes if wrong",
            "alternative": "Alternative interpretation"
        }}
    ],
    "next_actions": [
        {{"type": "experiment|code|read|collaborate", "description": "...", "priority": "high|medium|low"}}
    ]
}}"""

        try:
            # Call LLM (implementation depends on client type)
            response = self._call_llm(prompt)
            parsed = json.loads(response)

            # Convert to dataclasses
            for claim_data in parsed.get("claims", []):
                supporting = [
                    evidence_list[i-1] for i in claim_data.get("supporting_evidence_indices", [])
                    if 0 < i <= len(evidence_list)
                ]
                refuting = [
                    evidence_list[i-1] for i in claim_data.get("refuting_evidence_indices", [])
                    if 0 < i <= len(evidence_list)
                ]
                reasoning = [
                    ReasoningStep(j+1, step)
                    for j, step in enumerate(claim_data.get("reasoning", []))
                ]

                claim = Claim(
                    claim_id=str(uuid.uuid4()),
                    statement=claim_data.get("statement", ""),
                    confidence=claim_data.get("confidence", 0.5),
                    supporting_evidence=supporting,
                    refuting_evidence=refuting,
                    reasoning_chain=reasoning,
                    uncertainty={
                        "what_would_change_my_mind": claim_data.get("what_would_change_mind", ""),
                        "missing_evidence": claim_data.get("missing_evidence", "")
                    }
                )
                pack.claims.append(claim)

            for cf_data in parsed.get("counterfactuals", []):
                pack.counterfactuals.append(Counterfactual(
                    assumption=cf_data.get("assumption", ""),
                    if_wrong=cf_data.get("if_wrong", ""),
                    alternative_interpretation=cf_data.get("alternative")
                ))

            for action_data in parsed.get("next_actions", []):
                action_type = ActionType(action_data.get("type", "read"))
                pack.next_actions.append(NextAction(
                    action_type=action_type,
                    description=action_data.get("description", ""),
                    priority=action_data.get("priority", "medium")
                ))

        except Exception as e:
            logger.error(f"LLM generation failed: {e}")
            # Fall back to heuristic
            return self._generate_heuristic(pack, query, evidence_list)

        return pack

    def _call_llm(self, prompt: str) -> str:
        """Call the LLM and return response text"""
        if hasattr(self.llm, 'messages'):
            # Anthropic client
            response = self.llm.messages.create(
                model="claude-sonnet-4-20250514",
                max_tokens=2000,
                messages=[{"role": "user", "content": prompt}]
            )
            return response.content[0].text
        elif hasattr(self.llm, 'chat'):
            # OpenAI client
            response = self.llm.chat.completions.create(
                model="gpt-4",
                messages=[{"role": "user", "content": prompt}]
            )
            return response.choices[0].message.content
        else:
            raise ValueError("Unknown LLM client type")

    def _extract_year_range(self, evidence_list: List[Evidence]) -> str:
        """Extract year range from evidence sources"""
        years = []
        for e in evidence_list:
            # Try to extract year from source string
            import re
            match = re.search(r'\b(19|20)\d{2}\b', e.source)
            if match:
                years.append(int(match.group()))

        if years:
            return f"{min(years)}-{max(years)}"
        return "unknown"

    def save_pack(self, pack: AnswerPack, filepath: str = None) -> str:
        """Save answer pack to file and optionally ingest back"""
        if filepath is None:
            os.makedirs("data/answer_packs", exist_ok=True)
            filepath = f"data/answer_packs/{pack.pack_id}.json"

        with open(filepath, "w") as f:
            f.write(pack.to_json())

        # Also save markdown version
        md_path = filepath.replace(".json", ".md")
        with open(md_path, "w") as f:
            f.write(pack.to_markdown())

        logger.info(f"Saved Answer Pack to {filepath}")
        return filepath


# Convenience function
def generate_answer(query: str, **kwargs) -> AnswerPack:
    """Quick function to generate an Answer Pack"""
    generator = AnswerPackGenerator()
    return generator.generate(query, **kwargs)


if __name__ == "__main__":
    import sys

    if len(sys.argv) < 2:
        print("Usage: python answer_pack.py 'your query here'")
        sys.exit(1)

    query = " ".join(sys.argv[1:])
    pack = generate_answer(query, n_passages=15)
    print(pack.to_markdown())
