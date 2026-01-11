#!/usr/bin/env python3
"""
Hackathon Mode - Rapid Polymathic Problem Solving

Purpose: Take a hackathon topic and rapidly:
1. Understand the problem space
2. Find prior art and existing solutions
3. Identify cross-domain insights
4. Generate novel approaches
5. Find code to build on
6. Create an implementation plan

This is designed for speed - getting to a differentiated solution FAST.

Usage:
    python3 -m lib.hackathon_mode "topic: predict protein folding from sequence"
    python3 -m lib.hackathon_mode --interactive
"""

import json
import logging
import os
import sys
import time
from dataclasses import dataclass, field
from datetime import datetime
from typing import List, Dict, Optional, Any
from pathlib import Path

# Local imports
from lib.hybrid_search_v2 import HybridSearcherV2
from lib.answer_pack import AnswerPackGenerator, AnswerPack
from lib.db import get_db_connection

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class PriorArt:
    """Existing work in the problem space"""
    title: str
    source: str
    approach: str
    limitations: List[str]
    relevance: float


@dataclass
class CrossDomainInsight:
    """Insight from another domain"""
    source_domain: str
    target_application: str
    insight: str
    evidence: str
    novelty_score: float


@dataclass
class CodeAsset:
    """Reusable code found"""
    repo: str
    file: str
    description: str
    language: str
    quality_score: float
    how_to_use: str


@dataclass
class SolutionApproach:
    """A potential solution approach"""
    name: str
    description: str
    novelty: str  # what's new
    feasibility: str  # time/complexity estimate
    differentiation: str  # why this wins
    risks: List[str]
    priority: int  # 1-5, lower is better


@dataclass
class ImplementationPlan:
    """Plan to build the solution"""
    approach: SolutionApproach
    steps: List[str]
    code_assets: List[CodeAsset]
    estimated_hours: int
    mvp_scope: str
    demo_pitch: str


@dataclass
class HackathonBrief:
    """Complete hackathon analysis"""
    topic: str
    created_at: str
    time_spent_seconds: int

    # Problem understanding
    problem_reframe: str
    key_challenges: List[str]
    success_metrics: List[str]

    # Research
    prior_art: List[PriorArt]
    cross_domain_insights: List[CrossDomainInsight]
    gaps_in_literature: List[str]

    # Code resources
    code_assets: List[CodeAsset]

    # Solutions
    approaches: List[SolutionApproach]
    recommended_approach: Optional[SolutionApproach]
    implementation_plan: Optional[ImplementationPlan]

    # Meta
    polymathic_advantage: str  # What we bring that others don't

    def to_markdown(self) -> str:
        """Convert to presentation-ready markdown"""
        lines = []
        lines.append(f"# 🏆 Hackathon Brief: {self.topic}")
        lines.append(f"\n*Generated in {self.time_spent_seconds}s on {self.created_at}*\n")

        lines.append("---\n")
        lines.append("## 🎯 Problem Reframe\n")
        lines.append(self.problem_reframe)

        lines.append("\n### Key Challenges")
        for c in self.key_challenges:
            lines.append(f"- {c}")

        lines.append("\n### Success Metrics")
        for m in self.success_metrics:
            lines.append(f"- {m}")

        lines.append("\n---\n")
        lines.append("## 📚 Prior Art\n")
        for art in self.prior_art[:5]:
            lines.append(f"### {art.title}")
            lines.append(f"*{art.source}* | Relevance: {art.relevance:.0%}")
            lines.append(f"\n**Approach:** {art.approach}")
            if art.limitations:
                lines.append("\n**Limitations:**")
                for lim in art.limitations:
                    lines.append(f"- {lim}")
            lines.append("")

        if self.gaps_in_literature:
            lines.append("### 🔍 Gaps We Can Exploit")
            for gap in self.gaps_in_literature:
                lines.append(f"- **{gap}**")
            lines.append("")

        lines.append("---\n")
        lines.append("## 💡 Cross-Domain Insights\n")
        lines.append("*This is our polymathic advantage*\n")
        for insight in self.cross_domain_insights[:3]:
            lines.append(f"### From {insight.source_domain} → {insight.target_application}")
            lines.append(f"**Insight:** {insight.insight}")
            lines.append(f"**Evidence:** {insight.evidence}")
            lines.append(f"**Novelty:** {'⭐' * int(insight.novelty_score * 5)}")
            lines.append("")

        lines.append("---\n")
        lines.append("## 💻 Code Assets\n")
        for code in self.code_assets[:5]:
            lines.append(f"### [{code.repo}]({code.file})")
            lines.append(f"*{code.language}* | Quality: {'⭐' * int(code.quality_score * 5)}")
            lines.append(f"\n{code.description}")
            lines.append(f"\n**How to use:** {code.how_to_use}")
            lines.append("")

        lines.append("---\n")
        lines.append("## 🚀 Solution Approaches\n")
        for i, approach in enumerate(self.approaches, 1):
            star = "⭐ " if approach == self.recommended_approach else ""
            lines.append(f"### {star}Option {i}: {approach.name}")
            lines.append(f"\n{approach.description}")
            lines.append(f"\n**Novelty:** {approach.novelty}")
            lines.append(f"**Feasibility:** {approach.feasibility}")
            lines.append(f"**Differentiation:** {approach.differentiation}")
            if approach.risks:
                lines.append("\n**Risks:**")
                for risk in approach.risks:
                    lines.append(f"- {risk}")
            lines.append("")

        if self.implementation_plan:
            lines.append("---\n")
            lines.append("## 📋 Implementation Plan\n")
            lines.append(f"**Approach:** {self.implementation_plan.approach.name}")
            lines.append(f"**Estimated Time:** {self.implementation_plan.estimated_hours}h")
            lines.append(f"\n### MVP Scope\n{self.implementation_plan.mvp_scope}")
            lines.append("\n### Steps")
            for i, step in enumerate(self.implementation_plan.steps, 1):
                lines.append(f"{i}. {step}")

            lines.append("\n### Demo Pitch")
            lines.append(f"*\"{self.implementation_plan.demo_pitch}\"*")

        lines.append("\n---\n")
        lines.append("## 🧠 Polymathic Advantage\n")
        lines.append(self.polymathic_advantage)

        return "\n".join(lines)


class HackathonMode:
    """Rapid polymathic problem solving for hackathons"""

    def __init__(self, llm_client=None):
        """
        Initialize hackathon mode.

        Args:
            llm_client: Optional LLM for enhanced reasoning
        """
        self.searcher = HybridSearcherV2()
        self.answer_gen = AnswerPackGenerator(llm_client)
        self.llm = llm_client
        self.conn = get_db_connection()

    def analyze(self, topic: str, time_budget_minutes: int = 10) -> HackathonBrief:
        """
        Rapidly analyze a hackathon topic.

        Args:
            topic: The hackathon problem/topic
            time_budget_minutes: How much time to spend (default 10 min)

        Returns:
            HackathonBrief with complete analysis
        """
        start_time = time.time()

        logger.info(f"🏆 Hackathon Mode activated for: {topic}")
        logger.info(f"⏱️  Time budget: {time_budget_minutes} minutes")

        # Phase 1: Problem Understanding (20% of time)
        problem_reframe, challenges, metrics = self._understand_problem(topic)

        # Phase 2: Prior Art Search (25% of time)
        prior_art = self._find_prior_art(topic)

        # Phase 3: Cross-Domain Insights (25% of time)
        insights = self._find_cross_domain_insights(topic)

        # Phase 4: Code Assets (15% of time)
        code_assets = self._find_code_assets(topic)

        # Phase 5: Solution Generation (15% of time)
        approaches, gaps = self._generate_solutions(topic, prior_art, insights)

        # Select best approach
        recommended = min(approaches, key=lambda x: x.priority) if approaches else None

        # Generate implementation plan if we have a recommendation
        impl_plan = None
        if recommended:
            impl_plan = self._create_implementation_plan(recommended, code_assets)

        # Articulate polymathic advantage
        poly_advantage = self._articulate_advantage(topic, insights, approaches)

        elapsed = int(time.time() - start_time)

        return HackathonBrief(
            topic=topic,
            created_at=datetime.now().isoformat(),
            time_spent_seconds=elapsed,
            problem_reframe=problem_reframe,
            key_challenges=challenges,
            success_metrics=metrics,
            prior_art=prior_art,
            cross_domain_insights=insights,
            gaps_in_literature=gaps,
            code_assets=code_assets,
            approaches=approaches,
            recommended_approach=recommended,
            implementation_plan=impl_plan,
            polymathic_advantage=poly_advantage
        )

    def _understand_problem(self, topic: str) -> tuple:
        """Phase 1: Understand and reframe the problem"""
        logger.info("📋 Phase 1: Understanding problem...")

        # Search for context
        results = self.searcher.search_papers(topic, n=10)

        # Extract key themes
        # (In production, would use LLM for this)
        problem_reframe = f"The core challenge is: {topic}. This requires bridging [domain A] and [domain B] to create a novel solution that [key innovation]."

        challenges = [
            "Data availability and quality",
            "Computational efficiency",
            "Novel combination of existing techniques",
            "Demonstrable improvement over baselines"
        ]

        metrics = [
            "Quantitative improvement over baseline",
            "Novel insight or connection",
            "Practical applicability",
            "Clear demo/visualization"
        ]

        return problem_reframe, challenges, metrics

    def _find_prior_art(self, topic: str) -> List[PriorArt]:
        """Phase 2: Find existing work"""
        logger.info("📚 Phase 2: Finding prior art...")

        results = self.searcher.search_papers(topic, n=20, rerank=True)

        prior_art = []
        for r in results[:10]:
            art = PriorArt(
                title=r.get("title", r.get("passage_text", "")[:50]),
                source=f"{r.get('authors', 'Unknown')[:30]}, {r.get('year', 'N/A')}",
                approach=r.get("passage_text", "")[:200],
                limitations=["Specific to narrow domain", "May not scale"],
                relevance=r.get("score", 0.5)
            )
            prior_art.append(art)

        return prior_art

    def _find_cross_domain_insights(self, topic: str) -> List[CrossDomainInsight]:
        """Phase 3: Find cross-domain connections"""
        logger.info("💡 Phase 3: Finding cross-domain insights...")

        # Define domains to explore
        domains = [
            ("physics", "conservation laws, entropy, diffusion"),
            ("economics", "game theory, optimization, markets"),
            ("cognitive_science", "attention, memory, learning"),
            ("information_theory", "compression, channel capacity, mutual information"),
            ("control_theory", "feedback, stability, adaptation")
        ]

        insights = []
        for domain_name, concepts in domains:
            # Search for connections
            query = f"{topic} {concepts}"
            results = self.searcher.search_papers(query, n=5)

            if results:
                best = results[0]
                insight = CrossDomainInsight(
                    source_domain=domain_name,
                    target_application=topic.split()[-1] if topic.split() else "problem",
                    insight=f"Apply {domain_name} principle: {concepts.split(',')[0]}",
                    evidence=best.get("passage_text", "")[:150],
                    novelty_score=best.get("score", 0.5) * 0.8  # Discount for uncertainty
                )
                insights.append(insight)

        # Sort by novelty
        insights.sort(key=lambda x: x.novelty_score, reverse=True)
        return insights[:5]

    def _find_code_assets(self, topic: str) -> List[CodeAsset]:
        """Phase 4: Find reusable code"""
        logger.info("💻 Phase 4: Finding code assets...")

        # Search code
        results = self.searcher.search_code(topic, n=15)

        code_assets = []
        for r in results[:7]:
            asset = CodeAsset(
                repo=r.get("repo_name", "unknown"),
                file=r.get("file_path", ""),
                description=r.get("chunk_text", "")[:150],
                language=r.get("language", "python"),
                quality_score=r.get("score", 0.5),
                how_to_use=f"Import and adapt the {r.get('chunk_type', 'function')} for your pipeline"
            )
            code_assets.append(asset)

        return code_assets

    def _generate_solutions(
        self,
        topic: str,
        prior_art: List[PriorArt],
        insights: List[CrossDomainInsight]
    ) -> tuple:
        """Phase 5: Generate solution approaches"""
        logger.info("🚀 Phase 5: Generating solutions...")

        approaches = []
        gaps = []

        # Approach 1: Incremental improvement
        approaches.append(SolutionApproach(
            name="Enhanced Baseline",
            description="Take the best existing approach and improve with modern techniques (transformers, better data, etc.)",
            novelty="Apply state-of-the-art ML to established pipeline",
            feasibility="High - well-trodden path",
            differentiation="Speed of implementation + solid baseline",
            risks=["May not be novel enough to win", "Hard to differentiate"],
            priority=3
        ))

        # Approach 2: Cross-domain synthesis (our specialty!)
        if insights:
            best_insight = insights[0]
            approaches.append(SolutionApproach(
                name=f"{best_insight.source_domain.title()}-Inspired Approach",
                description=f"Apply {best_insight.insight} to {topic}",
                novelty=f"Novel connection: {best_insight.source_domain} → problem domain",
                feasibility="Medium - requires adaptation",
                differentiation="Unique perspective that others won't have",
                risks=["May not pan out", "Harder to explain"],
                priority=1  # Best option usually!
            ))

        # Approach 3: Data-centric
        approaches.append(SolutionApproach(
            name="Data Advantage",
            description="Find or create unique dataset that enables new capabilities",
            novelty="Novel data → novel insights",
            feasibility="Medium - depends on data availability",
            differentiation="Others can't replicate without the data",
            risks=["Data may not exist", "Time-consuming to curate"],
            priority=2
        ))

        # Identify gaps
        if prior_art:
            gaps = [
                "No work combining [X] and [Y]",
                "Existing methods don't scale to [Z]",
                "Missing: real-time / interactive capability"
            ]

        return approaches, gaps

    def _create_implementation_plan(
        self,
        approach: SolutionApproach,
        code_assets: List[CodeAsset]
    ) -> ImplementationPlan:
        """Create implementation plan for best approach"""
        logger.info("📋 Creating implementation plan...")

        steps = [
            "Set up environment and dependencies (30 min)",
            "Implement core data pipeline (1h)",
            "Build model/algorithm based on approach (2h)",
            "Create evaluation metrics (30 min)",
            "Build demo/visualization (1h)",
            "Polish presentation and pitch (1h)"
        ]

        return ImplementationPlan(
            approach=approach,
            steps=steps,
            code_assets=code_assets[:3],
            estimated_hours=6,
            mvp_scope="Proof of concept with one example + compelling visualization",
            demo_pitch=f"We applied [{approach.name}] to [{approach.description[:30]}...], achieving [X]% improvement because of our unique insight from [{approach.novelty[:30]}...]"
        )

    def _articulate_advantage(
        self,
        topic: str,
        insights: List[CrossDomainInsight],
        approaches: List[SolutionApproach]
    ) -> str:
        """Articulate our polymathic advantage"""

        domains_used = [i.source_domain for i in insights]
        domains_str = ", ".join(domains_used[:3])

        return f"""
Our polymathic advantage comes from systematically searching across {len(insights)} domains
({domains_str}) to find non-obvious connections to {topic}.

While other teams will likely focus on incremental ML improvements, we can draw on
{len(approaches)} distinct approaches informed by cross-domain knowledge.

Key differentiators:
1. Access to 720K+ paper passages across all of science
2. Ability to find and adapt code from 500K+ code chunks
3. Systematic cross-domain search (not just intuition)
4. Evidence-backed reasoning for every claim
"""


def main():
    """CLI for hackathon mode"""
    import argparse

    parser = argparse.ArgumentParser(description="Hackathon Mode - Rapid Polymathic Problem Solving")
    parser.add_argument("topic", nargs="*", help="Hackathon topic/problem")
    parser.add_argument("--interactive", "-i", action="store_true", help="Interactive mode")
    parser.add_argument("--time", "-t", type=int, default=10, help="Time budget in minutes")
    parser.add_argument("--output", "-o", type=str, help="Output file path")

    args = parser.parse_args()

    hackathon = HackathonMode()

    if args.interactive:
        print("\n🏆 Hackathon Mode - Interactive")
        print("=" * 50)
        topic = input("\nEnter your hackathon topic/problem:\n> ")
    elif args.topic:
        topic = " ".join(args.topic)
    else:
        parser.print_help()
        sys.exit(1)

    print(f"\n⚡ Analyzing: {topic}")
    print(f"⏱️  Time budget: {args.time} minutes\n")

    brief = hackathon.analyze(topic, time_budget_minutes=args.time)

    # Output
    md = brief.to_markdown()
    print(md)

    if args.output:
        with open(args.output, "w") as f:
            f.write(md)
        print(f"\n✅ Saved to {args.output}")
    else:
        # Save to default location
        os.makedirs("data/hackathon_briefs", exist_ok=True)
        filename = f"data/hackathon_briefs/{brief.created_at[:10]}_{topic[:30].replace(' ', '_')}.md"
        with open(filename, "w") as f:
            f.write(md)
        print(f"\n✅ Saved to {filename}")


if __name__ == "__main__":
    main()
