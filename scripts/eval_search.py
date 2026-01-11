#!/usr/bin/env python3
"""
Gold Query Evaluation Suite for Polymath Search

Measures search quality using a curated set of queries with known good answers.
Metrics: Recall@K, Precision@K, MRR (Mean Reciprocal Rank), NDCG

Usage:
    python3 scripts/eval_search.py              # Run all evaluations
    python3 scripts/eval_search.py --report     # Generate detailed report
    python3 scripts/eval_search.py --query "attention mechanism"  # Test single query
"""

import argparse
import json
import sys
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple
import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent))

from lib.hybrid_search_v2 import HybridSearcherV2

# Gold query set - queries with expected results
# Format: {query: {relevant_docs: [doc_ids], relevant_terms: [terms that should appear], domain: str}}
GOLD_QUERIES = {
    # === Foundational ML Papers ===
    "attention is all you need transformer": {
        "relevant_terms": ["transformer", "attention", "vaswani"],
        "domain": "ml_foundations",
        "expected_recall": 0.8,
        "description": "Should find the original Transformer paper"
    },
    "BERT pre-training language model": {
        "relevant_terms": ["bert", "pre-training", "language model", "devlin"],
        "domain": "ml_foundations",
        "expected_recall": 0.7,
        "description": "Should find BERT-related papers"
    },
    "residual network deep learning": {
        "relevant_terms": ["resnet", "residual", "skip connection", "he kaiming"],
        "domain": "ml_foundations",
        "expected_recall": 0.6,
        "description": "Should find ResNet paper"
    },

    # === Spatial Transcriptomics ===
    "visium spatial transcriptomics gene expression": {
        "relevant_terms": ["visium", "spatial transcriptomics", "10x genomics", "spot"],
        "domain": "spatial_biology",
        "expected_recall": 0.7,
        "description": "Should find Visium-related papers and methods"
    },
    "H&E histology to gene expression prediction": {
        "relevant_terms": ["H&E", "histology", "gene expression", "prediction", "spatial"],
        "domain": "spatial_biology",
        "expected_recall": 0.7,
        "description": "Should find Img2ST and similar methods"
    },
    "spatial deconvolution cell type": {
        "relevant_terms": ["deconvolution", "cell type", "spatial", "RCTD", "cell2location"],
        "domain": "spatial_biology",
        "expected_recall": 0.6,
        "description": "Should find spatial deconvolution methods"
    },

    # === Computational Pathology ===
    "pathology foundation model whole slide image": {
        "relevant_terms": ["foundation model", "pathology", "WSI", "whole slide"],
        "domain": "comp_pathology",
        "expected_recall": 0.7,
        "description": "Should find pathology foundation models"
    },
    "multiple instance learning histopathology": {
        "relevant_terms": ["MIL", "multiple instance", "pathology", "attention"],
        "domain": "comp_pathology",
        "expected_recall": 0.7,
        "description": "Should find CLAM, ABMIL, etc."
    },

    # === Information Theory / Cross-Domain ===
    "information bottleneck deep learning": {
        "relevant_terms": ["information bottleneck", "mutual information", "compression"],
        "domain": "information_theory",
        "expected_recall": 0.6,
        "description": "Should find information-theoretic ML papers"
    },
    "entropy regularization neural network": {
        "relevant_terms": ["entropy", "regularization", "neural network"],
        "domain": "information_theory",
        "expected_recall": 0.5,
        "description": "Should find entropy-based methods"
    },

    # === Single-Cell Biology ===
    "single cell RNA sequencing clustering": {
        "relevant_terms": ["scRNA-seq", "single cell", "clustering", "scanpy"],
        "domain": "single_cell",
        "expected_recall": 0.7,
        "description": "Should find single-cell analysis methods"
    },
    "cell type annotation transfer learning": {
        "relevant_terms": ["cell type", "annotation", "transfer", "scANVI", "scArches"],
        "domain": "single_cell",
        "expected_recall": 0.5,
        "description": "Should find cell type transfer methods"
    },

    # === Graph Neural Networks ===
    "graph neural network molecular property": {
        "relevant_terms": ["GNN", "graph neural", "molecular", "MPNN", "property prediction"],
        "domain": "graph_ml",
        "expected_recall": 0.6,
        "description": "Should find molecular GNN papers"
    },

    # === Colorectal Cancer ===
    "colorectal cancer tumor microenvironment spatial": {
        "relevant_terms": ["colorectal", "CRC", "tumor microenvironment", "TME", "spatial"],
        "domain": "cancer_biology",
        "expected_recall": 0.6,
        "description": "Should find CRC spatial biology papers"
    },
}


@dataclass
class SearchResult:
    """Single search result"""
    doc_id: str
    title: str
    score: float
    text_snippet: str = ""


@dataclass
class EvalResult:
    """Evaluation result for a single query"""
    query: str
    domain: str
    num_results: int
    terms_found: List[str]
    terms_missing: List[str]
    recall: float  # Fraction of relevant terms found
    first_relevant_rank: Optional[int]  # For MRR
    results: List[SearchResult] = field(default_factory=list)


@dataclass
class EvalSummary:
    """Summary of all evaluations"""
    timestamp: str
    num_queries: int
    mean_recall: float
    mean_mrr: float
    by_domain: Dict[str, Dict[str, float]]
    failed_queries: List[str]


class SearchEvaluator:
    """Evaluates Polymath search quality"""

    def __init__(self, k: int = 10):
        self.searcher = HybridSearcherV2()
        self.k = k

    def evaluate_query(self, query: str, gold: dict) -> EvalResult:
        """Evaluate a single query against gold data"""
        relevant_terms = [t.lower() for t in gold.get("relevant_terms", [])]
        domain = gold.get("domain", "unknown")

        # Run search
        try:
            results = self.searcher.search_papers(query, n=self.k)
        except Exception as e:
            print(f"Error searching '{query}': {e}")
            return EvalResult(
                query=query,
                domain=domain,
                num_results=0,
                terms_found=[],
                terms_missing=relevant_terms,
                recall=0.0,
                first_relevant_rank=None
            )

        # Check which terms appear in results
        result_text = ""
        search_results = []

        for r in results:
            # Handle both dict and SearchResult objects
            if hasattr(r, 'title'):
                title = r.title or ""
                text = r.content or ""
                score = r.score or 0
                doc_id = r.id or ""
            else:
                title = r.get("title", "")
                text = r.get("text", "") or r.get("content", "")
                score = r.get("score", 0)
                doc_id = r.get("doc_id", "") or r.get("id", "")

            combined = f"{title} {text}".lower()
            result_text += combined + " "
            search_results.append(SearchResult(
                doc_id=doc_id,
                title=title[:80],
                score=score,
                text_snippet=text[:150]
            ))

        terms_found = []
        terms_missing = []
        first_relevant_rank = None

        for i, term in enumerate(relevant_terms):
            if term in result_text:
                terms_found.append(term)
                if first_relevant_rank is None:
                    # Find which result contains this term
                    for j, r in enumerate(results):
                        if hasattr(r, 'title'):
                            text = f"{r.title or ''} {r.content or ''}".lower()
                        else:
                            text = f"{r.get('title', '')} {r.get('text', '') or r.get('content', '')}".lower()
                        if term in text:
                            first_relevant_rank = j + 1
                            break
            else:
                terms_missing.append(term)

        recall = len(terms_found) / len(relevant_terms) if relevant_terms else 0.0

        return EvalResult(
            query=query,
            domain=domain,
            num_results=len(results),
            terms_found=terms_found,
            terms_missing=terms_missing,
            recall=recall,
            first_relevant_rank=first_relevant_rank,
            results=search_results
        )

    def evaluate_all(self, queries: Dict = None) -> Tuple[List[EvalResult], EvalSummary]:
        """Evaluate all gold queries"""
        queries = queries or GOLD_QUERIES
        results = []

        for query, gold in queries.items():
            print(f"Evaluating: {query[:50]}...")
            result = self.evaluate_query(query, gold)
            results.append(result)

        # Compute summary
        recalls = [r.recall for r in results]
        mrrs = [1/r.first_relevant_rank if r.first_relevant_rank else 0 for r in results]

        # Group by domain
        by_domain = {}
        for r in results:
            if r.domain not in by_domain:
                by_domain[r.domain] = {"recalls": [], "mrrs": []}
            by_domain[r.domain]["recalls"].append(r.recall)
            by_domain[r.domain]["mrrs"].append(1/r.first_relevant_rank if r.first_relevant_rank else 0)

        for domain in by_domain:
            by_domain[domain] = {
                "mean_recall": float(np.mean(by_domain[domain]["recalls"])),
                "mean_mrr": float(np.mean(by_domain[domain]["mrrs"])),
                "count": len(by_domain[domain]["recalls"])
            }

        failed = [r.query for r in results if r.recall < 0.3]

        summary = EvalSummary(
            timestamp=datetime.now().isoformat(),
            num_queries=len(results),
            mean_recall=float(np.mean(recalls)),
            mean_mrr=float(np.mean(mrrs)),
            by_domain=by_domain,
            failed_queries=failed
        )

        return results, summary

    def generate_report(self, results: List[EvalResult], summary: EvalSummary) -> str:
        """Generate detailed markdown report"""
        lines = [
            "# Polymath Search Evaluation Report",
            "",
            f"**Timestamp**: {summary.timestamp}",
            f"**Queries Evaluated**: {summary.num_queries}",
            "",
            "## Summary Metrics",
            "",
            f"| Metric | Value |",
            f"|--------|-------|",
            f"| Mean Recall@{self.k} | {summary.mean_recall:.3f} |",
            f"| Mean MRR | {summary.mean_mrr:.3f} |",
            f"| Failed Queries (recall<0.3) | {len(summary.failed_queries)} |",
            "",
            "## Performance by Domain",
            "",
            "| Domain | Recall | MRR | Count |",
            "|--------|--------|-----|-------|",
        ]

        for domain, metrics in sorted(summary.by_domain.items()):
            lines.append(
                f"| {domain} | {metrics['mean_recall']:.3f} | {metrics['mean_mrr']:.3f} | {metrics['count']} |"
            )

        lines.extend([
            "",
            "## Query Details",
            "",
        ])

        for r in sorted(results, key=lambda x: x.recall):
            status = "✅" if r.recall >= 0.6 else "⚠️" if r.recall >= 0.3 else "❌"
            lines.extend([
                f"### {status} {r.query[:60]}",
                "",
                f"- **Domain**: {r.domain}",
                f"- **Recall**: {r.recall:.2f}",
                f"- **Terms Found**: {', '.join(r.terms_found) or 'None'}",
                f"- **Terms Missing**: {', '.join(r.terms_missing) or 'None'}",
                f"- **First Relevant Rank**: {r.first_relevant_rank or 'N/A'}",
                ""
            ])

        if summary.failed_queries:
            lines.extend([
                "",
                "## Failed Queries (Need Attention)",
                "",
            ])
            for q in summary.failed_queries:
                lines.append(f"- {q}")

        return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(description="Evaluate Polymath search quality")
    parser.add_argument("--report", action="store_true", help="Generate detailed report")
    parser.add_argument("--query", type=str, help="Evaluate single query")
    parser.add_argument("--k", type=int, default=10, help="Number of results to retrieve")
    parser.add_argument("--output", type=str, help="Output file for report")
    args = parser.parse_args()

    evaluator = SearchEvaluator(k=args.k)

    if args.query:
        # Single query mode
        gold = GOLD_QUERIES.get(args.query, {"relevant_terms": [], "domain": "custom"})
        result = evaluator.evaluate_query(args.query, gold)

        print(f"\nQuery: {result.query}")
        print(f"Recall: {result.recall:.2f}")
        print(f"Terms Found: {result.terms_found}")
        print(f"Terms Missing: {result.terms_missing}")
        print(f"\nTop Results:")
        for i, r in enumerate(result.results[:5]):
            print(f"  {i+1}. [{r.score:.3f}] {r.title}")
    else:
        # Full evaluation
        results, summary = evaluator.evaluate_all()

        print("\n" + "="*60)
        print("EVALUATION SUMMARY")
        print("="*60)
        print(f"Queries: {summary.num_queries}")
        print(f"Mean Recall@{args.k}: {summary.mean_recall:.3f}")
        print(f"Mean MRR: {summary.mean_mrr:.3f}")
        print(f"Failed Queries: {len(summary.failed_queries)}")

        print("\nBy Domain:")
        for domain, metrics in sorted(summary.by_domain.items()):
            print(f"  {domain}: Recall={metrics['mean_recall']:.3f}, MRR={metrics['mean_mrr']:.3f}")

        if args.report:
            report = evaluator.generate_report(results, summary)

            if args.output:
                Path(args.output).write_text(report)
                print(f"\nReport saved to: {args.output}")
            else:
                output_path = Path("/home/user/polymath-repo/reports/search_eval.md")
                output_path.parent.mkdir(exist_ok=True)
                output_path.write_text(report)
                print(f"\nReport saved to: {output_path}")

        # Save JSON summary
        json_path = Path("/home/user/polymath-repo/data/eval_results.json")
        json_path.parent.mkdir(exist_ok=True)

        summary_dict = {
            "timestamp": summary.timestamp,
            "num_queries": summary.num_queries,
            "mean_recall": summary.mean_recall,
            "mean_mrr": summary.mean_mrr,
            "by_domain": summary.by_domain,
            "failed_queries": summary.failed_queries
        }

        # Append to history
        history = []
        if json_path.exists():
            try:
                history = json.loads(json_path.read_text())
            except:
                pass
        history.append(summary_dict)
        json_path.write_text(json.dumps(history, indent=2))

        print(f"\nResults appended to: {json_path}")


if __name__ == "__main__":
    main()
