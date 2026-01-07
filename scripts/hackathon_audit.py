#!/usr/bin/env python3
"""
Hackathon audit for spatial multimodal coverage.

Generates a Markdown and JSON report describing what is already indexed and
where gaps likely exist. Read-only; safe to run during migration.
"""
import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from lib.db import Database
from lib.config import POSTGRES_DSN, NEO4J_URI, NEO4J_USER, NEO4J_PASSWORD

KEYWORD_GROUPS = {
    "core": [
        "spatial",
        "multimodal",
        "multi-omics",
        "multiomics",
        "multi modal",
        "multi-modal",
        "multiome",
        "integration",
        "fusion",
    ],
    "modalities": [
        "visium",
        "visium hd",
        "xenium",
        "merfish",
        "cosmx",
        "slide-seq",
        "slideseq",
        "seqfish",
        "stereo-seq",
        "stereoseq",
        "in situ sequencing",
        "smfish",
    ],
    "imaging": [
        "h&e",
        "h and e",
        "histology",
        "pathology",
        "immunofluorescence",
        "ihc",
    ],
    "tasks": [
        "alignment",
        "registration",
        "deconvolution",
        "segmentation",
        "imputation",
        "mapping",
        "label transfer",
        "cell type",
        "co-embedding",
        "cross-modal",
    ],
    "methods": [
        "optimal transport",
        "graph matching",
        "contrastive",
        "transformer",
        "graph neural",
        "gnn",
        "variational",
        "bayesian",
        "manifold",
    ],
    "tooling": [
        "seurat",
        "scanpy",
        "squidpy",
        "cell2location",
        "tangram",
        "stalign",
        "spatialde",
        "bayesprism",
        "destvi",
        "totalvi",
        "scvi",
    ],
}

CONCEPT_PATTERNS = ["spatial", "modal"]


class ConceptCounter:
    """Resolve concept coverage from Postgres or Neo4j (optional)."""

    def __init__(self, db: Database, use_neo4j: bool, neo4j_timeout: int) -> None:
        self.db = db
        self.source = "postgres"
        self.error: Optional[str] = None
        self.driver = None

        if not use_neo4j:
            return
        if not NEO4J_PASSWORD:
            self.error = "NEO4J_PASSWORD not set; skipping Neo4j concept coverage."
            return

        try:
            from neo4j import GraphDatabase
        except Exception as exc:
            self.error = f"Neo4j driver unavailable: {exc}"
            return

        try:
            self.driver = GraphDatabase.driver(
                NEO4J_URI,
                auth=(NEO4J_USER, NEO4J_PASSWORD),
                connection_timeout=neo4j_timeout,
            )
            self.driver.verify_connectivity()
            self.source = "neo4j"
        except Exception as exc:
            self.error = f"Neo4j connection failed: {exc}"
            if self.driver:
                self.driver.close()
                self.driver = None

    def close(self) -> None:
        if self.driver:
            self.driver.close()

    def count_docs(self, keyword: str) -> int:
        if self.driver:
            return self._count_docs_neo4j([keyword])
        return count_concept_docs_pg(self.db, keyword)

    def count_docs_any(self, keywords: list[str]) -> int:
        if self.driver:
            return self._count_docs_neo4j(keywords)
        return count_concept_docs_any_pg(self.db, keywords)

    def top_concepts(self, pattern: str, limit: int) -> list[dict]:
        if self.driver:
            return self._top_concepts_neo4j(pattern, limit)
        return top_concepts_pg(self.db, pattern, limit)

    def _count_docs_neo4j(self, keywords: list[str]) -> int:
        if not keywords:
            return 0
        lower = [keyword.lower() for keyword in keywords]
        query = (
            "MATCH (p:Paper)-[:MENTIONS]->(c:CONCEPT) "
            "WHERE any(kw IN $keywords WHERE toLower(c.name) CONTAINS kw) "
            "RETURN count(DISTINCT p) AS count"
        )
        with self.driver.session() as session:
            record = session.run(query, keywords=lower).single()
            return int(record["count"]) if record else 0

    def _top_concepts_neo4j(self, pattern: str, limit: int) -> list[dict]:
        query = (
            "MATCH (p:Paper)-[:MENTIONS]->(c:CONCEPT) "
            "WHERE toLower(c.name) CONTAINS $pattern "
            "RETURN c.name AS name, count(DISTINCT p) AS count "
            "ORDER BY count DESC LIMIT $limit"
        )
        with self.driver.session() as session:
            rows = session.run(
                query,
                pattern=pattern.lower(),
                limit=limit,
            )
            return [{"name": row["name"], "count": int(row["count"])} for row in rows]


def redact_dsn(dsn: str) -> str:
    parts = []
    for token in dsn.split():
        if token.startswith("password="):
            parts.append("password=***")
        else:
            parts.append(token)
    return " ".join(parts)


def fetch_count(db: Database, query: str, params: tuple = None) -> int:
    row = db.fetch_one(query, params)
    if not row:
        return 0
    value = row.get("count")
    return int(value) if value is not None else 0


def count_docs_by_title(db: Database, keyword: str) -> int:
    return fetch_count(
        db,
        "SELECT count(*) AS count FROM documents WHERE title ILIKE %s",
        (f"%{keyword}%",),
    )


def count_concept_docs_pg(db: Database, keyword: str) -> int:
    return fetch_count(
        db,
        """
        SELECT count(DISTINCT a.doc_id) AS count
        FROM artifact_concepts ac
        JOIN concepts c ON c.id = ac.concept_id
        JOIN artifacts a ON a.id = ac.artifact_id
        WHERE a.artifact_type = 'paper'
          AND a.doc_id IS NOT NULL
          AND c.name ILIKE %s
        """,
        (f"%{keyword}%",),
    )


def count_code_files(db: Database, keyword: str) -> int:
    return fetch_count(
        db,
        """
        SELECT count(*) AS count
        FROM code_files
        WHERE repo_name ILIKE %s OR file_path ILIKE %s
        """,
        (f"%{keyword}%", f"%{keyword}%"),
    )


def count_docs_by_title_any(db: Database, keywords: list[str]) -> int:
    if not keywords:
        return 0
    clauses = " OR ".join(["title ILIKE %s"] * len(keywords))
    params = tuple(f"%{keyword}%" for keyword in keywords)
    query = f"SELECT count(*) AS count FROM documents WHERE {clauses}"
    return fetch_count(db, query, params)


def count_concept_docs_any_pg(db: Database, keywords: list[str]) -> int:
    if not keywords:
        return 0
    clauses = " OR ".join(["c.name ILIKE %s"] * len(keywords))
    params = tuple(f"%{keyword}%" for keyword in keywords)
    query = f"""
        SELECT count(DISTINCT a.doc_id) AS count
        FROM artifact_concepts ac
        JOIN concepts c ON c.id = ac.concept_id
        JOIN artifacts a ON a.id = ac.artifact_id
        WHERE a.artifact_type = 'paper'
          AND a.doc_id IS NOT NULL
          AND ({clauses})
    """
    return fetch_count(db, query, params)


def count_code_files_any(db: Database, keywords: list[str]) -> int:
    if not keywords:
        return 0
    clauses = " OR ".join(["repo_name ILIKE %s", "file_path ILIKE %s"] * len(keywords))
    params = []
    for keyword in keywords:
        params.append(f"%{keyword}%")
        params.append(f"%{keyword}%")
    query = f"SELECT count(*) AS count FROM code_files WHERE {clauses}"
    return fetch_count(db, query, tuple(params))


def top_concepts_pg(db: Database, pattern: str, limit: int) -> list[dict]:
    rows = db.fetch_all(
        """
        SELECT c.name, count(*) AS count
        FROM artifact_concepts ac
        JOIN concepts c ON c.id = ac.concept_id
        JOIN artifacts a ON a.id = ac.artifact_id
        WHERE a.artifact_type = 'paper'
          AND c.name ILIKE %s
        GROUP BY c.name
        ORDER BY count DESC
        LIMIT %s
        """,
        (f"%{pattern}%", limit),
    )
    return [{"name": r["name"], "count": int(r["count"])} for r in rows]


def status_for_counts(title_hits: int, concept_hits: int, code_hits: int) -> str:
    paper_signal = max(title_hits, concept_hits)
    if paper_signal == 0 and code_hits == 0:
        return "missing"
    if paper_signal < 5 and code_hits < 2:
        return "thin"
    return "ok"


def build_report(db: Database, concept_counter: ConceptCounter, concept_limit: int) -> dict:
    counts = {
        "documents": fetch_count(db, "SELECT count(*) AS count FROM documents"),
        "passages": fetch_count(db, "SELECT count(*) AS count FROM passages"),
        "code_files": fetch_count(db, "SELECT count(*) AS count FROM code_files"),
        "code_chunks": fetch_count(db, "SELECT count(*) AS count FROM code_chunks"),
        "concepts": fetch_count(db, "SELECT count(*) AS count FROM concepts"),
        "artifact_concepts": fetch_count(db, "SELECT count(*) AS count FROM artifact_concepts"),
    }

    group_summaries = {}
    keyword_rows = []
    missing = []
    thin = []

    for group, keywords in KEYWORD_GROUPS.items():
        group_summaries[group] = {
            "title_hits": count_docs_by_title_any(db, keywords),
            "concept_docs": concept_counter.count_docs_any(keywords),
            "code_files": count_code_files_any(db, keywords),
        }

        for keyword in keywords:
            title_hits = count_docs_by_title(db, keyword)
            concept_hits = concept_counter.count_docs(keyword)
            code_hits = count_code_files(db, keyword)
            status = status_for_counts(title_hits, concept_hits, code_hits)
            row = {
                "group": group,
                "keyword": keyword,
                "title_hits": title_hits,
                "concept_docs": concept_hits,
                "code_files": code_hits,
                "status": status,
            }
            keyword_rows.append(row)
            if status == "missing":
                missing.append(keyword)
            elif status == "thin":
                thin.append(keyword)

    concept_summary = {
        pattern: concept_counter.top_concepts(pattern, concept_limit)
        for pattern in CONCEPT_PATTERNS
    }

    return {
        "generated_at": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
        "counts": counts,
        "group_summaries": group_summaries,
        "keywords": keyword_rows,
        "missing": sorted(set(missing)),
        "thin": sorted(set(thin)),
        "concept_summary": concept_summary,
        "concept_source": concept_counter.source,
        "concept_error": concept_counter.error,
    }


def write_markdown(report: dict, path: Path, dsn: str) -> None:
    lines = []
    lines.append("# Hackathon Audit - Spatial Multimodal Coverage")
    lines.append("")
    lines.append(f"Generated: {report['generated_at']}")
    lines.append(f"Postgres DSN: {redact_dsn(dsn)}")
    lines.append(f"Concept source: {report.get('concept_source', 'postgres')}")
    if report.get("concept_error"):
        lines.append(f"Concept source error: {report['concept_error']}")
    lines.append("")
    lines.append("## Global Counts")
    lines.append(f"- Documents: {report['counts']['documents']}")
    lines.append(f"- Passages: {report['counts']['passages']}")
    lines.append(f"- Code files: {report['counts']['code_files']}")
    lines.append(f"- Code chunks: {report['counts']['code_chunks']}")
    lines.append(f"- Concepts: {report['counts']['concepts']}")
    lines.append(f"- Concept links: {report['counts']['artifact_concepts']}")
    if report["counts"]["artifact_concepts"] == 0 and report.get("concept_source") == "postgres":
        if report.get("concept_error"):
            lines.append("- Note: Neo4j concept coverage unavailable; check NEO4J_USER/NEO4J_PASSWORD.")
        else:
            lines.append("- Note: Concept links are stored in Neo4j; rerun with --neo4j for concept coverage.")
    lines.append("")

    lines.append("## Group Coverage Summary")
    lines.append("| Group | Title hits | Concept docs | Code files |")
    lines.append("| --- | --- | --- | --- |")
    for group, summary in report["group_summaries"].items():
        lines.append(
            f"| {group} | {summary['title_hits']} | {summary['concept_docs']} | {summary['code_files']} |"
        )
    lines.append("")

    lines.append("## Keyword Coverage")
    for group in KEYWORD_GROUPS:
        lines.append("")
        lines.append(f"### {group}")
        lines.append("| Keyword | Title hits | Concept docs | Code files | Status |")
        lines.append("| --- | --- | --- | --- | --- |")
        for row in report["keywords"]:
            if row["group"] != group:
                continue
            lines.append(
                f"| {row['keyword']} | {row['title_hits']} | {row['concept_docs']} | {row['code_files']} | {row['status']} |"
            )
    lines.append("")

    lines.append("## Missing Keywords (No Paper or Code Signal)")
    if report["missing"]:
        lines.append("- " + ", ".join(report["missing"]))
    else:
        lines.append("- None")
    lines.append("")

    lines.append("## Thin Keywords (Low Coverage)")
    if report["thin"]:
        lines.append("- " + ", ".join(report["thin"]))
    else:
        lines.append("- None")
    lines.append("")

    lines.append("## Concept Summary (Patterns)")
    for pattern, rows in report["concept_summary"].items():
        lines.append("")
        lines.append(f"### Concepts matching: {pattern}")
        if not rows:
            lines.append("- None")
            continue
        lines.append("| Concept | Count |")
        lines.append("| --- | --- |")
        for row in rows:
            lines.append(f"| {row['name']} | {row['count']} |")

    lines.append("")
    lines.append("## Suggested Acquisition Actions")
    lines.append("- Pull missing modalities and benchmarks as PDFs, then ingest them.")
    lines.append("- Add missing code repos, then re-ingest your hackathon repo.")
    lines.append("- Re-run this audit after each major ingestion batch.")

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Audit spatial multimodal coverage")
    parser.add_argument(
        "--out-md",
        default=str(ROOT / "docs" / "hackathon_reports" / "HACKATHON_AUDIT.md"),
        help="Markdown output path",
    )
    parser.add_argument(
        "--out-json",
        default=str(ROOT / "docs" / "hackathon_reports" / "hackathon_audit.json"),
        help="JSON output path",
    )
    parser.add_argument(
        "--concept-limit",
        type=int,
        default=15,
        help="Max concepts per pattern",
    )
    parser.add_argument(
        "--dsn",
        default=POSTGRES_DSN,
        help="Postgres DSN (default from config)",
    )
    parser.add_argument(
        "--neo4j",
        action="store_true",
        help="Use Neo4j for concept coverage (optional)",
    )
    parser.add_argument(
        "--neo4j-timeout",
        type=int,
        default=5,
        help="Neo4j connection timeout in seconds",
    )

    args = parser.parse_args()

    db = Database(dsn=args.dsn)
    try:
        concept_counter = ConceptCounter(db, args.neo4j, args.neo4j_timeout)
        try:
            report = build_report(db, concept_counter, args.concept_limit)
        finally:
            concept_counter.close()
    finally:
        db.close()

    write_markdown(report, Path(args.out_md), args.dsn)
    Path(args.out_json).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out_json).write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")

    print(f"Wrote: {args.out_md}")
    print(f"Wrote: {args.out_json}")


if __name__ == "__main__":
    main()
