#!/usr/bin/env python3
"""
Consistency check across Postgres, ChromaDB, and Neo4j.

Checks:
- Count comparison (Postgres chunks vs Chroma chunks)
- Sampled ID existence (Chroma -> Postgres, Postgres -> Chroma)
- Sampled artifact nodes in Neo4j
"""

import argparse
import json
import os
from datetime import datetime, timezone
from typing import Dict, List, Optional
from urllib.parse import urlparse, urlencode, parse_qs, urlunparse


CHROMA_PATH = os.environ.get("CHROMA_PATH", "/home/user/work/polymax/chromadb/polymath_v2")
POSTGRES_URL = os.environ.get("POSTGRES_URL", "dbname=polymath user=polymath host=/var/run/postgresql")
NEO4J_URI = os.environ.get("NEO4J_URI", "bolt://localhost:7687")
NEO4J_PASSWORD = os.environ.get("NEO4J_PASSWORD", "polymathic2026")


def _get_postgres_conn():
    import psycopg2
    dsn = POSTGRES_URL
    if dsn.startswith("postgresql:///") or dsn.startswith("postgres:///"):
        if "user=" not in dsn:
            user = os.environ.get("PGUSER", "polymath")
            sep = "&" if "?" in dsn else "?"
            dsn = f"{dsn}{sep}user={user}"
        return psycopg2.connect(dsn)

    if dsn.startswith("postgresql://") or dsn.startswith("postgres://"):
        parsed = urlparse(dsn)
        if parsed.username is None:
            user = os.environ.get("PGUSER", "polymath")
            query = parse_qs(parsed.query)
            if "user" not in query:
                query["user"] = [user]
                parsed = parsed._replace(query=urlencode(query, doseq=True))
            dsn = urlunparse(parsed)
    else:
        if "user=" not in dsn:
            user = os.environ.get("PGUSER", "polymath")
            dsn = f"{dsn} user={user}"
    return psycopg2.connect(dsn)


def _get_chroma_collection():
    import chromadb
    client = chromadb.PersistentClient(path=CHROMA_PATH)
    return client.get_collection("polymath_corpus")


def _get_neo4j_driver():
    from neo4j import GraphDatabase
    return GraphDatabase.driver(NEO4J_URI, auth=("neo4j", NEO4J_PASSWORD))


def _unique(items: List[str]) -> List[str]:
    seen = set()
    ordered = []
    for item in items:
        if item not in seen:
            seen.add(item)
            ordered.append(item)
    return ordered


def _sample_chroma_ids(collection, sample_size: int) -> List[str]:
    if sample_size <= 0:
        return []
    results = collection.get(limit=sample_size)
    return results.get("ids", []) or []


def _sample_postgres_chunk_ids(conn, sample_size: int) -> List[str]:
    if sample_size <= 0:
        return []
    with conn.cursor() as cursor:
        cursor.execute("SELECT id FROM chunks ORDER BY random() LIMIT %s", (sample_size,))
        return [row[0] for row in cursor.fetchall()]


def _sample_postgres_artifacts(conn, sample_size: int) -> List[Dict[str, str]]:
    if sample_size <= 0:
        return []
    with conn.cursor() as cursor:
        cursor.execute(
            "SELECT file_hash, artifact_type FROM artifacts ORDER BY random() LIMIT %s",
            (sample_size,)
        )
        return [{"file_hash": row[0], "artifact_type": row[1]} for row in cursor.fetchall()]


def _fetch_postgres_ids(conn, ids: List[str]) -> List[str]:
    if not ids:
        return []
    with conn.cursor() as cursor:
        cursor.execute("SELECT id FROM chunks WHERE id = ANY(%s)", (ids,))
        return [row[0] for row in cursor.fetchall()]


def _fetch_neo4j_hashes(driver, label: str, prop: str, hashes: List[str]) -> List[str]:
    if not hashes:
        return []
    with driver.session() as session:
        result = session.run(
            f"MATCH (n:{label}) WHERE n.{prop} IN $hashes RETURN n.{prop} AS hash",
            hashes=hashes,
        )
        return [record["hash"] for record in result]


def run_checks(sample_size: int, skip_neo4j: bool) -> Dict[str, object]:
    summary: Dict[str, object] = {
        "timestamp": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
        "counts": {},
        "samples": {},
        "neo4j": {},
        "errors": [],
    }

    try:
        chroma = _get_chroma_collection()
        summary["counts"]["chromadb_chunks"] = chroma.count()
    except Exception as e:
        summary["errors"].append(f"ChromaDB error: {e}")
        chroma = None

    try:
        pg = _get_postgres_conn()
        with pg.cursor() as cursor:
            cursor.execute("SELECT COUNT(*) FROM chunks")
            summary["counts"]["postgres_chunks"] = cursor.fetchone()[0]
            cursor.execute("SELECT COUNT(*) FROM artifacts")
            summary["counts"]["postgres_artifacts"] = cursor.fetchone()[0]
    except Exception as e:
        summary["errors"].append(f"Postgres error: {e}")
        pg = None

    # Sampled ID checks
    if chroma and pg:
        chroma_ids = _sample_chroma_ids(chroma, sample_size)
        pg_ids = _sample_postgres_chunk_ids(pg, sample_size)

        if chroma_ids:
            found_pg_ids = _fetch_postgres_ids(pg, chroma_ids)
            missing_in_pg = sorted(set(chroma_ids) - set(found_pg_ids))
            summary["samples"]["chroma_to_postgres"] = {
                "sample_size": len(chroma_ids),
                "missing_count": len(missing_in_pg),
                "missing_ids": missing_in_pg[:10],
            }

        if pg_ids:
            chroma_found = chroma.get(ids=pg_ids).get("ids", [])
            missing_in_chroma = sorted(set(pg_ids) - set(chroma_found))
            summary["samples"]["postgres_to_chroma"] = {
                "sample_size": len(pg_ids),
                "missing_count": len(missing_in_chroma),
                "missing_ids": missing_in_chroma[:10],
            }

    # Neo4j artifact checks
    if not skip_neo4j and pg:
        try:
            driver = _get_neo4j_driver()
            artifacts = _sample_postgres_artifacts(pg, sample_size)
            by_type: Dict[str, List[str]] = {}
            for item in artifacts:
                file_hash = item.get("file_hash")
                artifact_type = item.get("artifact_type")
                if not file_hash or not artifact_type:
                    continue
                by_type.setdefault(artifact_type, []).append(file_hash)

            neo4j_results = {}
            for artifact_type, hashes in by_type.items():
                hashes = _unique(hashes)
                if artifact_type == "paper":
                    found = _fetch_neo4j_hashes(driver, "Paper", "file_hash", hashes)
                elif artifact_type == "code":
                    found = _fetch_neo4j_hashes(driver, "Code", "file_hash", hashes)
                elif artifact_type == "repo":
                    found = _fetch_neo4j_hashes(driver, "Repo", "repo_hash", hashes)
                else:
                    continue
                missing = sorted(set(hashes) - set(found))
                neo4j_results[artifact_type] = {
                    "sample_size": len(hashes),
                    "missing_count": len(missing),
                    "missing_hashes": missing[:10],
                }

            summary["neo4j"]["artifact_nodes"] = neo4j_results
            driver.close()
        except Exception as e:
            summary["errors"].append(f"Neo4j error: {e}")

    if pg:
        pg.close()

    return summary


def main() -> int:
    parser = argparse.ArgumentParser(description="Polymath consistency check")
    parser.add_argument("--sample-size", type=int, default=200, help="Sample size for ID checks")
    parser.add_argument("--skip-neo4j", action="store_true", help="Skip Neo4j checks")
    parser.add_argument("--output", help="Write JSON summary to path")
    parser.add_argument("--strict", action="store_true", help="Exit non-zero on mismatches")
    args = parser.parse_args()

    summary = run_checks(args.sample_size, args.skip_neo4j)

    print("Consistency Check Summary")
    print(json.dumps(summary, indent=2))

    if args.output:
        with open(args.output, "w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2)
        print(f"\nWrote report to {args.output}")

    if args.strict:
        sample_checks = summary.get("samples", {})
        neo4j_checks = summary.get("neo4j", {}).get("artifact_nodes", {})
        mismatches = 0

        for check in sample_checks.values():
            if check.get("missing_count", 0) > 0:
                mismatches += 1

        for check in neo4j_checks.values():
            if check.get("missing_count", 0) > 0:
                mismatches += 1

        if mismatches > 0:
            return 1

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
