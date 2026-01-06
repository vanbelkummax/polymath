#!/usr/bin/env python3
"""
Search Quality Smoke Test

Runs the smaller golden query set using the shared evaluation harness.
"""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from scripts.golden_query_eval import main as eval_main


if __name__ == "__main__":
    queries_path = ROOT / "tests" / "data" / "golden_queries.json"
    output_path = ROOT / "tests" / "golden_smoke_run.json"
    sys.argv = [
        sys.argv[0],
        "--queries",
        str(queries_path),
        "--output",
        str(output_path),
        "--k",
        "10",
    ]
    raise SystemExit(eval_main())
