#!/usr/bin/env python3
"""
Literature Sentry Configuration

API keys, rate limits, and thresholds.
"""

import os

# ============================================================
# API KEYS
# ============================================================

BRAVE_API_KEY = os.environ.get("BRAVE_API_KEY", "")
GITHUB_TOKEN = os.environ.get("GITHUB_TOKEN", os.environ.get("GITHUB_PAT", ""))

# ============================================================
# RATE LIMITS (requests per second)
# ============================================================

RATE_LIMITS = {
    "europepmc": 10.0,      # Europe PMC is generous
    "arxiv": 0.33,          # 1 request per 3 seconds
    "biorxiv": 1.0,
    "github": 0.5,          # 30 req/min unauthenticated
    "youtube": 1.0,
    "brave": 1.0,
    "unpaywall": 1.0,
}

# ============================================================
# GUARDRAILS
# ============================================================

GUARDRAILS = {
    "max_downloads_per_run": 50,
    "max_disk_usage_percent": 85,
    "min_free_space_gb": 20,
    "max_concurrent_downloads": 4,
    "download_timeout_seconds": 300,
    "max_file_size_mb": 100,
    "quarantine_on_failure": True,
}

# ============================================================
# PATHS
# ============================================================

POLYMAX_HOME = "/home/user/work/polymax"
STAGING_DIR = f"{POLYMAX_HOME}/ingest_staging"
GITHUB_REPOS_DIR = f"{POLYMAX_HOME}/data/github_repos"
CHROMADB_PATH = f"{POLYMAX_HOME}/chromadb/polymath_v2"

# ============================================================
# YOUTUBE CHANNEL WHITELIST (Academic Only)
# ============================================================

YOUTUBE_WHITELIST = {
    # Universities
    "MIT OpenCourseWare",
    "Stanford",
    "Stanford Online",
    "Harvard",
    "Caltech",
    "Berkeley",
    "Yale Courses",
    "Princeton",

    # Conferences
    "NeurIPS",
    "ICML",
    "ICLR",
    "CVPR",
    "ACL",
    "EMNLP",
    "MICCAI",
    "ISBI",

    # Research Labs
    "DeepMind",
    "OpenAI",
    "Anthropic",
    "Google Research",
    "Meta AI",
    "Microsoft Research",

    # Notable Educators
    "3Blue1Brown",
    "Yannic Kilcher",
    "Two Minute Papers",
    "Lex Fridman",
    "StatQuest",
}

# ============================================================
# EXPECTED INFORMATION GAIN (EIG) CONCEPTS
# ============================================================

# Questions the system is trying to answer (for Active Inference scoring)
OPEN_QUESTIONS = {
    "h_and_e_signal_sufficiency": {
        "question": "Does H&E contain sufficient high-frequency signal for RNA inference?",
        "related_concepts": ["spatial_transcriptomics", "compressed_sensing", "information_bottleneck"],
        "uncertainty": 0.7,  # High uncertainty = high EIG potential
    },
    "optimal_encoder_architecture": {
        "question": "What encoder architecture best captures morphological-transcriptomic mapping?",
        "related_concepts": ["transformer", "graph_neural_network", "attention", "foundation_model"],
        "uncertainty": 0.5,
    },
    "colibactin_detection_sensitivity": {
        "question": "What is the detection limit for pks+ E. coli in metagenomes?",
        "related_concepts": ["gene_regulatory_network", "metagenomics"],
        "uncertainty": 0.6,
    },
    "cell_type_deconvolution": {
        "question": "How to accurately deconvolve cell types from 2µm spots?",
        "related_concepts": ["single_cell", "spatial_transcriptomics", "sparse_coding"],
        "uncertainty": 0.8,
    },
}
