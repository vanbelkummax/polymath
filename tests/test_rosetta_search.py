import json
import chromadb
from sentence_transformers import SentenceTransformer
import numpy as np
from pathlib import Path
from datetime import datetime
import sys
import os

# Add root to path for imports
sys.path.insert(0, '/home/user/work/polymax')
from lib.rosetta_query_expander import expand_query

# Configuration
GOLDEN_SET_PATH = Path("/home/user/work/polymax/tests/data/golden_queries.json")
CHROMADB_PATH = "/home/user/work/polymax/chromadb/polymath_v2"
COLLECTION_NAME = "polymath_corpus"
EMBEDDING_MODEL = "all-mpnet-base-v2"
TOP_K = 10

def load_golden_set():
    with open(GOLDEN_SET_PATH) as f:
        return json.load(f)

def get_db_client():
    return chromadb.PersistentClient(path=CHROMADB_PATH)

def get_model():
    print(f"Loading embedding model: {EMBEDDING_MODEL}...")
    return SentenceTransformer(EMBEDDING_MODEL)

def evaluate_query(query_data, results, top_k):
    """
    Evaluate if results match expectations.
    Returns (score, reason)
    """
    found_matches = []
    
    # Check expected concepts (keywords in metadata or text)
    expected_concepts = query_data.get("expected_concepts", [])
    expected_papers = query_data.get("expected_papers_keywords", [])
    expected_repos = query_data.get("expected_repos", [])
    
    hits = 0
    
    for i, (doc, meta) in enumerate(zip(results['documents'][0], results['metadatas'][0])):
        if i >= top_k:
            break
            
        text_content = (doc + " " + str(meta)).lower()
        
        # Check concepts
        for concept in expected_concepts:
            if concept.lower() in text_content:
                found_matches.append(f"Concept: {concept} (Rank {i+1})")
                hits += 1
                
        # Check papers
        for keyword in expected_papers:
            if keyword.lower() in text_content:
                found_matches.append(f"Paper: {keyword} (Rank {i+1})")
                hits += 1

        # Check repos
        for repo in expected_repos:
            if repo.lower() in text_content:
                found_matches.append(f"Repo: {repo} (Rank {i+1})")
                hits += 1
                
    if hits > 0:
        return 1.0, f"Found matches: {', '.join(list(set(found_matches))[:3])}..."
    else:
        return 0.0, "No expected concepts or papers found in top results"

def run_tests():
    print(f"Running Search Quality Tests (With Rosetta Stone)")
    print(f"Date: {datetime.now().isoformat()}")
    print(f"DB Path: {CHROMADB_PATH}")
    
    queries = load_golden_set()
    client = get_db_client()
    collection = client.get_collection(COLLECTION_NAME)
    model = get_model()
    
    total_score = 0
    results_log = []
    
    print(f"\nExecuting {len(queries)} queries...")
    print("-" * 60)
    
    for q in queries:
        query_text = q['query']
        
        # Apply Rosetta Stone Expansion
        print(f"\nQuery: {query_text}")
        expanded_query = expand_query(query_text)
        if expanded_query != query_text:
            print(f"  Expanded: {expanded_query[:100]}...")
        else:
            print("  (No expansion)")
        
        # Embed and search
        embedding = model.encode([expanded_query]).tolist()
        results = collection.query(
            query_embeddings=embedding,
            n_results=TOP_K,
            include=["documents", "metadatas"]
        )
        
        # Evaluate
        score, reason = evaluate_query(q, results, q.get("min_relevant_rank", TOP_K))
        total_score += score
        
        status = "PASS" if score > 0 else "FAIL"
        print(f"  Status: {status} (Score: {score})")
        print(f"  Reason: {reason}")
        
        results_log.append({
            "id": q['id'],
            "query": query_text,
            "expanded_query": expanded_query,
            "status": status,
            "score": score,
            "reason": reason
        })
        
    print("-" * 60)
    avg_score = total_score / len(queries)
    print(f"Final Score: {total_score}/{len(queries)} ({avg_score*100:.1f}%)")
    
    # Save results
    output_file = f"/home/user/work/polymax/tests/results_rosetta_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    with open(output_file, 'w') as f:
        json.dump(results_log, f, indent=2)
        
    if avg_score < 0.9: # Raise bar to 90%
        print("\n❌ FAILED: Average score below 90%")
        sys.exit(1)
    else:
        print("\n✅ PASSED: Search quality excellent (>=90%)")
        sys.exit(0)

if __name__ == "__main__":
    run_tests()
