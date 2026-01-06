#!/bin/bash
# Quick test script for concept linking

echo "=== Concept Linking System Test ==="
echo

# Test 1: Script exists and is executable
echo "Test 1: Script exists"
if [ -x "/home/user/work/polymax/scripts/fix_concept_links.py" ]; then
    echo "✓ Script found and executable"
else
    echo "✗ Script not found or not executable"
    exit 1
fi

# Test 2: Can load resources
echo
echo "Test 2: Check dependencies"
python3 -c "
import chromadb
from sentence_transformers import SentenceTransformer
from neo4j import GraphDatabase
print('✓ All dependencies available')
"

# Test 3: Neo4j connection
echo
echo "Test 3: Neo4j connection"
python3 /home/user/work/polymax/polymath_cli.py graph "RETURN 'OK' as status" | grep -q OK
if [ $? -eq 0 ]; then
    echo "✓ Neo4j connected"
else
    echo "✗ Neo4j connection failed"
    exit 1
fi

# Test 4: ChromaDB accessible
echo
echo "Test 4: ChromaDB accessible"
COUNT=$(python3 /home/user/work/polymax/polymath_cli.py stats | grep "ChromaDB Chunks" | awk '{print $3}')
if [ ! -z "$COUNT" ]; then
    echo "✓ ChromaDB accessible ($COUNT chunks)"
else
    echo "✗ ChromaDB not accessible"
    exit 1
fi

# Test 5: Check current concept state
echo
echo "Test 5: Check target concepts (before fix)"
python3 /home/user/work/polymax/polymath_cli.py graph \
  "MATCH (c:CONCEPT) 
   WHERE c.name IN ['maximum_entropy', 'do_calculus', 'sparse_coding', 'reaction_diffusion', 'attractor_dynamics']
   RETURN c.name as concept, size([(c)<-[:MENTIONS]-(p:Paper) | p]) as papers" | grep -v "==="

echo
echo "=== All Tests Passed ==="
echo
echo "Ready to run:"
echo "  python3 /home/user/work/polymax/scripts/fix_concept_links.py --dry-run"
echo "  python3 /home/user/work/polymax/scripts/fix_concept_links.py --live"
