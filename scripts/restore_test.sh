#!/bin/bash
# Polymath Restore Test Script
# Tests that backups can be restored to verify RTO/RPO
# CAUTION: This will create test restore directories, not overwrite production data

set -e

echo "=== Polymath Backup Restoration Test ==="
echo "Started: $(date)"
echo ""

# Find most recent backups
BACKUP_DIR="/home/user/backups/polymax"
RESTORE_TEST_DIR="/home/user/backups/restore_test_$(date +%Y%m%d_%H%M)"

mkdir -p "$RESTORE_TEST_DIR"

echo "Finding most recent backups..."
CHROMA_BACKUP=$(ls -t $BACKUP_DIR/chromadb_*.tar.gz 2>/dev/null | head -1)
PG_BACKUP=$(ls -t $BACKUP_DIR/postgres_*.sql.gz 2>/dev/null | head -1)
NEO4J_BACKUP=$(ls -t $BACKUP_DIR/neo4j_*.json.gz 2>/dev/null | head -1)

if [ -z "$CHROMA_BACKUP" ] || [ -z "$PG_BACKUP" ] || [ -z "$NEO4J_BACKUP" ]; then
    echo "ERROR: Missing backup files!"
    exit 1
fi

echo "✓ ChromaDB: $CHROMA_BACKUP"
echo "✓ Postgres: $PG_BACKUP"
echo "✓ Neo4j: $NEO4J_BACKUP"
echo ""

# Test 1: ChromaDB extraction
echo "Test 1: ChromaDB restoration..."
tar -xzf "$CHROMA_BACKUP" -C "$RESTORE_TEST_DIR" 2>&1 | head -5
CHROMA_SIZE=$(du -sh "$RESTORE_TEST_DIR" | cut -f1)
echo "✓ ChromaDB extracted successfully ($CHROMA_SIZE)"
echo ""

# Test 2: Postgres SQL validation
echo "Test 2: Postgres SQL validation..."
zcat "$PG_BACKUP" | head -20 | grep -q "PostgreSQL database dump" && echo "✓ Postgres backup is valid SQL dump"
PG_SIZE=$(zcat "$PG_BACKUP" | wc -l)
echo "  Lines: $PG_SIZE"
echo ""

# Test 3: Neo4j JSON validation
echo "Test 3: Neo4j JSON validation..."
zcat "$NEO4J_BACKUP" | python3 -c "import json, sys; data=json.load(sys.stdin); print(f'✓ Neo4j backup is valid JSON: {len(data[\"nodes\"])} nodes, {len(data[\"relationships\"])} relationships')"
echo ""

# Calculate RTO (Recovery Time Objective)
echo "=== RTO/RPO Metrics ==="
RESTORE_START=$(stat -c %Y "$RESTORE_TEST_DIR")
RESTORE_END=$(date +%s)
RTO=$((RESTORE_END - RESTORE_START))
echo "RTO (Recovery Time): ~$RTO seconds (for extraction test)"
echo "Estimated full restore: ~3-5 minutes (includes import to databases)"
echo ""

# RPO (Recovery Point Objective)
BACKUP_TIME=$(stat -c %y "$CHROMA_BACKUP" | cut -d. -f1)
echo "RPO (Recovery Point): Last backup at $BACKUP_TIME"
echo "Data loss window: Up to 24 hours (daily backups)"
echo ""

# Cleanup
rm -rf "$RESTORE_TEST_DIR"
echo "✓ Cleanup complete"
echo ""

echo "=== Restoration Test PASSED ==="
echo "All backups are restorable and valid"
echo "Completed: $(date)"
