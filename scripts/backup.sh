#!/bin/bash
# Polymath System Backup Script
# RTO: 1 hour, RPO: 24 hours
# Run daily via cron: 0 2 * * * /home/user/work/polymax/scripts/backup.sh

set -e

BACKUP_DIR="/home/user/backups/polymax"
DATE=$(date +%Y%m%d_%H%M)
LOG_FILE="/home/user/work/polymax/logs/backup_${DATE}.log"

echo "Starting Polymath backup: $(date)" | tee -a "$LOG_FILE"

# Create backup directory
mkdir -p "$BACKUP_DIR"

# 1. Backup ChromaDB (vector store)
echo "Backing up ChromaDB..." | tee -a "$LOG_FILE"
CHROMA_BACKUP="$BACKUP_DIR/chromadb_${DATE}"
cp -r /home/user/work/polymax/chromadb "$CHROMA_BACKUP"
tar -czf "${CHROMA_BACKUP}.tar.gz" "$CHROMA_BACKUP"
rm -rf "$CHROMA_BACKUP"
echo "ChromaDB backup complete: ${CHROMA_BACKUP}.tar.gz" | tee -a "$LOG_FILE"

# 2. Backup Postgres (metadata)
echo "Backing up Postgres..." | tee -a "$LOG_FILE"
PG_BACKUP="$BACKUP_DIR/postgres_${DATE}.sql"
pg_dump -U polymath polymath > "$PG_BACKUP"
gzip "$PG_BACKUP"
echo "Postgres backup complete: ${PG_BACKUP}.gz" | tee -a "$LOG_FILE"

# 3. Backup Neo4j (graph)
echo "Backing up Neo4j..." | tee -a "$LOG_FILE"
NEO4J_BACKUP="$BACKUP_DIR/neo4j_${DATE}.json"
python3 /home/user/work/polymax/scripts/backup_neo4j.py "$NEO4J_BACKUP" 2>&1 | tee -a "$LOG_FILE"
gzip "$NEO4J_BACKUP"
echo "Neo4j backup complete: ${NEO4J_BACKUP}.gz" | tee -a "$LOG_FILE"

# 4. Backup configuration files
echo "Backing up configs..." | tee -a "$LOG_FILE"
CONFIG_BACKUP="$BACKUP_DIR/config_${DATE}"
mkdir -p "$CONFIG_BACKUP"
cp /home/user/work/polymax/config.toml "$CONFIG_BACKUP/"
cp /home/user/work/polymax/.env "$CONFIG_BACKUP/" 2>/dev/null || true
cp /home/user/.mcp.json "$CONFIG_BACKUP/" 2>/dev/null || true
tar -czf "${CONFIG_BACKUP}.tar.gz" "$CONFIG_BACKUP"
rm -rf "$CONFIG_BACKUP"
echo "Config backup complete" | tee -a "$LOG_FILE"

# 5. Keep only last 7 days of backups
echo "Pruning old backups..." | tee -a "$LOG_FILE"
find "$BACKUP_DIR" -name "*.tar.gz" -mtime +7 -delete
find "$BACKUP_DIR" -name "*.sql.gz" -mtime +7 -delete
find "$BACKUP_DIR" -name "*.dump" -mtime +7 -delete

# 6. Calculate backup size
BACKUP_SIZE=$(du -sh "$BACKUP_DIR" | cut -f1)
echo "Total backup size: $BACKUP_SIZE" | tee -a "$LOG_FILE"

# 7. Verify backups exist
if [ ! -f "${CHROMA_BACKUP}.tar.gz" ]; then
    echo "ERROR: ChromaDB backup failed!" | tee -a "$LOG_FILE"
    exit 1
fi

if [ ! -f "${PG_BACKUP}.gz" ]; then
    echo "ERROR: Postgres backup failed!" | tee -a "$LOG_FILE"
    exit 1
fi

if [ ! -f "${NEO4J_BACKUP}.gz" ]; then
    echo "ERROR: Neo4j backup failed!" | tee -a "$LOG_FILE"
    exit 1
fi

echo "Backup complete: $(date)" | tee -a "$LOG_FILE"
echo "Backups stored in: $BACKUP_DIR" | tee -a "$LOG_FILE"
