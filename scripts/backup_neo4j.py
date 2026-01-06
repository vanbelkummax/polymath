#!/usr/bin/env python3
"""
Neo4j Backup Script
Exports all nodes and relationships to JSON for backup/restore
"""

import json
import sys
from datetime import datetime
from neo4j import GraphDatabase
from neo4j.time import DateTime, Date, Time, Duration
from neo4j.spatial import Point

def backup_neo4j(output_file, uri="bolt://localhost:7687", user="neo4j", password="polymathic2026"):
    """
    Backup Neo4j graph to JSON file

    Args:
        output_file: Path to output JSON file
        uri: Neo4j connection URI
        user: Neo4j username
        password: Neo4j password
    """
    driver = GraphDatabase.driver(uri, auth=(user, password))

    backup_data = {
        "timestamp": datetime.now().isoformat(),
        "nodes": [],
        "relationships": []
    }

    with driver.session() as session:
        # Export all nodes
        print("Exporting nodes...")
        result = session.run("MATCH (n) RETURN n, labels(n) as labels, id(n) as id")
        for record in result:
            node = record["n"]
            backup_data["nodes"].append({
                "id": record["id"],
                "labels": record["labels"],
                "properties": dict(node)
            })

        print(f"Exported {len(backup_data['nodes'])} nodes")

        # Export all relationships
        print("Exporting relationships...")
        result = session.run("""
            MATCH (a)-[r]->(b)
            RETURN id(a) as start_id, type(r) as type, id(b) as end_id, properties(r) as props
        """)
        for record in result:
            backup_data["relationships"].append({
                "start_id": record["start_id"],
                "type": record["type"],
                "end_id": record["end_id"],
                "properties": record["props"]
            })

        print(f"Exported {len(backup_data['relationships'])} relationships")

    driver.close()

    # Custom JSON encoder to handle Neo4j types
    class Neo4jEncoder(json.JSONEncoder):
        def default(self, obj):
            # Handle Neo4j temporal types
            if isinstance(obj, (DateTime, Date, Time)):
                return obj.isoformat()
            # Handle Neo4j Duration
            elif isinstance(obj, Duration):
                return str(obj)
            # Handle Neo4j Point (spatial)
            elif isinstance(obj, Point):
                return {"type": "Point", "coordinates": list(obj), "srid": obj.srid}
            # Handle Python datetime
            elif isinstance(obj, datetime):
                return obj.isoformat()
            return super().default(obj)

    # Write to file
    with open(output_file, 'w') as f:
        json.dump(backup_data, f, indent=2, cls=Neo4jEncoder)

    print(f"Backup complete: {output_file}")
    return True

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: backup_neo4j.py <output_file>")
        sys.exit(1)

    output_file = sys.argv[1]
    try:
        backup_neo4j(output_file)
    except Exception as e:
        print(f"ERROR: {e}", file=sys.stderr)
        sys.exit(1)
