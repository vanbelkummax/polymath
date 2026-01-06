#!/usr/bin/env python3
"""
Neo4j MCP Server for PolyMax
Provides tools for Claude to interact with the Knowledge Graph

Tools:
- cypher_query: Execute read-only Cypher queries
- add_paper: Add a paper with entities to the graph
- find_connections: Find paths between concepts/methods
- get_graph_stats: Get current graph statistics
"""

import json
import sys
from neo4j import GraphDatabase

# Configuration
NEO4J_URI = "bolt://localhost:7687"
NEO4J_USER = "neo4j"
NEO4J_PASSWORD = "polymathic2026"

driver = GraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USER, NEO4J_PASSWORD))

def cypher_query(query: str, read_only: bool = True) -> dict:
    """
    Execute a Cypher query on the Knowledge Graph.

    Args:
        query: Cypher query string
        read_only: If True, only allow read operations (default: True)

    Returns:
        Query results as JSON
    """
    # Safety check for read_only mode
    if read_only:
        forbidden = ['CREATE', 'MERGE', 'DELETE', 'SET', 'REMOVE', 'DROP']
        query_upper = query.upper()
        for word in forbidden:
            if word in query_upper:
                return {"error": f"Read-only mode: {word} not allowed"}

    try:
        with driver.session() as session:
            result = session.run(query)
            records = [dict(record) for record in result]
            return {"results": records, "count": len(records)}
    except Exception as e:
        return {"error": str(e)}

def add_paper(title: str, entities: list, relationships: list = None) -> dict:
    """
    Add a paper and its entities to the Knowledge Graph.

    Args:
        title: Paper title
        entities: List of {"name": str, "type": "method|concept"}
        relationships: Optional list of {"source": str, "target": str, "type": str}

    Returns:
        Confirmation with paper_id
    """
    paper_id = title.lower().replace(' ', '_')[:50]

    try:
        with driver.session() as session:
            # Create paper node
            session.run("""
                MERGE (p:Paper {id: $id})
                SET p.title = $title,
                    p.added_by = 'mcp',
                    p.timestamp = timestamp()
            """, id=paper_id, title=title)

            # Add entities
            for entity in entities:
                entity_type = entity['type'].upper()
                session.run(f"""
                    MERGE (e:{entity_type} {{name: $name}})
                    WITH e
                    MATCH (p:Paper {{id: $paper_id}})
                    MERGE (p)-[:MENTIONS]->(e)
                """, name=entity['name'], paper_id=paper_id)

            # Add relationships
            if relationships:
                for rel in relationships:
                    session.run("""
                        MATCH (s {name: $source})
                        MATCH (t {name: $target})
                        MERGE (s)-[r:RELATES_TO {type: $rel_type}]->(t)
                    """, source=rel['source'], target=rel['target'],
                         rel_type=rel.get('type', 'RELATES_TO'))

            return {
                "success": True,
                "paper_id": paper_id,
                "entities_added": len(entities),
                "relationships_added": len(relationships) if relationships else 0
            }
    except Exception as e:
        return {"error": str(e)}

def find_connections(source: str, target: str, max_hops: int = 4) -> dict:
    """
    Find paths between two concepts or methods.

    Args:
        source: Starting entity name
        target: Target entity name
        max_hops: Maximum path length (default: 4)

    Returns:
        Paths found and intermediate nodes
    """
    query = f"""
        MATCH path = shortestPath(
            (s {{name: $source}})-[*..{max_hops}]-(t {{name: $target}})
        )
        RETURN [node in nodes(path) | coalesce(node.name, node.title)] AS path,
               length(path) AS hops
        LIMIT 5
    """

    try:
        with driver.session() as session:
            result = session.run(query, source=source, target=target)
            paths = []
            for record in result:
                paths.append({
                    "path": record["path"],
                    "hops": record["hops"]
                })
            return {"paths": paths, "count": len(paths)}
    except Exception as e:
        return {"error": str(e)}

def get_graph_stats() -> dict:
    """
    Get current Knowledge Graph statistics.

    Returns:
        Node counts, relationship counts, top entities
    """
    try:
        with driver.session() as session:
            # Node counts
            result = session.run("""
                MATCH (n)
                RETURN labels(n)[0] AS type, count(*) AS count
                ORDER BY count DESC
            """)
            node_counts = {record["type"]: record["count"] for record in result}

            # Relationship counts
            result = session.run("""
                MATCH ()-[r]->()
                RETURN type(r) AS type, count(*) AS count
                ORDER BY count DESC
            """)
            rel_counts = {record["type"]: record["count"] for record in result}

            # Top methods
            result = session.run("""
                MATCH (m:METHOD)<-[:MENTIONS]-(p:Paper)
                RETURN m.name AS name, count(p) AS papers
                ORDER BY papers DESC
                LIMIT 5
            """)
            top_methods = [{"name": r["name"], "papers": r["papers"]} for r in result]

            # Top concepts
            result = session.run("""
                MATCH (c:CONCEPT)<-[:MENTIONS]-(p:Paper)
                RETURN c.name AS name, count(p) AS papers
                ORDER BY papers DESC
                LIMIT 5
            """)
            top_concepts = [{"name": r["name"], "papers": r["papers"]} for r in result]

            return {
                "nodes": node_counts,
                "relationships": rel_counts,
                "top_methods": top_methods,
                "top_concepts": top_concepts
            }
    except Exception as e:
        return {"error": str(e)}

def find_unexplored_combinations(method_type: str = None) -> dict:
    """
    Find method-concept combinations that appear only once (research gaps).

    Returns:
        List of underexplored combinations
    """
    try:
        with driver.session() as session:
            result = session.run("""
                MATCH (p:Paper)-[:MENTIONS]->(m:METHOD)
                MATCH (p)-[:MENTIONS]->(c:CONCEPT)
                WITH m.name AS method, c.name AS concept, count(p) AS papers
                WHERE papers = 1
                RETURN method, concept
                LIMIT 20
            """)
            combinations = [{"method": r["method"], "concept": r["concept"]} for r in result]
            return {"unexplored": combinations, "count": len(combinations)}
    except Exception as e:
        return {"error": str(e)}

def generate_hypothesis(topic: str) -> dict:
    """
    Generate a polymathic hypothesis by finding cross-domain connections.

    Args:
        topic: Starting concept or method

    Returns:
        Hypothesis with supporting evidence from graph
    """
    try:
        with driver.session() as session:
            # Find what connects to this topic
            result = session.run("""
                MATCH (start {name: $topic})-[r]-(connected)
                RETURN labels(connected)[0] AS type,
                       connected.name AS name,
                       type(r) AS relationship
                LIMIT 10
            """, topic=topic)

            connections = [dict(r) for r in result]

            # Find papers mentioning this topic
            result = session.run("""
                MATCH (p:Paper)-[:MENTIONS]->(e {name: $topic})
                RETURN p.title AS paper
                LIMIT 5
            """, topic=topic)

            papers = [r["paper"] for r in result]

            return {
                "topic": topic,
                "connections": connections,
                "supporting_papers": papers,
                "hypothesis_template": f"Cross-domain insight: {topic} relates to " +
                                       ", ".join([c["name"] for c in connections[:3]])
            }
    except Exception as e:
        return {"error": str(e)}


# MCP Protocol Handler
def handle_request(request: dict) -> dict:
    """Handle MCP tool requests"""
    tool = request.get("tool")
    params = request.get("params", {})

    if tool == "cypher_query":
        return cypher_query(params.get("query", ""), params.get("read_only", True))
    elif tool == "add_paper":
        return add_paper(
            params.get("title", ""),
            params.get("entities", []),
            params.get("relationships", [])
        )
    elif tool == "find_connections":
        return find_connections(
            params.get("source", ""),
            params.get("target", ""),
            params.get("max_hops", 4)
        )
    elif tool == "get_graph_stats":
        return get_graph_stats()
    elif tool == "find_unexplored":
        return find_unexplored_combinations()
    elif tool == "generate_hypothesis":
        return generate_hypothesis(params.get("topic", ""))
    else:
        return {"error": f"Unknown tool: {tool}"}


if __name__ == "__main__":
    # Test mode
    if len(sys.argv) > 1 and sys.argv[1] == "test":
        print("Testing Neo4j MCP Server...")
        print("\n1. Graph Stats:")
        print(json.dumps(get_graph_stats(), indent=2))

        print("\n2. Find Connections (UNI → morphogenesis):")
        print(json.dumps(find_connections("UNI", "morphogenesis"), indent=2))

        print("\n3. Unexplored Combinations:")
        print(json.dumps(find_unexplored_combinations(), indent=2))

        print("\n4. Generate Hypothesis (reaction-diffusion):")
        print(json.dumps(generate_hypothesis("reaction-diffusion"), indent=2))
    else:
        # MCP mode - read from stdin
        for line in sys.stdin:
            request = json.loads(line)
            response = handle_request(request)
            print(json.dumps(response))
            sys.stdout.flush()
