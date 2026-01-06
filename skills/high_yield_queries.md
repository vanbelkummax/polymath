# High-Yield Queries
## Proven Cypher and RAG Queries for Polymathic Discovery

*Last Updated: 2026-01-04*

---

## Neo4j Cypher Queries

### 1. Find Cross-Domain Bridging Methods
```cypher
MATCH (p:Paper)-[:MENTIONS]->(m:METHOD)
MATCH (p)-[:MENTIONS]->(c:CONCEPT)
WITH m.name AS method, collect(DISTINCT c.name) AS concepts
WHERE size(concepts) >= 3
RETURN method, concepts, size(concepts) AS breadth
ORDER BY breadth DESC
LIMIT 10
```
**Use case**: Identify methods that connect multiple research areas

---

### 2. Find Unexplored Combinations
```cypher
MATCH (p:Paper)-[:MENTIONS]->(m:METHOD)
MATCH (p)-[:MENTIONS]->(c:CONCEPT)
WITH m.name AS method, c.name AS concept, count(p) AS papers
WHERE papers = 1
RETURN method, concept
LIMIT 20
```
**Use case**: Research gaps - combinations tried only once

---

### 3. Find Papers Connecting Classic + Modern
```cypher
MATCH (p:Paper)-[:MENTIONS]->(classic:CONCEPT)
MATCH (p)-[:MENTIONS]->(modern:METHOD)
WHERE classic.name IN ['morphogenesis', 'pattern formation', 'feedback loop', 'self-organization']
  AND modern.name IN ['UNI', 'transformer', 'foundation model', 'RAG']
RETURN p.title, classic.name, modern.name
```
**Use case**: Find papers bridging classic biology and modern AI

---

### 4. Method Evolution (Co-occurrence)
```cypher
MATCH (p:Paper)-[:MENTIONS]->(m1:METHOD)
MATCH (p)-[:MENTIONS]->(m2:METHOD)
WHERE m1.name < m2.name
WITH m1.name AS method1, m2.name AS method2, count(p) AS papers
WHERE papers >= 2
RETURN method1, method2, papers
ORDER BY papers DESC
```
**Use case**: Track which methods are used together (evolution proxy)

---

### 5. Hub Concepts (Most Connected)
```cypher
MATCH (c:CONCEPT)<-[:MENTIONS]-(p:Paper)
WITH c.name AS concept, count(p) AS papers
ORDER BY papers DESC
RETURN concept, papers
LIMIT 10
```
**Use case**: Find the most central concepts in the corpus

---

### 6. Shortest Path Between Concepts
```cypher
MATCH path = shortestPath(
  (start:CONCEPT {name: $concept1})-[*..5]-(end:CONCEPT {name: $concept2})
)
RETURN [node in nodes(path) | coalesce(node.name, node.title)] AS path,
       length(path) AS hops
```
**Use case**: Trace connections between distant concepts

---

## RAG Semantic Search Patterns

### 1. Application Query
```
"[METHOD] applied to [DOMAIN]"
Example: "attention mechanism applied to pathology"
```

### 2. Combination Query
```
"combining [A] with [B] for [TASK]"
Example: "combining foundation models with spatial deconvolution"
```

### 3. Challenge Query
```
"challenges in [FIELD]"
Example: "challenges in spatial transcriptomics at single-cell resolution"
```

### 4. Comparison Query
```
"[METHOD1] vs [METHOD2] for [TASK]"
Example: "Cell2location vs RCTD for deconvolution"
```

### 5. History Query
```
"history of [CONCEPT]"
Example: "history of reaction-diffusion in developmental biology"
```

---

## Combined KG + RAG Workflows

### Workflow 1: Deep Topic Exploration
1. KG: Find all connections to topic
2. RAG: Get context from highest-connected papers
3. KG: Find unexplored combinations with this topic
4. Synthesize: Cross-domain hypothesis

### Workflow 2: Paper Addition Pipeline
1. RAG: Search if similar paper exists
2. If new: Extract entities
3. KG: Add paper with entities
4. KG: Check new connections formed

### Workflow 3: Hypothesis Validation
1. KG: Check if hypothesis already explored
2. RAG: Find supporting/contradicting evidence
3. KG: Find related successful approaches
4. Generate: Kill-shot test design

---

*Add new high-yield queries as you discover them*
