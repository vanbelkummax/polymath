// Neo4j Schema for Polymath
// Run these commands to set up constraints and indexes

// Constraints (enforce uniqueness)
CREATE CONSTRAINT paper_artifact_id IF NOT EXISTS
FOR (p:Paper) REQUIRE p.artifact_id IS UNIQUE;

CREATE CONSTRAINT concept_name IF NOT EXISTS
FOR (c:CONCEPT) REQUIRE c.name IS UNIQUE;

CREATE CONSTRAINT repo_name IF NOT EXISTS
FOR (r:Repo) REQUIRE r.name IS UNIQUE;

// Indexes for performance
CREATE INDEX paper_title IF NOT EXISTS
FOR (p:Paper) ON (p.title);

CREATE INDEX paper_year IF NOT EXISTS
FOR (p:Paper) ON (p.year);

CREATE INDEX paper_doc_id IF NOT EXISTS
FOR (p:Paper) ON (p.doc_id);

// Full-text search index
CREATE FULLTEXT INDEX paper_search IF NOT EXISTS
FOR (p:Paper) ON EACH [p.title];

// Relationship types used:
// (Paper)-[:MENTIONS]->(CONCEPT)
// (CONCEPT)-[:CO_OCCURS]-(CONCEPT)
// (Paper)-[:CITES]->(Paper)
// (Code)-[:IMPLEMENTS]->(Paper)
// (Code)-[:USES]->(CONCEPT)
