// Neo4j 5.x: constraints, range/fulltext/vector indexes.
// VECTOR_DIMENSIONS deve corresponder ao modelo de embeddings configurado no .env.
CREATE CONSTRAINT entity_id IF NOT EXISTS FOR (n:Entity) REQUIRE n.id IS UNIQUE;
CREATE CONSTRAINT source_id IF NOT EXISTS FOR (s:Source) REQUIRE s.id IS UNIQUE;
CREATE CONSTRAINT chunk_id IF NOT EXISTS FOR (c:Chunk) REQUIRE c.id IS UNIQUE;
CREATE INDEX entity_name IF NOT EXISTS FOR (n:Entity) ON (n.name);
CREATE INDEX source_date IF NOT EXISTS FOR (s:Source) ON (s.date);
CREATE FULLTEXT INDEX entity_text IF NOT EXISTS FOR (n:Entity) ON EACH [n.name, n.synonyms];
CREATE FULLTEXT INDEX chunk_text IF NOT EXISTS FOR (c:Chunk) ON EACH [c.text];
CREATE VECTOR INDEX chunk_embedding_768 IF NOT EXISTS FOR (c:Chunk) ON (c.embedding)
  OPTIONS {indexConfig: {`vector.dimensions`: 768, `vector.similarity_function`: 'cosine'}};
