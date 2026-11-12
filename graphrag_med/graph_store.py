"""
Armazenamento exclusivo do grafo LPG em Neo4j.

Responsabilidades:
- Persistência de entidades, fontes, chunks e relações.
- Busca vetorial e Fulltext.
- Recuperação de caminhos multi-hop.
- Execução controlada de Cypher somente leitura.
- Controle de importação.
- Recuperação completa de nós e relacionamentos.
"""

import os
import re
from itertools import combinations


# ============================================================
# CONFIGURAÇÕES DO GRAFO
# ============================================================

RELS = [
    "INTERACTS_WITH",
    "TREATS",
    "CONTRAINDICATED_IN",
    "CAUSES",
    "BELONGS_TO",
]

LABELS = {
    "Drug",
    "Condition",
    "AdverseEffect",
    "DrugClass",
    "Procedure",
    "Anatomy",
}

VECTOR_INDEX = "chunk_embedding_768"
CHUNK_FULLTEXT_INDEX = "chunk_text"
ENTITY_FULLTEXT_INDEX = "entity_text"


# ============================================================
# CONSULTAS MULTI-HOP
# ============================================================

def path_queries(seeds, hops, hint=None, limit=30):
    """Gera consultas Cypher multi-hop exclusivamente entre entidades."""

    seeds = list(dict.fromkeys(str(x) for x in (seeds or [])))

    if not seeds:
        return []

    hops = max(1, min(int(hops), 5))
    limit = max(1, int(limit))

    conditions = (
        "all(n IN nodes(p) WHERE n:Entity) "
        "AND all(r IN rels "
        "WHERE type(r) IN $allowed_rels "
        "AND coalesce(r.negated, false) = false)"
    )

    if len(seeds) == 1:
        extra = ""

        if hint:
            extra = " AND type(rels[0]) IN $hint"

        query = (
            "MATCH p = "
            "(a:Entity {id: $a})"
            f"-[rels*1..{hops}]-(x:Entity) "
            f"WHERE {conditions}"
            f"{extra} "
            "RETURN p LIMIT $limit"
        )

        params = {
            "a": seeds[0],
            "limit": limit,
            "hint": list(hint or []),
            "allowed_rels": RELS,
        }

        return [(query, params)]

    queries = []

    for a, b in combinations(seeds, 2):
        query = (
            "MATCH p = "
            "(a:Entity {id: $a})"
            f"-[rels*1..{hops}]-(b:Entity {{id: $b}}) "
            f"WHERE {conditions} "
            "RETURN p "
            "ORDER BY length(p) "
            "LIMIT $limit"
        )

        params = {
            "a": a,
            "b": b,
            "limit": limit,
            "allowed_rels": RELS,
        }

        queries.append((query, params))

    return queries


# ============================================================
# CLASSE PRINCIPAL
# ============================================================

def shortest_path_query(max_hops=5):
    """Cypher do menor caminho entre duas entidades (sem relações negadas)."""

    max_hops = max(1, min(int(max_hops), 6))

    return (
        "MATCH (a:Entity {id: $source}), (b:Entity {id: $target})\n"
        "MATCH p = shortestPath("
        "(a)-[:" + "|".join(RELS) + "*.." + str(max_hops) + "]-(b))\n"
        "WHERE all(r IN relationships(p) "
        "WHERE coalesce(r.negated, false) = false)\n"
        "RETURN p LIMIT 1"
    )


class Neo4jStore:
    """Persistência exclusiva em Neo4j 5.x."""

    name = "Neo4j"

    def __init__(
        self,
        uri,
        user,
        password,
        database="neo4j",
        schema_path=None,
    ):
        from neo4j import GraphDatabase

        self.database = database

        self.vector_dimensions = int(
            os.getenv("VECTOR_DIMENSIONS", "768")
        )

        self.driver = GraphDatabase.driver(
            uri,
            auth=(user, password),
            connection_timeout=30,
            max_connection_lifetime=300,
        )

        try:
            self.driver.verify_connectivity()

            if schema_path:
                self.ensure_schema(schema_path)

        except Exception:
            self.driver.close()
            raise

    # ========================================================
    # CONEXÃO
    # ========================================================

    def close(self):
        if self.driver:
            self.driver.close()

    def _run(self, query, **params):
        with self.driver.session(
            database=self.database
        ) as session:
            return list(session.run(query, **params))

    # ========================================================
    # ESQUEMA
    # ========================================================

    def ensure_schema(self, path):
        with open(path, "r", encoding="utf-8") as file:
            content = "\n".join(
                line
                for line in file
                if not line.strip().startswith("//")
            )

        statements = [
            statement.strip()
            for statement in content.split(";")
            if statement.strip()
        ]

        for statement in statements:
            self._run(statement)

    # ========================================================
    # PROCESSAMENTO EM LOTES
    # ========================================================

    @staticmethod
    def _batches(items, size=100):
        size = max(1, int(size))

        for start in range(0, len(items), size):
            yield items[start:start + size]

    # ========================================================
    # ENTIDADES
    # ========================================================

    def add_entities(self, entities):
        entities = list(entities or [])

        for label in LABELS:
            rows = []

            for entity in entities:
                if entity.get("type") != label:
                    continue

                rows.append({
                    "id": str(entity["id"]),
                    "name": entity["name"],
                    "synonyms": list(
                        entity.get("synonyms") or []
                    ),
                })

            query = (
                "UNWIND $rows AS row "
                "MERGE (n:Entity {id:row.id}) "
                f"SET n:{label}, "
                "n.name=row.name, "
                "n.type=$label, "
                "n.synonyms=row.synonyms"
            )

            for batch in self._batches(rows, 250):
                self._run(
                    query,
                    rows=batch,
                    label=label,
                )

    def add_entity(self, entity):
        label = entity["type"]

        if label not in LABELS:
            raise ValueError(
                "Tipo de entidade inválido: " + str(label)
            )

        query = (
            "MERGE (n:Entity {id:$id}) "
            f"SET n:{label}, "
            "n.name=$name, "
            "n.type=$type, "
            "n.synonyms=$synonyms"
        )

        self._run(
            query,
            id=str(entity["id"]),
            name=entity["name"],
            type=label,
            synonyms=list(entity.get("synonyms") or []),
        )

    # ========================================================
    # FONTES
    # ========================================================

    def add_sources(self, docs):
        rows = []

        for doc in docs:
            meta = doc.get("metadata") or {}

            rows.append({
                "id": doc["id"],
                "title": doc.get(
                    "title",
                    "Documento sem título",
                ),
                "date": (
                    doc.get("date")
                    or meta.get("date")
                    or meta.get("data_publicacao")
                ),
                "dataset_id": (
                    meta.get("dataset_id")
                    or meta.get("dataset")
                    or "ChatBulário"
                ),
            })

        query = (
            "UNWIND $rows AS row "
            "MERGE (s:Source {id:row.id}) "
            "SET s.title=row.title, "
            "s.date=row.date, "
            "s.dataset_id=row.dataset_id"
        )

        for batch in self._batches(rows, 500):
            self._run(query, rows=batch)

    def add_source(self, doc):
        meta = doc.get("metadata") or {}

        query = (
            "MERGE (s:Source {id:$id}) "
            "SET s.title=$title, "
            "s.date=$date, "
            "s.dataset_id=$dataset_id"
        )

        self._run(
            query,
            id=doc["id"],
            title=doc.get(
                "title",
                "Documento sem título",
            ),
            date=(
                doc.get("date")
                or meta.get("date")
                or meta.get("data_publicacao")
            ),
            dataset_id=(
                meta.get("dataset_id")
                or meta.get("dataset")
                or "ChatBulário"
            ),
        )

    # ========================================================
    # CHUNKS
    # ========================================================

    def _validate_embedding(self, embedding):
        if embedding is None:
            raise ValueError("Embedding não pode ser None.")

        vector = [float(value) for value in embedding]

        if len(vector) != self.vector_dimensions:
            raise ValueError(
                "Dimensão do embedding incompatível: "
                "{}. Esperado: {}.".format(
                    len(vector),
                    self.vector_dimensions,
                )
            )

        return vector

    def add_chunks(self, chunks):
        """
        Grava chunks em lotes.

        A gravação do chunk não depende da existência de entidades.
        A relação FROM é criada quando a fonte existe.
        """

        rows = []

        for chunk in chunks:
            rows.append({
                "id": chunk["id"],
                "text": chunk["text"],
                "embedding": self._validate_embedding(
                    chunk["embedding"]
                ),
                "source_id": chunk.get("source_id"),
                "entities": list(
                    chunk.get("entities") or []
                ),
            })

        if not rows:
            return

        query_chunks = (
            "UNWIND $rows AS row "
            "MERGE (c:Chunk {id:row.id}) "
            "SET c.text=row.text, "
            "c.embedding=row.embedding, "
            "c.source_id=row.source_id "
            "WITH c, row "
            "OPTIONAL MATCH (s:Source {id:row.source_id}) "
            "FOREACH (_ IN CASE WHEN s IS NULL "
            "THEN [] ELSE [1] END | "
            "MERGE (c)-[:FROM]->(s))"
        )

        query_entities = (
            "UNWIND $rows AS row "
            "MATCH (c:Chunk {id:row.id}) "
            "UNWIND row.entities AS eid "
            "MATCH (e:Entity {id:eid}) "
            "MERGE (c)-[:MENTIONS]->(e)"
        )

        for batch in self._batches(rows, 50):
            self._run(query_chunks, rows=batch)
            self._run(query_entities, rows=batch)

    def add_chunk(self, chunk):
        self.add_chunks([chunk])

    # ========================================================
    # RELAÇÕES / TRIPLAS
    # ========================================================

    def add_triples(self, triples):
        grouped = {}

        for triple in triples:
            rel = triple["rel"]

            if rel not in RELS:
                raise ValueError(
                    "Relação inválida: " + str(rel)
                )

            subject = str(triple["subject"])
            obj = str(triple["object"])

            if rel == "INTERACTS_WITH":
                subject, obj = sorted((subject, obj))

            grouped.setdefault(rel, []).append({
                "subject": subject,
                "object": obj,
                "source_id": triple.get("source_id"),
                "evidence": triple.get("evidence", ""),
                "confidence": float(
                    triple.get("confidence", 0.0)
                ),
                "negated": bool(
                    triple.get("negated", False)
                ),
                "certainty": triple.get(
                    "certainty",
                    "asserted",
                ),
                "effect": triple.get("effect"),
            })

        for rel, rows in grouped.items():
            query = (
                "UNWIND $rows AS row "
                "MATCH (a:Entity {id:row.subject}), "
                "(b:Entity {id:row.object}) "
                f"MERGE (a)-[r:{rel}]->(b) "

                "ON CREATE SET "
                "r.source_ids=[], "
                "r.evidence_texts=[], "
                "r.confidence=0.0, "
                "r.negated=true, "
                "r.certainty=row.certainty, "
                "r.effect=row.effect "

                "SET "
                "r.source_ids = CASE "
                "WHEN row.source_id IS NULL "
                "OR row.source_id IN r.source_ids "
                "THEN r.source_ids "
                "ELSE r.source_ids + [row.source_id] END, "

                "r.evidence_texts = CASE "
                "WHEN row.evidence IN r.evidence_texts "
                "THEN r.evidence_texts "
                "ELSE r.evidence_texts + [row.evidence] END, "

                "r.confidence = CASE "
                "WHEN row.confidence > r.confidence "
                "THEN row.confidence "
                "ELSE r.confidence END, "

                "r.negated = r.negated AND row.negated, "

                "r.certainty = CASE "
                "WHEN row.certainty='asserted' "
                "THEN 'asserted' "
                "ELSE r.certainty END, "

                "r.effect = coalesce(r.effect, row.effect)"
            )

            for batch in self._batches(rows, 100):
                self._run(query, rows=batch)

    def add_triple(self, triple):
        self.add_triples([triple])

    # ========================================================
    # CONTROLE DE IMPORTAÇÃO
    # ========================================================

    def is_import_complete(
        self,
        import_id,
        dataset_id,
        expected_docs,
        expected_chunks,
    ):
        marker = self._run(
            "MATCH (i:DatasetImport {id:$id}) "
            "RETURN i.completed AS completed, "
            "i.document_count AS documents, "
            "i.chunk_count AS chunks",
            id=import_id,
        )

        if not marker:
            return False

        if not marker[0]["completed"]:
            return False

        if marker[0]["documents"] != expected_docs:
            return False

        if marker[0]["chunks"] != expected_chunks:
            return False

        sources = self._run(
            "MATCH (s:Source {dataset_id:$dataset_id}) "
            "RETURN count(s) AS c",
            dataset_id=dataset_id,
        )[0]["c"]

        chunks = self._run(
            "MATCH (c:Chunk)-[:FROM]->"
            "(s:Source {dataset_id:$dataset_id}) "
            "RETURN count(DISTINCT c) AS c",
            dataset_id=dataset_id,
        )[0]["c"]

        return (
            sources == expected_docs
            and chunks == expected_chunks
        )

    def mark_import_complete(
        self,
        import_id,
        dataset_id,
        documents,
        chunks,
        triples,
    ):
        self._run(
            "MERGE (i:DatasetImport {id:$id}) "
            "SET i.dataset_id=$dataset_id, "
            "i.document_count=$documents, "
            "i.chunk_count=$chunks, "
            "i.triple_count=$triples, "
            "i.completed=true, "
            "i.completed_at=datetime()",
            id=import_id,
            dataset_id=dataset_id,
            documents=documents,
            chunks=chunks,
            triples=triples,
        )

    # ========================================================
    # CONSULTAS DE ENTIDADES
    # ========================================================

    def node(self, nid):
        """
        Recupera uma entidade pelo identificador.

        toString(n.id) evita incompatibilidade entre IDs
        numéricos e textuais recebidos pela visualização.
        """

        if nid is None:
            return None

        nid = str(nid)

        result = self._run(
            "MATCH (n:Entity) "
            "WHERE toString(n.id) = $id "
            "RETURN toString(n.id) AS id, "
            "n.name AS name, "
            "n.type AS type, "
            "n.synonyms AS synonyms "
            "LIMIT 1",
            id=nid,
        )

        if not result:
            return None

        record = result[0]

        return {
            "id": str(record["id"]),
            "name": record["name"] or nid,
            "type": record["type"] or "Entity",
            "synonyms": record["synonyms"] or [],
        }

    def all_nodes(self):
        """
        Recupera todas as entidades do grafo, inclusive isoladas.

        Retorna uma lista de dicionários compatível com a
        visualização.
        """

        records = self._run(
            "MATCH (n:Entity) "
            "RETURN toString(n.id) AS id, "
            "n.name AS name, "
            "n.type AS type, "
            "n.synonyms AS synonyms "
            "ORDER BY n.name"
        )

        return [
            {
                "id": str(record["id"]),
                "name": record["name"] or str(record["id"]),
                "type": record["type"] or "Entity",
                "synonyms": record["synonyms"] or [],
            }
            for record in records
        ]

    # ========================================================
    # BUSCA FULLTEXT
    # ========================================================

    @staticmethod
    def _lucene_query(query):
        """Prepara uma consulta para o mecanismo Lucene."""

        terms = re.findall(
            r"[\wÀ-ÿ]+",
            str(query or ""),
        )

        if not terms:
            return ""

        special = re.compile(
            r'([+\-!(){}\[\]^"~*?:\\/|&])'
        )

        escaped = [
            special.sub(r"\\\1", term)
            for term in terms
        ]

        return " OR ".join(escaped)

    def fulltext_search(self, query, k=10):
        """Busca textual nos chunks."""

        lucene = self._lucene_query(query)

        if not lucene:
            return []

        k = max(1, min(int(k), 1000))

        records = self._run(
            "CALL db.index.fulltext.queryNodes("
            "$index_name, $search_text"
            ") YIELD node, score "
            "RETURN node.id AS id, "
            "node.text AS text, "
            "node.source_id AS source_id, "
            "score "
            "ORDER BY score DESC "
            "LIMIT $k",
            index_name=CHUNK_FULLTEXT_INDEX,
            search_text=lucene,
            k=k,
        )

        return [
            {
                "id": record["id"],
                "text": record["text"] or "",
                "source_id": record["source_id"],
                "score": float(record["score"]),
            }
            for record in records
        ]

    def entity_fulltext_search(self, query, k=10):
        """Busca textual nas entidades."""

        lucene = self._lucene_query(query)

        if not lucene:
            return []

        k = max(1, min(int(k), 1000))

        records = self._run(
            "CALL db.index.fulltext.queryNodes("
            "$index_name, $search_text"
            ") YIELD node, score "
            "RETURN node.id AS id, "
            "node.name AS name, "
            "node.type AS type, "
            "node.synonyms AS synonyms, "
            "score "
            "ORDER BY score DESC "
            "LIMIT $k",
            index_name=ENTITY_FULLTEXT_INDEX,
            search_text=lucene,
            k=k,
        )

        return [
            {
                "id": record["id"],
                "name": record["name"],
                "type": record["type"],
                "synonyms": record["synonyms"] or [],
                "score": float(record["score"]),
            }
            for record in records
        ]

    # ========================================================
    # BUSCA VETORIAL
    # ========================================================

    def vector_search(self, vector, k=10):
        """Busca chunks semelhantes usando o índice vetorial."""

        vector = self._validate_embedding(vector)
        k = max(1, min(int(k), 1000))

        records = self._run(
            "CALL db.index.vector.queryNodes("
            "$index_name, $k, $embedding"
            ") YIELD node, score "
            "RETURN node.id AS id, "
            "node.text AS text, "
            "node.source_id AS source_id, "
            "score "
            "ORDER BY score DESC",
            index_name=VECTOR_INDEX,
            k=k,
            embedding=vector,
        )

        return [
            {
                "id": record["id"],
                "text": record["text"] or "",
                "source_id": record["source_id"],
                "score": float(record["score"]),
            }
            for record in records
        ]

    # ========================================================
    # CYPHER SOMENTE LEITURA
    # ========================================================

    @staticmethod
    def _validate_readonly_query(query):
        """
        Bloqueia operações de escrita e comandos administrativos.

        Observação: esta validação é uma barreira adicional.
        A execução também utiliza uma transação de leitura.
        """

        if not isinstance(query, str) or not query.strip():
            raise ValueError("Consulta Cypher vazia.")

        query = query.strip()

        if ";" in query.rstrip(";"):
            raise ValueError(
                "Somente uma instrução Cypher é permitida."
            )

        normalized = re.sub(
            r"/\*.*?\*/|//[^\n]*",
            " ",
            query,
            flags=re.DOTALL,
        )

        forbidden = re.compile(
            r"\b("
            r"CREATE|MERGE|DELETE|DETACH|SET|REMOVE|DROP|"
            r"ALTER|RENAME|GRANT|DENY|REVOKE|"
            r"LOAD\s+CSV|"
            r"START\s+DATABASE|STOP\s+DATABASE"
            r")\b",
            re.IGNORECASE,
        )

        if forbidden.search(normalized):
            raise ValueError(
                "Consulta bloqueada: operações de escrita "
                "ou administrativas não são permitidas."
            )

        return query.rstrip(";").strip()

    def execute_readonly_cypher(self, query, params=None):
        """
        Executa Cypher em transação de leitura.

        Preserva objetos Path do Neo4j nos resultados.
        """

        query = self._validate_readonly_query(query)
        params = dict(params or {})

        with self.driver.session(
            database=self.database
        ) as session:

            def read_transaction(tx):
                result = tx.run(query, **params)

                return [
                    record.data()
                    for record in result
                ]

            return session.execute_read(read_transaction)

    # ========================================================
    # CONVERSÃO DE RELAÇÕES
    # ========================================================

    @staticmethod
    def _edge(relationship):
        """
        Converte um relacionamento Neo4j em dicionário.

        Compatível com objetos Relationship do Neo4j.
        """

        source_ids = relationship.get(
            "source_ids",
            [],
        ) or []

        evidence_texts = relationship.get(
            "evidence_texts",
            [],
        ) or []

        evidence = [
            {
                "source_id": source_id,
                "text": text,
                "confidence": relationship.get(
                    "confidence",
                    0.9,
                ),
                "certainty": relationship.get(
                    "certainty",
                    "asserted",
                ),
                "negated": relationship.get(
                    "negated",
                    False,
                ),
            }
            for source_id, text in zip(
                source_ids,
                evidence_texts,
            )
        ]

        return {
            "src": str(relationship.start_node["id"]),
            "dst": str(relationship.end_node["id"]),
            "rel": relationship.type,
            "confidence": relationship.get(
                "confidence",
                0.9,
            ),
            "certainty": relationship.get(
                "certainty",
                "asserted",
            ),
            "negated": relationship.get(
                "negated",
                False,
            ),
            "effect": relationship.get("effect"),
            "evidence": evidence,
        }

    @staticmethod
    def _edge_from_record(record):
        """
        Converte uma linha retornada por all_edges()
        em um dicionário padronizado.
        """

        props = dict(record["props"] or {})

        source_ids = props.get("source_ids") or []
        evidence_texts = props.get("evidence_texts") or []

        evidence = [
            {
                "source_id": source_id,
                "text": text,
                "confidence": props.get("confidence", 0.9),
                "certainty": props.get("certainty", "asserted"),
                "negated": props.get("negated", False),
            }
            for source_id, text in zip(
                source_ids,
                evidence_texts,
            )
        ]

        return {
            "src": str(record["src"]),
            "dst": str(record["dst"]),
            "rel": record["rel"],
            "confidence": props.get("confidence", 0.9),
            "certainty": props.get("certainty", "asserted"),
            "negated": props.get("negated", False),
            "effect": props.get("effect"),
            "evidence": evidence,
        }

    # ========================================================
    # CONSULTA DE ARESTAS
    # ========================================================

    def all_edges(self):
        """
        Recupera todas as relações entre entidades.

        Retorna os IDs dos nós de origem e destino diretamente
        na consulta, sem depender de consultas adicionais.
        """

        records = self._run(
            "MATCH (a:Entity)-[r]->(b:Entity) "
            "RETURN "
            "toString(a.id) AS src, "
            "toString(b.id) AS dst, "
            "type(r) AS rel, "
            "properties(r) AS props"
        )

        return [
            self._edge_from_record(record)
            for record in records
        ]

    # ========================================================
    # BUSCA MULTI-HOP
    # ========================================================

    def find_paths(
        self,
        seeds,
        hops,
        rel_hint=None,
        max_paths=30,
    ):
        paths = []

        queries = path_queries(
            seeds,
            hops,
            rel_hint,
            max_paths,
        )

        for query, params in queries:
            records = self._run(query, **params)

            for record in records:
                path = record["p"]

                nodes = [
                    str(node["id"])
                    for node in path.nodes
                ]

                if len(set(nodes)) != len(nodes):
                    continue

                if not all(
                    "Entity" in node.labels
                    for node in path.nodes
                ):
                    continue

                if not all(
                    rel.type in RELS
                    and not rel.get("negated", False)
                    for rel in path.relationships
                ):
                    continue

                edges = [
                    self._edge(rel)
                    for rel in path.relationships
                ]

                paths.append({
                    "nodes": nodes,
                    "edges": edges,
                })

        paths.sort(
            key=lambda item: len(item["edges"])
        )

        return paths[:max_paths]

    # ========================================================
    # SHORTEST PATH
    # ========================================================

    def shortest_path(self, source, target, max_hops=5):
        """
        Menor caminho entre duas entidades via Cypher shortestPath().

        Retorna (caminho, cypher). O caminho segue o formato de
        find_paths() ou é None quando não existe.
        """

        query = shortest_path_query(max_hops)

        records = self._run(
            query,
            source=str(source),
            target=str(target),
        )

        if not records:
            return None, query

        path = records[0]["p"]

        nodes = [str(node["id"]) for node in path.nodes]

        if len(set(nodes)) != len(nodes):
            return None, query

        edges = [self._edge(rel) for rel in path.relationships]

        return {"nodes": nodes, "edges": edges}, query

    # ========================================================
    # RELAÇÕES NEGADAS
    # ========================================================

    def negated_between(self, seeds):
        seeds = set(str(x) for x in (seeds or []))

        if not seeds:
            return []

        return [
            edge
            for edge in self.all_edges()
            if edge["negated"]
            and (
                (
                    edge["src"] in seeds
                    and edge["dst"] in seeds
                )
                if len(seeds) > 1
                else (
                    edge["src"] in seeds
                    or edge["dst"] in seeds
                )
            )
        ]

    # ========================================================
    # ESTATÍSTICAS
    # ========================================================

    def stats(self):
        nodes = self._run(
            "MATCH (n:Entity) "
            "RETURN count(n) AS total"
        )[0]["total"]

        edges = self._run(
            "MATCH (:Entity)-[r]->(:Entity) "
            "RETURN count(r) AS total"
        )[0]["total"]

        by_type_records = self._run(
            "MATCH (n:Entity) "
            "RETURN n.type AS type, count(n) AS total"
        )

        by_rel_records = self._run(
            "MATCH (:Entity)-[r]->(:Entity) "
            "RETURN type(r) AS type, count(r) AS total"
        )

        by_type = {
            record["type"]: record["total"]
            for record in by_type_records
        }

        by_rel = {
            record["type"]: record["total"]
            for record in by_rel_records
        }

        return {
            "nodes": nodes,
            "edges": edges,
            "by_type": by_type,
            "by_rel": by_rel,
        }

    # ========================================================
    # ANÁLISES DO GRAFO
    # ========================================================

    def _analytics_key(self):
        """Assinatura barata do grafo para invalidar o cache."""

        nodes = self._run(
            "MATCH (n:Entity) RETURN count(n) AS total"
        )[0]["total"]

        edges = self._run(
            "OPTIONAL MATCH (:Entity)-[r]->(:Entity) "
            "RETURN count(r) AS total"
        )[0]["total"]

        return (nodes, edges)

    def analytics(self, refresh=False):
        """
        PageRank, Louvain e Leiden (Neo4j GDS) com fallback em Python.

        O resultado fica em cache e só é recalculado quando o número de
        entidades/relações muda ou quando refresh=True. O motor pode ser
        forçado com GRAPH_ANALYTICS_ENGINE = auto | gds | python.
        """

        from .graph_algorithms import compute_analytics, default_engine

        engine = default_engine()

        key = (self._analytics_key(), engine)

        cache = getattr(self, "_analytics_cache", None)

        if cache and not refresh and cache["key"] == key:
            return cache["data"]

        data = compute_analytics(self, engine)

        self._analytics_cache = {"key": key, "data": data}

        return data