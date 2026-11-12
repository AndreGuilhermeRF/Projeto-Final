"""
Recuperação híbrida real:
- Índice vetorial do Neo4j.
- Fulltext do Neo4j.
- Entity Linking.
- Travessia Cypher.
- Caminhos gerados opcionalmente pela LLM.
"""

import logging
import os
import re

from collections import Counter
from dataclasses import dataclass

from .graph_store import path_queries, RELS
from .graph_algorithms import build_graph_context, shortest_paths_between


logger = logging.getLogger(__name__)


MIN_SIM = 0.15


OUT_OF_SCOPE = re.compile(
    r"\b(price|cost|preço|preços|valor comercial|brand name)\b",
    re.I,
)


HINTS = [
    (r"interag|interact", ["INTERACTS_WITH"]),
    (
        r"\btrata|tratamento|indicado|indicação|treat|therapy",
        ["TREATS"],
    ),
    (
        r"contraindic|evitar|não deve|nao deve",
        ["CONTRAINDICATED_IN"],
    ),
    (
        r"efeito adverso|reação adversa|reacao adversa|"
        r"causa|provoca|side effect|adverse",
        ["CAUSES"],
    ),
    (r"\bclasse\b|class|tipo de", ["BELONGS_TO"]),
]


# ============================================================
# ÍNDICE HÍBRIDO
# ============================================================

class ChunkIndex:
    """
    Consulta os índices do Neo4j sem manter vetores em memória.

    Combina:
    - Busca vetorial.
    - Busca Fulltext.
    - Reciprocal Rank Fusion (RRF).

    Se um mecanismo falhar, tenta utilizar o outro.
    """

    def __init__(self, chunks, embedder, store):
        self.chunks = chunks or []
        self.embedder = embedder
        self.store = store

        self.last_mode = "não executada"
        self.last_errors = []

        self.by_id = {
            chunk["id"]: chunk
            for chunk in self.chunks
            if isinstance(chunk, dict) and chunk.get("id")
        }

    def search(self, query, k=3):
        """
        Executa a recuperação híbrida.

        Retorna:
            Lista de tuplas (chunk, score).
        """

        merged = {}

        vector_ok = False
        fulltext_ok = False

        errors = []

        search_k = max(int(k) * 3, 10)

        # ----------------------------------------------------
        # BUSCA VETORIAL
        # ----------------------------------------------------

        try:
            vector = self.embedder.embed(query)

            vector_results = self.store.vector_search(
                vector,
                k=search_k,
            )

            for rank, item in enumerate(vector_results, 1):

                chunk_id = item.get("id")

                if not chunk_id:
                    continue

                score = float(item.get("score", 0.0))

                merged[chunk_id] = {
                    "item": item,
                    "rrf": 1.0 / (60 + rank),
                    "vector_score": score,
                    "fulltext_score": 0.0,
                }

            vector_ok = True

        except Exception as exc:
            message = "Busca vetorial: {}".format(exc)

            errors.append(message)

            logger.exception(
                "Falha na busca vetorial do Neo4j."
            )

        # ----------------------------------------------------
        # BUSCA FULLTEXT
        # ----------------------------------------------------

        try:
            fulltext_results = self.store.fulltext_search(
                query,
                k=search_k,
            )

            for rank, item in enumerate(fulltext_results, 1):

                chunk_id = item.get("id")

                if not chunk_id:
                    continue

                entry = merged.setdefault(
                    chunk_id,
                    {
                        "item": item,
                        "rrf": 0.0,
                        "vector_score": 0.0,
                        "fulltext_score": 0.0,
                    },
                )

                # Prioriza os dados retornados pelo Neo4j.
                if item.get("text"):
                    entry["item"] = item

                entry["rrf"] += 1.0 / (60 + rank)

                entry["fulltext_score"] = float(
                    item.get("score", 0.0)
                )

            fulltext_ok = True

        except Exception as exc:
            message = "Busca Fulltext: {}".format(exc)

            errors.append(message)

            logger.exception(
                "Falha na busca Fulltext do Neo4j."
            )

        # ----------------------------------------------------
        # DIAGNÓSTICO
        # ----------------------------------------------------

        self.last_errors = errors

        if vector_ok and fulltext_ok:
            self.last_mode = "Vector + Fulltext"

        elif vector_ok:
            self.last_mode = "Vector (Fulltext indisponível)"

        elif fulltext_ok:
            self.last_mode = "Fulltext (índice vetorial indisponível)"

        else:
            self.last_mode = "sem índices disponíveis"

        # ----------------------------------------------------
        # ORDENAÇÃO RRF
        # ----------------------------------------------------

        ranked = sorted(
            merged.values(),
            key=lambda item: item["rrf"],
            reverse=True,
        )[:k]

        results = []

        for row in ranked:

            item = row["item"]

            chunk_id = item.get("id")

            if not chunk_id:
                continue

            chunk = self.by_id.get(chunk_id)

            if chunk is None:
                chunk = {
                    "id": chunk_id,
                    "text": item.get("text", ""),
                    "source_id": item.get("source_id"),
                    "entities": item.get("entities", []),
                }

            # RRF é pontuação de ranking.
            # Não representa probabilidade clínica.
            score = min(
                1.0,
                row["rrf"] * 60.0,
            )

            results.append((chunk, score))

        return results


# ============================================================
# SISTEMA
# ============================================================

@dataclass
class System:
    store: object
    index: ChunkIndex
    lex: object
    triples: list


# ============================================================
# IDENTIFICAÇÃO DE RELAÇÕES
# ============================================================

def detect_hint(question):
    rels = []

    for pattern, relation in HINTS:

        if re.search(pattern, question, re.I):
            rels += relation

    return list(dict.fromkeys(rels)) or None


# ============================================================
# CONVERSÃO DE CAMINHOS EM TEXTO
# ============================================================

def path_text(store, path):
    """
    Converte um caminho em texto.

    Não pressupõe que todos os nós ainda existam no banco.
    """

    nodes = path.get("nodes") or []
    edges = path.get("edges") or []

    if not nodes:
        return ""

    current = store.node(nodes[0])

    if current is None:
        return ""

    parts = [current["name"]]
    cur = nodes[0]

    for edge in edges:

        forward = edge["src"] == cur

        nxt = (
            edge["dst"]
            if forward
            else edge["src"]
        )

        node = store.node(nxt)

        if node is None:
            return ""

        if forward:
            parts.append(
                "—{}→".format(edge["rel"])
            )
        else:
            parts.append(
                "←{}—".format(edge["rel"])
            )

        parts.append(node["name"])

        cur = nxt

    return " ".join(parts)


# ============================================================
# CONVERSÃO DE RELAÇÕES EM FATOS
# ============================================================

def _fact(store, edge, fid, hop):
    """
    Converte uma relação em fato.

    Retorna None quando uma das entidades não existe.
    """

    src = store.node(edge["src"])
    dst = store.node(edge["dst"])

    if src is None or dst is None:
        return None

    evidence = edge.get("evidence") or []

    evidence = [
        item
        for item in evidence
        if isinstance(item, dict)
    ]

    ev = evidence[0] if evidence else {}

    return {
        "id": fid,
        "src": edge["src"],
        "dst": edge["dst"],
        "rel": edge["rel"],
        "hop": hop,
        "src_name": src["name"],
        "dst_name": dst["name"],
        "confidence": edge.get("confidence", 0.0),
        "certainty": edge.get("certainty", "asserted"),
        "negated": edge.get("negated", False),
        "effect": edge.get("effect"),
        "evidence": ev.get("text", ""),
        "sources": sorted({
            item["source_id"]
            for item in evidence
            if item.get("source_id")
        }),
        "evidence_count": len(evidence),
    }


# ============================================================
# VALIDAÇÃO DE CAMINHOS
# ============================================================

def _valid_path(store, path):
    """
    Verifica se todos os nós e relações do caminho são válidos.

    Também verifica se cada relação conecta os nós
    consecutivos apresentados no caminho.
    """

    nodes = path.get("nodes") or []
    edges = path.get("edges") or []

    if not nodes:
        return False

    if len(nodes) != len(edges) + 1:
        return False

    if len(nodes) != len(set(nodes)):
        return False

    for node_id in nodes:

        if store.node(node_id) is None:
            return False

    for index, edge in enumerate(edges):

        if edge.get("rel") not in RELS:
            return False

        if edge.get("negated", False):
            return False

        expected_a = nodes[index]
        expected_b = nodes[index + 1]

        actual_a = edge.get("src")
        actual_b = edge.get("dst")

        if not (
            (actual_a == expected_a and actual_b == expected_b)
            or
            (actual_a == expected_b and actual_b == expected_a)
        ):
            return False

    return True


# ============================================================
# CYPHER GERADO PELA LLM
# ============================================================

def _llm_paths(store, question, seeds, hops):
    """
    Geração de Cypher opcional.

    Aceita caminhos retornados individualmente ou em listas.
    Descarta resultados que não sejam objetos Path válidos.

    Retorna:
        paths, query, error
    """

    if os.getenv(
        "ENABLE_LLM_CYPHER",
        "true",
    ).strip().lower() not in ("1", "true", "yes", "on"):

        return [], "", None

    query = ""

    try:
        from .llm import generate_cypher, llm_available

        if not llm_available():
            return [], "", "LLM indisponível"

        query = generate_cypher(
            question,
            seeds,
            hops,
        )

        if not query or not query.strip():
            return [], "", "A LLM não gerou uma consulta Cypher."

        records = store.execute_readonly_cypher(
            query,
            {
                "seed_ids": seeds,
                "allowed_rels": list(RELS),
                "limit": 30,
            },
        )

        paths = []
        seen_paths = set()

        for record in records:

            if not hasattr(record, "get"):
                logger.warning(
                    "Registro Cypher ignorado: tipo inesperado %s.",
                    type(record).__name__,
                )
                continue

            raw_path = record.get("p")

            if raw_path is None:
                continue

            candidates = (
                raw_path
                if isinstance(raw_path, (list, tuple))
                else [raw_path]
            )

            for path in candidates:

                try:
                    path_nodes = getattr(path, "nodes", None)

                    path_relationships = getattr(
                        path,
                        "relationships",
                        None,
                    )

                    if path_nodes is None or path_relationships is None:
                       # logger.warning(
                        #    "Resultado Cypher ignorado: "
                        #   "esperado Path, recebido %s.",
                        #  type(path).__name__,
                        #)
                        continue

                    if not path_nodes:
                        continue

                    if not all(
                        "Entity" in node.labels
                        for node in path_nodes
                    ):
                        continue

                    if not all(
                        relationship.type in RELS
                        for relationship in path_relationships
                    ):
                        continue

                    nodes = [
                        node.get("id")
                        for node in path_nodes
                    ]

                    if any(node_id is None for node_id in nodes):
                        continue

                    if len(nodes) != len(set(nodes)):
                        continue

                    edges = [
                        store._edge(relationship)
                        for relationship in path_relationships
                    ]

                    candidate = {
                        "nodes": nodes,
                        "edges": edges,
                    }

                    if not _valid_path(store, candidate):
                        continue

                    path_key = (
                        tuple(nodes),
                        tuple(
                            (
                                edge.get("src"),
                                edge.get("rel"),
                                edge.get("dst"),
                            )
                            for edge in edges
                        ),
                    )

                    if path_key in seen_paths:
                        continue

                    seen_paths.add(path_key)
                    paths.append(candidate)

                except Exception as exc:
                    logger.warning(
                        "Caminho Cypher inválido ignorado: %s",
                        exc,
                    )
                    continue

        return paths, query, None

    except Exception as exc:

        logger.exception(
            "Falha na geração ou execução de Cypher pela LLM."
        )

        return (
            [],
            query,
            "Falha na consulta Cypher da LLM: {}".format(exc),
        )


# ============================================================
# RECUPERAÇÃO PRINCIPAL
# ============================================================

def retrieve(system, question, hops=2, top_k=3):

    store = system.store
    index = system.index
    lex = system.lex

    result = {
        "question": question,
        "refused": False,
        "reason": None,
        "seeds": [],
        "seed_source": None,
        "hint": None,
        "chunks": [],
        "paths": [],
        "path_citations": [],
        "facts": [],
        "negated_facts": [],
        "graph_context": None,
        "cypher": "",
        "confidence": 0.0,
        "confidence_basis": "pontuação heurística de suporte",
        "retrieval_only": False,
        "retrieval_errors": [],
    }

    def refuse(reason):
        result.update(
            refused=True,
            reason=reason,
        )

        return result

    errors = []

    # --------------------------------------------------------
    # CONTROLE DE ESCOPO
    # --------------------------------------------------------

    if OUT_OF_SCOPE.search(question):

        return refuse(
            "A pergunta pede informação comercial "
            "que não faz parte do ChatBulário."
        )

    # --------------------------------------------------------
    # BUSCA HÍBRIDA
    # --------------------------------------------------------

    try:
        hits = index.search(
            question,
            top_k,
        )

        result["chunks"] = hits
        result["retrieval_mode"] = index.last_mode

        errors.extend(index.last_errors)

    except Exception as exc:

        logger.exception(
            "Falha inesperada na recuperação híbrida."
        )

        hits = []

        result["retrieval_mode"] = "erro"

        errors.append(
            "Recuperação híbrida: {}".format(exc)
        )

    # --------------------------------------------------------
    # IDENTIFICAÇÃO DAS ENTIDADES
    # --------------------------------------------------------

    try:

        mentions = lex.find(question)

        seeds = list(
            dict.fromkeys(
                mention.id
                for mention in mentions
                if getattr(mention, "id", None)
            )
        )

    except Exception as exc:

        logger.exception(
            "Falha no Entity Linking."
        )

        seeds = []

        errors.append(
            "Entity Linking: {}".format(exc)
        )

    result["seed_source"] = (
        "entity linking (léxico + aliases)"
    )

    # --------------------------------------------------------
    # FALLBACK: FULLTEXT DE ENTIDADES
    # --------------------------------------------------------

    if not seeds:

        try:

            entity_hits = store.entity_fulltext_search(
                question,
                k=4,
            )

        except Exception as exc:

            logger.exception(
                "Falha na busca Fulltext de entidades."
            )

            entity_hits = []

            errors.append(
                "Fulltext de entidades: {}".format(exc)
            )

        seeds = [
            item["id"]
            for item in entity_hits[:3]
            if item.get("id")
        ]

        if seeds:

            result["seed_source"] = (
                "índice Fulltext de entidades"
            )

    # --------------------------------------------------------
    # FALLBACK: ENTIDADES DOS CHUNKS
    # --------------------------------------------------------

    if not seeds and hits:

        count = Counter(
            entity
            for chunk, _score in hits
            for entity in chunk.get("entities", [])
        )

        seeds = [
            entity
            for entity, _ in count.most_common(2)
        ]

        result["seed_source"] = (
            "entidades dos chunks recuperados"
        )

    # --------------------------------------------------------
    # NENHUMA ENTIDADE OU TRECHO
    # --------------------------------------------------------

    if not seeds and not hits:

        result["retrieval_errors"] = errors

        return refuse(
            "Nenhuma entidade reconhecida e nenhum "
            "trecho recuperado pelos índices."
        )

    # --------------------------------------------------------
    # VALIDAÇÃO DAS ENTIDADES
    # --------------------------------------------------------

    seeds = seeds[:4]

    valid_seeds = []

    for seed in seeds:

        try:

            if store.node(seed) is not None:
                valid_seeds.append(seed)

        except Exception as exc:

            logger.exception(
                "Falha ao validar entidade %s.",
                seed,
            )

            errors.append(
                "Validação da entidade {}: {}".format(
                    seed,
                    exc,
                )
            )

    seeds = valid_seeds

    if not seeds:

        result["retrieval_errors"] = errors

        return refuse(
            "As entidades identificadas não foram "
            "encontradas no grafo Neo4j."
        )

    # --------------------------------------------------------
    # DETECÇÃO DO TIPO DE RELAÇÃO
    # --------------------------------------------------------

    hint = (
        detect_hint(question)
        if len(seeds) == 1
        else None
    )

    result.update(
        seeds=seeds,
        hint=hint,
    )

    # --------------------------------------------------------
    # CAMINHOS DA LLM
    # --------------------------------------------------------

    llm_paths, llm_query, llm_error = _llm_paths(
        store,
        question,
        seeds,
        hops,
    )

    if llm_error:
        errors.append(llm_error)

    # --------------------------------------------------------
    # CONSULTAS CONVENCIONAIS
    # --------------------------------------------------------

    try:

        static_queries = path_queries(
            seeds,
            hops,
            hint,
        )

    except Exception as exc:

        logger.exception(
            "Falha ao construir consultas convencionais."
        )

        static_queries = []

        errors.append(
            "Construção das consultas convencionais: {}".format(
                exc
            )
        )

    if llm_query:

        result["cypher"] = (
            "// Cypher gerado pela LLM "
            "(validado como read-only)\n"
            + llm_query
        )

    else:

        result["cypher"] = "\n\n".join(
            query
            for query, _params in static_queries
        )

    # --------------------------------------------------------
    # EXECUÇÃO DOS CAMINHOS
    # --------------------------------------------------------

    paths = llm_paths

    if not paths:

        try:

            paths = store.find_paths(
                seeds,
                hops,
                hint,
            )

        except Exception as exc:

            logger.exception(
                "Falha na recuperação convencional de caminhos."
            )

            paths = []

            errors.append(
                "Busca convencional de caminhos: {}".format(
                    exc
                )
            )

    # --------------------------------------------------------
    # VALIDAÇÃO DOS CAMINHOS
    # --------------------------------------------------------

    valid_paths = []

    for path in paths:

        try:

            if _valid_path(store, path):
                valid_paths.append(path)

        except Exception as exc:

            logger.exception(
                "Falha ao validar caminho do grafo."
            )

            errors.append(
                "Validação de caminho: {}".format(exc)
            )

    paths = valid_paths

    # --------------------------------------------------------
    # ALGORITMOS DE GRAFO: SHORTEST PATH, PAGERANK E COMUNIDADES
    # --------------------------------------------------------

    graph_analytics = None

    try:

        graph_analytics = store.analytics()

    except Exception as exc:

        logger.exception(
            "Falha nas análises de grafo."
        )

        errors.append(
            "Análises de grafo: {}".format(exc)
        )

    if len(seeds) >= 2:

        try:

            sp_paths, sp_cypher, _sp_engine = shortest_paths_between(
                store,
                seeds,
            )

        except Exception as exc:

            logger.exception(
                "Falha no Shortest Path."
            )

            sp_paths, sp_cypher = [], ""

            errors.append(
                "Shortest Path: {}".format(exc)
            )

        known = {}

        for path in paths:

            known[tuple(path["nodes"])] = path
            known[tuple(reversed(path["nodes"]))] = path

        for sp in sp_paths:

            existing = known.get(tuple(sp["nodes"]))

            if existing is not None:

                existing["algorithm"] = "shortestPath"
                existing["engine"] = sp.get("engine")

                continue

            try:

                valid = _valid_path(store, sp)

            except Exception as exc:

                logger.exception(
                    "Falha ao validar caminho do Shortest Path."
                )

                errors.append(
                    "Validação do Shortest Path: {}".format(exc)
                )

                valid = False

            if valid:

                paths.append(sp)

                known[tuple(sp["nodes"])] = sp
                known[tuple(reversed(sp["nodes"]))] = sp

        if sp_cypher:

            result["cypher"] = (
                (result["cypher"] + "\n\n")
                if result["cypher"]
                else ""
            ) + sp_cypher

    # Ordena por tamanho; em empate, Shortest Path primeiro e,
    # depois, caminhos cujos nós têm maior PageRank médio.

    if graph_analytics and graph_analytics.get("available"):

        scores = graph_analytics.get("pagerank") or {}

        def _path_rank(path):

            nodes = path.get("nodes") or [None]

            mean_score = sum(
                scores.get(node, 0.0) for node in nodes
            ) / float(len(nodes))

            return (
                len(path["edges"]),
                0 if path.get("algorithm") == "shortestPath" else 1,
                -mean_score,
            )

        paths = sorted(paths, key=_path_rank)

    try:

        result["graph_context"] = build_graph_context(
            graph_analytics,
            store,
            seeds,
            paths,
        )

    except Exception as exc:

        logger.exception(
            "Falha ao montar o contexto estrutural do grafo."
        )

        errors.append(
            "Contexto estrutural: {}".format(exc)
        )

    # --------------------------------------------------------
    # RELAÇÕES NEGADAS
    # --------------------------------------------------------

    try:

        negated = store.negated_between(seeds)

    except Exception as exc:

        logger.exception(
            "Falha ao recuperar relações negadas."
        )

        negated = []

        errors.append(
            "Recuperação de relações negadas: {}".format(exc)
        )

    # --------------------------------------------------------
    # CONSTRUÇÃO DOS FATOS
    # --------------------------------------------------------

    facts = []
    seen = {}

    for path in paths:

        for i, edge in enumerate(path["edges"]):

            key = (
                edge["src"],
                edge["rel"],
                edge["dst"],
            )

            if key in seen:
                continue

            fact = _fact(
                store,
                edge,
                "F{}".format(len(facts) + 1),
                i + 1,
            )

            if fact is not None:

                seen[key] = fact
                facts.append(fact)

    # --------------------------------------------------------
    # CITAÇÕES DOS CAMINHOS
    # --------------------------------------------------------

    visible_fact_ids = {
        fact["id"]
        for fact in facts[:40]
    }

    path_citations = []

    for i, path in enumerate(paths[:15], 1):

        path_fact_ids = []

        for edge in path["edges"]:

            key = (
                edge["src"],
                edge["rel"],
                edge["dst"],
            )

            fact = seen.get(key)

            if (
                fact
                and fact["id"] in visible_fact_ids
            ):
                path_fact_ids.append(fact["id"])

        try:

            text = path_text(
                store,
                path,
            )

        except Exception as exc:

            logger.exception(
                "Falha ao converter caminho em texto."
            )

            errors.append(
                "Conversão de caminho em texto: {}".format(exc)
            )

            continue

        if not text:
            continue

        path_citations.append({
            "id": "P{}".format(i),
            "text": text,
            "fact_ids": list(
                dict.fromkeys(path_fact_ids)
            ),
            "algorithm": path.get("algorithm"),
        })

    # --------------------------------------------------------
    # FATOS NEGADOS
    # --------------------------------------------------------

    negated_facts = []

    for i, edge in enumerate(negated):

        fact = _fact(
            store,
            edge,
            "N{}".format(i + 1),
            1,
        )

        if fact is not None:
            negated_facts.append(fact)

    # --------------------------------------------------------
    # RESULTADO PARCIAL
    # --------------------------------------------------------

    result.update(
        paths=paths,
        path_citations=path_citations,
        facts=facts[:40],
        negated_facts=negated_facts,
        retrieval_errors=errors,
    )

    # --------------------------------------------------------
    # RECUPERAÇÃO SOMENTE POR TRECHOS
    # --------------------------------------------------------

    if not facts and not negated_facts:

        if hits:

            result["retrieval_only"] = True

            result["confidence"] = round(
                min(0.75, hits[0][1]),
                2,
            )

            return result

        return refuse(
            "Entidades reconhecidas, mas não há caminhos "
            "nem trechos recuperados."
        )

    # --------------------------------------------------------
    # CONFIANÇA HEURÍSTICA
    # --------------------------------------------------------

    used = facts or negated_facts

    mean_confidence = (
        sum(
            fact["confidence"]
            for fact in used
        )
        / float(len(used))
    )

    source_bonus = min(
        0.10,
        sum(
            fact["evidence_count"]
            for fact in used
        ) * 0.02,
    )

    asserted_ratio = (
        sum(
            1
            for fact in used
            if fact["certainty"] == "asserted"
        )
        / float(len(used))
    )

    result["confidence"] = round(
        min(
            1.0,
            mean_confidence * 0.7
            + source_bonus
            + asserted_ratio * 0.1,
        ),
        2,
    )

    return result