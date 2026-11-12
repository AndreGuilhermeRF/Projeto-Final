"""
Algoritmos de grafo usados para enriquecer o contexto do GraphRAG.

Algoritmos:
- PageRank (centralidade das entidades).
- Louvain e Leiden (detecção de comunidades).
- Shortest Path (menor caminho entre entidades da pergunta).

Motores:
- "gds":    Neo4j Graph Data Science (PageRank, Louvain e Leiden).
- "python": implementação própria (PageRank, Louvain e BFS), usada
            quando o GDS não está instalado ou falha. O Leiden só existe
            no GDS; no fallback ele é informado como indisponível.

Convenções:
- O grafo analisado é o de entidades (:Entity), tratado como NÃO
  direcionado, ignorando relações negadas, auto-laços e tipos fora de RELS.
- Os escores de PageRank são normalizados (soma = 1) nos dois motores,
  para que os valores sejam comparáveis.
- As comunidades são renumeradas por tamanho (0 = maior comunidade).

O módulo não depende de bibliotecas externas e é compatível com Python 3.8.
"""

import logging
import os
import time

from collections import defaultdict, deque
from itertools import combinations

from .graph_store import RELS


logger = logging.getLogger(__name__)


GDS_GRAPH_NAME = "graphrag_med_entities"
DEFAULT_MAX_HOPS = 5
MAX_SEED_PAIRS = 6


# ============================================================
# CONSTRUÇÃO DO GRAFO EM MEMÓRIA
# ============================================================

def _usable(edge):
    """Aresta válida para análise estrutural."""

    if not isinstance(edge, dict):
        return False

    if edge.get("rel") not in RELS:
        return False

    if edge.get("negated", False):
        return False

    src = edge.get("src")
    dst = edge.get("dst")

    if src is None or dst is None or str(src) == str(dst):
        return False

    return True


def build_adjacency(edges):
    """
    Constrói a adjacência não direcionada e ponderada.

    O peso entre dois nós é o número de relações (de qualquer tipo)
    que os conectam.
    """

    adjacency = defaultdict(lambda: defaultdict(float))

    for edge in edges or []:

        if not _usable(edge):
            continue

        a = str(edge["src"])
        b = str(edge["dst"])

        adjacency[a][b] += 1.0
        adjacency[b][a] += 1.0

    return {
        node: dict(neighbors)
        for node, neighbors in adjacency.items()
    }


def _normalize(scores):
    total = float(sum(scores.values()))

    if total <= 0:
        return dict(scores)

    return {
        node: value / total
        for node, value in scores.items()
    }


def _relabel_by_size(communities):
    """Renumera comunidades: 0 é a maior; empates por menor ID de nó."""

    members = defaultdict(list)

    for node, community in communities.items():
        members[community].append(node)

    ordered = sorted(
        members.values(),
        key=lambda nodes: (-len(nodes), min(nodes)),
    )

    relabeled = {}

    for new_id, nodes in enumerate(ordered):
        for node in nodes:
            relabeled[node] = new_id

    return relabeled


# ============================================================
# PAGERANK (PYTHON)
# ============================================================

def pagerank(adjacency, damping=0.85, max_iter=100, tol=1e-9):
    """
    PageRank por iteração de potência em grafo não direcionado ponderado.

    Nós isolados (sem vizinhos) redistribuem sua massa uniformemente.
    O resultado soma 1.
    """

    nodes = sorted(adjacency)
    n = len(nodes)

    if n == 0:
        return {}

    strength = {
        node: sum(adjacency[node].values())
        for node in nodes
    }

    rank = {node: 1.0 / n for node in nodes}

    for _ in range(max_iter):

        dangling = sum(
            rank[node]
            for node in nodes
            if strength[node] == 0
        )

        base = (1.0 - damping) / n + damping * dangling / n

        updated = {}

        for node in nodes:

            incoming = 0.0

            for neighbor, weight in adjacency[node].items():
                incoming += rank[neighbor] * weight / strength[neighbor]

            updated[node] = base + damping * incoming

        diff = sum(abs(updated[node] - rank[node]) for node in nodes)

        rank = updated

        if diff < tol:
            break

    return _normalize(rank)


# ============================================================
# MODULARIDADE E LOUVAIN (PYTHON)
# ============================================================

def modularity(adjacency, communities, resolution=1.0):
    """Modularidade de Newman-Girvan de uma partição."""

    total = sum(
        sum(neighbors.values())
        for neighbors in adjacency.values()
    )

    if total <= 0:
        return 0.0

    internal = defaultdict(float)
    degree = defaultdict(float)

    for node, neighbors in adjacency.items():

        community = communities.get(node)

        if community is None:
            continue

        for neighbor, weight in neighbors.items():

            degree[community] += weight

            if communities.get(neighbor) == community:
                internal[community] += weight

    return sum(
        internal[c] / total
        - resolution * (degree[c] / total) ** 2
        for c in degree
    )


def louvain(adjacency, resolution=1.0, max_levels=10):
    """
    Louvain (Blondel et al., 2008) determinístico.

    Retorna {nó: comunidade}.
    """

    nodes = sorted(adjacency)

    if not nodes:
        return {}

    # Nível 0: cópia do grafo; membership acompanha o mapeamento
    # dos nós originais para as comunidades do nível atual.
    graph = {
        node: dict(adjacency[node])
        for node in nodes
    }

    membership = {node: node for node in nodes}

    for _ in range(max_levels):

        strength = {
            node: sum(neighbors.values())
            for node, neighbors in graph.items()
        }

        total = sum(strength.values())

        if total <= 0:
            break

        local, moved = _louvain_local_moves(
            graph,
            strength,
            total,
            resolution,
        )

        if not moved:
            break

        # Propaga a mudança para os nós originais.
        membership = {
            node: local[membership[node]]
            for node in nodes
        }

        # Agrega o grafo: cada comunidade vira um super-nó.
        aggregated = defaultdict(lambda: defaultdict(float))

        for node, neighbors in graph.items():
            for neighbor, weight in neighbors.items():
                aggregated[local[node]][local[neighbor]] += weight

        new_graph = {
            community: dict(neighbors)
            for community, neighbors in aggregated.items()
        }

        if len(new_graph) == len(graph):
            break

        graph = new_graph

    return _relabel_by_size(membership)


def _louvain_local_moves(graph, strength, total, resolution):
    """Fase 1 do Louvain: movimentação local de nós."""

    community = {node: node for node in graph}
    community_total = dict(strength)

    order = sorted(graph, key=str)

    moved_any = False
    improved = True
    rounds = 0

    while improved and rounds < 100:

        improved = False
        rounds += 1

        for node in order:

            current = community[node]
            degree = strength[node]

            links = defaultdict(float)

            for neighbor, weight in graph[node].items():
                if neighbor != node:
                    links[community[neighbor]] += weight

            community_total[current] -= degree

            best = current
            best_gain = (
                links.get(current, 0.0)
                - resolution * community_total[current] * degree / total
            )

            for candidate in sorted(links, key=str):

                gain = (
                    links[candidate]
                    - resolution
                    * community_total[candidate]
                    * degree
                    / total
                )

                if gain > best_gain + 1e-12:
                    best = candidate
                    best_gain = gain

            community_total[best] += degree

            if best != current:
                community[node] = best
                improved = True
                moved_any = True

    return community, moved_any


# ============================================================
# SHORTEST PATH (PYTHON/BFS)
# ============================================================

def shortest_path_bfs(edges, source, target, max_hops=DEFAULT_MAX_HOPS):
    """
    Menor caminho (em número de saltos) por BFS, ignorando direção.

    Retorna {"nodes": [...], "edges": [...]} no mesmo formato de
    Neo4jStore.find_paths, ou None quando não há caminho.
    """

    source = str(source)
    target = str(target)

    if source == target:
        return None

    index = defaultdict(list)

    for edge in edges or []:

        if not _usable(edge):
            continue

        a = str(edge["src"])
        b = str(edge["dst"])

        index[a].append((b, edge))
        index[b].append((a, edge))

    for node in index:
        index[node].sort(
            key=lambda item: (item[0], item[1].get("rel", ""))
        )

    if source not in index or target not in index:
        return None

    previous = {source: None}
    queue = deque([(source, 0)])

    while queue:

        node, depth = queue.popleft()

        if node == target:
            break

        if depth >= max_hops:
            continue

        for neighbor, edge in index[node]:

            if neighbor in previous:
                continue

            previous[neighbor] = (node, edge)
            queue.append((neighbor, depth + 1))

    if target not in previous:
        return None

    nodes = [target]
    path_edges = []

    cursor = target

    while previous[cursor] is not None:
        parent, edge = previous[cursor]
        path_edges.append(edge)
        nodes.append(parent)
        cursor = parent

    nodes.reverse()
    path_edges.reverse()

    return {"nodes": nodes, "edges": path_edges}


def shortest_paths_between(store, seeds, max_hops=DEFAULT_MAX_HOPS):
    """
    Menor caminho entre cada par de entidades-semente.

    Usa Cypher shortestPath() no Neo4j e, se falhar, BFS em memória.

    Retorna (caminhos, cypher_text, engine). Cada caminho recebe
    "algorithm": "shortestPath" e "engine".
    """

    seeds = list(dict.fromkeys(str(seed) for seed in (seeds or [])))

    if len(seeds) < 2:
        return [], "", None

    pairs = list(combinations(seeds, 2))[:MAX_SEED_PAIRS]

    paths = []
    queries = []
    engine = "neo4j-cypher"
    cached_edges = None

    for source, target in pairs:

        path = None

        try:
            path, query = store.shortest_path(
                source,
                target,
                max_hops,
            )
            queries.append(query)

        except Exception:

            logger.exception(
                "shortestPath via Cypher falhou; usando BFS em memória."
            )

            engine = "python-bfs"

            if cached_edges is None:
                cached_edges = store.all_edges()

            path = shortest_path_bfs(
                cached_edges,
                source,
                target,
                max_hops,
            )

        if path:
            path = dict(path)
            path["algorithm"] = "shortestPath"
            path["engine"] = engine
            paths.append(path)

    cypher_text = ""

    if queries:
        cypher_text = (
            "// Shortest Path entre entidades-semente\n"
            + queries[0]
        )

    return paths, cypher_text, engine


# ============================================================
# GDS (NEO4J GRAPH DATA SCIENCE)
# ============================================================

def gds_available(store):
    """True se o plugin GDS responde a gds.version()."""

    try:
        store._run("CALL gds.version() YIELD version RETURN version")
        return True
    except Exception:
        return False


def _gds_drop(store):
    try:
        store._run(
            "CALL gds.graph.drop($name, false) YIELD graphName "
            "RETURN graphName",
            name=GDS_GRAPH_NAME,
        )
    except Exception:
        logger.debug("Projeção GDS inexistente ou não removível.")


def compute_gds(store):
    """
    PageRank, Louvain e Leiden pelo Neo4j GDS.

    Projeta o grafo de entidades como não direcionado, sem relações
    negadas, executa os três algoritmos e remove a projeção.
    """

    rels = list(RELS)

    _gds_drop(store)

    store._run(
        "MATCH (s:Entity)-[r]->(t:Entity) "
        "WHERE type(r) IN $rels "
        "AND coalesce(r.negated, false) = false "
        "AND s <> t "
        "WITH gds.graph.project("
        "$name, s, t, {}, "
        "{undirectedRelationshipTypes: ['*']}"
        ") AS g "
        "RETURN g.graphName AS name",
        name=GDS_GRAPH_NAME,
        rels=rels,
    )

    try:

        pr_records = store._run(
            "CALL gds.pageRank.stream($name) "
            "YIELD nodeId, score "
            "RETURN toString(gds.util.asNode(nodeId).id) AS id, "
            "score",
            name=GDS_GRAPH_NAME,
        )

        pr = {
            record["id"]: float(record["score"])
            for record in pr_records
        }

        louvain_records = store._run(
            "CALL gds.louvain.stream($name) "
            "YIELD nodeId, communityId "
            "RETURN toString(gds.util.asNode(nodeId).id) AS id, "
            "communityId",
            name=GDS_GRAPH_NAME,
        )

        louvain_map = {
            record["id"]: int(record["communityId"])
            for record in louvain_records
        }

        leiden_records = store._run(
            "CALL gds.leiden.stream($name, "
            "{randomSeed: 42, concurrency: 1}) "
            "YIELD nodeId, communityId "
            "RETURN toString(gds.util.asNode(nodeId).id) AS id, "
            "communityId",
            name=GDS_GRAPH_NAME,
        )

        leiden_map = {
            record["id"]: int(record["communityId"])
            for record in leiden_records
        }

    finally:
        _gds_drop(store)

    return {
        "pagerank": _normalize(pr),
        "louvain": _relabel_by_size(louvain_map),
        "leiden": _relabel_by_size(leiden_map),
    }


# ============================================================
# ORQUESTRAÇÃO
# ============================================================

def compute_python(edges):
    """PageRank e Louvain em Python; Leiden indisponível."""

    adjacency = build_adjacency(edges)

    return {
        "pagerank": pagerank(adjacency),
        "louvain": louvain(adjacency),
        "leiden": {},
    }


def _community_count(mapping):
    return len(set(mapping.values())) if mapping else 0


def compute_analytics(store, engine="auto"):
    """
    Calcula as análises do grafo.

    engine: "auto" (GDS se disponível), "gds" ou "python".

    Estrutura retornada (chaves usadas pela interface e pela viz):
      available, engine, message, pagerank, louvain, leiden,
      community (Leiden se existir, senão Louvain), modularity,
      communities, nodes, edges, notes, computed_at.
    """

    engine = (engine or "auto").strip().lower()
    notes = []

    edges = store.all_edges() or []
    adjacency = build_adjacency(edges)

    result = None
    used = None

    if engine in ("auto", "gds"):

        if gds_available(store):

            try:
                result = compute_gds(store)
                used = "gds"

            except Exception as exc:

                logger.exception("Falha ao executar o GDS.")

                _gds_drop(store)

                notes.append(
                    "GDS falhou ({}); usado fallback em Python."
                    .format(str(exc)[:200])
                )

        else:
            notes.append(
                "Plugin GDS não encontrado; usado fallback em Python."
            )

    if result is None:
        result = compute_python(edges)
        used = "python"

        notes.append(
            "Leiden requer Neo4j GDS e não é calculado no fallback "
            "em Python; a coluna de comunidade usa Louvain."
        )

    pr = result["pagerank"]
    louvain_map = result["louvain"]
    leiden_map = result["leiden"]

    community = leiden_map or louvain_map

    if not pr:
        return {
            "available": False,
            "engine": used,
            "message": (
                "O grafo de entidades não possui relações válidas "
                "para análise."
            ),
            "pagerank": {},
            "community": {},
            "louvain": {},
            "leiden": {},
            "modularity": {},
            "communities": {},
            "nodes": 0,
            "edges": 0,
            "notes": notes,
            "computed_at": time.time(),
        }

    modularity_values = {
        "louvain": round(modularity(adjacency, louvain_map), 4)
    }

    if leiden_map:
        modularity_values["leiden"] = round(
            modularity(adjacency, leiden_map),
            4,
        )

    usable_edges = sum(1 for edge in edges if _usable(edge))

    label = (
        "Neo4j GDS (PageRank, Louvain, Leiden)"
        if used == "gds"
        else "Python (PageRank, Louvain)"
    )

    return {
        "available": True,
        "engine": used,
        "message": "Análises calculadas com {}: {} nós, {} relações."
        .format(label, len(pr), usable_edges),
        "pagerank": pr,
        "community": community,
        "louvain": louvain_map,
        "leiden": leiden_map,
        "modularity": modularity_values,
        "communities": {
            "louvain": _community_count(louvain_map),
            "leiden": _community_count(leiden_map),
        },
        "nodes": len(pr),
        "edges": usable_edges,
        "notes": notes,
        "computed_at": time.time(),
    }


def default_engine():
    return os.getenv("GRAPH_ANALYTICS_ENGINE", "auto")


# ============================================================
# CONTEXTO ESTRUTURAL PARA A LLM
# ============================================================

def community_members(mapping, pagerank_scores, community_id, top=5):
    """Principais membros de uma comunidade, ordenados por PageRank."""

    members = [
        node
        for node, community in mapping.items()
        if community == community_id
    ]

    members.sort(
        key=lambda node: (-pagerank_scores.get(node, 0.0), node)
    )

    return members[:top]


def build_graph_context(analytics, store, seeds, paths=None, top=5):
    """
    Resume as métricas de grafo que acompanham o contexto da LLM.

    Retorna um dict com:
      engine, seeds (PageRank, posto e comunidade de cada semente),
      communities (comunidades das sementes com principais membros),
      shortest_paths (caminhos obtidos por Shortest Path).

    Retorna None quando as análises não estão disponíveis.
    """

    if not analytics or not analytics.get("available"):
        return None

    pr = analytics.get("pagerank") or {}
    mapping = analytics.get("community") or {}

    if not pr:
        return None

    ranking = sorted(pr, key=lambda node: (-pr[node], node))
    position = {node: i + 1 for i, node in enumerate(ranking)}

    sizes = defaultdict(int)

    for community in mapping.values():
        sizes[community] += 1

    def name_of(node_id):
        try:
            node = store.node(node_id)
        except Exception:
            node = None

        return (node or {}).get("name") or node_id

    seed_rows = []
    seen_communities = []

    for seed in seeds or []:

        seed = str(seed)

        if seed not in pr:
            continue

        community = mapping.get(seed)

        seed_rows.append({
            "id": seed,
            "name": name_of(seed),
            "pagerank": round(pr[seed], 5),
            "rank": position[seed],
            "of": len(pr),
            "community": community,
            "community_size": sizes.get(community, 0),
        })

        if community is not None and community not in seen_communities:
            seen_communities.append(community)

    community_rows = []

    for community in seen_communities:

        members = community_members(mapping, pr, community, top)

        community_rows.append({
            "id": community,
            "size": sizes.get(community, 0),
            "top_members": [
                {
                    "id": member,
                    "name": name_of(member),
                    "pagerank": round(pr[member], 5),
                }
                for member in members
            ],
        })

    shortest = [
        {
            "nodes": path.get("nodes", []),
            "hops": len(path.get("edges", [])),
            "engine": path.get("engine"),
        }
        for path in (paths or [])
        if path.get("algorithm") == "shortestPath"
    ]

    return {
        "engine": analytics.get("engine"),
        "seeds": seed_rows,
        "communities": community_rows,
        "shortest_paths": shortest,
    }