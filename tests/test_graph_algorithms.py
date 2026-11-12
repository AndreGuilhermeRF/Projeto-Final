import unittest

from graphrag_med import graph_algorithms as ga
from graphrag_med.graph_store import shortest_path_query
from graphrag_med.llm import facts_block, graph_metrics_lines


def edge(src, dst, rel="INTERACTS_WITH", negated=False):
    return {"src": src, "dst": dst, "rel": rel, "negated": negated}


def two_cliques():
    """Duas comunidades de 4 nós ligadas pela ponte a1—b1."""

    edges = []

    for group in (["a1", "a2", "a3", "a4"], ["b1", "b2", "b3", "b4"]):
        for i in range(len(group)):
            for j in range(i + 1, len(group)):
                edges.append(edge(group[i], group[j]))

    edges.append(edge("a1", "b1", "TREATS"))

    # Devem ser ignoradas: negada, tipo desconhecido e auto-laço.
    edges.append(edge("a2", "b2", "CAUSES", negated=True))
    edges.append(edge("x", "y", "FOO"))
    edges.append(edge("a3", "a3"))

    return edges


class FakeStore:
    """Neo4j falso: sem GDS e com shortestPath indisponível."""

    def __init__(self, edges):
        self._edges = edges
        self.names = {}

    def _run(self, query, **params):
        raise RuntimeError("GDS indisponível")

    def all_edges(self):
        return self._edges

    def shortest_path(self, source, target, max_hops=5):
        raise RuntimeError("Cypher indisponível")

    def node(self, node_id):
        return {
            "id": node_id,
            "name": self.names.get(node_id, node_id.upper()),
        }


class AdjacencyTests(unittest.TestCase):
    def test_ignores_negated_unknown_and_self_loops(self):
        adjacency = ga.build_adjacency(two_cliques())

        self.assertNotIn("x", adjacency)
        self.assertNotIn("y", adjacency)
        self.assertNotIn("b2", adjacency["a2"])
        self.assertNotIn("a3", adjacency["a3"])
        self.assertIn("b1", adjacency["a1"])


class PageRankTests(unittest.TestCase):
    def test_scores_sum_to_one(self):
        scores = ga.pagerank(ga.build_adjacency(two_cliques()))

        self.assertAlmostEqual(sum(scores.values()), 1.0, places=6)

    def test_bridge_nodes_are_most_central(self):
        scores = ga.pagerank(ga.build_adjacency(two_cliques()))

        top_two = sorted(scores, key=scores.get, reverse=True)[:2]

        self.assertEqual(set(top_two), {"a1", "b1"})

    def test_empty_graph(self):
        self.assertEqual(ga.pagerank({}), {})


class LouvainTests(unittest.TestCase):
    def test_finds_two_communities(self):
        adjacency = ga.build_adjacency(two_cliques())
        communities = ga.louvain(adjacency)

        self.assertEqual(len(set(communities.values())), 2)
        self.assertEqual(
            len({communities[n] for n in ("a1", "a2", "a3", "a4")}), 1
        )
        self.assertEqual(
            len({communities[n] for n in ("b1", "b2", "b3", "b4")}), 1
        )
        self.assertNotEqual(communities["a1"], communities["b1"])

    def test_modularity_is_positive_and_beats_single_community(self):
        adjacency = ga.build_adjacency(two_cliques())
        communities = ga.louvain(adjacency)

        single = {node: 0 for node in adjacency}

        self.assertGreater(ga.modularity(adjacency, communities), 0.3)
        self.assertGreater(
            ga.modularity(adjacency, communities),
            ga.modularity(adjacency, single),
        )

    def test_deterministic(self):
        adjacency = ga.build_adjacency(two_cliques())

        self.assertEqual(ga.louvain(adjacency), ga.louvain(adjacency))

    def test_ids_start_at_zero_for_largest(self):
        edges = two_cliques() + [edge("c1", "c2")]
        communities = ga.louvain(ga.build_adjacency(edges))

        self.assertEqual(min(communities.values()), 0)
        self.assertEqual(communities["c1"], communities["c2"])
        self.assertEqual(
            max(communities.values()), len(set(communities.values())) - 1
        )


class ShortestPathTests(unittest.TestCase):
    def test_bfs_crosses_bridge(self):
        path = ga.shortest_path_bfs(two_cliques(), "a3", "b4")

        self.assertEqual(path["nodes"], ["a3", "a1", "b1", "b4"])
        self.assertEqual(len(path["edges"]), 3)

    def test_bfs_ignores_negated_edges(self):
        edges = [edge("a", "b", "CAUSES", negated=True)]

        self.assertIsNone(ga.shortest_path_bfs(edges, "a", "b"))

    def test_bfs_respects_max_hops(self):
        edges = [edge("a", "b"), edge("b", "c"), edge("c", "d")]

        self.assertIsNone(ga.shortest_path_bfs(edges, "a", "d", max_hops=2))
        self.assertIsNotNone(
            ga.shortest_path_bfs(edges, "a", "d", max_hops=3)
        )

    def test_fallback_to_bfs_when_cypher_fails(self):
        store = FakeStore(two_cliques())

        with self.assertLogs(ga.logger, level="ERROR"):
            paths, _cypher, engine = ga.shortest_paths_between(
                store, ["a3", "b4"]
            )

        self.assertEqual(engine, "python-bfs")
        self.assertEqual(len(paths), 1)
        self.assertEqual(paths[0]["algorithm"], "shortestPath")

    def test_needs_two_seeds(self):
        paths, cypher, engine = ga.shortest_paths_between(
            FakeStore(two_cliques()), ["a1"]
        )

        self.assertEqual((paths, cypher, engine), ([], "", None))

    def test_cypher_is_read_only_and_excludes_negated(self):
        query = shortest_path_query(3)

        self.assertIn("shortestPath", query)
        self.assertIn("*..3", query)
        self.assertIn("negated", query)
        self.assertNotRegex(query, r"\b(CREATE|MERGE|DELETE|SET)\b")


class AnalyticsTests(unittest.TestCase):
    def test_python_fallback_when_gds_missing(self):
        store = FakeStore(two_cliques())

        data = ga.compute_analytics(store, "auto")

        self.assertTrue(data["available"])
        self.assertEqual(data["engine"], "python")
        self.assertEqual(data["leiden"], {})
        self.assertEqual(data["community"], data["louvain"])
        self.assertEqual(data["nodes"], 8)
        self.assertEqual(data["communities"]["louvain"], 2)
        self.assertGreater(data["modularity"]["louvain"], 0.3)
        self.assertTrue(data["notes"])

    def test_empty_graph_is_unavailable(self):
        data = ga.compute_analytics(FakeStore([]), "python")

        self.assertFalse(data["available"])


class GraphContextTests(unittest.TestCase):
    def setUp(self):
        self.store = FakeStore(two_cliques())
        self.store.names = {"a1": "Metformina", "a2": "Insulina"}
        self.analytics = ga.compute_analytics(self.store, "python")

    def test_seed_metrics_and_community_members(self):
        context = ga.build_graph_context(
            self.analytics, self.store, ["a1", "zzz"]
        )

        self.assertEqual(len(context["seeds"]), 1)

        seed = context["seeds"][0]

        self.assertEqual(seed["name"], "Metformina")
        self.assertEqual(seed["of"], 8)
        self.assertLessEqual(seed["rank"], 2)
        self.assertEqual(seed["community_size"], 4)
        self.assertEqual(len(context["communities"]), 1)

    def test_none_when_analytics_unavailable(self):
        self.assertIsNone(
            ga.build_graph_context({"available": False}, self.store, ["a1"])
        )

    def test_llm_context_labels_metrics_as_non_clinical(self):
        context = ga.build_graph_context(
            self.analytics, self.store, ["a1"]
        )

        lines = graph_metrics_lines(context)

        self.assertEqual(len(lines), 1)
        self.assertIn("[G1]", lines[0])
        self.assertIn("não é fato clínico", lines[0])
        self.assertIn("Metformina", lines[0])

        block = facts_block({"graph_context": context})

        self.assertIn("[G1]", block)

    def test_shortest_path_is_marked_in_llm_block(self):
        block = facts_block({
            "path_citations": [{
                "id": "P1",
                "text": "A —TREATS→ B",
                "fact_ids": ["F1"],
                "algorithm": "shortestPath",
            }]
        })

        self.assertIn("Shortest Path", block)


if __name__ == "__main__":
    unittest.main()