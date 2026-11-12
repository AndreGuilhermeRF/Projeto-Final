import unittest

from graphrag_med.lexicon import Lexicon
from graphrag_med.pipeline import extract_from_sentence
from graphrag_med.clinical_extractor import _json_from_response
from graphrag_med.evaluation import extraction_metrics
from graphrag_med.graph_store import Neo4jStore


class CoreTests(unittest.TestCase):
    def test_lexicon_entity_resolution(self):
        lex = Lexicon()
        mentions = lex.find("O paciente apresenta diabetes tipo 2.")
        self.assertTrue(any(item.type == "Condition" for item in mentions))

    def test_rule_relation_extraction(self):
        lex = Lexicon()
        lex.add_entity({
            "id": "DRUG:metformin", "type": "Drug",
            "name": "Metformin", "synonyms": ["metformina"]
        })
        triples = extract_from_sentence(
            "A metformina pode causar acidose láctica.", lex
        )
        self.assertTrue(triples)
        self.assertEqual(triples[0]["rel"], "CAUSES")
        self.assertEqual(triples[0]["certainty"], "hypothetical")

    def test_json_extractor_parser(self):
        parsed = _json_from_response(
            '{"entities":[{"text":"diabetes","type":"Condition"}],"relations":[]}'
        )
        self.assertEqual(parsed["entities"][0]["text"], "diabetes")

    def test_extraction_metrics(self):
        result = extraction_metrics(
            [("metformina", "Drug"), ("diabetes", "Condition")],
            [("metformina", "Drug"), ("asma", "Condition")],
        )
        self.assertEqual(result["tp"], 1)
        self.assertEqual(result["fp"], 1)
        self.assertEqual(result["fn"], 1)

    def test_readonly_cypher_blocks_write(self):
        store = Neo4jStore.__new__(Neo4jStore)
        store._run = lambda query, **params: []
        with self.assertRaises(ValueError):
            store.execute_readonly_cypher(
                "MATCH (n) DETACH DELETE n RETURN n"
            )

    def test_readonly_cypher_accepts_read(self):
        class FakeRecord:
            def data(self):
                return {"ok": True}

        class FakeTx:
            def run(self, query, **params):
                self.query = query
                self.params = params
                return [FakeRecord()]

        class FakeSession:
            def __enter__(self):
                return self

            def __exit__(self, *args):
                return False

            def execute_read(self, work):
                return work(FakeTx())

        class FakeDriver:
            def session(self, database=None):
                return FakeSession()

        store = Neo4jStore.__new__(Neo4jStore)
        store.driver = FakeDriver()
        store.database = "neo4j"

        result = store.execute_readonly_cypher(
            "MATCH (n:Entity) RETURN n LIMIT $limit", {"limit": 1}
        )
        self.assertEqual(result[0]["ok"], True)


if __name__ == "__main__":
    unittest.main()