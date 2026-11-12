import csv
import os
import tempfile
import unittest
from unittest import mock

from graphrag_med import experiment as ex


BLOCK = (
    "[F1] Tramadol INTERACTS_WITH Paracetamol | certainty=asserted conf=0.9 "
    "fontes=D1 | evidência: \"Tramadol pode interagir com paracetamol.\"\n"
    "[F2] Tramadol CAUSES Náusea | certainty=asserted conf=0.9 fontes=D1 | "
    "evidência: \"Náusea é comum com tramadol, em 10% dos pacientes.\"\n"
    "[P1] caminho completo: Tramadol —CAUSES→ Náusea | fatos: F2\n"
    "[G1] métrica estrutural (python; não é fato clínico): Tramadol — "
    "PageRank=0.1000 (posto 1 de 9), comunidade 0 com 4 entidades\n"
    "[T1] trecho de bula (similaridade=0.800; documento=D1): O tramadol é um\n"
    "analgésico opioide usado para dor moderada a intensa."
)


class EvidenceTests(unittest.TestCase):
    def test_parse_evidence_with_continuation_lines(self):
        items = ex.parse_evidence(BLOCK)

        self.assertEqual(set(items), {"F1", "F2", "P1", "G1", "T1"})
        self.assertIn("analgésico opioide", items["T1"])

    def test_graph_metrics_are_not_clinical_evidence(self):
        clinical = ex.clinical_items(ex.parse_evidence(BLOCK))

        self.assertNotIn("G1", clinical)
        self.assertIn("F1", clinical)


class ClaimSplitTests(unittest.TestCase):
    def test_splits_and_drops_disclaimers_and_headings(self):
        answer = (
            "**Fatos diretos:**\n"
            "- Tramadol interage com paracetamol [F1].\n"
            "- Náusea é um efeito do tramadol. [F2]\n"
            "Consulte um profissional de saúde.\n"
        )

        claims = ex.split_claims(answer)

        self.assertEqual(len(claims), 2)
        self.assertEqual(claims[0]["cites"], ["F1"])
        self.assertEqual(claims[1]["cites"], ["F2"])

    def test_multiple_tags_in_one_bracket(self):
        claims = ex.split_claims(
            "Tramadol interage com paracetamol [F1, P1]."
        )

        self.assertEqual(claims[0]["cites"], ["F1", "P1"])


class CheckClaimTests(unittest.TestCase):
    def setUp(self):
        self.items = ex.parse_evidence(BLOCK)

    def grade(self, text):
        return ex.check_claim(ex.split_claims(text)[0], self.items)

    def test_paraphrase_with_relation_translation_is_supported(self):
        result = self.grade(
            "O tramadol pode interagir com o paracetamol [F1]."
        )

        self.assertTrue(result["supported"])
        self.assertEqual(result["cite_status"], "ok")

    def test_invented_drug_is_unsupported(self):
        result = self.grade(
            "O tramadol interage com a Varfarina e aumenta o sangramento."
        )

        self.assertFalse(result["supported"])
        self.assertIn("Varfarina", result["missing_entities"])

    def test_number_absent_from_evidence_is_unsupported(self):
        result = self.grade(
            "A náusea ocorre em 35% dos pacientes que usam tramadol."
        )

        self.assertFalse(result["supported"])
        self.assertTrue(result["missing_numbers"])

    def test_number_present_in_evidence_is_supported(self):
        result = self.grade(
            "A náusea é comum com tramadol, em 10% dos pacientes."
        )

        self.assertTrue(result["supported"])

    def test_invalid_citation_tags(self):
        for tag in ("[F9]", "[G1]"):
            result = self.grade(
                "O tramadol pode interagir com o paracetamol {}.".format(tag)
            )

            self.assertEqual(result["cite_status"], "invalid", tag)

    def test_citation_mismatch(self):
        result = self.grade(
            "O tramadol pode interagir com o paracetamol [F2]."
        )

        self.assertTrue(result["supported"])
        self.assertEqual(result["cite_status"], "mismatch")

    def test_no_citation(self):
        result = self.grade(
            "O tramadol é um analgésico opioide para dor intensa."
        )

        self.assertEqual(result["cite_status"], "none")


class AbstentionTests(unittest.TestCase):
    def test_detects_abstention(self):
        self.assertTrue(ex.is_abstention(
            "Não há evidências suficientes para responder."
        ))
        self.assertTrue(ex.is_abstention(
            "Não posso responder com segurança. A pergunta pede "
            "informação comercial que não faz parte do ChatBulário."
        ))

    def test_factual_answer_is_not_abstention(self):
        self.assertFalse(ex.is_abstention(
            "Tramadol interage com paracetamol [F1]. "
            "Náusea é um efeito do tramadol [F2]."
        ))


class GradeAnswerTests(unittest.TestCase):
    def test_counts_and_judge(self):
        items = ex.parse_evidence(BLOCK)

        answer = (
            "Tramadol pode interagir com paracetamol [F1]. "
            "Tramadol também causa Hepatotoxicidade grave."
        )

        grade = ex.grade_answer(
            answer, items, judge=lambda claim, evidence: "Hepato" not in claim
        )

        self.assertEqual(grade["n_claims"], 2)
        self.assertEqual(grade["n_unsupported"], 1)
        self.assertEqual(grade["n_judged"], 2)
        self.assertEqual(grade["n_judge_unsupported"], 1)
        self.assertEqual(grade["n_cited"], 1)
        self.assertEqual(grade["n_uncited"], 1)

    def test_judge_parses_verdict(self):
        class Response:
            def raise_for_status(self):
                pass

            def json(self):
                return {"message": {"content": '{"verdict": "supported"}'}}

        with mock.patch("requests.post", return_value=Response()):
            self.assertTrue(ex.judge_claim("a", "b"))

        with mock.patch("requests.post", side_effect=RuntimeError("off")):
            self.assertIsNone(ex.judge_claim("a", "b"))


class RecallTests(unittest.TestCase):
    NAMES = {
        "DRUG:tramadol": ("Tramadol", []),
        "DRUG:paracetamol": ("Paracetamol", ["acetaminofeno"]),
        "AE:nausea": ("Náusea", []),
    }

    def test_recall_nodes(self):
        self.assertEqual(
            ex.recall_nodes(["a", "b"], ["a", "c"]), 0.5
        )
        self.assertIsNone(ex.recall_nodes([], ["a"]))

    def test_recall_text_uses_names_and_synonyms(self):
        value = ex.recall_text(
            ["DRUG:paracetamol", "AE:nausea"],
            ["O acetaminofeno é um analgésico."],
            self.NAMES,
        )

        self.assertEqual(value, 0.5)

    def test_recall_edges_cooccurrence_needs_same_item(self):
        edge = [["DRUG:tramadol", "INTERACTS_WITH", "DRUG:paracetamol"]]

        same = ex.recall_edges_cooccurrence(
            edge, ["Tramadol e paracetamol juntos."], self.NAMES
        )
        split = ex.recall_edges_cooccurrence(
            edge, ["Tramadol sozinho.", "Paracetamol sozinho."], self.NAMES
        )

        self.assertEqual(same, 1.0)
        self.assertEqual(split, 0.0)

    def test_recall_edges_exact_any_direction_and_wildcard(self):
        facts = [{"src": "B", "dst": "A", "rel": "CAUSES"}]

        self.assertEqual(
            ex.recall_edges_exact([["A", "CAUSES", "B"]], facts), 1.0
        )
        self.assertEqual(
            ex.recall_edges_exact([["A", "TREATS", "B"]], facts), 0.0
        )
        self.assertEqual(
            ex.recall_edges_exact([["A", "*", "B"]], facts), 1.0
        )


class StatsTests(unittest.TestCase):
    def test_bootstrap_is_deterministic_and_brackets_point(self):
        rows = [0, 0, 1, 1, 1, 0, 1, 0, 0, 1]
        stat = lambda s: sum(s) / float(len(s))

        a = ex.bootstrap_ci(rows, stat, n_boot=500, seed=1)
        b = ex.bootstrap_ci(rows, stat, n_boot=500, seed=1)

        self.assertEqual(a, b)
        self.assertLessEqual(a[1], a[0])
        self.assertGreaterEqual(a[2], a[0])

    def test_bootstrap_single_row_has_no_interval(self):
        point, lo, hi = ex.bootstrap_ci([1], lambda s: 1.0)

        self.assertEqual((point, lo, hi), (1.0, None, None))

    def test_kappa(self):
        self.assertEqual(ex.cohen_kappa([1, 0, 1, 0], [1, 0, 1, 0]), 1.0)
        self.assertAlmostEqual(
            ex.cohen_kappa([1, 1, 0, 0], [1, 0, 1, 0]), 0.0
        )


def fake_runner(answer, found, facts, items_block=BLOCK, latency=1.0):
    def run(item):
        return {
            "answer": answer,
            "items": ex.parse_evidence(items_block),
            "found_ids": found,
            "facts": facts,
            "latency": latency,
        }

    return run


class ExperimentFlowTests(unittest.TestCase):
    QUESTIONS = [
        {
            "q": "Quais efeitos do tramadol?",
            "hops": 2,
            "expected": ["DRUG:tramadol", "AE:nausea"],
            "expected_edges": [["DRUG:tramadol", "CAUSES", "AE:nausea"]],
        },
        {"q": "Qual o preço do tramadol?", "expect_refusal": True},
    ]

    def setUp(self):
        names = RecallTests.NAMES

        graph_answer = "Tramadol causa Náusea [F2]. Tramadol interage com paracetamol [F1]."
        rag_answer = "Tramadol causa Náusea. Tramadol interage com Varfarina."

        def graph(item):
            if item.get("expect_refusal"):
                return {
                    "answer": "Não posso responder com segurança.",
                    "items": {},
                    "found_ids": [],
                    "facts": [],
                    "latency": 0.1,
                }
            return fake_runner(
                graph_answer,
                ["DRUG:tramadol", "AE:nausea"],
                [{"src": "DRUG:tramadol", "dst": "AE:nausea", "rel": "CAUSES"}],
            )(item)

        def rag(item):
            if item.get("expect_refusal"):
                return fake_runner(rag_answer, [], None)(item)
            return fake_runner(
                rag_answer,
                ["DRUG:tramadol"],
                None,
                items_block="[T1] O tramadol é um analgésico opioide.",
            )(item)

        self.conditions = ["rag_vetorial", "graphrag"]

        self.results = ex.run_experiment(
            self.QUESTIONS,
            self.conditions,
            {"rag_vetorial": rag, "graphrag": graph},
            name_lookup=lambda entity: names.get(entity),
        )

    def test_recall_favors_graph(self):
        first = self.results[0]["cond"]

        self.assertEqual(first["graphrag"]["recall_nodes"], 1.0)
        self.assertEqual(first["rag_vetorial"]["recall_nodes"], 0.5)
        self.assertEqual(first["graphrag"]["recall_edges_exact"], 1.0)
        self.assertIsNone(first["rag_vetorial"]["recall_edges_exact"])
        self.assertEqual(first["graphrag"]["recall_edges_cooc"], 1.0)
        self.assertEqual(first["rag_vetorial"]["recall_edges_cooc"], 0.0)

    def test_question_kinds_and_multihop(self):
        self.assertEqual(self.results[0]["kind"], "respondivel")
        self.assertTrue(self.results[0]["multihop"])
        self.assertEqual(self.results[1]["kind"], "nao_respondivel")

    def test_aggregate_hallucination_and_abstention(self):
        summary = ex.aggregate(self.results, self.conditions, n_boot=200)

        graph = summary["graphrag"]
        rag = summary["rag_vetorial"]

        self.assertEqual(graph["unsupported_rate"][0], 0.0)
        self.assertGreater(rag["unsupported_rate"][0], 0.0)
        self.assertEqual(graph["abstention_correct"][0], 1.0)
        self.assertEqual(rag["abstention_correct"][0], 0.0)

    def test_paired_difference_negative_for_hallucination(self):
        diff = ex.paired_differences(
            self.results, "graphrag", "rag_vetorial", n_boot=200
        )

        self.assertLess(diff["unsupported_rate"][0], 0)
        self.assertGreater(diff["recall_nodes"][0], 0)

    def test_tables_markdown_and_latex(self):
        summary = ex.aggregate(self.results, self.conditions, n_boot=100)
        diff = ex.paired_differences(
            self.results, "graphrag", "rag_vetorial", n_boot=100
        )

        tables = ex.build_tables(
            summary,
            self.conditions,
            {"GraphRAG − RAG vetorial": diff},
        )

        self.assertEqual(
            [t["id"] for t in tables],
            ["recuperacao", "fidelidade", "abstencao", "diferencas"],
        )

        markdown = ex.to_markdown(tables)
        latex = ex.to_latex(tables)

        self.assertIn("GraphRAG", markdown)
        self.assertIn("\\toprule", latex)
        self.assertIn("\\%", latex)
        self.assertEqual(latex.count("\\begin{table}"), 4)

    def test_regrade_changes_threshold(self):
        before = self.results[0]["cond"]["graphrag"]["grade"]["n_unsupported"]

        ex.regrade(self.results, tau=1.01)

        after = self.results[0]["cond"]["graphrag"]["grade"]["n_unsupported"]

        self.assertGreaterEqual(after, before)

    def test_runner_error_is_recorded_not_raised(self):
        results = ex.run_experiment(
            [{"q": "x y z"}],
            ["graphrag"],
            {"graphrag": lambda item: {"error": "LLM offline"}},
        )

        record = results[0]["cond"]["graphrag"]

        self.assertEqual(record["error"], "LLM offline")
        self.assertNotIn("grade", record)

        summary = ex.aggregate(results, ["graphrag"], n_boot=10)

        self.assertEqual(summary["graphrag"]["n_errors"], 1)


class AuditTests(unittest.TestCase):
    def test_roundtrip_and_agreement(self):
        results = ex.run_experiment(
            [{"q": "q1", "expected": ["DRUG:tramadol"]}],
            ["graphrag"],
            {"graphrag": fake_runner(
                "Tramadol causa Náusea [F2]. Tramadol interage com Varfarina.",
                ["DRUG:tramadol"],
                [],
            )},
        )

        sample = ex.audit_sample(results, ["graphrag"], n=10, seed=1)

        self.assertEqual(len(sample), 2)

        with tempfile.TemporaryDirectory() as folder:
            path = os.path.join(folder, "auditoria.csv")

            for row in sample:
                # O "humano" concorda com o verificador em tudo.
                row["humano_suportada"] = str(row["auto_suportada"])

            ex.write_csv(path, sample)

            report = ex.evaluate_audit(path)

        self.assertEqual(report["n"], 2)
        self.assertEqual(report["concordancia"], 1.0)
        self.assertEqual(report["taxa_humana_sem_suporte"]["graphrag"], 0.5)

    def test_empty_audit(self):
        with tempfile.TemporaryDirectory() as folder:
            path = os.path.join(folder, "a.csv")

            with open(path, "w", encoding="utf-8", newline="") as handle:
                writer = csv.writer(handle)
                writer.writerow(["condicao", "auto_suportada", "humano_suportada"])
                writer.writerow(["graphrag", "1", ""])

            self.assertEqual(ex.evaluate_audit(path)["n"], 0)


if __name__ == "__main__":
    unittest.main()