import importlib.util
import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from graphrag_med import experiment as ex

ROOT = Path(__file__).resolve().parent.parent


def load_tool():
    spec = importlib.util.spec_from_file_location(
        "avaliar_rag_vs_graphrag",
        ROOT / "tools" / "avaliar_rag_vs_graphrag.py",
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


BLOCK = "[F1] Tramadol CAUSES Náusea | certainty=asserted | evidência: \"Náusea é comum com tramadol.\""


class CliSmokeTests(unittest.TestCase):
    def test_full_run_writes_all_outputs(self):
        tool = load_tool()

        questions = [
            {"q": "Quais efeitos do tramadol?", "hops": 2,
             "expected": ["DRUG:tramadol", "AE:nausea"]},
            {"q": "Qual o preço do tramadol?", "expect_refusal": True},
        ]

        def graph(item):
            return {"answer": "Tramadol causa Náusea [F1].",
                    "items": ex.parse_evidence(BLOCK),
                    "found_ids": ["DRUG:tramadol", "AE:nausea"],
                    "facts": [], "latency": 0.2}

        def rag(item):
            return {"answer": "Tramadol interage com Varfarina.",
                    "items": ex.parse_evidence("[T1] O tramadol é um opioide."),
                    "found_ids": ["DRUG:tramadol"], "facts": None,
                    "latency": 0.1}

        fake_system = mock.Mock()

        with tempfile.TemporaryDirectory() as folder:
            qfile = Path(folder) / "q.json"
            qfile.write_text(json.dumps(questions), encoding="utf-8")

            argv = ["avaliar", "--perguntas", str(qfile),
                    "--saida", folder, "--boot", "50"]

            with mock.patch.object(tool, "build_system", return_value=fake_system), \
                 mock.patch.object(tool, "make_runners",
                                   return_value={"rag_vetorial": rag, "graphrag": graph}), \
                 mock.patch.object(tool, "make_name_lookup",
                                   return_value=lambda e: (e.split(":")[-1], [])), \
                 mock.patch("graphrag_med.llm.llm_available", return_value=True), \
                 mock.patch.object(sys, "argv", argv):
                tool.main()

            runs = [p for p in Path(folder).iterdir() if p.is_dir()]

            self.assertEqual(len(runs), 1)

            names = {p.name for p in runs[0].iterdir()}

            for expected in ("resultados.json", "resumo.json", "tabelas.md",
                             "tabelas.tex", "por_pergunta.csv",
                             "auditoria_amostra.csv"):
                self.assertIn(expected, names)

            summary = json.loads((runs[0] / "resumo.json").read_text(encoding="utf-8"))

            self.assertEqual(
                summary["summary"]["graphrag"]["unsupported_rate"][0], 0.0
            )

            # Reagregar reaproveita as respostas sem chamar o LLM.
            argv2 = ["avaliar", "--reagregar", str(runs[0] / "resultados.json"),
                     "--saida", folder, "--boot", "50", "--tau", "0.5"]

            with mock.patch.object(sys, "argv", argv2):
                tool.main()

            self.assertEqual(
                len([p for p in Path(folder).iterdir() if p.is_dir()]), 2
            )


if __name__ == "__main__":
    unittest.main() 