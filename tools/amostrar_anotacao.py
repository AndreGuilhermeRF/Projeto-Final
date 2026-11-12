"""
Gera um modelo de conjunto-ouro para anotação manual de NER/RE.

Amostra frases do ChatBulário (estratificadas por seção) e grava um JSON
no formato esperado pela aba "Avaliação" (Avaliar NER e RE):

  [{"text": "...",
    "entities":  [{"text": "diabetes", "type": "Condition"}],
    "relations": [{"subject": "metformina", "relation": "TREATS",
                   "object": "diabetes"}]}]

Uso:
  python tools/amostrar_anotacao.py --n 60 --seed 42 \
      --saida graphrag_med/data/anotacao_modelo.json

Regras de anotação (combine com a equipe e registre no artigo):
- Tipos: Drug, Condition, AdverseEffect, Procedure, Anatomy, DrugClass.
- Relações: TREATS, CONTRAINDICATED_IN, CAUSES, INTERACTS_WITH, BELONGS_TO,
  HAS_SYMPTOM, DIAGNOSED_WITH, ADMINISTERED_BY.
- Use a grafia literal do texto, sem "®" e sem pontuação final. A métrica
  compara texto (sem diferenciar maiúsculas) e tipo por igualdade exata.
- Anote só o que a frase afirma. Relações negadas ou hipotéticas ficam
  de fora (a métrica atual não trata negação).
- Duas pessoas devem anotar as mesmas frases; calcule a concordância
  (Cohen's kappa) e resolva divergências antes de medir o sistema.
- Apague o campo "contexto" se quiser; ele é ignorado pela avaliação.
"""

import argparse
import json
import random
import re
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DEFAULT_DATA = ROOT / "graphrag_med" / "data" / "chatbulario.json"
SENTENCE_SPLIT = re.compile(r"(?<=[.!?;])\s+(?=[A-ZÁÉÍÓÚÂÊÔÃÕÇ])")


def iter_records(path):
    """Lê o JSON linha a linha (um objeto por linha); recorre a json.load."""

    try:
        with open(path, encoding="utf-8") as handle:
            for line in handle:
                line = line.strip().rstrip(",")
                if line in ("", "[", "]"):
                    continue
                yield json.loads(line)
        return
    except json.JSONDecodeError:
        pass

    with open(path, encoding="utf-8") as handle:
        for record in json.load(handle):
            yield record


def sentences(text):
    text = re.sub(r"\s+", " ", text or "").strip()
    for sentence in SENTENCE_SPLIT.split(text):
        if 60 <= len(sentence) <= 400 and "Folheto" not in sentence:
            yield sentence


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[1])
    parser.add_argument("--dados", default=str(DEFAULT_DATA))
    parser.add_argument("--saida", default="anotacao_modelo.json")
    parser.add_argument("--n", type=int, default=60)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-registros", type=int, default=3000)
    args = parser.parse_args()

    rng = random.Random(args.seed)
    by_section = defaultdict(list)

    for count, record in enumerate(iter_records(args.dados)):
        if count >= args.max_registros:
            break
        for sentence in sentences(record.get("resposta")):
            by_section[record.get("secao_id")].append({
                "text": sentence,
                "contexto": {
                    "registro": record.get("registro"),
                    "produto": record.get("nome_produto"),
                    "secao_id": record.get("secao_id"),
                },
            })

    if not by_section:
        raise SystemExit("Nenhuma frase encontrada em " + args.dados)

    sections = sorted(by_section, key=str)

    for pool in by_section.values():
        rng.shuffle(pool)

    chosen, seen = [], set()
    index = 0

    while len(chosen) < args.n and any(by_section.values()):
        pool = by_section[sections[index % len(sections)]]
        index += 1
        while pool:
            item = pool.pop()
            if item["text"] not in seen:
                seen.add(item["text"])
                item["entities"] = []
                item["relations"] = []
                chosen.append(item)
                break

    with open(args.saida, "w", encoding="utf-8") as handle:
        json.dump(chosen, handle, ensure_ascii=False, indent=2)

    print("{} frases gravadas em {} (seed={}).".format(
        len(chosen), args.saida, args.seed))


if __name__ == "__main__":
    main()