"""
Lista entidades e relações reais do Neo4j para montar o evaluation.json.

As perguntas de recuperação precisam de IDs esperados que existam no grafo
da sua máquina. Rode este script, escolha entidades/relações e escreva
perguntas cuja resposta você consiga verificar na bula original:

  {"q": "Quais medicamentos interagem com X?", "hops": 2,
   "expected": ["DRUG:x", "DRUG:y"]}

Também há dois tipos de caso que não usam "expected":
  {"q": "...", "expect_refusal": true}   # pergunta fora do escopo
  {"q": "...", "expect_negated": true}   # fonte afirma que NÃO há relação

Uso:
  python tools/listar_entidades.py --top 30 --relacoes 40
"""

import argparse
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--top", type=int, default=30)
    parser.add_argument("--relacoes", type=int, default=40)
    args = parser.parse_args()

    from dotenv import load_dotenv
    from graphrag_med.graph_store import Neo4jStore

    load_dotenv(ROOT / ".env", override=False)

    store = Neo4jStore(
        os.getenv("NEO4J_URI", "bolt://127.0.0.1:7687"),
        os.getenv("NEO4J_USER", "neo4j"),
        os.getenv("NEO4J_PASSWORD", ""),
        os.getenv("NEO4J_DATABASE", "neo4j"),
    )

    try:
        print("== Entidades com mais relações ==")
        for row in store._run(
            "MATCH (n:Entity) "
            "OPTIONAL MATCH (n)-[r]-(:Entity) "
            "RETURN n.id AS id, n.name AS name, n.type AS type, "
            "count(r) AS grau ORDER BY grau DESC LIMIT $k",
            k=args.top,
        ):
            print("{:<40} {:<28} {:<14} grau={}".format(
                row["id"], row["name"], row["type"], row["grau"]))

        print("\n== Relações (não negadas) ==")
        for row in store._run(
            "MATCH (a:Entity)-[r]->(b:Entity) "
            "WHERE coalesce(r.negated,false)=false "
            "RETURN a.id AS a, type(r) AS rel, b.id AS b, "
            "r.source_ids AS fontes LIMIT $k",
            k=args.relacoes,
        ):
            print("{} -[{}]-> {}  fontes={}".format(
                row["a"], row["rel"], row["b"], row["fontes"]))

        print("\n== Relações negadas (candidatas a expect_negated) ==")
        for row in store._run(
            "MATCH (a:Entity)-[r]->(b:Entity) "
            "WHERE r.negated = true "
            "RETURN a.id AS a, type(r) AS rel, b.id AS b LIMIT 20"
        ):
            print("{} -[{}]-> {}".format(row["a"], row["rel"], row["b"]))
    finally:
        store.close()


if __name__ == "__main__":
    main()