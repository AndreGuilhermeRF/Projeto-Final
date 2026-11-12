"""
Experimento RAG × GraphRAG: recall multi-hop, taxa de alucinação e tabelas.

Pré-requisitos: Neo4j com a ingestão concluída e Ollama com o modelo de
geração e o de embeddings.

Uso típico
----------
  # Experimento completo (RAG vetorial × GraphRAG)
  python tools/avaliar_rag_vs_graphrag.py

  # Com as duas linhas de base e juiz LLM (mais lento)
  python tools/avaliar_rag_vs_graphrag.py \
      --condicoes rag_vetorial,rag_hibrido,graphrag --juiz

  # Reavaliar respostas já geradas com outro limiar, sem chamar o LLM
  python tools/avaliar_rag_vs_graphrag.py \
      --reagregar resultados/20260101-120000/resultados.json --tau 0.5

  # Validar o verificador automático com a auditoria humana preenchida
  python tools/avaliar_rag_vs_graphrag.py \
      --auditoria-csv resultados/20260101-120000/auditoria_amostra.csv

Formato das perguntas (graphrag_med/data/experimento.json)
----------------------------------------------------------
  {"q": "...", "hops": 2}                         # só fidelidade
  {"q": "...", "hops": 2,
   "expected": ["DRUG:a", "AE:b"],                # recall de nós/textual
   "expected_edges": [["DRUG:a", "CAUSES", "AE:b"]]}   # recall de arestas
  {"q": "...", "nao_respondivel": true}           # deve haver abstenção
  {"q": "...", "expect_refusal": true}            # idem (pergunta fora de escopo)

Campos opcionais: "id", "multihop" (padrão: hops >= 2). Use
tools/listar_entidades.py para obter IDs reais do seu grafo.

Condições
---------
  rag_vetorial : só busca vetorial no Neo4j (top-k trechos) + LLM.
  rag_hibrido  : vetor + Fulltext com RRF (top-k trechos) + LLM.
  graphrag     : recuperação por grafo (fatos, caminhos, algoritmos) + os
                 mesmos top-k trechos + LLM, com o prompt do sistema.

Todas usam o mesmo LLM, o mesmo top-k e o mesmo formato de evidência
([F#], [P#], [T#]). A diferença medida é o efeito do grafo.
"""

import argparse
import json
import logging
import os
import sys
import time

from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from dotenv import load_dotenv  # noqa: E402

load_dotenv(ROOT / ".env", override=False)

from graphrag_med import experiment as ex  # noqa: E402

logger = logging.getLogger("experimento")

ALL_CONDITIONS = ("rag_vetorial", "rag_hibrido", "graphrag")
EMPTY_ANSWER = "Evidências insuficientes: nenhum trecho foi recuperado."


# ============================================================
# SISTEMA
# ============================================================

def load_chunks(store):
    """Chunks já existentes no Neo4j, com as entidades mencionadas."""

    records = store._run(
        "MATCH (c:Chunk) "
        "OPTIONAL MATCH (c)-[:MENTIONS]->(e:Entity) "
        "RETURN c.id AS id, c.text AS text, c.source_id AS source_id, "
        "collect(DISTINCT e.id) AS entities ORDER BY c.id"
    )

    return [
        {
            "id": record["id"],
            "text": record["text"],
            "source_id": record.get("source_id"),
            "entities": [e for e in (record.get("entities") or []) if e],
        }
        for record in records
        if record.get("id") and record.get("text") is not None
    ]


def build_system():
    from graphrag_med.embeddings import OllamaEmbedder
    from graphrag_med.graph_store import Neo4jStore
    from graphrag_med.lexicon import Lexicon
    from graphrag_med.retrieval import ChunkIndex, System

    store = Neo4jStore(
        uri=os.getenv("NEO4J_URI", "bolt://127.0.0.1:7687"),
        user=os.getenv("NEO4J_USER", "neo4j"),
        password=os.getenv("NEO4J_PASSWORD", ""),
        database=os.getenv("NEO4J_DATABASE", "neo4j"),
        schema_path=str(ROOT / "graphrag_med" / "schema.cypher"),
    )

    chunks = load_chunks(store)

    if not chunks:
        raise SystemExit(
            "Nenhum Chunk no Neo4j. Conclua a ingestão antes do experimento."
        )

    index = ChunkIndex(chunks, OllamaEmbedder(), store)

    return System(
        store=store,
        index=index,
        lex=Lexicon.for_chatbulario(),
        triples=[],
    )


# ============================================================
# EXECUTORES DAS CONDIÇÕES
# ============================================================

def make_runners(system, args):
    from graphrag_med.llm import (
        answer_template,
        answer_with_llm,
        facts_block,
    )
    from graphrag_med.retrieval import retrieve

    by_id = system.index.by_id

    def enrich(chunk):
        chunk = dict(chunk)
        chunk["entities"] = (
            by_id.get(chunk.get("id"), {}).get("entities")
            or chunk.get("entities")
            or []
        )
        return chunk

    def generate(question, ctx):
        """Retorna (resposta, erro)."""

        try:
            return answer_with_llm(question, ctx), None

        except Exception as exc:
            logger.exception("Falha ao gerar resposta.")
            return None, "LLM: {}".format(exc)

    def chunk_entities(hits):
        return {
            entity
            for chunk, _score in hits
            for entity in chunk.get("entities", [])
        }

    def rag_vetorial(item):
        start = time.time()

        try:
            vector = system.index.embedder.embed(item["q"])

            hits = [
                (enrich({
                    "id": r["id"],
                    "text": r["text"],
                    "source_id": r.get("source_id"),
                }), r["score"])
                for r in system.store.vector_search(vector, k=args.top_k)
            ]

        except Exception as exc:
            return {"error": "Recuperação: {}".format(exc)}

        return finish_rag(item, hits, start)

    def rag_hibrido(item):
        start = time.time()

        try:
            hits = [
                (enrich(chunk), score)
                for chunk, score in system.index.search(
                    item["q"], args.top_k
                )
            ]

        except Exception as exc:
            return {"error": "Recuperação: {}".format(exc)}

        return finish_rag(item, hits, start)

    def finish_rag(item, hits, start):
        ctx = {"chunks": hits}

        if hits:
            answer, error = generate(item["q"], ctx)
        else:
            answer, error = EMPTY_ANSWER, None

        return {
            "answer": answer,
            "error": error,
            "items": ex.parse_evidence(facts_block(ctx)),
            "found_ids": chunk_entities(hits),
            "facts": None,
            "latency": time.time() - start,
        }

    def graphrag(item):
        start = time.time()

        try:
            ctx = retrieve(
                system,
                item["q"],
                hops=item.get("hops", args.hops),
                top_k=args.top_k,
            )

        except Exception as exc:
            return {"error": "Recuperação: {}".format(exc)}

        if ctx.get("refused"):
            answer, error = answer_template(ctx), None
        else:
            answer, error = generate(item["q"], ctx)

        found = {
            node
            for path in ctx.get("paths", [])
            for node in path.get("nodes", [])
        }

        for fact in ctx.get("facts", []):
            found.update([fact.get("src"), fact.get("dst")])

        found.update(chunk_entities(
            [(enrich(c), s) for c, s in ctx.get("chunks", [])]
        ))

        found.discard(None)

        return {
            "answer": answer,
            "error": error,
            "items": ex.parse_evidence(facts_block(ctx)),
            "found_ids": found,
            "facts": ctx.get("facts", []),
            "latency": time.time() - start,
        }

    return {
        "rag_vetorial": rag_vetorial,
        "rag_hibrido": rag_hibrido,
        "graphrag": graphrag,
    }


def make_name_lookup(system):

    def lookup(entity_id):
        node = system.store.node(entity_id)

        if not node:
            return None

        return node.get("name") or entity_id, node.get("synonyms") or []

    return lookup


# ============================================================
# SAÍDAS
# ============================================================

def json_default(value):
    if isinstance(value, set):
        return sorted(value)

    return str(value)


def write_outputs(folder, results, conditions, args, config):
    folder.mkdir(parents=True, exist_ok=True)

    summary = ex.aggregate(results, conditions, args.boot, args.seed)

    differences = {}

    if "graphrag" in conditions:
        for baseline in conditions:
            if baseline == "graphrag":
                continue

            differences["GraphRAG − {}".format(ex.label(baseline))] = (
                ex.paired_differences(
                    results, "graphrag", baseline, args.boot, args.seed
                )
            )

    tables = ex.build_tables(
        summary,
        conditions,
        differences,
        judge_used=bool(args.juiz),
    )

    markdown = ex.to_markdown(tables)

    with open(folder / "resultados.json", "w", encoding="utf-8") as handle:
        json.dump(
            {
                "config": config,
                "conditions": conditions,
                "results": results,
            },
            handle,
            ensure_ascii=False,
            indent=2,
            default=json_default,
        )

    with open(folder / "resumo.json", "w", encoding="utf-8") as handle:
        json.dump(
            {"summary": summary, "differences": differences},
            handle,
            ensure_ascii=False,
            indent=2,
            default=json_default,
        )

    (folder / "tabelas.md").write_text(markdown, encoding="utf-8")
    (folder / "tabelas.tex").write_text(
        ex.to_latex(tables), encoding="utf-8"
    )

    ex.write_csv(
        folder / "por_pergunta.csv",
        ex.per_question_rows(results, conditions),
    )

    sample = ex.audit_sample(
        results, conditions, args.auditoria_amostra, args.seed
    )

    ex.write_csv(folder / "auditoria_amostra.csv", sample)

    return markdown, summary, len(sample)


def load_questions(path, limit):
    with open(path, encoding="utf-8") as handle:
        questions = json.load(handle)

    if not isinstance(questions, list) or not questions:
        raise SystemExit("O arquivo de perguntas deve ser uma lista não vazia.")

    for item in questions:
        if "q" not in item:
            raise SystemExit("Item sem campo 'q': {}".format(item))

    return questions[:limit] if limit else questions


def make_judge(args):
    if not args.juiz:
        return None

    from graphrag_med.llm import ollama_config

    config = ollama_config()
    model = args.juiz_modelo or config["model"]

    def judge(claim, evidence):
        return ex.judge_claim(
            claim, evidence, base_url=config["base_url"], model=model
        )

    return judge


# ============================================================
# PRINCIPAL
# ============================================================

def main():
    parser = argparse.ArgumentParser(
        description="Experimento RAG × GraphRAG (fidelidade e recall)."
    )
    parser.add_argument(
        "--perguntas",
        default=str(ROOT / "graphrag_med" / "data" / "experimento.json"),
    )
    parser.add_argument("--saida", default=str(ROOT / "resultados"))
    parser.add_argument(
        "--condicoes",
        default="rag_vetorial,graphrag",
        help="Separadas por vírgula: " + ", ".join(ALL_CONDITIONS),
    )
    parser.add_argument("--top-k", type=int, default=3)
    parser.add_argument("--hops", type=int, default=2)
    parser.add_argument("--tau", type=float, default=0.6,
                        help="Cobertura mínima de termos (0-1).")
    parser.add_argument("--juiz", action="store_true",
                        help="Usa um LLM-juiz além do verificador léxico.")
    parser.add_argument("--juiz-modelo", default=None)
    parser.add_argument("--boot", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--limite", type=int, default=0,
                        help="Usa só as N primeiras perguntas (teste rápido).")
    parser.add_argument("--auditoria-amostra", type=int, default=60)
    parser.add_argument("--reagregar", default=None)
    parser.add_argument("--auditoria-csv", default=None)
    args = parser.parse_args()

    logging.basicConfig(level=logging.WARNING)

    # ---- Modo: validar auditoria humana ----
    if args.auditoria_csv:
        print(json.dumps(
            ex.evaluate_audit(args.auditoria_csv),
            ensure_ascii=False, indent=2,
        ))
        return

    judge = make_judge(args)

    # ---- Modo: reavaliar respostas já geradas ----
    if args.reagregar:
        with open(args.reagregar, encoding="utf-8") as handle:
            saved = json.load(handle)

        results = ex.regrade(saved["results"], tau=args.tau, judge=judge)
        conditions = saved["conditions"]
        config = dict(saved.get("config", {}), tau=args.tau, reagregado=True)

        folder = Path(args.saida) / (
            datetime.now().strftime("%Y%m%d-%H%M%S") + "-reagregado"
        )

    # ---- Modo: experimento completo ----
    else:
        conditions = [c.strip() for c in args.condicoes.split(",") if c.strip()]

        unknown = [c for c in conditions if c not in ALL_CONDITIONS]

        if unknown:
            raise SystemExit("Condições inválidas: {}".format(unknown))

        from graphrag_med.llm import llm_available, ollama_config

        if not llm_available():
            raise SystemExit(
                "Ollama indisponível ou modelo não instalado "
                "(OLLAMA_BASE_URL/OLLAMA_MODEL)."
            )

        questions = load_questions(args.perguntas, args.limite)

        system = build_system()

        try:
            results = ex.run_experiment(
                questions,
                conditions,
                make_runners(system, args),
                name_lookup=make_name_lookup(system),
                tau=args.tau,
                judge=judge,
                progress=lambda i, n, c, q: print(
                    "[{}/{}] {:<13} {}".format(i, n, c, q[:70]),
                    flush=True,
                ),
            )

        finally:
            system.store.close()

        config = {
            "data": datetime.now().isoformat(timespec="seconds"),
            "perguntas": str(args.perguntas),
            "n_perguntas": len(questions),
            "top_k": args.top_k,
            "hops_padrao": args.hops,
            "tau": args.tau,
            "juiz": bool(args.juiz),
            "juiz_modelo": args.juiz_modelo,
            "seed": args.seed,
            "bootstrap": args.boot,
            "ollama": ollama_config(),
            "embed_model": os.getenv("OLLAMA_EMBED_MODEL", "nomic-embed-text"),
            "chatbulario_limit": os.getenv("CHATBULARIO_LIMIT"),
            "enable_model_ner": os.getenv("ENABLE_MODEL_NER"),
            "enable_llm_extraction": os.getenv("ENABLE_LLM_EXTRACTION"),
            "enable_llm_cypher": os.getenv("ENABLE_LLM_CYPHER"),
            "graph_analytics_engine": os.getenv(
                "GRAPH_ANALYTICS_ENGINE", "auto"
            ),
        }

        folder = Path(args.saida) / datetime.now().strftime("%Y%m%d-%H%M%S")

    markdown, summary, audit_n = write_outputs(
        folder, results, conditions, args, config
    )

    print("\n" + markdown)

    errors = {c: summary[c]["n_errors"] for c in conditions}

    if any(errors.values()):
        print("ATENÇÃO: perguntas com erro por condição:", errors)

    print("Arquivos em: {}".format(folder))
    print(
        "Auditoria humana: preencha 'humano_suportada' (1/0) em "
        "auditoria_amostra.csv ({} afirmações) e rode com "
        "--auditoria-csv.".format(audit_n)
    )


if __name__ == "__main__":
    main()