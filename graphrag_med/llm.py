"""Geração de resposta ancorada em fatos do grafo e trechos do ChatBulário."""
import os


SYSTEM = (
    "Você responde perguntas sobre medicamentos usando EXCLUSIVAMENTE as evidências fornecidas: "
    "fatos do grafo [F1], [F2]..., caminhos [P1], [P2]... e trechos [T1], [T2]... "

    "Regras obrigatórias: "

    "(1) Cite cada afirmação com a evidência correspondente. "
    "Não faça afirmações clínicas sem uma citação [F#], [P#] ou [T#]. "

    "(2) Ao citar fatos do grafo, utilize o formato "
    "[F#: nome do medicamento]. "
    "Exemplo: [F1: cloridrato de tramadol + paracetamol]. "
    "O nome deve corresponder ao medicamento efetivamente associado ao fato. "
    "Utilize exclusivamente o nome presente nas evidências fornecidas. "
    "Nunca invente, deduza ou substitua o nome do medicamento. "

    "(3) Para caminhos, mantenha o identificador original [P#]. "
    "Para trechos documentais, mantenha o identificador original [T#]. "
    "Não altere os identificadores das evidências. "

    "(4) Se as evidências fornecidas não forem suficientes para responder, "
    "informe explicitamente que não há evidências suficientes no material recuperado. "
    "Não complete lacunas com conhecimento externo. "

    "(5) Nunca acrescente conhecimento externo, informações presumidas "
    "ou extrapolações. Não extrapole informações de uma bula para outro medicamento. "

    "(6) Preserve a classificação das evidências: "
    "fatos afirmados devem ser descritos como afirmados; "
    "fatos hipotéticos devem ser descritos como possibilidades; "
    "fatos negados devem ser apresentados explicitamente como negados. "

    "(7) Nunca transforme uma evidência hipotética em uma afirmação definitiva. "
    "Nunca transforme uma evidência negada em uma afirmação positiva. "

    "(8) A ausência de uma informação nas evidências não significa que "
    "essa informação seja falsa ou que o evento não aconteça. "
    "Diferencie ausência de evidência de evidência de ausência. "

    "(9) Quando houver contradição entre fatos do grafo [F#], caminhos [P#] "
    "e trechos documentais [T#], informe a divergência explicitamente. "
    "Não escolha arbitrariamente uma das versões e não tente resolver "
    "a contradição usando conhecimento externo. "

    "(10) Responda sempre em português brasileiro, com linguagem clara, "
    "objetiva e tecnicamente adequada. "

    "(11) Não forneça diagnóstico, prescrição, indicação de tratamento, "
    "ajuste de dose ou orientação médica individualizada. "

    "(12) As linhas [G#] representam métricas estruturais do grafo, "
    "como PageRank e comunidades. "
    "Elas servem exclusivamente para indicar a relevância estrutural "
    "das entidades na rede. "
    "Nunca utilize métricas [G#] como evidência clínica, farmacológica "
    "ou terapêutica. Não cite [G#] como fundamento de afirmações médicas. "

    "(13) Não invente identificadores, nomes de medicamentos, relações, "
    "citações ou trechos. Toda referência utilizada deve existir "
    "nas evidências fornecidas. "

    "(14) Ao mencionar uma relação entre entidades, descreva corretamente "
    "o sentido da relação. Não inverta sujeito e objeto de uma relação "
    "do grafo. "

    "(15) Organize a resposta de forma legível, evitando repetições. "
    "Quando possível, apresente primeiro a resposta direta e depois "
    "as evidências que a sustentam."
)


def graph_metrics_lines(graph_context):
    """Linhas [G#] com PageRank e comunidades das entidades-semente."""

    if not graph_context:
        return []

    engine = graph_context.get("engine") or "?"
    communities = {
        item["id"]: item
        for item in graph_context.get("communities", [])
    }

    lines = []

    for i, seed in enumerate(graph_context.get("seeds", []), 1):

        community = communities.get(seed.get("community"), {})

        members = ", ".join(
            member["name"]
            for member in community.get("top_members", [])
            if member["id"] != seed["id"]
        )

        lines.append(
            "[G{}] métrica estrutural ({}; não é fato clínico): "
            "{} — PageRank={:.4f} (posto {} de {}), "
            "comunidade {} com {} entidades{}".format(
                i,
                engine,
                seed["name"],
                seed["pagerank"],
                seed["rank"],
                seed["of"],
                seed["community"],
                seed["community_size"],
                (
                    "; entidades centrais da comunidade: " + members
                    if members else ""
                ),
            )
        )

    return lines


def facts_block(ctx):
    lines = []

    for fact in ctx.get("facts", []) + ctx.get("negated_facts", []):
        extra = (
            f" (efeito: {fact['effect']})"
            if fact.get("effect") else ""
        )

        lines.append(
            f"[{fact['id']}] {fact['src_name']} {fact['rel']} "
            f"{fact['dst_name']}{extra} | "
            f"certainty={fact['certainty']} conf={fact['confidence']} "
            f"fontes={','.join(fact['sources'])} | "
            f"evidência: \"{fact['evidence']}\""
        )

    for i, path in enumerate(ctx.get("path_citations", []), 1):
        marker = (
            " [menor caminho — Shortest Path]"
            if path.get("algorithm") == "shortestPath"
            else ""
        )

        lines.append(
            "[P{}] caminho completo: {}{} | fatos: {}".format(
                i,
                path.get("text", ""),
                marker,
                ", ".join(path.get("fact_ids", []))
            )
        )

    lines.extend(graph_metrics_lines(ctx.get("graph_context")))

    for i, (chunk, similarity) in enumerate(
        ctx.get("chunks", []), 1
    ):
        lines.append(
            f"[T{i}] trecho de bula "
            f"(similaridade={similarity:.3f}; "
            f"documento={chunk.get('source_id') or 'não identificado'}): "
            f"{chunk.get('text', '')}"
        )

    return "\n".join(lines)


def ollama_config():
    """Lê a configuração do Ollama exclusivamente das variáveis de ambiente."""
    return {
        "base_url": os.getenv(
            "OLLAMA_BASE_URL",
            "http://127.0.0.1:11434"
        ).rstrip("/"),
        "model": os.getenv(
            "OLLAMA_MODEL",
            "qwen2.5:3b"
        ).strip(),
    }


def llm_available():
    """Retorna True quando o servidor Ollama responde e o modelo está instalado."""
    try:
        import requests

        config = ollama_config()

        response = requests.get(
            f"{config['base_url']}/api/tags",
            timeout=3
        )
        response.raise_for_status()

        models = response.json().get("models", [])

        return any(
            m.get("name") == config["model"]
            or m.get("name", "").startswith(
                config["model"].split(":")[0] + ":"
            )
            for m in models
        )

    except Exception:
        return False


def ollama_status():
    """Retorna (conectado, mensagem) para exibição na interface."""
    try:
        import requests

        config = ollama_config()

        response = requests.get(
            f"{config['base_url']}/api/tags",
            timeout=3
        )
        response.raise_for_status()

        models = [
            m.get("name", "")
            for m in response.json().get("models", [])
        ]

        if config["model"] not in models:
            return (
                False,
                f"Modelo {config['model']} não encontrado. "
                f"Baixe-o com: ollama pull {config['model']}"
            )

        return True, f"Ollama conectado · modelo {config['model']}"

    except Exception as exc:
        return (
            False,
            f"Não foi possível conectar ao Ollama em "
            f"{ollama_config()['base_url']}: {exc}"
        )


def generate_cypher(question, seed_ids, hops=2):
    """
    Gera uma consulta Cypher de leitura de forma determinística.

    A consulta não é criada pela LLM. Isso evita consultas malformadas
    e garante que os IDs, a profundidade e os tipos de relações sejam
    controlados pela aplicação.

    Os parâmetros seed_ids, allowed_rels e limit são fornecidos
    separadamente por retrieval.py ao executar a consulta.
    """
    import re

    del question  # A pergunta não é necessária para montar a consulta.

    if not isinstance(seed_ids, (list, tuple, set)):
        raise ValueError("seed_ids precisa ser uma lista de IDs.")

    seed_ids = [
        str(item).strip()
        for item in seed_ids
        if item is not None and str(item).strip()
    ]

    if not seed_ids:
        raise ValueError("Nenhuma entidade-semente foi fornecida.")

    max_hops = max(1, min(int(hops), 5))

    # Tipos de relações permitidos no grafo.
    allowed_rels = [
        "INTERACTS_WITH",
        "TREATS",
        "CONTRAINDICATED_IN",
        "CAUSES",
        "BELONGS_TO",
        "HAS_SYMPTOM",
        "DIAGNOSED_WITH",
        "ADMINISTERED_BY",
    ]

    # Os nomes de relações são fixos e não vêm da LLM.
    if not all(re.fullmatch(r"[A-Z_]+", rel) for rel in allowed_rels):
        raise ValueError("Tipo de relacionamento inválido na allowlist.")

    query = """
MATCH (start:Entity)
WHERE start.id IN $seed_ids
MATCH p=(start)-[rels*1..{}]-(end:Entity)
WHERE all(r IN rels WHERE type(r) IN $allowed_rels)
RETURN p
LIMIT $limit
""".format(max_hops).strip()

    return query


def answer_with_llm(question, ctx):
    """Gera resposta local com Ollama, limitada às evidências recuperadas."""
    import requests

    config = ollama_config()

    prompt = (
        f"Pergunta: {question}\n\n"
        f"Evidências disponíveis:\n{facts_block(ctx)}\n\n"
        "Responda somente com base nas evidências acima. "
        "Cite cada afirmação. "
        "Se as evidências forem insuficientes, diga explicitamente. "
        "Não invente dados, doses, indicações ou contraindicações. "
        "Responda em português e não dê orientação médica individualizada."
    )
#com [F#], [P#] ou [T#]
    response = requests.post(
        f"{config['base_url']}/api/chat",
        json={
            "model": config["model"],
            "messages": [
                {"role": "system", "content": SYSTEM},
                {"role": "user", "content": prompt}
            ],
            "stream": False,
            "options": {
                "temperature": 0.1,
                "num_predict": 700
            }
        },
        timeout=(10, 240)
    )

    response.raise_for_status()

    content = (
        response.json()
        .get("message", {})
        .get("content", "")
        .strip()
    )

    if not content:
        raise RuntimeError("O Ollama retornou uma resposta vazia.")

    return content


def answer_template(ctx):
    """Resposta determinística: mostra relações e trechos recuperados com suas fontes."""
    if ctx["refused"]:
        return f"Não posso responder com segurança. {ctx['reason']}"

    out = []

    if ctx.get("retrieval_only"):
        out.append(
            "**Trechos de bula recuperados "
            "(sem relação confirmada no grafo):**"
        )

        for i, (chunk, similarity) in enumerate(
            ctx.get("chunks", []), 1
        ):
            out.append(
                f"- [T{i}] (similaridade {similarity:.2f}) "
                f"{chunk['text']}"
            )

        out.append(
            "\nA busca textual localizou estes trechos. Confira a bula "
            "e a identificação do medicamento antes de interpretar a informação."
        )

        return "\n".join(out)

    direct = [
        fact for fact in ctx["facts"]
        if fact["hop"] == 1
    ]

    if len(ctx["seeds"]) == 1 and direct:
        out.append("**Fatos diretos (1 salto):**")

        for fact in direct:
            tag = (
                " _(possível)_"
                if fact["certainty"] == "hypothetical"
                else ""
            )

            out.append(
                f"- {fact['src_name']} **{fact['rel']}** "
                f"{fact['dst_name']} [{fact['id']}]" + tag
            )

    elif ctx["facts"]:
        out.append("**Conexões encontradas no grafo (multi-hop):**")

        for fact in ctx["facts"]:
            out.append(
                f"- {fact['src_name']} **{fact['rel']}** "
                f"{fact['dst_name']} [{fact['id']}]"
            )

    for fact in ctx["negated_facts"]:
        out.append(
            f"- ⚠️ As fontes afirmam que **NÃO** há relação "
            f"{fact['rel']} entre {fact['src_name']} e "
            f"{fact['dst_name']} [{fact['id']}]"
        )

    if ctx.get("chunks"):
        out.append("\n**Trechos de bula relacionados:**")

        for i, (chunk, similarity) in enumerate(
            ctx["chunks"], 1
        ):
            out.append(
                f"- [T{i}] ({similarity:.2f}) {chunk['text']}"
            )

    return "\n".join(out)