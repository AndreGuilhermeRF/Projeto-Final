"""Interface Streamlit do GraphRAG Med usando exclusivamente Neo4j."""

import os
import json
import logging
from pathlib import Path

import pandas as pd
import streamlit as st
import streamlit.components.v1 as components

from dotenv import load_dotenv


# ============================================================
# CONFIGURAÇÃO
# ============================================================

ROOT = Path(__file__).resolve().parent

load_dotenv(ROOT / ".env", override=False)

st.set_page_config(
    page_title="GraphRAG Med — ChatBulário",
    page_icon="💊",
    layout="wide",
)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

DATA_LIMIT = int(os.getenv("CHATBULARIO_LIMIT", "100"))


# ============================================================
# IMPORTAÇÕES DO PROJETO
# ============================================================

from graphrag_med import evaluation

from graphrag_med.chatbulario import (
    DATASET_ID,
    DATASET_URL,
    load_chatbulario_docs,
)

from graphrag_med.graph_store import Neo4jStore
from graphrag_med.lexicon import Lexicon

from graphrag_med.llm import (
    answer_template,
    answer_with_llm,
    ollama_status,
)

from graphrag_med.retrieval import (
    ChunkIndex,
    System,
    retrieve,
)

from graphrag_med.embeddings import OllamaEmbedder

from graphrag_med.viz import graph_html


# ============================================================
# DOCUMENTOS DO CHATBULÁRIO
# ============================================================

@st.cache_data(show_spinner="Carregando o ChatBulário...")
def get_docs(limit):
    return load_chatbulario_docs(
        split="train",
        limit=limit,
        min_fuzzy_score=80.0,
    )


try:
    DOCS = get_docs(DATA_LIMIT)

except Exception:
    DOCS = []

    logger.exception(
        "Não foi possível carregar os documentos do ChatBulário."
    )


# ============================================================
# EXEMPLOS
# ============================================================

EXAMPLES = [
    "Quais são as relações entre cloridrato de tramadol e paracetamol no grafo de conhecimento?",
    "Quais medicamentos estão associados ao tratamento da dor?",
    "Quais efeitos adversos estão relacionados ao tramadol?",
    "Quais relações existem entre paracetamol, tramadol e seus efeitos adversos?",
    "Quais medicamentos do grafo pertencem à mesma classe farmacológica que o tramadol?",
]


# ============================================================
# CARREGAMENTO DOS CHUNKS EXISTENTES
# ============================================================

def load_existing_chunks(store):
    """
    Carrega os chunks já existentes no Neo4j.

    Não executa ingestão.
    Não gera embeddings.
    Não modifica os dados.
    """

    query = """
    MATCH (c:Chunk)
    OPTIONAL MATCH (c)-[:MENTIONS]->(e:Entity)
    RETURN
        c.id AS id,
        c.text AS text,
        c.source_id AS source_id,
        collect(DISTINCT e.id) AS entities
    ORDER BY c.id
    """

    records = store._run(query)

    chunks = []

    for record in records:

        chunk_id = record.get("id")
        chunk_text = record.get("text")

        if not chunk_id or chunk_text is None:
            continue

        chunks.append({
            "id": chunk_id,
            "text": chunk_text,
            "source_id": record.get("source_id"),
            "entities": [
                entity
                for entity in (record.get("entities") or [])
                if entity
            ],
        })

    return chunks


# ============================================================
# CONSTRUÇÃO DO SISTEMA
# ============================================================

def build_system(store):
    """
    Inicializa o sistema usando os dados já existentes.

    Não chama pipeline.build().
    Não executa extração NER/RE.
    Não recalcula embeddings.
    """

    lex = Lexicon.for_chatbulario()

    embedder = OllamaEmbedder()

    chunks = load_existing_chunks(store)

    if not chunks:
        raise RuntimeError(
            "Nenhum Chunk foi encontrado no Neo4j. "
            "Verifique se a ingestão inicial foi concluída."
        )

    index = ChunkIndex(
        chunks,
        embedder,
        store,
    )

    return System(
        store=store,
        index=index,
        lex=lex,
        triples=[],
    )


# ============================================================
# CONEXÃO NEO4J
# ============================================================

def neo4j_system(uri, user, password, database):

    store = Neo4jStore(
        uri=uri,
        user=user,
        password=password,
        database=database,
        schema_path=str(
            ROOT / "graphrag_med" / "schema.cypher"
        ),
    )

    try:
        return build_system(store)

    except Exception:
        store.close()
        raise


# ============================================================
# CONSTRUÇÃO DAS ARESTAS DO SUBGRAFO
# ============================================================

def normalize_context_edge(edge, negated=False):
    """
    Normaliza uma relação recuperada pelo sistema.

    Aceita nomes alternativos para origem, destino
    e tipo de relação.
    """

    if not isinstance(edge, dict):
        return None

    src = (
        edge.get("src")
        or edge.get("src_id")
        or edge.get("source_id")
        or edge.get("source")
        or edge.get("subject_id")
    )

    dst = (
        edge.get("dst")
        or edge.get("dst_id")
        or edge.get("target_id")
        or edge.get("target")
        or edge.get("object_id")
    )

    rel = (
        edge.get("rel")
        or edge.get("relation")
        or edge.get("type")
    )

    if src is None or dst is None or not rel:
        return None

    normalized = dict(edge)

    normalized["src"] = src
    normalized["dst"] = dst
    normalized["rel"] = str(rel)

    if negated:
        normalized["negated"] = True

    evidence = normalized.get("evidence")

    if isinstance(evidence, str):
        normalized["evidence"] = (
            [{"text": evidence}]
            if evidence
            else []
        )

    elif evidence is None:
        normalized["evidence"] = []

    return normalized


def build_context_edges(ctx):
    """
    Monta o subgrafo usando:

    1. Fatos positivos.
    2. Fatos negados.
    3. Arestas dos caminhos multi-hop.

    Evita duplicar relações com a mesma origem,
    tipo e destino.
    """

    edge_map = {}

    # --------------------------------------------------------
    # FATOS POSITIVOS
    # --------------------------------------------------------

    for fact in ctx.get("facts", []):

        edge = normalize_context_edge(fact)

        if edge is None:
            continue

        key = (
            edge["src"],
            edge["rel"],
            edge["dst"],
        )

        edge_map[key] = edge

    # --------------------------------------------------------
    # FATOS NEGADOS
    # --------------------------------------------------------

    for fact in ctx.get("negated_facts", []):

        edge = normalize_context_edge(
            fact,
            negated=True,
        )

        if edge is None:
            continue

        key = (
            edge["src"],
            edge["rel"],
            edge["dst"],
        )

        edge_map[key] = edge

    # --------------------------------------------------------
    # CAMINHOS MULTI-HOP
    # --------------------------------------------------------

    for path in ctx.get("paths", []):

        if not isinstance(path, dict):
            continue

        for path_edge in path.get("edges", []):

            edge = normalize_context_edge(path_edge)

            if edge is None:
                continue

            key = (
                edge["src"],
                edge["rel"],
                edge["dst"],
            )

            if key not in edge_map:
                edge_map[key] = edge

    return list(edge_map.values())


# ============================================================
# DIAGNÓSTICO DA RECUPERAÇÃO
# ============================================================

def show_retrieval_diagnostics(ctx):
    """
    Exibe os dados retornados por retrieve()
    para identificar incompatibilidades nos campos.
    """

    with st.expander(
        "Diagnóstico dos fatos e caminhos",
        expanded=True,
    ):

        facts = ctx.get("facts", []) or []
        negated_facts = ctx.get("negated_facts", []) or []
        paths = ctx.get("paths", []) or []

        st.write("**Quantidade de fatos positivos:**")
        st.write(len(facts))

        st.write("**Quantidade de fatos negados:**")
        st.write(len(negated_facts))

        st.write("**Quantidade de caminhos:**")
        st.write(len(paths))

        st.divider()

        st.write("**Exemplo de fato positivo:**")

        if facts:
            st.json(facts[0])
        else:
            st.warning(
                "Nenhum fato positivo recuperado."
            )

        st.divider()

        st.write("**Exemplo de fato negado:**")

        if negated_facts:
            st.json(negated_facts[0])
        else:
            st.info(
                "Nenhum fato negado recuperado."
            )

        st.divider()

        st.write("**Exemplo de caminho:**")

        if paths:
            st.json(paths[0])
        else:
            st.warning(
                "Nenhum caminho multi-hop recuperado."
            )

        st.divider()

        st.write("**Arestas normalizadas:**")

        try:

            diagnostic_edges = build_context_edges(ctx)

            st.write(
                "Quantidade de arestas:",
                len(diagnostic_edges),
            )

            if diagnostic_edges:
                st.json(diagnostic_edges)
            else:
                st.warning(
                    "Nenhuma aresta foi produzida pela função "
                    "build_context_edges()."
                )

        except Exception as diagnostic_error:

            st.error(
                "Erro ao normalizar as arestas: {}".format(
                    diagnostic_error
                )
            )

        st.divider()

        st.write("**Entidades-semente:**")
        st.write(ctx.get("seeds", []))

        st.write("**Erros da recuperação:**")
        st.write(ctx.get("retrieval_errors", []))


# ============================================================
# CALLBACKS
# ============================================================

def set_question(value):
    st.session_state.q = value


# ============================================================
# BARRA LATERAL
# ============================================================

st.sidebar.header("Configuração")

st.sidebar.caption(
    "Banco de grafos: Neo4j "
    "(configurado exclusivamente pelo .env)"
)

hops = st.sidebar.slider(
    "Máx. de saltos (multi-hop)",
    1,
    3,
    2,
)

top_k = st.sidebar.slider(
    "Top-k chunks (Vector + Fulltext)",
    1,
    8,
    3,
)

ollama_ok, ollama_message = ollama_status()

st.sidebar.caption(ollama_message)

use_llm = st.sidebar.checkbox(
    "Gerar resposta com LLM (Ollama local)",
    value=ollama_ok,
    disabled=not ollama_ok,
    help=(
        "Requer o Ollama em execução e o modelo "
        "definido no .env."
    ),
)


# ============================================================
# CONFIGURAÇÕES DO NEO4J
# ============================================================

uri = os.getenv(
    "NEO4J_URI",
    "",
).strip()

user = os.getenv(
    "NEO4J_USER",
    "",
).strip()

password = os.getenv(
    "NEO4J_PASSWORD",
    "",
)

database = os.getenv(
    "NEO4J_DATABASE",
    "neo4j",
).strip() or "neo4j"


if not uri or not user or not password:

    st.error(
        "Configuração Neo4j incompleta. Confira "
        "NEO4J_URI, NEO4J_USER e NEO4J_PASSWORD no .env."
    )

    st.stop()


# ============================================================
# INICIALIZAÇÃO
# ============================================================

status = st.status(
    "Conectando ao Neo4j...",
    expanded=True,
)

progress = st.progress(
    0,
    text="Inicializando conexão...",
)


try:

    if "graphrag_med_system" not in st.session_state:

        system = neo4j_system(
            uri,
            user,
            password,
            database,
        )

        st.session_state["graphrag_med_system"] = system

    system = st.session_state["graphrag_med_system"]

    progress.progress(
        1.0,
        text="Chunks carregados do Neo4j.",
    )

    status.update(
        label="Neo4j conectado. Sistema pronto.",
        state="complete",
        expanded=False,
    )

except Exception as exc:

    status.update(
        label="Falha ao preparar o GraphRAG Med.",
        state="error",
        expanded=True,
    )

    st.error(
        "Erro na inicialização: {}".format(exc)
    )

    st.exception(exc)

    st.stop()


store = system.store


# ============================================================
# CABEÇALHO
# ============================================================

st.title(
    "🕸️ GraphRAG Med — Fármacos, Interações e Doenças"
)

st.warning(
    "Protótipo acadêmico baseado no ChatBulário. "
    "Confira sempre a bula vigente; as informações "
    "não substituem orientação de profissionais de saúde."
)

st.caption(
    "Fonte: {} · {} chunks carregados do Neo4j · "
    "Dataset: {} registros disponíveis · "
    "limite configurado: {:,}.".format(
        DATASET_ID,
        len(system.index.chunks),
        len(DOCS),
        DATA_LIMIT,
    )
)

st.caption(
    "Os dados são consultados diretamente no Neo4j. "
    "A inicialização da interface não executa nova ingestão."
)

if not DOCS:

    st.warning(
        "Não foi possível carregar o conjunto de documentos "
        "original do ChatBulário. A recuperação pelo Neo4j "
        "continua disponível."
    )


# ============================================================
# ABAS
# ============================================================

tab_ask, tab_graph, tab_pipe, tab_eval = st.tabs([
    "💬 Perguntar",
    "🌐 Grafo completo",
    "⚙️ Pipeline de extração",
    "📊 Avaliação",
])


# ============================================================
# ABA: PERGUNTAR
# ============================================================

with tab_ask:

    st.caption(
        "Faça perguntas em português sobre as informações "
        "presentes nos documentos recuperados."
    )

    cols = st.columns(len(EXAMPLES))

    for col, example in zip(cols, EXAMPLES):

        col.button(
            example,
            on_click=set_question,
            args=(example,),
            use_container_width=True,
        )

    if "q" not in st.session_state:
        st.session_state.q = EXAMPLES[0]

    question = st.text_input(
        "Pergunta",
        key="q",
    )

    if question.strip():

        try:

            ctx = retrieve(
                system,
                question,
                hops=hops,
                top_k=top_k,
            )

        except Exception as exc:

            logger.exception(
                "Erro durante a recuperação."
            )

            st.error(
                "Falha durante a recuperação: {}".format(exc)
            )

            st.exception(exc)

            st.stop()

        # ----------------------------------------------------
        # DIAGNÓSTICO TEMPORÁRIO
        # ----------------------------------------------------

        show_retrieval_diagnostics(ctx)

        # ----------------------------------------------------
        # RESPOSTA DA LLM
        # ----------------------------------------------------

        if use_llm and not ctx.get("refused", False):

            try:

                answer = answer_with_llm(
                    question,
                    ctx,
                )

            except Exception as exc:

                logger.exception(
                    "Falha ao gerar resposta com Ollama."
                )

                answer = (
                    answer_template(ctx)
                    + "\n\n_(LLM indisponível: {})_".format(exc)
                )

        else:

            answer = answer_template(ctx)

        # ----------------------------------------------------
        # APRESENTAÇÃO DA RESPOSTA
        # ----------------------------------------------------

        left, right = st.columns([3, 1])

        with left:

            st.subheader("Resposta")

            if ctx.get("refused", False):
                st.error(answer)
            else:
                st.success(answer)

        with right:

            st.metric(
                "Suporte das evidências",
                "{:.0%}".format(
                    ctx.get("confidence", 0)
                ),
            )

            st.caption(
                "Pontuação heurística; "
                "não é probabilidade clínica validada."
            )

        # ----------------------------------------------------
        # DIAGNÓSTICOS DA RECUPERAÇÃO
        # ----------------------------------------------------

        retrieval_errors = ctx.get(
            "retrieval_errors",
            [],
        )

        if retrieval_errors:

            with st.expander(
                "Avisos e erros da recuperação",
                expanded=False,
            ):

                for error in retrieval_errors:
                    st.warning(error)

        # ----------------------------------------------------
        # DETALHES
        # ----------------------------------------------------

        if not ctx.get("refused", False):

            st.subheader("Subgrafo de contexto")

            context_edges = build_context_edges(ctx)

            if context_edges:

                components.html(
                    graph_html(
                        store,
                        context_edges,
                        highlight=ctx.get("seeds", []),
                        include_analytics=True,
                    ),
                    height=540,
                    scrolling=False,
                )

            else:

                st.info(
                    "Nenhuma relação entre entidades foi "
                    "recuperada para esta pergunta. "
                    "Verifique os fatos, caminhos e avisos "
                    "da recuperação acima."
                )

            c1, c2 = st.columns(2)

            # ------------------------------------------------
            # PROCEDÊNCIA
            # ------------------------------------------------

            with c1:

                st.subheader("Procedência")

                facts = (
                    ctx.get("facts", [])
                    + ctx.get("negated_facts", [])
                )

                rows = []

                for fact in facts:

                    rows.append({
                        "id": fact.get("id"),
                        "salto": fact.get("hop"),
                        "fato": "{} {} {}".format(
                            fact.get("src_name", fact.get("src", "")),
                            fact.get("rel", fact.get("relation", "")),
                            fact.get("dst_name", fact.get("dst", "")),
                        ),
                        "certeza": fact.get("certainty"),
                        "conf.": fact.get("confidence"),
                        "fontes": ", ".join(
                            str(source)
                            for source in fact.get("sources", [])
                        ),
                        "evidência": fact.get("evidence"),
                    })

                if rows:

                    st.dataframe(
                        pd.DataFrame(rows),
                        use_container_width=True,
                        hide_index=True,
                    )

                else:

                    st.info(
                        "Nenhum fato estruturado foi recuperado."
                    )

            # ------------------------------------------------
            # CAMINHOS
            # ------------------------------------------------

            with c2:

                st.subheader("Caminhos multi-hop")

                path_citations = ctx.get(
                    "path_citations",
                    [],
                )

                if path_citations:

                    for path in path_citations[:10]:

                        st.markdown(
                            "**[{}]** {}{}".format(
                                path.get("id", ""),
                                path.get("text", ""),
                                (
                                    "  \n🔹 _menor caminho "
                                    "(Shortest Path)_"
                                    if path.get("algorithm")
                                    == "shortestPath"
                                    else ""
                                ),
                            )
                        )

                        st.caption(
                            "Fatos associados: {}".format(
                                ", ".join(
                                    str(fact_id)
                                    for fact_id in path.get(
                                        "fact_ids",
                                        [],
                                    )
                                )
                            )
                        )

                else:

                    st.info(
                        "Nenhum caminho multi-hop foi encontrado. "
                        "Os fatos diretos ainda podem ser "
                        "visualizados no subgrafo."
                    )

            # ------------------------------------------------
            # DETALHES DA RECUPERAÇÃO
            # ------------------------------------------------

            with st.expander(
                "Cypher e detalhes da recuperação"
            ):

                st.code(
                    ctx.get("cypher", ""),
                    language="cypher",
                )

                st.write({
                    "entidades-semente": ctx.get("seeds", []),
                    "origem": ctx.get("seed_source"),
                    "modo de recuperação": ctx.get(
                        "retrieval_mode"
                    ),
                    "filtro": ctx.get("hint"),
                    "erros": retrieval_errors,
                    "arestas do subgrafo": len(context_edges),
                })

                graph_context = ctx.get("graph_context")

                if graph_context:

                    st.markdown(
                        "**Métricas de grafo enviadas ao LLM "
                        "([G#], motor: {})**".format(
                            graph_context.get("engine")
                        )
                    )

                    if graph_context.get("seeds"):
                        st.dataframe(
                            pd.DataFrame(graph_context["seeds"]),
                            use_container_width=True,
                            hide_index=True,
                        )

                    for community in graph_context.get("communities", []):
                        st.caption(
                            "Comunidade {} ({} entidades): {}".format(
                                community["id"],
                                community["size"],
                                ", ".join(
                                    member["name"]
                                    for member in community["top_members"]
                                ),
                            )
                        )

                for chunk, similarity in ctx.get("chunks", []):

                    st.write(
                        "- `{}` (pontuação RRF={:.2f}): {}".format(
                            chunk["id"],
                            similarity,
                            chunk["text"],
                        )
                    )

        else:

            with st.expander("Detalhes"):

                st.write(
                    ctx.get("reason", "")
                )

                st.write({
                    "entidades-semente": ctx.get("seeds", []),
                    "origem": ctx.get("seed_source"),
                    "modo de recuperação": ctx.get(
                        "retrieval_mode"
                    ),
                    "erros": retrieval_errors,
                })



# ============================================================
# ABA: GRAFO COMPLETO
# ============================================================

with tab_graph:

    st.subheader("Visualização completa do grafo Neo4j")

    # --------------------------------------------------------
    # 1. ESTATÍSTICAS
    # --------------------------------------------------------

    try:
        stats = store.stats()

        a, b, c = st.columns(3)

        a.metric("Nós", stats.get("nodes", 0))
        b.metric("Arestas", stats.get("edges", 0))
        c.metric(
            "Tipos de relação",
            len(stats.get("by_rel", {})),
        )

        col1, col2 = st.columns(2)

        with col1:
            st.markdown("**Entidades por tipo**")
            st.json(stats.get("by_type", {}))

        with col2:
            st.markdown("**Relacionamentos por tipo**")
            st.json(stats.get("by_rel", {}))

    except Exception as exc:
        logger.exception("Falha ao carregar estatísticas.")
        st.error("Erro ao consultar estatísticas: {}".format(exc))

    st.divider()

    # --------------------------------------------------------
    # 2. ANÁLISES: PAGERANK, LOUVAIN E LEIDEN
    # --------------------------------------------------------

    with st.expander("PageRank, Louvain e Leiden", expanded=True):

        try:
            refresh = st.button(
                "Recalcular análises",
                key="refresh_graph_analytics",
            )

            analytics = store.analytics(refresh=refresh) or {}

            if analytics.get("available"):

                st.success(
                    analytics.get(
                        "message",
                        "Análises calculadas.",
                    )
                )

                for note in analytics.get("notes", []):
                    st.caption(note)

                pagerank = analytics.get("pagerank") or {}
                louvain = analytics.get("louvain") or {}
                leiden = analytics.get("leiden") or {}
                modularity_values = analytics.get("modularity") or {}
                community_counts = analytics.get("communities") or {}

                m1, m2, m3, m4 = st.columns(4)

                m1.metric("Motor", analytics.get("engine", "-"))
                m2.metric(
                    "Comunidades (Louvain)",
                    community_counts.get("louvain", 0),
                )
                m3.metric(
                    "Comunidades (Leiden)",
                    community_counts.get("leiden", 0)
                    if leiden else "n/d",
                )
                m4.metric(
                    "Modularidade (Louvain)",
                    modularity_values.get("louvain", 0),
                )

                try:
                    names = {
                        str(node.get("id")): node.get("name")
                        for node in (store.all_nodes() or [])
                        if isinstance(node, dict)
                    }
                except Exception:
                    names = {}

                df_analytics = pd.DataFrame([
                    {
                        "entidade": names.get(node_id) or node_id,
                        "id": node_id,
                        "PageRank": round(score, 5),
                        "Louvain": louvain.get(node_id),
                        "Leiden": leiden.get(node_id),
                    }
                    for node_id, score in sorted(
                        pagerank.items(),
                        key=lambda item: -item[1],
                    )
                ])

                if not df_analytics.empty:

                    st.markdown("**Entidades mais centrais (PageRank)**")

                    st.dataframe(
                        df_analytics.head(30),
                        use_container_width=True,
                        hide_index=True,
                    )

                    community_column = (
                        "Leiden" if leiden else "Louvain"
                    )

                    summary = (
                        df_analytics
                        .groupby(community_column)
                        .agg(
                            entidades=("id", "count"),
                            pagerank_total=("PageRank", "sum"),
                            principal=("entidade", "first"),
                        )
                        .reset_index()
                        .sort_values("entidades", ascending=False)
                    )

                    st.markdown(
                        "**Comunidades ({})**".format(community_column)
                    )

                    st.dataframe(
                        summary.head(20),
                        use_container_width=True,
                        hide_index=True,
                    )

                else:
                    st.info("Nenhum resultado de PageRank disponível.")

            else:
                st.warning(
                    analytics.get(
                        "message",
                        "Análises de grafo indisponíveis.",
                    )
                )

        except Exception as exc:
            logger.exception("Falha nas análises do grafo.")
            st.warning(
                "Não foi possível executar as análises: {}".format(exc)
            )

    st.divider()

    # --------------------------------------------------------
    # 3. VISUALIZAÇÃO INTERATIVA
    # --------------------------------------------------------

    st.markdown("### Grafo de entidades e relacionamentos")

    try:
        all_edges = store.all_edges() or []

        st.caption(
            "Relacionamentos recuperados do Neo4j: {}".format(
                len(all_edges)
            )
        )

        if all_edges:

            graph = graph_html(
                store=store,
                edges=all_edges,
                height=700,
                include_analytics=True,
            )

            if not graph:
                st.error("A função graph_html não retornou conteúdo.")
            else:
                components.html(
                    graph,
                    height=720,
                    scrolling=False,
                )

        else:
            st.warning(
                "O Neo4j não retornou relações para a visualização."
            )

    except Exception as exc:
        logger.exception(
            "Falha ao renderizar o grafo completo."
        )

        st.error(
            "Não foi possível renderizar o grafo: {}".format(exc)
        )


# ============================================================
# ABA: PIPELINE DE EXTRAÇÃO
# ============================================================

with tab_pipe:

    st.write(
        "O pipeline de extração foi executado previamente. "
        "Esta interface consulta os dados existentes no Neo4j "
        "e não executa novamente NER, RE ou geração de embeddings."
    )

    st.info(
        "A ingestão e a migração de embeddings são operações "
        "separadas da inicialização do aplicativo."
    )

    st.metric(
        "Chunks carregados",
        len(system.index.chunks),
    )

    pipeline_stats = store.stats()

    total_entities = sum(
        pipeline_stats.get("by_type", {}).values()
    )

    st.metric(
        "Entidades no grafo",
        total_entities,
    )

    if DOCS:

        selected = st.selectbox(
            "Documento",
            DOCS,
            format_func=lambda doc: "{} — {}".format(
                doc["id"],
                doc["title"],
            ),
        )

        st.subheader("Conteúdo do documento")

        st.info(
            selected["text"]
        )

        st.caption(
            "Documento carregado do conjunto original. "
            "A exibição não altera os dados do Neo4j."
        )

    else:

        st.warning(
            "Documentos originais indisponíveis nesta sessão. "
            "Os chunks existentes continuam disponíveis para recuperação."
        )


# ============================================================
# ABA: AVALIAÇÃO
# ============================================================

with tab_eval:

    st.write(
        "Avaliação reprodutível da recuperação. "
        "Para medir NER/RE, utilize exemplos anotados manualmente."
    )

    eval_file = (
        ROOT
        / "graphrag_med"
        / "data"
        / "evaluation.json"
    )

    if eval_file.exists():

        try:

            rows = evaluation.run(
                system,
                str(eval_file),
                default_hops=2,
            )

            visible = [
                {
                    key: value
                    for key, value in row.items()
                    if not key.startswith("_")
                }

                for row in rows
            ]

            st.dataframe(
                pd.DataFrame(visible),
                use_container_width=True,
                hide_index=True,
            )

            st.json(
                evaluation.summary(rows)
            )

        except Exception as exc:

            logger.exception(
                "Falha na avaliação."
            )

            st.error(
                "Falha na avaliação: {}".format(exc)
            )

    else:

        st.info(
            "Arquivo evaluation.json não encontrado."
        )

    st.caption(
        "A avaliação de extração NER/RE depende de um "
        "conjunto-ouro anotado. Não declare precisão ou "
        "recall sem essas anotações."
    )

    annotation_file = st.file_uploader(
        "Enviar conjunto-ouro anotado (JSON)",
        type=["json"],
    )

    if (
        annotation_file is not None
        and st.button("Avaliar NER e RE")
    ):

        try:

            from graphrag_med.ner import MedicalNER

            from graphrag_med.clinical_extractor import (
                OllamaClinicalExtractor,
            )

            samples = json.loads(
                annotation_file.getvalue().decode("utf-8")
            )

            if not isinstance(samples, list) or not samples:

                raise ValueError(
                    "O JSON deve conter uma lista de exemplos."
                )

            with st.spinner(
                "Executando avaliação NER/RE..."
            ):

                result = evaluation.evaluate_annotated_samples(
                    samples,
                    MedicalNER(),
                    OllamaClinicalExtractor(),
                )

            st.json(result)

        except Exception as exc:

            logger.exception(
                "Falha na avaliação NER/RE."
            )

            st.error(
                "Não foi possível avaliar o conjunto-ouro: {}".format(
                    exc
                )
            )

    st.code(
        '{"entities":[{"text":"metformina","type":"Drug"}],'
        '"relations":[{"subject":"metformina","relation":"CAUSES",'
        '"object":"acidose láctica","evidence":"..."}]}',
        language="json",
    )