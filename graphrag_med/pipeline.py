
"""Pipeline de ingestão otimizado com controle de importação."""

import os
import re
import json
import math
import hashlib


# ============================================================
# CONFIGURAÇÕES
# ============================================================

PIPELINE_VERSION = "3"

TRIGGERS = [
    (
        "INTERACTS_WITH",
        r"\b(?:interage com|interagem com|interação com|"
        r"interactions? with)\b",
    ),
    (
        "TREATS",
        r"\b(?:indicado para|indicação|indicações|"
        r"utilizado para tratar|utilizada para tratar|"
        r"trata|tratam|treats?)\b",
    ),
    (
        "CONTRAINDICATED_IN",
        r"\b(?:contraindicado em|contraindicada em|"
        r"contraindicados em|contraindicadas em|"
        r"não deve ser usado por|não deve ser utilizada por|"
        r"não utilize se|contraindicated in)\b",
    ),
    (
        "CAUSES",
        r"\b(?:"
        r"pode(?:m)?\s+(?:causar|provocar|ocasionar|"
        r"desencadear|produzir|gerar)|"
        r"poderá(?:o)?\s+(?:causar|provocar|ocasionar|"
        r"produzir|gerar)|"
        r"causa(?:m)?|causou|causaram|"
        r"provoca(?:m)?|provocou|provocaram|"
        r"ocasiona(?:m)?|ocasionou|"
        r"desencadeia(?:m)?|desencadeou|"
        r"produz(?:em)?|produziu|"
        r"causes?|produces?|induces?"
        r")\b",
    ),
    (
        "BELONGS_TO",
        r"\b(?:é um|é uma|são um|são uma|"
        r"pertence à classe|pertencem à classe|"
        r"inibidor da|inibidora da|"
        r"antagonista da|antagonista do)\b",
    ),
]


TRIG_RE = re.compile(
    "|".join(
        "(?P<{}>{})".format(name, pattern)
        for name, pattern in TRIGGERS
    ),
    re.I,
)


NEG_RE = re.compile(
    r"\b(?:"
    r"não\s+(?:"
    r"há|houve|foi|foram|"
    r"é|são|"
    r"causa(?:m)?|causou|"
    r"provoca(?:m)?|provocou|"
    r"produz(?:em)?|produziu|"
    r"apresenta(?:m)?|apresentou|"
    r"se observa(?:m)?|se observou|"
    r"se verifica(?:m)?|se verificou|"
    r"deve(?:m)?\s+(?:causar|provocar|produzir)|"
    r"está associado|está associada|"
    r"estão associados|estão associadas"
    r")"
    r"|ausência de"
    r"|sem evidência de"
    r"|sem evidências de"
    r"|não foram observad[oa]s?"
    r"|não foi observado"
    r"|não foram identificad[oa]s?"
    r"|não foi identificado"
    r"|não existem relatos"
    r"|não há relatos"
    r"|não se observou"
    r"|não se verificou"
    r"|not observed"
    r"|no evidence of"
    r"|no reports of"
    r")\b",
    re.I,
)


HEDGE_RE = re.compile(
    r"\b(?:"
    r"pode(?:m)?\s+(?:"
    r"causar|provocar|ocasionar|desencadear|"
    r"ocorrer|apresentar|produzir|gerar|"
    r"estar associado(?:s|as)?"
    r")"
    r"|poderá(?:o)?\s+(?:"
    r"causar|provocar|ocasionar|ocorrer|"
    r"apresentar|produzir|gerar"
    r")"
    r"|possível"
    r"|possivelmente"
    r"|eventual(?:mente)?"
    r"|talvez"
    r"|pode ser observado"
    r"|podem ser observados"
    r"|may"
    r"|might"
    r"|possibly"
    r"|could"
    r")\b",
    re.I,
)


OBJECT_TYPE = {
    "TREATS": "Condition",
    "CONTRAINDICATED_IN": "Condition",
    "CAUSES": "AdverseEffect",
    "BELONGS_TO": "DrugClass",
}


SECTION_REL = {
    1: "TREATS",
    3: "CONTRAINDICATED_IN",
    8: "CAUSES",
}


VALID_RELATIONS = {
    "INTERACTS_WITH",
    "TREATS",
    "CONTRAINDICATED_IN",
    "CAUSES",
    "BELONGS_TO",
}


# Separadores entre orações independentes.
# Vírgulas comuns não são separadores, pois podem enumerar efeitos.
CLAUSE_SEPARATOR_RE = re.compile(
    r";|[.!?]\s+|"
    r",\s*(?:mas|porém|contudo|entretanto|todavia)\s+|"
    r"\b(?:mas|porém|contudo|entretanto|todavia)\b",
    re.I,
)


# ============================================================
# CLASSIFICAÇÃO DA EVIDÊNCIA
# ============================================================

def _classify_evidence(text):
    """
    Classifica a evidência em:

    asserted:
        Afirmação sem negação ou modalização.

    hypothetical:
        Possibilidade ou hipótese.

    negated:
        Negação explícita.

    A negação sempre tem prioridade sobre a hipótese.
    """

    text = str(text or "")

    negated = bool(NEG_RE.search(text))
    hedged = bool(HEDGE_RE.search(text))

    if negated:
        certainty = "negated"
        confidence = 0.62

    elif hedged:
        certainty = "hypothetical"
        confidence = 0.68

    else:
        certainty = "asserted"
        confidence = 0.82

    return {
        "negated": negated,
        "certainty": certainty,
        "confidence": confidence,
    }


def _classification_for_span(text, start, end):
    """
    Classifica a oração que contém determinada entidade.

    Isso evita que uma negação em uma parte da frase
    seja aplicada automaticamente a todos os efeitos.

    Exemplo:
        "Não foram observadas náuseas, mas houve tontura."

    Náuseas:
        negated

    Tontura:
        asserted
    """

    text = str(text or "")

    start = max(0, int(start))
    end = max(start, int(end))

    boundaries = [0]

    for match in CLAUSE_SEPARATOR_RE.finditer(text):
        boundaries.append(match.end())

    boundaries.append(len(text))

    clause_start = 0
    clause_end = len(text)

    for index in range(len(boundaries) - 1):

        current_start = boundaries[index]
        current_end = boundaries[index + 1]

        if current_start <= start < current_end:
            clause_start = current_start
            clause_end = current_end
            break

    clause = text[clause_start:clause_end]

    return _classify_evidence(clause)


# ============================================================
# IDENTIFICAÇÃO DA IMPORTAÇÃO
# ============================================================

def _dataset_id(docs):
    """Obtém o identificador do conjunto de documentos."""

    dataset_ids = set()

    for doc in docs:
        meta = doc.get("metadata") or {}

        dataset_ids.add(
            str(
                meta.get("dataset_id")
                or meta.get("dataset")
                or "ChatBulário"
            )
        )

    if not dataset_ids:
        return "ChatBulário"

    if len(dataset_ids) > 1:
        raise ValueError(
            "O pipeline recebeu documentos de diferentes datasets. "
            "Separe os datasets antes da ingestão."
        )

    return next(iter(dataset_ids))


def _fingerprint(docs, embedder):
    """
    Gera uma identificação determinística da importação.
    """

    documents = []

    for doc in docs:
        documents.append({
            "id": str(doc.get("id", "")),
            "text": str(doc.get("text", "")),
            "title": str(doc.get("title", "")),
            "date": str(doc.get("date", "")),
            "metadata": doc.get("metadata") or {},
        })

    documents.sort(key=lambda item: item["id"])

    configuration = {
        "pipeline_version": PIPELINE_VERSION,
        "embedding_model": getattr(
            embedder,
            "model",
            os.getenv("OLLAMA_EMBED_MODEL", "unknown"),
        ),
        "embedding_dimensions": getattr(
            embedder,
            "dimensions",
            int(os.getenv("VECTOR_DIMENSIONS", "768")),
        ),
        "model_ner": os.getenv(
            "ENABLE_MODEL_NER", "false"
        ).lower() in ("1", "true", "yes"),
        "llm_extraction": os.getenv(
            "ENABLE_LLM_EXTRACTION", "false"
        ).lower() in ("1", "true", "yes"),
    }

    payload = {
        "documents": documents,
        "configuration": configuration,
    }

    serialized = json.dumps(
        payload,
        ensure_ascii=False,
        sort_keys=True,
        default=str,
    )

    return hashlib.sha256(
        serialized.encode("utf-8")
    ).hexdigest()


def _expected_chunk_count(docs):
    """Calcula quantos chunks serão produzidos."""

    total = 0

    for doc in docs:
        total += len(
            split_sentences(str(doc.get("text", "")))
        )

    return total


def _load_existing_chunks(store, docs):
    """
    Recupera chunks existentes diretamente do Neo4j.

    Não calcula embeddings e não modifica os dados.
    """

    source_ids = [
        str(doc["id"])
        for doc in docs
    ]

    if not source_ids:
        return []

    records = store._run(
        "MATCH (c:Chunk) "
        "WHERE c.source_id IN $source_ids "
        "OPTIONAL MATCH (c)-[:MENTIONS]->(e:Entity) "
        "RETURN "
        "c.id AS id, "
        "c.text AS text, "
        "c.source_id AS source_id, "
        "collect(DISTINCT e.id) AS entities "
        "ORDER BY c.id",
        source_ids=source_ids,
    )

    return [
        {
            "id": record["id"],
            "text": record["text"] or "",
            "source_id": record["source_id"],
            "entities": record["entities"] or [],
        }
        for record in records
    ]


# ============================================================
# DIVISÃO DE SENTENÇAS
# ============================================================

def split_sentences(text):
    return [
        sentence.strip()
        for sentence in re.split(
            r"(?<=[.!?])\s+|\n+",
            str(text or ""),
        )
        if sentence.strip()
    ]


def _section_id(doc):
    """Obtém o número da seção do documento."""

    meta = doc.get("metadata") or {}

    try:
        return int(meta.get("secao_id"))
    except (TypeError, ValueError):
        return None


def _answer_evidence(sentence):
    """
    Remove os marcadores Pergunta/Resposta.

    Retorna None quando a frase contém somente uma pergunta.

    Quando Pergunta e Resposta aparecem juntas, conserva
    somente o conteúdo posterior ao marcador Resposta.
    """

    sentence = str(sentence or "").strip()

    if not sentence:
        return None

    lower = sentence.lower()

    answer_pos = lower.find("resposta:")

    if answer_pos >= 0:
        evidence = sentence[
            answer_pos + len("resposta:"):
        ].strip()

        return evidence or None

    if re.search(r"\bpergunta\s*:", lower):
        return None

    if re.match(r"^\s*resposta\s*:", lower):
        return None

    return sentence


# ============================================================
# RESOLUÇÃO DO MEDICAMENTO
# ============================================================

def _resolve_chatbulario_subject(doc, lex):
    """
    Resolve o medicamento usando os metadados do ChatBulário.

    Primeiro tenta o identificador canônico esperado.
    Depois tenta localizar o medicamento pelo léxico.
    """

    meta = doc.get("metadata") or {}

    active = str(
        meta.get("principio_ativo")
        or meta.get("principio_ativo_csv")
        or ""
    ).strip()

    product = str(
        meta.get("nome_produto")
        or meta.get("nome_comercial")
        or ""
    ).strip()

    canonical = active or product

    if not canonical:
        return None

    expected_id = "DRUG:" + lex._slug(canonical)

    if expected_id in lex.by_id:
        return expected_id

    for candidate in (active, product):
        if not candidate:
            continue

        mentions = [
            mention
            for mention in lex.find(candidate)
            if mention.type == "Drug"
        ]

        if mentions:
            return mentions[0].id

    return None


# ============================================================
# EXTRAÇÃO DE RELAÇÕES POR REGRAS
# ============================================================

def extract_from_sentence(sent, lex):
    """
    Extrai relações explicitamente indicadas por gatilhos.

    Para cada gatilho:
    - seleciona o medicamento mais próximo antes dele;
    - procura o objeto depois do gatilho;
    - classifica a evidência da oração correspondente.
    """

    sent = str(sent or "")

    mentions = lex.find(sent)

    if not mentions:
        return []

    triggers = [
        (match.lastgroup, match.start(), match.end())
        for match in TRIG_RE.finditer(sent)
    ]

    if not triggers:
        return []

    triples = []

    for index, (rel, start, end) in enumerate(triggers):

        subjects = [
            mention
            for mention in mentions
            if mention.type == "Drug"
            and mention.end <= start
        ]

        if not subjects:
            continue

        subject = max(
            subjects,
            key=lambda mention: mention.end,
        )

        segment_end = (
            triggers[index + 1][1]
            if index + 1 < len(triggers)
            else len(sent)
        )

        segment = [
            mention
            for mention in mentions
            if mention.start >= end
            and mention.end <= segment_end
        ]

        if rel == "INTERACTS_WITH":

            objects = [
                mention
                for mention in segment
                if mention.type == "Drug"
                and mention.id != subject.id
            ][:1]

        else:

            expected_type = OBJECT_TYPE.get(rel)

            if not expected_type:
                continue

            objects = [
                mention
                for mention in segment
                if mention.type == expected_type
                and mention.id != subject.id
            ]

        if not objects:
            continue

        for obj in objects:

            classification = _classification_for_span(
                sent,
                obj.start,
                obj.end,
            )

            triples.append({
                "subject": subject.id,
                "rel": rel,
                "object": obj.id,
                "effect": None,
                "evidence": sent,
                "confidence": classification["confidence"],
                "certainty": classification["certainty"],
                "negated": classification["negated"],
            })

    return triples


# ============================================================
# EXTRAÇÃO DO CHATBULÁRIO
# ============================================================

def _chatbulario_triples(doc, lex, evidence):
    """
    Extrai relações a partir do contexto das seções do ChatBulário.

    A seção 8 representa o contexto de efeitos adversos.

    Uma menção a uma entidade AdverseEffect nessa seção pode
    formar uma relação CAUSES mesmo sem um verbo causal explícito.

    Restrições:
    - A pergunta não é utilizada como evidência.
    - O objeto precisa existir no léxico.
    - A seção 8 não inventa efeitos adversos.
    - A negação é classificada por oração.
    - Possibilidades são classificadas como hipotéticas.
    """

    rel = SECTION_REL.get(_section_id(doc))

    if not rel:
        return []

    evidence = str(evidence or "").strip()

    if not evidence:
        return []

    subject = _resolve_chatbulario_subject(
        doc,
        lex,
    )

    if not subject:
        return []

    expected_type = OBJECT_TYPE[rel]

    mentions = lex.find(evidence)

    objects = []
    seen = set()

    for mention in mentions:

        if mention.type != expected_type:
            continue

        if mention.id == subject:
            continue

        if mention.id in seen:
            continue

        seen.add(mention.id)
        objects.append(mention)

    if not objects:
        return []

    triples = []

    for obj in objects:

        classification = _classification_for_span(
            evidence,
            obj.start,
            obj.end,
        )

        triples.append({
            "subject": subject,
            "rel": rel,
            "object": obj.id,
            "effect": None,
            "evidence": evidence,
            "confidence": classification["confidence"],
            "certainty": classification["certainty"],
            "negated": classification["negated"],
        })

    return triples


# ============================================================
# NORMALIZAÇÃO DAS RELAÇÕES DA LLM
# ============================================================

def _normalize_model_relation(relation, doc, lex):
    """
    Valida e normaliza uma relação produzida pela LLM.

    Não aceita:
    - tipos de relação desconhecidos;
    - entidades inexistentes;
    - sujeito que não seja medicamento;
    - objeto incompatível com o tipo da relação;
    - evidência vazia.

    A negação textual tem prioridade sobre a classificação
    fornecida pela LLM.
    """

    relation = dict(relation or {})

    subject = str(
        relation.get("subject") or ""
    )

    obj = str(
        relation.get("object") or ""
    )

    rel = str(
        relation.get("rel") or ""
    ).upper()

    if rel not in VALID_RELATIONS:
        return None

    if subject not in lex.by_id:
        return None

    if obj not in lex.by_id:
        return None

    subject_type = lex.by_id[subject].type
    object_type = lex.by_id[obj].type

    if subject_type != "Drug":
        return None

    if rel == "INTERACTS_WITH":

        if object_type != "Drug":
            return None

    elif object_type != OBJECT_TYPE.get(rel):
        return None

    evidence = str(
        relation.get("evidence") or ""
    ).strip()

    if not evidence:
        return None

    classification = _classify_evidence(evidence)

    supplied_certainty = str(
        relation.get("certainty") or ""
    ).lower().strip()

    supplied_negated = bool(
        relation.get("negated", False)
    )

    # A negação encontrada no texto tem prioridade absoluta.
    if classification["negated"]:

        certainty = "negated"
        negated = True

    elif supplied_negated or supplied_certainty == "negated":

        certainty = "negated"
        negated = True

    elif (
        supplied_certainty == "hypothetical"
        or classification["certainty"] == "hypothetical"
    ):

        certainty = "hypothetical"
        negated = False

    else:

        certainty = "asserted"
        negated = False

    try:
        confidence = float(
            relation.get(
                "confidence",
                classification["confidence"],
            )
        )

        if not math.isfinite(confidence):
            confidence = classification["confidence"]

    except (TypeError, ValueError):
        confidence = classification["confidence"]

    confidence = max(
        0.0,
        min(confidence, 1.0),
    )

    return {
        "subject": subject,
        "rel": rel,
        "object": obj,
        "effect": relation.get("effect"),
        "evidence": evidence,
        "confidence": confidence,
        "certainty": certainty,
        "negated": negated,
    }


# ============================================================
# ASSOCIAÇÃO DA EVIDÊNCIA AO CHUNK
# ============================================================

def _normalize_text(text):
    """Normaliza espaços e caixa para comparação textual."""

    return re.sub(
        r"\s+",
        " ",
        str(text or ""),
    ).strip().casefold()


def _find_evidence_chunk(source_id, evidence, sentence_records):
    """
    Localiza o chunk que contém a evidência da relação.

    Primeiro tenta correspondência textual direta.
    Depois tenta uma correspondência por sobreposição de termos.

    Se não houver correspondência confiável, retorna um
    identificador de evidência sem atribuir um chunk incorreto.
    """

    normalized_evidence = _normalize_text(evidence)

    if not normalized_evidence:
        return str(source_id) + "#evidence"

    candidates = [
        row
        for row in sentence_records
        if str(row.get("source_id")) == str(source_id)
    ]

    for row in candidates:

        normalized_chunk = _normalize_text(
            row.get("text", "")
        )

        if (
            normalized_evidence in normalized_chunk
            or normalized_chunk in normalized_evidence
        ):
            return row["id"]

    evidence_terms = set(
        re.findall(
            r"\w+",
            normalized_evidence,
            re.UNICODE,
        )
    )

    if not evidence_terms:
        return str(source_id) + "#evidence"

    best_row = None
    best_score = 0.0

    for row in candidates:

        chunk_text = _normalize_text(
            row.get("text", "")
        )

        chunk_terms = set(
            re.findall(
                r"\w+",
                chunk_text,
                re.UNICODE,
            )
        )

        if not chunk_terms:
            continue

        overlap = len(
            evidence_terms.intersection(chunk_terms)
        )

        score = overlap / float(len(evidence_terms))

        if score > best_score:
            best_score = score
            best_row = row

    if best_row is not None and best_score >= 0.50:
        return best_row["id"]

    return str(source_id) + "#evidence"


# ============================================================
# DEDUPLICAÇÃO DAS RELAÇÕES
# ============================================================

def _deduplicate_triples(triples):
    """
    Deduplica relações mantendo classificações distintas.

    A chave inclui:
    - fonte;
    - sujeito;
    - relação;
    - objeto;
    - certainty;
    - negated.

    Assim, uma evidência afirmativa não elimina uma evidência
    hipotética ou negada da mesma relação.

    Evidências equivalentes são reunidas em um único texto.
    """

    grouped = {}

    for triple in triples:

        item = dict(triple)

        key = (
            str(item.get("source_id") or ""),
            str(item.get("subject") or ""),
            str(item.get("rel") or ""),
            str(item.get("object") or ""),
            str(item.get("certainty") or "asserted"),
            bool(item.get("negated", False)),
        )

        if key not in grouped:

            grouped[key] = item
            continue

        current = grouped[key]

        old_evidence = str(
            current.get("evidence") or ""
        ).strip()

        new_evidence = str(
            item.get("evidence") or ""
        ).strip()

        if (
            new_evidence
            and new_evidence != old_evidence
        ):

            evidence_parts = [
                part.strip()
                for part in old_evidence.split(" | ")
                if part.strip()
            ]

            if new_evidence not in evidence_parts:
                evidence_parts.append(new_evidence)

            current["evidence"] = " | ".join(
                evidence_parts
            )

        current["confidence"] = max(
            float(current.get("confidence", 0.0)),
            float(item.get("confidence", 0.0)),
        )

        if not current.get("effect"):
            current["effect"] = item.get("effect")

    return list(grouped.values())


# ============================================================
# PERSISTÊNCIA EM LOTES
# ============================================================

def _persist_batches(
    store,
    method_name,
    fallback_name,
    items,
    batch_size,
):
    if not items:
        return

    batch_size = max(1, int(batch_size))

    batch_method = getattr(
        store,
        method_name,
        None,
    )

    if batch_method:

        for start in range(0, len(items), batch_size):

            batch_method(
                items[start:start + batch_size]
            )

    else:

        fallback = getattr(
            store,
            fallback_name,
        )

        for item in items:
            fallback(item)


# ============================================================
# VALIDAÇÃO DOS EMBEDDINGS
# ============================================================

def _validate_vectors(vectors, expected_count, dimensions):
    """Valida quantidade e dimensão dos embeddings."""

    if len(vectors) != expected_count:
        raise RuntimeError(
            "Quantidade incorreta de embeddings: "
            "{} recebidos para {} textos.".format(
                len(vectors),
                expected_count,
            )
        )

    validated = []

    for index, vector in enumerate(vectors):

        values = [
            float(value)
            for value in vector
        ]

        if len(values) != dimensions:
            raise RuntimeError(
                "Embedding {} possui dimensão {}. "
                "Esperado: {}.".format(
                    index,
                    len(values),
                    dimensions,
                )
            )

        if not all(math.isfinite(value) for value in values):
            raise RuntimeError(
                "Embedding {} contém valores não finitos.".format(
                    index
                )
            )

        validated.append(values)

    return validated


# ============================================================
# PIPELINE PRINCIPAL
# ============================================================

def build(
    store,
    docs,
    lex,
    embedder,
    batch_size=50,
    progress_callback=None,
):
    """
    Constrói o grafo usando:

    - Fingerprint de importação.
    - Controle de conclusão no Neo4j.
    - Embeddings em lotes.
    - Persistência em lotes.
    - Extração contextual das seções do ChatBulário.
    - Classificação de relações afirmadas, hipotéticas e negadas.
    - Validação das relações produzidas pela LLM.
    - Associação de evidências aos chunks correspondentes.
    """

    docs = list(docs or [])

    if not docs:
        return [], []

    batch_size = max(1, int(batch_size))

    total = len(docs)

    # --------------------------------------------------------
    # CONFIGURAÇÃO
    # --------------------------------------------------------

    use_model_ner = os.getenv(
        "ENABLE_MODEL_NER", "false"
    ).lower() in ("1", "true", "yes")

    use_llm_extraction = os.getenv(
        "ENABLE_LLM_EXTRACTION", "false"
    ).lower() in ("1", "true", "yes")

    dimensions = int(
        os.getenv("VECTOR_DIMENSIONS", "768")
    )

    dataset_id = _dataset_id(docs)

    import_id = _fingerprint(
        docs,
        embedder,
    )

    expected_chunks = _expected_chunk_count(docs)

    # Atualiza o léxico antes de verificar a importação.
    lex.extend_from_docs(docs)

    # --------------------------------------------------------
    # VERIFICAÇÃO DE IMPORTAÇÃO CONCLUÍDA
    # --------------------------------------------------------

    if store.is_import_complete(
        import_id=import_id,
        dataset_id=dataset_id,
        expected_docs=total,
        expected_chunks=expected_chunks,
    ):

        print(
            "Importação já concluída. "
            "Embeddings não serão recalculados."
        )

        chunks = _load_existing_chunks(
            store,
            docs,
        )

        if len(chunks) != expected_chunks:
            raise RuntimeError(
                "A importação consta como concluída, "
                "mas a quantidade de chunks recuperados "
                "não corresponde ao esperado. "
                "Esperados: {}. Recuperados: {}.".format(
                    expected_chunks,
                    len(chunks),
                )
            )

        if progress_callback:
            progress_callback(
                total,
                total,
                "Importação existente recuperada do Neo4j."
            )

        return chunks, []

    print(
        "Nova importação identificada: {}".format(
            import_id[:16]
        )
    )

    # --------------------------------------------------------
    # INICIALIZAÇÃO DOS EXTRATORES
    # --------------------------------------------------------

    ner = None
    extractor = None

    if use_model_ner:

        from .ner import MedicalNER

        ner = MedicalNER()

    if use_llm_extraction:

        from .llm import llm_available

        if llm_available():

            from .clinical_extractor import OllamaClinicalExtractor

            extractor = OllamaClinicalExtractor()

        else:

            print(
                "Aviso: Ollama indisponível; "
                "extração por regras/seções ativada."
            )

    # --------------------------------------------------------
    # ESTRUTURAS DE DADOS
    # --------------------------------------------------------

    chunks = []
    triples = []
    sources = list(docs)

    sentence_records = []
    model_relations = []

    # --------------------------------------------------------
    # EXTRAÇÃO DE ENTIDADES, CHUNKS E RELAÇÕES
    # --------------------------------------------------------

    for doc_index, doc in enumerate(docs, 1):

        text = str(doc.get("text", ""))
        meta = doc.get("metadata") or {}

        active = str(
            meta.get("principio_ativo")
            or meta.get("principio_ativo_csv")
            or ""
        ).strip()

        product = str(
            meta.get("nome_produto")
            or meta.get("nome_comercial")
            or ""
        ).strip()

        canonical = active or product

        remap = {}

        # ----------------------------------------------------
        # NER
        # ----------------------------------------------------

        if ner:

            try:

                _predictions, remap = ner.add_to_lexicon(
                    text,
                    lex,
                )

            except Exception as exc:

                print(
                    "Aviso: NER falhou em {}: {}".format(
                        doc.get("id"),
                        exc,
                    )
                )

                remap = {}

        # ----------------------------------------------------
        # EXTRAÇÃO COM LLM
        # ----------------------------------------------------

        if extractor:

            try:

                extracted = extractor.extract(text)

                for item in extracted.get("entities", []):

                    matches = lex.find(item["name"])

                    if matches:

                        remap[item["id"]] = matches[0].id

                    else:

                        lex.add_entity(
                            {
                                "id": item["id"],
                                "type": item["type"],
                                "name": item["name"],
                                "synonyms": [],
                            },
                            compile_now=False,
                        )

                        remap[item["id"]] = item["id"]

                lex._compile()

                for relation in extracted.get(
                    "relations",
                    [],
                ):

                    relation = dict(relation)

                    relation["subject"] = remap.get(
                        relation.get("subject"),
                        relation.get("subject"),
                    )

                    relation["object"] = remap.get(
                        relation.get("object"),
                        relation.get("object"),
                    )

                    normalized = _normalize_model_relation(
                        relation,
                        doc,
                        lex,
                    )

                    if normalized:

                        model_relations.append(
                            (doc, normalized)
                        )

            except Exception as exc:

                print(
                    "Aviso: Ollama RE falhou em {}: {}".format(
                        doc.get("id"),
                        exc,
                    )
                )

        # ----------------------------------------------------
        # DIVISÃO EM CHUNKS
        # ----------------------------------------------------

        sentences = split_sentences(text)

        section = _section_id(doc)

        for sentence_index, sentence in enumerate(sentences):

            chunk_text = (
                "Medicamento: {}. Princípio ativo: {}. {}".format(
                    product or canonical,
                    canonical,
                    sentence,
                )
                if canonical
                else sentence
            )

            cid = "{}#s{}".format(
                doc["id"],
                sentence_index,
            )

            mentions = lex.find(chunk_text)

            entity_ids = list(
                dict.fromkeys(
                    mention.id
                    for mention in mentions
                )
            )

            sentence_records.append({
                "id": cid,
                "source_id": doc["id"],
                "text": chunk_text,
                "entities": entity_ids,
            })

            # ------------------------------------------------
            # EXTRAÇÃO DE RELAÇÕES
            # ------------------------------------------------

            evidence = _answer_evidence(sentence)

            if not evidence:
                continue

            if section in SECTION_REL:

                found_triples = _chatbulario_triples(
                    doc,
                    lex,
                    evidence,
                )

            else:

                found_triples = extract_from_sentence(
                    evidence,
                    lex,
                )

            for triple in found_triples:

                triple.update({
                    "source_id": doc["id"],
                    "chunk_id": cid,
                })

                triples.append(triple)

        # ----------------------------------------------------
        # PROGRESSO
        # ----------------------------------------------------

        if progress_callback and (
            doc_index % batch_size == 0
            or doc_index == total
        ):

            progress_callback(
                doc_index,
                total,
                "Extraindo documentos: {}/{}".format(
                    doc_index,
                    total,
                ),
            )

    # --------------------------------------------------------
    # PERSISTÊNCIA DE ENTIDADES
    # --------------------------------------------------------

    print("Persistindo entidades no Neo4j...")

    _persist_batches(
        store,
        "add_entities",
        "add_entity",
        lex.entities(),
        batch_size,
    )

    # --------------------------------------------------------
    # PERSISTÊNCIA DE FONTES
    # --------------------------------------------------------

    print("Persistindo fontes no Neo4j...")

    _persist_batches(
        store,
        "add_sources",
        "add_source",
        sources,
        batch_size,
    )

    # --------------------------------------------------------
    # GERAÇÃO DE EMBEDDINGS
    # --------------------------------------------------------

    print(
        "Gerando embeddings para {} chunks...".format(
            len(sentence_records)
        )
    )

    embed_batch = max(
        1,
        int(os.getenv("EMBED_BATCH_SIZE", "64")),
    )

    for start in range(
        0,
        len(sentence_records),
        embed_batch,
    ):

        records = sentence_records[
            start:start + embed_batch
        ]

        texts = [
            row["text"]
            for row in records
        ]

        if hasattr(embedder, "embed_many"):

            vectors = embedder.embed_many(texts)

        else:

            vectors = [
                embedder.embed(item)
                for item in texts
            ]

        vectors = _validate_vectors(
            vectors,
            len(records),
            dimensions,
        )

        batch_chunks = []

        for row, vector in zip(records, vectors):

            chunk = dict(row)
            chunk["embedding"] = vector

            chunks.append(chunk)
            batch_chunks.append(chunk)

        _persist_batches(
            store,
            "add_chunks",
            "add_chunk",
            batch_chunks,
            batch_size,
        )

        if progress_callback:

            processed = min(
                start + len(records),
                len(sentence_records),
            )

            progress_callback(
                processed,
                max(1, len(sentence_records)),
                "Gerando embeddings: {}/{}".format(
                    processed,
                    len(sentence_records),
                ),
            )

    # --------------------------------------------------------
    # ASSOCIAÇÃO DAS RELAÇÕES DA LLM AOS CHUNKS
    # --------------------------------------------------------

    for doc, relation in model_relations:

        relation = dict(relation)

        relation["source_id"] = doc["id"]

        relation["chunk_id"] = _find_evidence_chunk(
            doc["id"],
            relation.get("evidence", ""),
            sentence_records,
        )

        triples.append(relation)

    # --------------------------------------------------------
    # DEDUPLICAÇÃO DAS RELAÇÕES
    # --------------------------------------------------------

    triples = _deduplicate_triples(triples)

    # --------------------------------------------------------
    # PERSISTÊNCIA DAS RELAÇÕES
    # --------------------------------------------------------

    print("Persistindo relações no Neo4j...")

    _persist_batches(
        store,
        "add_triples",
        "add_triple",
        triples,
        batch_size,
    )

    # --------------------------------------------------------
    # MARCA IMPORTAÇÃO COMO CONCLUÍDA
    # --------------------------------------------------------

    # Esta etapa só acontece depois da persistência de:
    # entidades, fontes, chunks e relações.

    print("Validando importação no Neo4j...")

    store.mark_import_complete(
        import_id=import_id,
        dataset_id=dataset_id,
        documents=total,
        chunks=len(chunks),
        triples=len(triples),
    )

    # --------------------------------------------------------
    # VALIDAÇÃO FINAL
    # --------------------------------------------------------

    if not store.is_import_complete(
        import_id=import_id,
        dataset_id=dataset_id,
        expected_docs=total,
        expected_chunks=len(chunks),
    ):

        raise RuntimeError(
            "A importação foi gravada, mas a validação "
            "final no Neo4j falhou."
        )

    print(
        "Importação concluída: "
        "{} documentos, {} chunks e {} relações.".format(
            total,
            len(chunks),
            len(triples),
        )
    )

    return chunks, triples
