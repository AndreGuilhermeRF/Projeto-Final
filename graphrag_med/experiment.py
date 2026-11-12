"""
Métricas do experimento RAG × GraphRAG.

Este módulo não depende de Neo4j nem de Ollama (exceto o juiz opcional),
para que a lógica de medição seja testável e reprodutível.

O que é medido
--------------
1. Recall de recuperação (perguntas com ``expected``):
   - recall de nós: entidades esperadas que aparecem no que foi recuperado
     (GraphRAG: nós de caminhos/fatos + entidades dos trechos; RAG: entidades
     mencionadas pelos trechos);
   - recall textual: entidades esperadas cujo nome ou sinônimo aparece no
     texto de evidência entregue ao LLM (justo para o RAG, que não usa grafo);
   - recall de arestas por coocorrência: ambas as pontas da relação esperada
     aparecem no mesmo item de evidência (vale para os dois sistemas);
   - recall de arestas exato: relação presente nos fatos do grafo (só GraphRAG).
   O recall multi-hop é o mesmo cálculo sobre as perguntas com 2+ saltos.

2. Fidelidade da resposta (taxa de alucinação):
   uma *afirmação* é cada frase factual da resposta. Ela é considerada
   **sem suporte** quando não é sustentada pelo texto de evidência que o LLM
   recebeu (verificador léxico determinístico) e, opcionalmente, quando um
   LLM-juiz a julga sem suporte. Também são medidas validade e correção das
   citações [F#]/[P#]/[T#].

3. Abstenção: respostas corretas de "não sei" nas perguntas não respondíveis
   e recusas indevidas nas respondíveis.

Limites do verificador léxico (declare no artigo)
-------------------------------------------------
- Mede sobreposição de termos (radicais de 5 letras), números e nomes
  próprios; não entende negação nem a direção da relação.
- Paráfrases muito livres podem ser marcadas como sem suporte (falso
  positivo); frases com os termos certos e relação errada podem passar
  (falso negativo). Por isso o módulo exporta uma amostra para auditoria
  humana e calcula concordância (kappa) e sensibilidade/especificidade.

Compatível com Python 3.8.
"""

import csv
import json
import math
import random
import re
import unicodedata

from collections import defaultdict


# ============================================================
# NORMALIZAÇÃO
# ============================================================

def norm(text):
    """Minúsculas e sem acentos."""

    decomposed = unicodedata.normalize("NFKD", str(text or ""))

    return "".join(
        char for char in decomposed
        if not unicodedata.combining(char)
    ).lower()


def tokens(text):
    return re.findall(r"[a-z0-9]+", norm(text))


def stem(token):
    """Radical grosseiro: primeiras 5 letras."""

    return token[:5]


STOPWORDS = set("""
a o as os um uma uns umas de do da dos das em no na nos nas por para com sem
que se ao aos ou e mas como mais menos ser sao foi era pode podem ha tem tem
entre sobre este esta estes estas esse essa esses essas isso isto seu sua seus
suas tambem quando onde qual quais outro outra outros outras cada todo toda
todos todas apenas ja nao sim medicamento medicamentos paciente pacientes
produto produtos uso caso casos grafo evidencia evidencias fato fatos relacao
relacoes conforme segundo indica indicam mostra mostram associado associados
associada associadas relacionado relacionados relacionada relacionadas existe
existem ainda alem desde ate sobre durante apos antes contra pelo pela pelos
pelas num numa aqui assim entao porque pois sendo foram sera serem
""".split())


# Tradução dos tipos de relação (em inglês no grafo) para termos que
# aparecem nas respostas em português.
REL_PT = {
    "TREATS": "trata tratamento tratar indicado indicacao indicada usado",
    "CONTRAINDICATED_IN": "contraindicado contraindicacao evitar",
    "CAUSES": "causa causar provoca efeito adverso reacao",
    "INTERACTS_WITH": "interage interagir interacao",
    "BELONGS_TO": "classe pertence pertencem",
    "HAS_SYMPTOM": "sintoma sintomas",
    "DIAGNOSED_WITH": "diagnostico diagnosticado",
    "ADMINISTERED_BY": "administrado administracao via",
}


# ============================================================
# EVIDÊNCIA E AFIRMAÇÕES
# ============================================================

CITE_GROUP = re.compile(r"\[((?:[FPTNG]\d+(?:\s*[,;/]\s*)?)+)\]")
CITE_TAG = re.compile(r"[FPTNG]\d+")
TAG_LINE = re.compile(r"^\[([FPTNG]\d+)\]\s*(.*)$")

ABSTAIN = re.compile(
    r"(nao (ha|existem?|foram? encontrad\w+|consigo|posso|e possivel|"
    r"foi possivel)\b|evidencias? (sao |foram )?insuficientes?|"
    r"sem evidencias?|informac\w+ insuficientes?|"
    r"nao ha informac\w+|nao encontrei)"
)

DISCLAIMER = re.compile(
    r"(orientacao medica|profissional de saude|consulte|procure|"
    r"nao substitui|uso academico|bula oficial|confira a bula|"
    r"farmaceutico|nao (de|da|dou) orientacao)"
)


def parse_evidence(block):
    """
    Converte o bloco de evidências do prompt (linhas "[F1] ...") em
    {tag: texto}. Linhas sem tag continuam o item anterior.
    """

    items = {}
    current = None

    for line in str(block or "").splitlines():

        match = TAG_LINE.match(line.strip())

        if match:
            current = match.group(1)
            items[current] = match.group(2)

        elif current is not None and line.strip():
            items[current] += " " + line.strip()

    return items


def clinical_items(items):
    """Remove as métricas estruturais [G#], que não são evidência clínica."""

    return {
        tag: text
        for tag, text in items.items()
        if not tag.startswith("G")
    }


def expand_evidence(text):
    """Texto de evidência com sinônimos em português dos tipos de relação."""

    extra = [
        words
        for rel, words in REL_PT.items()
        if rel in text
    ]

    return text + " " + " ".join(extra)


def content_stems(text):
    return {
        stem(token)
        for token in tokens(text)
        if len(token) >= 4
        and token not in STOPWORDS
        and not token.isdigit()
    }


def numbers(text):
    return {
        value.replace(",", ".")
        for value in re.findall(r"\d+(?:[.,]\d+)?", norm(text))
    }


def extract_cites(text):
    cites = []

    for group in CITE_GROUP.findall(text or ""):
        cites.extend(CITE_TAG.findall(group))

    return list(dict.fromkeys(cites))


def split_claims(answer):
    """
    Divide a resposta em afirmações verificáveis.

    Descarta títulos terminados em ":", frases de aviso/ressalva e frases
    sem pelo menos 2 termos de conteúdo. Frases de abstenção também ficam
    de fora (ver is_abstention).
    """

    text = re.sub(r"[*_`#>]", "", answer or "")

    parts = re.split(
        r"\n+|(?<=[.!?;])\s+(?=[A-ZÁÉÍÓÚÂÊÔÃÕÇ\[])",
        text,
    )

    merged = []

    for part in parts:

        part = part.strip().lstrip("-•–— ").strip()

        if not part:
            continue

        only_cites = not CITE_GROUP.sub("", part).strip(" .;,")

        if only_cites and merged:
            merged[-1] += " " + part
        else:
            merged.append(part)

    claims = []

    for part in merged:

        clean = CITE_GROUP.sub("", part).strip()

        if clean.endswith(":") and not extract_cites(part):
            continue

        normalized = norm(clean)

        if DISCLAIMER.search(normalized) or ABSTAIN.search(normalized):
            continue

        if len(content_stems(clean)) < 2:
            continue

        claims.append({
            "text": part,
            "clean": clean,
            "cites": extract_cites(part),
        })

    return claims


def is_abstention(answer):
    """
    True quando a resposta declara falta de evidência e quase não afirma
    mais nada (no máximo uma frase factual além da justificativa).
    """

    text = re.sub(r"[*_`#>]", "", answer or "")

    sentences = [
        sentence.strip()
        for sentence in re.split(r"\n+|(?<=[.!?;])\s+", text)
        if sentence.strip()
    ]

    if not any(ABSTAIN.search(norm(item)) for item in sentences):
        return False

    factual = [
        item for item in sentences
        if not ABSTAIN.search(norm(item))
        and not DISCLAIMER.search(norm(item))
        and len(content_stems(item)) >= 2
    ]

    return len(factual) <= 1


# ============================================================
# VERIFICAÇÃO DE UMA AFIRMAÇÃO
# ============================================================

PROPER = re.compile(r"\b[A-ZÁÉÍÓÚÂÊÔÃÕÇ][A-Za-zÀ-ÿ\-]{3,}\b")


def _coverage(claim_stems, evidence_stems):
    if not claim_stems:
        return 1.0, []

    missing = sorted(claim_stems - evidence_stems)

    return 1.0 - len(missing) / float(len(claim_stems)), missing


def check_claim(claim, items, tau=0.6):
    """
    Verifica uma afirmação contra a evidência entregue ao LLM.

    Suportada quando: cobertura de termos >= tau, todos os números da
    afirmação aparecem na evidência e nenhum nome próprio (fora do início
    da frase) está ausente da evidência.

    Status da citação: "none" (sem citação), "invalid" (tag inexistente ou
    [G#]), "mismatch" (tags válidas, mas o conteúdo não está nelas) ou "ok".
    """

    evidence = clinical_items(items)

    full_text = expand_evidence(" ".join(evidence.values()))
    full_stems = content_stems(full_text)

    claim_stems = content_stems(claim["clean"])

    coverage, missing_terms = _coverage(claim_stems, full_stems)

    missing_numbers = sorted(
        numbers(claim["clean"]) - numbers(full_text)
    )

    first = claim["clean"].split(" ", 1)[0]

    missing_entities = sorted({
        word
        for word in PROPER.findall(claim["clean"])
        if word != first
        and stem(norm(word)) not in full_stems
    })

    supported = (
        coverage >= tau
        and not missing_numbers
        and not missing_entities
    )

    cited = claim["cites"]
    valid = [tag for tag in cited if tag in evidence]

    if not cited:
        cite_status = "none"

    elif len(valid) != len(cited):
        cite_status = "invalid"

    else:
        cited_text = expand_evidence(
            " ".join(evidence[tag] for tag in valid)
        )

        cited_coverage, _ = _coverage(
            claim_stems,
            content_stems(cited_text),
        )

        cite_status = "ok" if cited_coverage >= tau else "mismatch"

    return {
        "text": claim["text"],
        "cites": cited,
        "supported": supported,
        "coverage": round(coverage, 3),
        "missing_terms": missing_terms[:8],
        "missing_numbers": missing_numbers,
        "missing_entities": missing_entities,
        "cite_status": cite_status,
        "judge_supported": None,
    }


def judge_claim(
    claim_text,
    evidence_text,
    base_url="http://127.0.0.1:11434",
    model="qwen2.5:3b",
    timeout=120,
):
    """
    LLM-juiz opcional (Ollama). Retorna True (suportada), False (sem
    suporte) ou None (falha/resposta ilegível).

    Um modelo pequeno é um juiz fraco: prefira um modelo maior via
    --juiz-modelo e valide com a auditoria humana.
    """

    import requests

    prompt = (
        "Você verifica fidelidade. Dada a EVIDÊNCIA e a AFIRMAÇÃO, "
        "responda se a afirmação é sustentada integralmente pela evidência, "
        "sem usar conhecimento externo. Uma afirmação que acrescenta "
        "medicamento, dose, efeito ou relação ausente da evidência é "
        "'unsupported'.\n\n"
        "EVIDÊNCIA:\n{}\n\nAFIRMAÇÃO:\n{}\n\n"
        "Responda somente JSON: "
        "{{\"verdict\": \"supported\" ou \"unsupported\"}}"
    ).format(evidence_text[:6000], claim_text)

    try:

        response = requests.post(
            base_url.rstrip("/") + "/api/chat",
            json={
                "model": model,
                "messages": [{"role": "user", "content": prompt}],
                "stream": False,
                "format": "json",
                "options": {"temperature": 0, "num_predict": 40},
            },
            timeout=timeout,
        )

        response.raise_for_status()

        content = response.json().get("message", {}).get("content", "")

        verdict = str(json.loads(content).get("verdict", "")).lower()

    except Exception:
        return None

    if verdict == "supported":
        return True

    if verdict == "unsupported":
        return False

    return None


def grade_answer(answer, items, tau=0.6, judge=None):
    """
    Avalia uma resposta.

    judge: função (afirmação, texto_de_evidência) -> True/False/None.
    """

    claims = [
        check_claim(claim, items, tau)
        for claim in split_claims(answer)
    ]

    if judge is not None:

        evidence = clinical_items(items)
        all_text = "\n".join(evidence.values())

        for graded in claims:

            cited = [
                evidence[tag]
                for tag in graded["cites"]
                if tag in evidence
            ]

            graded["judge_supported"] = judge(
                graded["text"],
                "\n".join(cited) if cited else all_text,
            )

    cited_claims = [c for c in claims if c["cites"]]

    judged = [
        c for c in claims
        if c["judge_supported"] is not None
    ]

    return {
        "abstained": is_abstention(answer),
        "claims": claims,
        "n_claims": len(claims),
        "n_unsupported": sum(1 for c in claims if not c["supported"]),
        "n_judged": len(judged),
        "n_judge_unsupported": sum(
            1 for c in judged if not c["judge_supported"]
        ),
        "n_cited": len(cited_claims),
        "n_cite_ok": sum(
            1 for c in cited_claims if c["cite_status"] == "ok"
        ),
        "n_cite_invalid": sum(
            1 for c in cited_claims if c["cite_status"] == "invalid"
        ),
        "n_uncited": len(claims) - len(cited_claims),
    }


# ============================================================
# RECALL
# ============================================================

def _mean(values):
    values = [v for v in values if v is not None]

    return sum(values) / float(len(values)) if values else None


def recall_nodes(expected, found):
    expected = set(str(x) for x in (expected or []))

    if not expected:
        return None

    return len(expected & set(str(x) for x in found)) / float(len(expected))


def name_variants(name, synonyms=()):
    variants = {norm(name)}

    for synonym in synonyms or []:
        variants.add(norm(synonym))

    return {item for item in variants if len(item) >= 3}


def _present(variants, normalized_text):
    return any(variant in normalized_text for variant in variants)


def recall_text(expected, texts, names):
    """
    Fração das entidades esperadas cujo nome/sinônimo aparece na evidência.

    names: {id: (nome, sinônimos)}.
    """

    expected = list(expected or [])

    if not expected:
        return None

    blob = norm(" ".join(texts))

    found = 0

    for entity in expected:

        name, synonyms = names.get(entity, (entity.split(":")[-1], ()))

        if _present(name_variants(name.replace("_", " "), synonyms), blob):
            found += 1

    return found / float(len(expected))


def recall_edges_cooccurrence(edges, texts, names):
    """Ambas as pontas da aresta aparecem no mesmo item de evidência."""

    edges = list(edges or [])

    if not edges:
        return None

    normalized = [norm(text) for text in texts]

    def variants(entity):
        name, synonyms = names.get(entity, (entity.split(":")[-1], ()))

        return name_variants(name.replace("_", " "), synonyms)

    hit = 0

    for src, _rel, dst in edges:

        a, b = variants(src), variants(dst)

        if any(_present(a, t) and _present(b, t) for t in normalized):
            hit += 1

    return hit / float(len(edges))


def recall_edges_exact(edges, facts):
    """Aresta esperada presente entre os fatos do grafo (qualquer direção)."""

    edges = list(edges or [])

    if not edges:
        return None

    hit = 0

    for src, rel, dst in edges:

        for fact in facts or []:

            same_pair = {fact.get("src"), fact.get("dst")} == {src, dst}
            same_rel = rel in ("*", None) or fact.get("rel") == rel

            if same_pair and same_rel:
                hit += 1
                break

    return hit / float(len(edges))


# ============================================================
# ESTATÍSTICA
# ============================================================

def bootstrap_ci(rows, stat, n_boot=2000, seed=42, alpha=0.05):
    """
    Intervalo percentil por bootstrap sobre perguntas.

    Retorna (estimativa, inferior, superior); limites None se n < 2.
    """

    point = stat(rows)

    if point is None or len(rows) < 2:
        return point, None, None

    rng = random.Random(seed)

    values = []

    for _ in range(n_boot):

        sample = [rows[rng.randrange(len(rows))] for _ in rows]

        value = stat(sample)

        if value is not None:
            values.append(value)

    if not values:
        return point, None, None

    values.sort()

    lo = values[int((alpha / 2) * (len(values) - 1))]
    hi = values[int((1 - alpha / 2) * (len(values) - 1))]

    return point, lo, hi


def cohen_kappa(a, b):
    """Kappa de Cohen para dois vetores binários."""

    if not a or len(a) != len(b):
        return None

    n = float(len(a))

    observed = sum(1 for x, y in zip(a, b) if x == y) / n

    pa = sum(a) / n
    pb = sum(b) / n

    expected = pa * pb + (1 - pa) * (1 - pb)

    if expected == 1:
        return 1.0

    return (observed - expected) / (1 - expected)


def _median(values):
    values = sorted(v for v in values if v is not None)

    if not values:
        return None

    mid = len(values) // 2

    if len(values) % 2:
        return values[mid]

    return (values[mid - 1] + values[mid]) / 2.0


# ============================================================
# EXECUÇÃO DO EXPERIMENTO
# ============================================================

def question_kind(item):
    if (
        item.get("expect_refusal")
        or item.get("nao_respondivel")
        or item.get("tipo") == "nao_respondivel"
    ):
        return "nao_respondivel"

    return "respondivel"


def run_experiment(
    questions,
    conditions,
    runners,
    name_lookup=None,
    tau=0.6,
    judge=None,
    progress=None,
):
    """
    Executa todas as condições sobre todas as perguntas.

    runners[condição](item) -> dict com:
      answer, items ({tag: texto}), found_ids, facts, latency, error
    name_lookup(id) -> (nome, sinônimos) ou None.
    """

    cache = {}

    def names_for(ids):

        result = {}

        for entity in ids:

            if entity not in cache:
                try:
                    cache[entity] = name_lookup(entity) if name_lookup else None
                except Exception:
                    cache[entity] = None

            if cache[entity]:
                result[entity] = cache[entity]

        return result

    results = []

    for index, item in enumerate(questions):

        hops = item.get("hops")

        entry = {
            "id": item.get("id", "Q{}".format(index + 1)),
            "q": item["q"],
            "kind": question_kind(item),
            "hops": hops,
            "multihop": bool(
                item.get("multihop", (hops or 0) >= 2)
            ),
            "expected": list(item.get("expected", [])),
            "expected_edges": [
                list(edge) for edge in item.get("expected_edges", [])
            ],
            "cond": {},
        }

        edges = entry["expected_edges"]

        ids = set(entry["expected"])

        for _src, _rel, _dst in edges:
            ids.update([_src, _dst])

        names = names_for(ids)

        for condition in conditions:

            if progress:
                progress(index + 1, len(questions), condition, item["q"])

            run = runners[condition](item)

            record = {
                "answer": run.get("answer"),
                "error": run.get("error"),
                "latency_s": run.get("latency"),
                "items": run.get("items", {}),
            }

            if run.get("error") or run.get("answer") is None:
                entry["cond"][condition] = record
                continue

            items = run.get("items", {})

            texts = list(clinical_items(items).values())

            record["recall_nodes"] = recall_nodes(
                entry["expected"],
                run.get("found_ids", []),
            )

            record["recall_text"] = recall_text(
                entry["expected"],
                texts,
                names,
            )

            record["recall_edges_cooc"] = recall_edges_cooccurrence(
                edges,
                texts,
                names,
            )

            record["recall_edges_exact"] = (
                recall_edges_exact(edges, run.get("facts", []))
                if run.get("facts") is not None
                else None
            )

            record["grade"] = grade_answer(
                run["answer"],
                items,
                tau=tau,
                judge=judge,
            )

            entry["cond"][condition] = record

        results.append(entry)

    return results


def regrade(results, tau=0.6, judge=None):
    """Reavalia as respostas já geradas (por exemplo, com outro tau)."""

    for entry in results:
        for record in entry["cond"].values():

            if record.get("answer") is None or record.get("error"):
                continue

            record["grade"] = grade_answer(
                record["answer"],
                record.get("items", {}),
                tau=tau,
                judge=judge,
            )

    return results


# ============================================================
# AGREGAÇÃO
# ============================================================

def _ok_rows(results, condition):
    return [
        entry for entry in results
        if "grade" in entry["cond"].get(condition, {})
    ]


def _rate(rows, condition, numerator, denominator):

    def stat(sample):
        num = sum(
            r["cond"][condition]["grade"][numerator] for r in sample
        )
        den = sum(
            r["cond"][condition]["grade"][denominator] for r in sample
        )

        return num / float(den) if den else None

    return stat


def _mean_metric(condition, key):

    def stat(sample):
        return _mean([
            r["cond"][condition].get(key) for r in sample
        ])

    return stat


def aggregate(results, conditions, n_boot=2000, seed=42):
    """Métricas agregadas por condição, com IC95% por bootstrap."""

    summary = {}

    for condition in conditions:

        rows = _ok_rows(results, condition)

        answerable = [r for r in rows if r["kind"] == "respondivel"]
        unanswerable = [r for r in rows if r["kind"] == "nao_respondivel"]

        with_recall = [
            r for r in answerable
            if r["cond"][condition].get("recall_nodes") is not None
        ]

        multihop = [r for r in with_recall if r["multihop"]]

        with_edges = [
            r for r in answerable
            if r["cond"][condition].get("recall_edges_cooc") is not None
        ]

        def ci(sample, stat):
            return bootstrap_ci(sample, stat, n_boot, seed)

        def per_answer(sample):

            def stat(s):
                values = []

                for r in s:
                    g = r["cond"][condition]["grade"]

                    if g["n_claims"]:
                        values.append(g["n_unsupported"] / float(g["n_claims"]))

                return _mean(values)

            return stat

        any_unsupported = (
            lambda s: _mean([
                1.0 if r["cond"][condition]["grade"]["n_unsupported"] else 0.0
                for r in s
                if r["cond"][condition]["grade"]["n_claims"]
            ])
        )

        abstained = lambda s: _mean([
            1.0 if r["cond"][condition]["grade"]["abstained"] else 0.0
            for r in s
        ])

        summary[condition] = {
            "n_questions": len(rows),
            "n_answerable": len(answerable),
            "n_unanswerable": len(unanswerable),
            "n_claims": sum(
                r["cond"][condition]["grade"]["n_claims"] for r in rows
            ),
            "n_errors": len(results) - len(rows),
            "recall_nodes": ci(with_recall, _mean_metric(condition, "recall_nodes")),
            "recall_text": ci(with_recall, _mean_metric(condition, "recall_text")),
            "recall_edges_cooc": ci(with_edges, _mean_metric(condition, "recall_edges_cooc")),
            "recall_edges_exact": ci(with_edges, _mean_metric(condition, "recall_edges_exact")),
            "recall_nodes_multihop": ci(multihop, _mean_metric(condition, "recall_nodes")),
            "recall_text_multihop": ci(multihop, _mean_metric(condition, "recall_text")),
            "n_recall": len(with_recall),
            "n_multihop": len(multihop),
            "unsupported_rate": ci(
                rows, _rate(rows, condition, "n_unsupported", "n_claims")
            ),
            "unsupported_per_answer": ci(rows, per_answer(rows)),
            "answers_with_unsupported": ci(rows, any_unsupported),
            "judge_unsupported_rate": ci(
                rows, _rate(rows, condition, "n_judge_unsupported", "n_judged")
            ),
            "citations_valid": ci(
                rows,
                lambda s: _safe_ratio(
                    sum(r["cond"][condition]["grade"]["n_cited"]
                        - r["cond"][condition]["grade"]["n_cite_invalid"]
                        for r in s),
                    sum(r["cond"][condition]["grade"]["n_cited"] for r in s),
                ),
            ),
            "citations_correct": ci(
                rows, _rate(rows, condition, "n_cite_ok", "n_cited")
            ),
            "uncited_rate": ci(
                rows, _rate(rows, condition, "n_uncited", "n_claims")
            ),
            "abstention_correct": ci(unanswerable, abstained),
            "abstention_wrong": ci(answerable, abstained),
            "latency_median_s": _median([
                r["cond"][condition].get("latency_s") for r in rows
            ]),
        }

    return summary


def _safe_ratio(num, den):
    return num / float(den) if den else None


def paired_differences(results, system, baseline, n_boot=2000, seed=42):
    """
    Diferença pareada system − baseline (mesmas perguntas), com IC95%.

    Para a taxa de alucinação, valores negativos favorecem o `system`.
    """

    rows = [
        r for r in results
        if "grade" in r["cond"].get(system, {})
        and "grade" in r["cond"].get(baseline, {})
    ]

    def diff_rate(sample):
        a = _rate(sample, system, "n_unsupported", "n_claims")(sample)
        b = _rate(sample, baseline, "n_unsupported", "n_claims")(sample)

        return None if a is None or b is None else a - b

    def diff_mean(key):

        def stat(sample):
            sample = [
                r for r in sample
                if r["cond"][system].get(key) is not None
                and r["cond"][baseline].get(key) is not None
            ]

            if not sample:
                return None

            return _mean([
                r["cond"][system][key] - r["cond"][baseline][key]
                for r in sample
            ])

        return stat

    recall_rows = [r for r in rows if r["kind"] == "respondivel"]
    multihop_rows = [r for r in recall_rows if r["multihop"]]

    return {
        "n_pairs": len(rows),
        "unsupported_rate": bootstrap_ci(rows, diff_rate, n_boot, seed),
        "recall_nodes": bootstrap_ci(
            recall_rows, diff_mean("recall_nodes"), n_boot, seed
        ),
        "recall_text": bootstrap_ci(
            recall_rows, diff_mean("recall_text"), n_boot, seed
        ),
        "recall_nodes_multihop": bootstrap_ci(
            multihop_rows, diff_mean("recall_nodes"), n_boot, seed
        ),
        "recall_text_multihop": bootstrap_ci(
            multihop_rows, diff_mean("recall_text"), n_boot, seed
        ),
    }


# ============================================================
# TABELAS (MARKDOWN E LATEX)
# ============================================================

LABELS = {
    "rag_vetorial": "RAG vetorial",
    "rag_hibrido": "RAG híbrido (vetor + texto)",
    "graphrag": "GraphRAG",
}


def label(condition):
    return LABELS.get(condition, condition)


def fmt(metric, pct=True, signed=False):
    """Formata (estimativa, lo, hi) como '12,3 [5,0; 20,1]'."""

    if metric is None:
        return "n/d"

    if not isinstance(metric, tuple):
        return "{:.2f}".format(metric)

    point, lo, hi = metric

    if point is None:
        return "n/d"

    scale = 100.0 if pct else 1.0

    pattern = "{:+.1f}" if signed else "{:.1f}"

    text = pattern.format(point * scale)

    if lo is not None and hi is not None:
        text += " [{:.1f}; {:.1f}]".format(lo * scale, hi * scale)

    return text.replace(".", ",")


def build_tables(summary, conditions, differences=None, judge_used=False):
    """
    Tabelas do artigo como lista de dicts:
    {"id", "title", "header", "rows", "note"}.
    """

    tables = []

    rows = []

    for c in conditions:
        s = summary[c]

        rows.append([
            label(c),
            str(s["n_recall"]),
            fmt(s["recall_nodes"]),
            fmt(s["recall_text"]),
            fmt(s["recall_edges_cooc"]),
            "{} ({})".format(
                fmt(s["recall_nodes_multihop"]),
                s["n_multihop"],
            ),
            fmt(s["recall_text_multihop"]),
        ])

    tables.append({
        "id": "recuperacao",
        "title": "Recuperação: recall médio (%) com IC95% por bootstrap",
        "header": [
            "Sistema", "n", "Recall de nós", "Recall textual",
            "Recall de arestas", "Nós (multi-hop, n)", "Textual (multi-hop)",
        ],
        "rows": rows,
        "note": (
            "Recall de arestas por coocorrência das duas pontas no mesmo "
            "item de evidência. Multi-hop: perguntas com 2+ saltos."
        ),
    })

    rows = []

    for c in conditions:
        s = summary[c]

        row = [
            label(c),
            str(s["n_questions"]),
            str(s["n_claims"]),
            fmt(s["unsupported_rate"]),
            fmt(s["answers_with_unsupported"]),
            fmt(s["citations_valid"]),
            fmt(s["citations_correct"]),
            fmt(s["uncited_rate"]),
        ]

        if judge_used:
            row.append(fmt(s["judge_unsupported_rate"]))

        rows.append(row)

    header = [
        "Sistema", "Respostas", "Afirmações", "Sem suporte (%)",
        "Respostas com ≥1 sem suporte (%)", "Citações válidas (%)",
        "Citações corretas (%)", "Sem citação (%)",
    ]

    if judge_used:
        header.append("Sem suporte, juiz LLM (%)")

    tables.append({
        "id": "fidelidade",
        "title": "Fidelidade: afirmações sem suporte nas evidências",
        "header": header,
        "rows": rows,
        "note": (
            "Verificador léxico determinístico (cobertura de termos, "
            "números e nomes próprios). Menor é melhor em 'sem suporte'."
        ),
    })

    rows = []

    for c in conditions:
        s = summary[c]

        rows.append([
            label(c),
            str(s["n_unanswerable"]),
            fmt(s["abstention_correct"]),
            str(s["n_answerable"]),
            fmt(s["abstention_wrong"]),
            "n/d" if s["latency_median_s"] is None
            else "{:.1f}".format(s["latency_median_s"]).replace(".", ","),
        ])

    tables.append({
        "id": "abstencao",
        "title": "Abstenção e latência",
        "header": [
            "Sistema", "Não respondíveis (n)", "Abstenção correta (%)",
            "Respondíveis (n)", "Recusa indevida (%)", "Latência mediana (s)",
        ],
        "rows": rows,
        "note": "Abstenção correta: maior é melhor; recusa indevida: menor é melhor.",
    })

    if differences:

        rows = []

        for name, diff in differences.items():
            rows.append([
                name,
                str(diff["n_pairs"]),
                fmt(diff["unsupported_rate"], signed=True),
                fmt(diff["recall_nodes"], signed=True),
                fmt(diff["recall_text"], signed=True),
                fmt(diff["recall_nodes_multihop"], signed=True),
            ])

        tables.append({
            "id": "diferencas",
            "title": "Diferença pareada (pontos percentuais, IC95%)",
            "header": [
                "Comparação", "Pares", "Δ sem suporte", "Δ recall de nós",
                "Δ recall textual", "Δ recall de nós (multi-hop)",
            ],
            "rows": rows,
            "note": (
                "Δ negativo em 'sem suporte' favorece o GraphRAG; Δ positivo "
                "em recall favorece o GraphRAG. IC que inclui 0 indica "
                "diferença não conclusiva."
            ),
        })

    return tables


def to_markdown(tables):
    out = []

    for table in tables:

        out.append("**{}**\n".format(table["title"]))
        out.append("| " + " | ".join(table["header"]) + " |")
        out.append("|" + "|".join(["---"] * len(table["header"])) + "|")

        for row in table["rows"]:
            out.append("| " + " | ".join(row) + " |")

        out.append("\n_{}_\n".format(table["note"]))

    return "\n".join(out)


def _tex(text):
    text = str(text)

    for char, rep in (
        ("\\", "\\textbackslash{}"),
        ("&", "\\&"),
        ("%", "\\%"),
        ("_", "\\_"),
        ("#", "\\#"),
        ("≥", "$\\geq$"),
        ("Δ", "$\\Delta$"),
        ("–", "--"),
    ):
        text = text.replace(char, rep)

    return text


def to_latex(tables):
    """Tabelas com booktabs, prontas para colar no artigo SBC/IEEE."""

    out = ["% Requer \\usepackage{booktabs}"]

    for table in tables:

        columns = "l" + "r" * (len(table["header"]) - 1)

        out.append("\\begin{table}[t]")
        out.append("\\centering\\small")
        out.append("\\caption{" + _tex(table["title"]) + ".}")
        out.append("\\label{tab:" + table["id"] + "}")
        out.append("\\resizebox{\\linewidth}{!}{%")
        out.append("\\begin{tabular}{" + columns + "}")
        out.append("\\toprule")
        out.append(" & ".join(_tex(h) for h in table["header"]) + " \\\\")
        out.append("\\midrule")

        for row in table["rows"]:
            out.append(" & ".join(_tex(cell) for cell in row) + " \\\\")

        out.append("\\bottomrule")
        out.append("\\end{tabular}}")
        out.append("\\end{table}")
        out.append("")

    return "\n".join(out)


# ============================================================
# EXPORTAÇÕES E AUDITORIA HUMANA
# ============================================================

def per_question_rows(results, conditions):
    """Linhas planas (uma por pergunta × condição) para CSV."""

    rows = []

    for entry in results:
        for c in conditions:

            record = entry["cond"].get(c, {})
            grade = record.get("grade", {})

            rows.append({
                "id": entry["id"],
                "pergunta": entry["q"],
                "tipo": entry["kind"],
                "multihop": int(entry["multihop"]),
                "condicao": c,
                "erro": record.get("error") or "",
                "latencia_s": record.get("latency_s"),
                "abstencao": grade.get("abstained"),
                "afirmacoes": grade.get("n_claims"),
                "sem_suporte": grade.get("n_unsupported"),
                "recall_nos": record.get("recall_nodes"),
                "recall_textual": record.get("recall_text"),
                "recall_arestas_cooc": record.get("recall_edges_cooc"),
                "recall_arestas_exato": record.get("recall_edges_exact"),
                "resposta": record.get("answer"),
            })

    return rows


def write_csv(path, rows):

    if not rows:
        return

    with open(path, "w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def audit_sample(results, conditions, n=60, seed=42):
    """
    Amostra de afirmações para auditoria humana, balanceada entre as
    condições. A coluna `humano_suportada` fica em branco (1 = suportada,
    0 = sem suporte).
    """

    rng = random.Random(seed)

    pool = defaultdict(list)

    for entry in results:
        for c in conditions:

            grade = entry["cond"].get(c, {}).get("grade")

            if not grade:
                continue

            evidence = clinical_items(entry["cond"][c].get("items", {}))

            for claim in grade["claims"]:

                cited = " || ".join(
                    evidence[tag][:300]
                    for tag in claim["cites"]
                    if tag in evidence
                )

                pool[c].append({
                    "id_pergunta": entry["id"],
                    "condicao": c,
                    "pergunta": entry["q"],
                    "afirmacao": claim["text"],
                    "evidencia_citada": cited,
                    "auto_suportada": int(claim["supported"]),
                    "humano_suportada": "",
                })

    per_condition = max(1, n // max(1, len(conditions)))

    sample = []

    for c in conditions:

        items = pool[c]

        rng.shuffle(items)

        sample.extend(items[:per_condition])

    rng.shuffle(sample)

    return sample


def evaluate_audit(path):
    """
    Lê o CSV de auditoria preenchido e compara humano × verificador.

    Positivo = "sem suporte". Retorna concordância, kappa, sensibilidade
    e especificidade do detector e a taxa humana de afirmações sem suporte
    por condição.
    """

    rows = []

    with open(path, encoding="utf-8-sig", newline="") as handle:
        for row in csv.DictReader(handle):

            human = str(row.get("humano_suportada", "")).strip()

            if human in ("0", "1"):
                rows.append((
                    row["condicao"],
                    int(row["auto_suportada"]),
                    int(human),
                ))

    if not rows:
        return {"n": 0, "nota": "Nenhuma linha preenchida."}

    auto = [r[1] for r in rows]
    human = [r[2] for r in rows]

    tp = sum(1 for a, h in zip(auto, human) if a == 0 and h == 0)
    fp = sum(1 for a, h in zip(auto, human) if a == 0 and h == 1)
    fn = sum(1 for a, h in zip(auto, human) if a == 1 and h == 0)
    tn = sum(1 for a, h in zip(auto, human) if a == 1 and h == 1)

    by_condition = defaultdict(list)

    for condition, _a, h in rows:
        by_condition[condition].append(1 - h)

    return {
        "n": len(rows),
        "concordancia": round(
            sum(1 for a, h in zip(auto, human) if a == h) / float(len(rows)), 3
        ),
        "kappa": (
            None if cohen_kappa(auto, human) is None
            else round(cohen_kappa(auto, human), 3)
        ),
        "sensibilidade_sem_suporte": _round(_safe_ratio(tp, tp + fn)),
        "especificidade": _round(_safe_ratio(tn, tn + fp)),
        "precisao_sem_suporte": _round(_safe_ratio(tp, tp + fp)),
        "taxa_humana_sem_suporte": {
            condition: _round(sum(values) / float(len(values)))
            for condition, values in by_condition.items()
        },
    }


def _round(value):
    return None if value is None else round(value, 3)