"""Extração assistida por LLM local com validação estrutural e de evidências."""
import json
import os
import re
import unicodedata
import requests

ENTITY_TYPES = {
    "drug": "Drug",
    "medication": "Drug",
    "medicamento": "Drug",
    "disease": "Condition",
    "condition": "Condition",
    "doenca": "Condition",
    "condicao": "Condition",
    "symptom": "AdverseEffect",
    "sintoma": "AdverseEffect",
    "adverse_effect": "AdverseEffect",
    "adverseeffect": "AdverseEffect",
    "efeito_adverso": "AdverseEffect",
    "procedure": "Procedure",
    "procedimento": "Procedure",
    "anatomy": "Anatomy",
    "anatomia": "Anatomy",
    "drug_class": "DrugClass",
    "drugclass": "DrugClass",
    "classe_farmacologica": "DrugClass",
}

RELATIONS = {
    "TREATS",
    "CONTRAINDICATED_IN",
    "CAUSES",
    "INTERACTS_WITH",
    "BELONGS_TO",
    "HAS_SYMPTOM",
    "DIAGNOSED_WITH",
    "ADMINISTERED_BY",
}

RELATION_PROMPT = """Você é um extrator de informação biomédica em português brasileiro.
Extraia somente entidades e relações explicitamente presentes no texto.
Não infira conhecimento médico externo. Preserve a grafia literal dos termos.
Tipos de entidade permitidos: Drug, Condition, AdverseEffect, Procedure, Anatomy, DrugClass.
Relações permitidas: TREATS, CONTRAINDICATED_IN, CAUSES, INTERACTS_WITH,
BELONGS_TO, HAS_SYMPTOM, DIAGNOSED_WITH, ADMINISTERED_BY.
Marque negated=true somente quando a própria evidência negar a relação.
Marque certainty como asserted, hypothetical ou negated.
Retorne exclusivamente JSON neste formato:
{"entities":[{"text":"termo literal","type":"Drug"}],
"relations":[{"subject":"termo literal","relation":"CAUSES","object":"termo literal",
"evidence":"frase literal do texto","negated":false,"certainty":"asserted"}]}
Se não houver elementos, use listas vazias."""


def slug(value):
    value = unicodedata.normalize("NFKD", str(value))
    value = value.encode("ascii", "ignore").decode("ascii")
    return re.sub(r"[^a-z0-9]+", "_", value.lower()).strip("_") or "unknown"


def _json_from_response(content):
    content = (content or "").strip()

    if content.startswith("```"):
        content = re.sub(
            r"^```(?:json)?\s*|\s*```$",
            "",
            content,
            flags=re.I,
        )

    start, end = content.find("{"), content.rfind("}")

    if start < 0 or end <= start:
        raise ValueError("Resposta do extrator não contém JSON.")

    return json.loads(content[start:end + 1])


class OllamaClinicalExtractor:
    """Extrai entidades/relações em blocos curtos e valida as evidências."""

    def __init__(self, base_url=None, model=None, timeout=None):
        self.base_url = (
            base_url or os.getenv(
                "OLLAMA_BASE_URL",
                "http://127.0.0.1:11434"
            )
        ).rstrip("/")

        self.model = model or os.getenv("OLLAMA_MODEL", "qwen2.5:3b")
        self.timeout = int(timeout or os.getenv("OLLAMA_RE_TIMEOUT", "120"))
        self.chunk_chars = int(os.getenv("OLLAMA_RE_CHUNK_CHARS", "1800"))
        self.max_chunks = int(os.getenv("OLLAMA_RE_MAX_CHUNKS", "5"))
        self.num_predict = int(os.getenv("OLLAMA_RE_NUM_PREDICT", "350"))

    def _chunks(self, text):
        """Agrupa sentenças sem ultrapassar o limite de caracteres por chamada."""
        text = str(text).strip()

        if not text:
            return []

        sentences = [
            part.strip()
            for part in re.split(r"(?<=[.!?])\s+|\n+", text)
            if part.strip()
        ]

        chunks, current = [], ""

        for sentence in sentences:
            pieces = []
            remaining = sentence

            while len(remaining) > self.chunk_chars:
                cut = remaining.rfind(" ", 0, self.chunk_chars)

                if cut < self.chunk_chars // 2:
                    cut = self.chunk_chars

                pieces.append(remaining[:cut].strip())
                remaining = remaining[cut:].strip()

            if remaining:
                pieces.append(remaining)

            for piece in pieces:
                candidate = (
                    (current + " " + piece).strip()
                    if current else piece
                )

                if len(candidate) > self.chunk_chars and current:
                    chunks.append(current)
                    current = piece
                else:
                    current = candidate

        if current:
            chunks.append(current)

        return chunks[:max(1, self.max_chunks)]

    def _extract_chunk(self, text):
        response = requests.post(
            self.base_url + "/api/chat",
            json={
                "model": self.model,
                "format": "json",
                "stream": False,
                "messages": [
                    {"role": "system", "content": RELATION_PROMPT},
                    {"role": "user", "content": text},
                ],
                "options": {
                    "temperature": 0,
                    "num_predict": self.num_predict,
                    "num_ctx": 2048,
                },
            },
            timeout=(10, self.timeout),
        )

        response.raise_for_status()

        content = response.json().get("message", {}).get("content", "")
        return _json_from_response(content)

    def extract(self, text):
        if not text or not str(text).strip():
            return {"entities": [], "relations": []}

        all_entities, all_relations = {}, {}
        chunks = self._chunks(text)
        failures = []

        for index, chunk in enumerate(chunks, 1):
            try:
                raw = self._extract_chunk(chunk)
            except (requests.RequestException, ValueError, KeyError) as exc:
                failures.append("bloco {}: {}".format(index, exc))
                continue

            found = {}

            for item in raw.get("entities", []):
                surface = str(item.get("text", "")).strip()

                etype = ENTITY_TYPES.get(
                    str(item.get("type", ""))
                    .strip()
                    .lower()
                    .replace(" ", "_")
                )

                if (
                    not surface
                    or not etype
                    or surface.casefold() not in chunk.casefold()
                ):
                    continue

                pos = chunk.casefold().find(surface.casefold())

                eid = {
                    "Drug": "DRUG",
                    "Condition": "COND",
                    "AdverseEffect": "AE",
                    "DrugClass": "CLASS",
                    "Procedure": "PROC",
                    "Anatomy": "ANAT",
                }[etype] + ":" + slug(surface)

                entity = {
                    "id": eid,
                    "type": etype,
                    "name": surface,
                    "start": pos,
                }

                found[surface.casefold()] = entity
                all_entities[surface.casefold()] = entity

            for item in raw.get("relations", []):
                subject = str(item.get("subject", "")).strip()
                obj = str(item.get("object", "")).strip()
                rel = str(item.get("relation", "")).strip().upper()
                evidence = str(item.get("evidence", "")).strip()

                if (
                    rel not in RELATIONS
                    or not evidence
                    or evidence.casefold() not in chunk.casefold()
                ):
                    continue

                src = found.get(subject.casefold())
                dst = found.get(obj.casefold())

                if not src or not dst or src["id"] == dst["id"]:
                    continue

                negated = bool(item.get("negated", False))
                certainty = str(
                    item.get("certainty", "asserted")
                ).lower()

                if certainty not in ("asserted", "hypothetical", "negated"):
                    certainty = "asserted"

                if certainty == "negated":
                    negated = True

                if negated:
                    certainty = "negated"

                relation = {
                    "subject": src["id"],
                    "rel": rel,
                    "object": dst["id"],
                    "evidence": evidence,
                    "confidence": 0.55,
                    "certainty": certainty,
                    "negated": negated,
                    "effect": None,
                }

                key = (
                    relation["subject"],
                    rel,
                    relation["object"],
                    evidence.casefold(),
                )

                all_relations[key] = relation

        if failures and not all_entities and not all_relations:
            raise RuntimeError(
                "Falha na extração Ollama: " + " | ".join(failures)
            )

        if failures:
            print("Aviso: extração RE parcial: " + " | ".join(failures))

        return {
            "entities": list(all_entities.values()),
            "relations": list(all_relations.values()),
        }