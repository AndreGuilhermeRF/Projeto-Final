"""Carregamento eficiente e tolerante a truncamento do ChatBulário."""
import json
import random
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

DATASET_ID = "walmeidadf/ChatBulario"
DATASET_URL = "https://huggingface.co/datasets/walmeidadf/ChatBulario"


def _clean(value: Any) -> str:
    if value is None:
        return ""
    return " ".join(str(value).split()).strip()


def normalize_row(row: Dict[str, Any], index: int = 0) -> Optional[Dict[str, Any]]:
    answer = _clean(row.get("resposta"))
    question = _clean(row.get("pergunta"))

    if not answer:
        return None

    registration = _clean(row.get("registro")) or "sem-registro-{}".format(index)
    section = row.get("secao_id")

    try:
        section = int(section) if section is not None else None
    except (TypeError, ValueError):
        section = None

    product = _clean(row.get("nome_produto")) or _clean(row.get("nome_comercial"))
    active = _clean(row.get("principio_ativo")) or _clean(row.get("principio_ativo_csv"))
    title = product or active or "Medicamento registro {}".format(registration)

    metadata = {
        "dataset": "ChatBulário",
        "dataset_id": DATASET_ID,
        "dataset_url": DATASET_URL,
        "license": "CC BY 4.0 (compilação); o texto das bulas pode ter direitos dos fabricantes",
        "registro": registration,
        "nome_produto": product,
        "nome_comercial": _clean(row.get("nome_comercial")),
        "principio_ativo": active,
        "principio_ativo_csv": _clean(row.get("principio_ativo_csv")),
        "classe_terapeutica": _clean(row.get("classe_terapeutica")),
        "categoria_regulatoria": _clean(row.get("categoria_regulatoria")),
        "fabricante": _clean(row.get("fabricante")),
        "forma_farmaceutica": _clean(row.get("forma_farmaceutica")),
        "via_administracao": _clean(row.get("via_administracao")),
        "apresentacao": _clean(row.get("apresentacao")),
        "composicao": _clean(row.get("composicao")),
        "uso": _clean(row.get("uso")),
        "secao_id": section,
        "fuzzy_score": row.get("fuzzy_score"),
        "split": row.get("split", "train"),
    }

    safe_registration = "".join(
        c for c in registration if c.isalnum() or c in "-_ "
    ).replace(" ", "_")

    doc_id = "chatbulario:{}:secao-{}".format(
        safe_registration,
        section if section is not None else "na"
    )

    return {
        "id": doc_id,
        "title": "{} — {}".format(title, question or "Informação da bula"),
        "text": (
            "Medicamento: {}. Princípio ativo: {}.\n"
            "Pergunta: {}\nResposta: {}"
        ).format(title, active or "não informado", question, answer),
        "metadata": metadata,
    }


def _iter_json_array(path):
    """Lê objetos de um array JSON e tolera truncamento no último registro."""
    decoder = json.JSONDecoder()

    with Path(path).open("r", encoding="utf-8-sig") as handle:
        raw = handle.read()

    pos = 0
    length = len(raw)

    while pos < length and raw[pos].isspace():
        pos += 1

    if pos >= length or raw[pos] != "[":
        raise ValueError("O arquivo local do ChatBulário precisa conter um array JSON.")

    pos += 1
    index = 0

    while pos < length:
        while pos < length and (raw[pos].isspace() or raw[pos] == ","):
            pos += 1

        if pos >= length or raw[pos] == "]":
            break

        try:
            item, end = decoder.raw_decode(raw, pos)

        except json.JSONDecodeError:
            next_obj = raw.find("\n  {", pos + 1)

            if next_obj < 0:
                print(
                    "Aviso: JSON local truncado; "
                    "último registro incompleto ignorado."
                )
                break

            print(
                "Aviso: registro JSON inválido próximo ao índice {}; "
                "tentando continuar.".format(index)
            )
            pos = next_obj + 3
            continue

        pos = end

        if isinstance(item, dict):
            yield index, item

        index += 1


def _load_local(path, limit, min_fuzzy_score):
    """Seleciona registros com amostragem determinística."""
    rng = random.Random(42)
    selected = []
    valid_count = 0

    for index, row in _iter_json_array(path):
        score = row.get("fuzzy_score")

        try:
            if score is not None and float(score) < min_fuzzy_score:
                continue
        except (TypeError, ValueError):
            pass

        valid_count += 1

        if limit is None or limit <= 0:
            selected.append((index, row))

        elif len(selected) < limit:
            selected.append((index, row))

        else:
            slot = rng.randrange(valid_count)

            if slot < limit:
                selected[slot] = (index, row)

    selected.sort(key=lambda pair: pair[0])

    docs = []

    for index, row in selected:
        doc = normalize_row(row, index)

        if doc:
            docs.append(doc)

    print(
        "ChatBulário: {} registros válidos; {} selecionados.".format(
            valid_count, len(docs)
        )
    )

    return docs


def load_chatbulario_docs(
    split: str = "train",
    limit: Optional[int] = None,
    min_fuzzy_score: float = 80.0,
    local_json: Optional[Union[str, Path]] = None,
) -> List[Dict[str, Any]]:

    # O JSON está dentro de graphrag_med/data.
    local_path = Path(local_json) if local_json else (
        Path(__file__).resolve().parent / "data" / "chatbulario.json"
    )

    if local_path.exists():
        return _load_local(local_path, limit, min_fuzzy_score)

    try:
        from datasets import load_dataset

    except ImportError as exc:
        raise RuntimeError(
            "Instale 'datasets' ou coloque data/chatbulario.json no projeto."
        ) from exc

    dataset = load_dataset(DATASET_ID, split=split)

    if "fuzzy_score" in dataset.column_names:
        dataset = dataset.filter(
            lambda row: row["fuzzy_score"] is None
            or float(row["fuzzy_score"]) >= min_fuzzy_score
        )

    if limit and len(dataset) > limit:
        dataset = dataset.shuffle(seed=42).select(range(limit))

    docs = []

    for index, row in enumerate(dataset):
        doc = normalize_row(dict(row), index)

        if doc:
            docs.append(doc)

    return docs