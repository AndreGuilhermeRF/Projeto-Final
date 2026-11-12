"""Métricas de avaliação para recuperação e extração anotada."""
import json
from .retrieval import retrieve


def vector_only(system, question, k=3):
    hits = system.index.search(question, k)
    return {entity for chunk, _score in hits for entity in chunk.get("entities", [])}


def run(system, questions_path, default_hops=2):
    with open(questions_path, encoding="utf-8") as handle:
        questions = json.load(handle)
    rows = []
    for item in questions:
        question, hops = item["q"], item.get("hops", default_hops)
        ctx = retrieve(system, question, hops=hops)
        graph_nodes = {node for path in ctx["paths"] for node in path["nodes"]}
        row = {"pergunta": question, "saltos": hops}
        if item.get("expect_refusal"):
            ok = bool(ctx["refused"])
            row.update(tipo="recusa", grafo="OK" if ok else "FALHOU", vetorial="N/A")
            row["_g"] = float(ok)  # sem baseline: não entra na média vetorial
        elif item.get("expect_negated"):
            ok = bool(ctx["negated_facts"])
            row.update(tipo="negação", grafo="OK" if ok else "FALHOU", vetorial="N/A")
            row["_g"] = float(ok)  # sem baseline: não entra na média vetorial
        else:
            expected = set(item.get("expected", []))
            if not expected:
                continue
            graph_recall = len(expected & graph_nodes) / float(len(expected))
            vector_recall = len(expected & vector_only(system, question)) / float(len(expected))
            row.update(tipo="recuperação", grafo="{:.0%}".format(graph_recall),
                       vetorial="{:.0%}".format(vector_recall))
            row["_g"], row["_v"] = graph_recall, vector_recall
        rows.append(row)
    return rows


def summary(rows):
    graph = [row["_g"] for row in rows if "_g" in row]
    vector = [row["_v"] for row in rows if "_v" in row]
    return {
        "casos_avaliados": len(rows),
        "recall_medio_grafo": round(sum(graph) / len(graph), 4) if graph else None,
        "recall_medio_baseline": round(sum(vector) / len(vector), 4) if vector else None,
        "nota": "Métricas válidas somente para as perguntas e IDs esperados anotados no arquivo.",
    }


def extraction_metrics(expected, predicted):
    """Precision/recall/F1 por correspondência exata de (texto, tipo) ou (sujeito, relação, objeto)."""
    expected_set, predicted_set = set(expected), set(predicted)
    true_positive = len(expected_set & predicted_set)
    precision = true_positive / float(len(predicted_set)) if predicted_set else 0.0
    recall = true_positive / float(len(expected_set)) if expected_set else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return {"precision": precision, "recall": recall, "f1": f1,
            "tp": true_positive, "fp": len(predicted_set - expected_set),
            "fn": len(expected_set - predicted_set)}



def evaluate_annotated_samples(samples, ner, relation_extractor):
    """Avalia NER/RE contra anotações humanas em correspondência exata."""
    entity_expected, entity_predicted = set(), set()
    relation_expected, relation_predicted = set(), set()
    for sample in samples:
        text = sample["text"]
        expected_entities = sample.get("entities", [])
        expected_relations = sample.get("relations", [])
        for entity in expected_entities:
            entity_expected.add((
                str(entity["text"]).casefold(),
                str(entity["type"]),
            ))
        for relation in expected_relations:
            relation_expected.add((
                str(relation["subject"]).casefold(),
                str(relation["relation"]).upper(),
                str(relation["object"]).casefold(),
            ))
        for entity in ner.extract(text):
            entity_predicted.add((
                str(entity["name"]).casefold(),
                str(entity["type"]),
            ))
        extracted = relation_extractor.extract(text)
        for relation in extracted.get("relations", []):
            # A avaliação usa as superfícies literais retornadas pelo extrator.
            entity_by_id = {
                item["id"]: item["name"] for item in extracted.get("entities", [])
            }
            subject = entity_by_id.get(relation["subject"])
            obj = entity_by_id.get(relation["object"])
            if subject and obj:
                relation_predicted.add((
                    subject.casefold(), relation["rel"].upper(), obj.casefold()
                ))
    return {
        "ner": extraction_metrics(entity_expected, entity_predicted),
        "re": extraction_metrics(relation_expected, relation_predicted),
        "samples": len(samples),
        "note": "Correspondência exata por superfície e rótulo; normalize/anote o corpus antes de comparar.",
    }