"""NER clínico pt-BR com modelo token-classification OpenMed-BR (CPU)."""
import os
import re
import unicodedata

LABEL_MAP = {
    "DOENCA": "Condition",
    "SINAL_SINTOMA": "AdverseEffect",
    "MEDICAMENTO": "Drug",
    "PROCEDIMENTO": "Procedure",
    "ANATOMIA": "Anatomy",
}
PREFIX = {
    "Condition": "COND", "AdverseEffect": "AE", "Drug": "DRUG",
    "Procedure": "PROC", "Anatomy": "ANAT",
}


def _slug(value):
    value = unicodedata.normalize("NFKD", str(value))
    value = value.encode("ascii", "ignore").decode("ascii")
    return re.sub(r"[^a-z0-9]+", "_", value.lower()).strip("_") or "unknown"


class MedicalNER:
    """Modelo NER treinado em português brasileiro, executado localmente em CPU."""

    def __init__(self, model_name=None, threshold=None):
        self.model_name = model_name or os.getenv(
            "NER_MODEL", "BrazilAI/openmed-br-clinical-pt"
        )
        self.threshold = float(
            threshold if threshold is not None else os.getenv("NER_THRESHOLD", "0.40")
        )
        self._pipeline = None

    def _load(self):
        if self._pipeline is None:
            import torch
            from transformers import AutoTokenizer, pipeline

            torch.set_num_threads(int(os.getenv("NER_CPU_THREADS", "2")))
            tokenizer = AutoTokenizer.from_pretrained(
                self.model_name,
                use_fast=True,
            )
            tokenizer.model_max_length = 512
            self._tokenizer = tokenizer
            self._pipeline = pipeline(
                "token-classification",
                model=self.model_name,
                tokenizer=tokenizer,
                aggregation_strategy="simple",
                device=-1,
            )
        return self._pipeline

    def _token_windows(self, text, max_tokens=450, overlap=48):
        """Divide pelo número real de tokens e preserva offsets no texto original."""
        encoded = self._tokenizer(
            text,
            add_special_tokens=False,
            return_offsets_mapping=True,
            truncation=False,
        )
        offsets = encoded.get("offset_mapping", [])
        offsets = [
            (int(start), int(end))
            for start, end in offsets
            if int(end) > int(start)
        ]
        if not offsets:
            return [(0, len(text), text)] if text else []

        windows = []
        token_start = 0
        total = len(offsets)
        while token_start < total:
            token_end = min(total, token_start + max_tokens)
            char_start = offsets[token_start][0]
            char_end = offsets[token_end - 1][1]
            if char_end > char_start:
                windows.append((char_start, char_end, text[char_start:char_end]))
            if token_end >= total:
                break
            token_start = max(token_start + 1, token_end - overlap)
        return windows

    def extract(self, text):
        if not text or not str(text).strip():
            return []
        source = str(text)
        model = self._load()
        predictions = []
        seen = set()

        for base, _end, window in self._token_windows(source):
            for item in model(window):
                label = str(item.get("entity_group", item.get("entity", ""))).upper()
                label = label.replace("B-", "").replace("I-", "")
                entity_type = LABEL_MAP.get(label)
                score = float(item.get("score", 0.0))
                start = int(item.get("start", -1)) + base
                end = int(item.get("end", -1)) + base

                if not entity_type or score < self.threshold:
                    continue
                if start < 0 or end <= start or end > len(source):
                    continue

                surface = source[start:end].strip()
                key = (start, end, entity_type)
                if not surface or key in seen:
                    continue

                seen.add(key)
                entity_id = PREFIX[entity_type] + ":" + _slug(surface)
                predictions.append({
                    "id": entity_id,
                    "type": entity_type,
                    "name": surface,
                    "start": start,
                    "end": end,
                    "score": score,
                })

        return predictions

    def add_to_lexicon(self, text, lexicon):
        predictions = self.extract(text)
        remap = {}

        for item in predictions:
            matches = lexicon.find(item["name"])
            if matches:
                remap[item["id"]] = matches[0].id
            else:
                lexicon.add_entity({
                    "id": item["id"],
                    "type": item["type"],
                    "name": item["name"],
                    "synonyms": [],
                }, compile_now=False)
                remap[item["id"]] = item["id"]

        lexicon._compile()
        return predictions, remap