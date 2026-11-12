"""Embeddings semânticos via Ollama, executados localmente em CPU."""
import os
import requests
import numpy as np


class OllamaEmbedder:
    """Gera vetores reais com /api/embed; não usa hashing lexical."""

    def __init__(self, base_url=None, model=None, timeout=180, dimensions=None):
        self.base_url = (base_url or os.getenv(
            "OLLAMA_BASE_URL", "http://127.0.0.1:11434"
        )).rstrip("/")
        self.model = model or os.getenv("OLLAMA_EMBED_MODEL", "nomic-embed-text")
        self.timeout = int(timeout)
        self.dimensions = int(dimensions or os.getenv("VECTOR_DIMENSIONS", "768"))

    def embed(self, text):
        response = requests.post(
            self.base_url + "/api/embed",
            json={"model": self.model, "input": str(text)},
            timeout=self.timeout,
        )
        response.raise_for_status()
        data = response.json()
        vectors = data.get("embeddings") or []
        if not vectors:
            raise RuntimeError("Ollama não retornou embedding. Verifique o modelo de embeddings.")
        vector = np.asarray(vectors[0], dtype=np.float32)
        if self.dimensions and len(vector) != self.dimensions:
            raise RuntimeError(
                "Dimensão do embedding ({}) diferente de VECTOR_DIMENSIONS ({}). "
                "Ajuste .env e recrie o índice vetorial.".format(len(vector), self.dimensions)
            )
        norm = float(np.linalg.norm(vector))
        return vector / norm if norm else vector

    def embed_many(self, texts):
        if not texts:
            return []
        response = requests.post(
            self.base_url + "/api/embed",
            json={"model": self.model, "input": [str(x) for x in texts]},
            timeout=self.timeout,
        )
        response.raise_for_status()
        vectors = response.json().get("embeddings") or []
        if len(vectors) != len(texts):
            raise RuntimeError("Quantidade de embeddings retornados não corresponde aos textos.")
        result = []
        for raw in vectors:
            vector = np.asarray(raw, dtype=np.float32)
            if self.dimensions and len(vector) != self.dimensions:
                raise RuntimeError("Dimensão de embedding incompatível com VECTOR_DIMENSIONS.")
            norm = float(np.linalg.norm(vector))
            result.append(vector / norm if norm else vector)
        return result
