"""Text embeddings: turn text into vectors whose closeness reflects closeness in meaning.

Two implementations share one small interface:

  FastEmbedEmbedder  a real model (BAAI/bge-small-en-v1.5, 384 dimensions, ~70 MB) run locally
                     with ONNX Runtime. No API key, no data leaves the machine.
  HashingEmbedder    a deterministic bag-of-words stand-in used by the tests, so they run fast
                     and offline. It only captures shared words, not meaning.

Vectors are L2-normalised, so cosine similarity is a plain dot product.
"""

from __future__ import annotations

import hashlib
import os
import re
from pathlib import Path
from typing import Protocol

import numpy as np

DEFAULT_MODEL = "BAAI/bge-small-en-v1.5"


class Embedder(Protocol):
    name: str

    def embed_documents(self, texts: list[str]) -> np.ndarray: ...

    def embed_query(self, text: str) -> np.ndarray: ...


def _normalise(m: np.ndarray) -> np.ndarray:
    m = np.asarray(m, dtype=np.float32)
    norms = np.linalg.norm(m, axis=-1, keepdims=True)
    return m / np.where(norms == 0, 1.0, norms)


class FastEmbedEmbedder:
    """Local sentence-embedding model. Loads lazily, so starting the server stays instant."""

    def __init__(self, model: str = DEFAULT_MODEL, cache_dir: str | Path | None = None):
        self.name = model
        self.cache_dir = str(Path(cache_dir or os.environ.get("KB_MODEL_DIR", "~/.kb-mcp/models")).expanduser())
        self._model = None

    def _load(self):
        if self._model is None:
            from fastembed import TextEmbedding  # imported here: heavy, and only needed for semantic search

            self._model = TextEmbedding(model_name=self.name, cache_dir=self.cache_dir)
        return self._model

    def embed_documents(self, texts: list[str]) -> np.ndarray:
        if not texts:
            return np.zeros((0, 384), dtype=np.float32)
        return _normalise(np.stack(list(self._load().passage_embed(texts))))

    def embed_query(self, text: str) -> np.ndarray:
        return _normalise(np.stack(list(self._load().query_embed(text)))[0])


class HashingEmbedder:
    """Deterministic test double: hashed word counts, normalised. Similar words-in-common, similar vectors."""

    def __init__(self, dim: int = 256, name: str = "test-hashing-256"):
        self.dim = dim
        self.name = name

    def _vec(self, text: str) -> np.ndarray:
        v = np.zeros(self.dim, dtype=np.float32)
        for tok in re.findall(r"\w+", text.lower()):
            h = int(hashlib.md5(tok.encode()).hexdigest(), 16)
            v[h % self.dim] += 1.0
        return v

    def embed_documents(self, texts: list[str]) -> np.ndarray:
        if not texts:
            return np.zeros((0, self.dim), dtype=np.float32)
        return _normalise(np.stack([self._vec(t) for t in texts]))

    def embed_query(self, text: str) -> np.ndarray:
        return _normalise(self._vec(text))


def chunk_text(title: str, body: str, max_words: int = 180, overlap: int = 40) -> list[str]:
    """Split a note into overlapping word windows, each prefixed with the title.

    Short notes become one chunk. Long notes are split so each vector describes one focused
    passage (a single vector for a long document blurs its topics together); the overlap keeps
    sentences that straddle a boundary findable from either side.
    """
    words = body.split()
    if len(words) <= max_words:
        return [f"{title}\n{body}".strip()]
    step = max_words - overlap
    chunks = []
    for start in range(0, len(words), step):
        window = words[start : start + max_words]
        chunks.append(f"{title}\n{' '.join(window)}")
        if start + max_words >= len(words):
            break
    return chunks


# --------------------------------------------------------------------------- re-ranking (Phase 3)

DEFAULT_RERANKER = "Xenova/ms-marco-MiniLM-L-6-v2"


class Reranker(Protocol):
    name: str

    def score(self, query: str, documents: list[str]) -> list[float]: ...


class FastEmbedReranker:
    """Cross-encoder: reads the query and a candidate *together* and scores how well it answers.

    Much more accurate than comparing two independent vectors, but far slower, so it only ever
    re-orders the top few candidates that cheaper search already found (~80 MB model, local).
    """

    def __init__(self, model: str = DEFAULT_RERANKER, cache_dir: str | Path | None = None):
        self.name = model
        self.cache_dir = str(Path(cache_dir or os.environ.get("KB_MODEL_DIR", "~/.kb-mcp/models")).expanduser())
        self._model = None

    def score(self, query: str, documents: list[str]) -> list[float]:
        if not documents:
            return []
        if self._model is None:
            from fastembed.rerank.cross_encoder import TextCrossEncoder

            self._model = TextCrossEncoder(model_name=self.name, cache_dir=self.cache_dir)
        return [float(s) for s in self._model.rerank(query, documents)]


class OverlapReranker:
    """Test double: score = number of distinct query words present in the document."""

    name = "test-overlap"

    def score(self, query: str, documents: list[str]) -> list[float]:
        q = set(re.findall(r"\w+", query.lower()))
        return [float(len(q & set(re.findall(r"\w+", d.lower())))) for d in documents]
