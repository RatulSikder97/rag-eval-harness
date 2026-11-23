"""In-memory store: exact cosine search + Okapi BM25. Used by tests and offline CI runs."""

from __future__ import annotations

import json
import math
from collections import Counter
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import numpy as np

from rag_eval.text import tokenize
from rag_eval.types import Chunk, ScoredChunk


class MemoryStore:
    def __init__(self, dim: int, *, k1: float = 1.5, b: float = 0.75) -> None:
        self.dim = dim
        self.k1 = k1
        self.b = b
        self._reset()

    def _reset(self) -> None:
        self._chunks: list[Chunk] = []
        self._index: dict[str, int] = {}
        self._matrix = np.zeros((0, self.dim), dtype=np.float32)
        self._tf: list[Counter[str]] = []
        self._df: Counter[str] = Counter()
        self._lengths: list[int] = []

    def add(self, chunks: Sequence[Chunk], embeddings: Sequence[Sequence[float]]) -> None:
        if len(chunks) != len(embeddings):
            raise ValueError("chunks and embeddings must have the same length")
        if not chunks:
            return
        matrix = np.asarray(embeddings, dtype=np.float32)
        if matrix.shape[1] != self.dim:
            raise ValueError(f"embedding dim {matrix.shape[1]} != store dim {self.dim}")
        latest = {c.id: (c, row) for c, row in zip(chunks, matrix, strict=True)}
        new_rows: list[np.ndarray] = []
        for chunk, row in latest.values():
            if chunk.id in self._index:
                self._replace(self._index[chunk.id], chunk, row)
            else:
                self._append(chunk)
                new_rows.append(row)
        if new_rows:
            self._matrix = np.vstack([self._matrix, np.stack(new_rows)])

    def _append(self, chunk: Chunk) -> None:
        tf = Counter(tokenize(chunk.text))
        self._index[chunk.id] = len(self._chunks)
        self._chunks.append(chunk)
        self._tf.append(tf)
        self._df.update(tf.keys())
        self._lengths.append(sum(tf.values()))

    def _replace(self, pos: int, chunk: Chunk, row: np.ndarray) -> None:
        self._df.subtract(self._tf[pos].keys())
        tf = Counter(tokenize(chunk.text))
        self._chunks[pos] = chunk
        self._matrix[pos] = row
        self._tf[pos] = tf
        self._df.update(tf.keys())
        self._lengths[pos] = sum(tf.values())

    def vector_search(self, embedding: Sequence[float], k: int) -> list[ScoredChunk]:
        if not self._chunks:
            return []
        query = np.asarray(embedding, dtype=np.float32)
        norms = np.linalg.norm(self._matrix, axis=1) * (np.linalg.norm(query) or 1.0)
        scores = (self._matrix @ query) / np.where(norms == 0, 1.0, norms)
        # Round before ranking so BLAS-level float noise across CPUs cannot reorder near-ties;
        # the stable sort then breaks exact ties by insertion order. Keeps CI baselines portable.
        scores = np.round(scores.astype(np.float64), 6)
        top = np.argsort(-scores, kind="stable")[:k]
        return [ScoredChunk(self._chunks[i], float(scores[i]), "vector") for i in top]

    def keyword_search(self, query: str, k: int) -> list[ScoredChunk]:
        terms = set(tokenize(query))
        if not self._chunks or not terms:
            return []
        n = len(self._chunks)
        avg_len = (sum(self._lengths) / n) or 1.0
        scored: list[tuple[float, int]] = []
        for pos, tf in enumerate(self._tf):
            score = 0.0
            for term in terms & tf.keys():
                df = self._df[term]
                idf = math.log(1 + (n - df + 0.5) / (df + 0.5))
                freq = tf[term]
                denom = freq + self.k1 * (1 - self.b + self.b * self._lengths[pos] / avg_len)
                score += idf * freq * (self.k1 + 1) / denom
            if score > 0:
                scored.append((score, pos))
        scored.sort(key=lambda item: (-item[0], item[1]))
        return [ScoredChunk(self._chunks[pos], score, "keyword") for score, pos in scored[:k]]

    def count(self) -> int:
        return len(self._chunks)

    def clear(self) -> None:
        self._reset()

    # -- persistence (lets the CLI ingest once and query many times without Postgres) --

    def save(self, path: Path, *, embedder: str) -> None:
        payload: dict[str, Any] = {
            "version": 1,
            "dim": self.dim,
            "embedder": embedder,
            "chunks": [
                {
                    "id": c.id,
                    "doc_id": c.doc_id,
                    "text": c.text,
                    "ordinal": c.ordinal,
                    "metadata": c.metadata,
                    "embedding": [round(float(v), 6) for v in self._matrix[i]],
                }
                for i, c in enumerate(self._chunks)
            ],
        }
        path.write_text(json.dumps(payload), encoding="utf-8")

    @classmethod
    def load(cls, path: Path, *, embedder: str) -> MemoryStore:
        payload = json.loads(path.read_text(encoding="utf-8"))
        if payload.get("embedder") != embedder:
            raise ValueError(
                f"index at {path} was built with {payload.get('embedder')!r}, "
                f"but the current embedder is {embedder!r}; re-run `rag-eval ingest`"
            )
        store = cls(int(payload["dim"]))
        rows = payload["chunks"]
        store.add(
            [
                Chunk(r["id"], r["doc_id"], r["text"], int(r["ordinal"]), r["metadata"])
                for r in rows
            ],
            [r["embedding"] for r in rows],
        )
        return store
