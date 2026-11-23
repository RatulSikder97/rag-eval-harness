"""Vector store interface: dense search plus lexical (BM25-style) search."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Protocol

from rag_eval.types import Chunk, ScoredChunk


class VectorStore(Protocol):
    def add(self, chunks: Sequence[Chunk], embeddings: Sequence[Sequence[float]]) -> None: ...

    def vector_search(self, embedding: Sequence[float], k: int) -> list[ScoredChunk]: ...

    def keyword_search(self, query: str, k: int) -> list[ScoredChunk]: ...

    def count(self) -> int: ...

    def clear(self) -> None: ...
