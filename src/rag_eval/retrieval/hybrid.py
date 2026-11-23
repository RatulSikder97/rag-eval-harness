"""Hybrid retrieval: dense + lexical candidates fused with Reciprocal Rank Fusion."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Literal

from rag_eval.embeddings.base import Embedder
from rag_eval.retrieval.rerank import Reranker
from rag_eval.store.base import VectorStore
from rag_eval.types import Chunk, ScoredChunk

RetrievalMode = Literal["hybrid", "vector", "keyword"]


def reciprocal_rank_fusion(
    rankings: Sequence[Sequence[ScoredChunk]], *, k: int = 60
) -> list[ScoredChunk]:
    """Fuse ranked lists by ``sum(1 / (k + rank))``.

    RRF only looks at ranks, so it needs no score calibration between cosine similarity and
    BM25 — the main reason it is the default fusion method in production hybrid search.
    Ties are broken by chunk id for determinism.
    """
    scores: dict[str, float] = {}
    chunks: dict[str, Chunk] = {}
    for ranking in rankings:
        for rank, hit in enumerate(ranking, start=1):
            scores[hit.chunk.id] = scores.get(hit.chunk.id, 0.0) + 1.0 / (k + rank)
            chunks.setdefault(hit.chunk.id, hit.chunk)
    ordered = sorted(scores.items(), key=lambda item: (-item[1], item[0]))
    return [ScoredChunk(chunks[cid], score, "rrf") for cid, score in ordered]


class HybridRetriever:
    def __init__(
        self,
        embedder: Embedder,
        store: VectorStore,
        *,
        mode: RetrievalMode = "hybrid",
        candidate_k: int = 20,
        rrf_k: int = 60,
        reranker: Reranker | None = None,
    ) -> None:
        self.embedder = embedder
        self.store = store
        self.mode = mode
        self.candidate_k = candidate_k
        self.rrf_k = rrf_k
        self.reranker = reranker

    def retrieve(self, query: str, k: int) -> list[ScoredChunk]:
        depth = max(k, self.candidate_k)
        rankings: list[list[ScoredChunk]] = []
        if self.mode in ("hybrid", "vector"):
            rankings.append(self.store.vector_search(self.embedder.embed_query(query), depth))
        if self.mode in ("hybrid", "keyword"):
            rankings.append(self.store.keyword_search(query, depth))
        fused = reciprocal_rank_fusion(rankings, k=self.rrf_k) if len(rankings) > 1 else rankings[0]
        if self.reranker is not None:
            return self.reranker.rerank(query, fused, k)
        return fused[:k]
