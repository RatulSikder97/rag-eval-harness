"""Optional second-stage reranking hook."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Protocol

import httpx

from rag_eval.http import post_json
from rag_eval.types import ScoredChunk


class Reranker(Protocol):
    def rerank(self, query: str, candidates: Sequence[ScoredChunk], k: int) -> list[ScoredChunk]:
        """Return the top ``k`` candidates in a new order with reranker scores."""
        ...


class VoyageReranker:
    """Cross-encoder reranking via Voyage AI's ``/rerank`` endpoint."""

    def __init__(
        self,
        *,
        api_key: str,
        model: str = "rerank-2.5",
        timeout_s: float = 60.0,
        max_retries: int = 4,
        client: httpx.Client | None = None,
    ) -> None:
        self.model = model
        self._max_retries = max_retries
        self._client = client or httpx.Client(
            timeout=timeout_s, headers={"Authorization": f"Bearer {api_key}"}
        )

    def rerank(self, query: str, candidates: Sequence[ScoredChunk], k: int) -> list[ScoredChunk]:
        if not candidates:
            return []
        data = post_json(
            self._client,
            "https://api.voyageai.com/v1/rerank",
            {
                "model": self.model,
                "query": query,
                "documents": [c.chunk.text for c in candidates],
                "top_k": min(k, len(candidates)),
            },
            max_retries=self._max_retries,
        )
        return [
            ScoredChunk(candidates[row["index"]].chunk, float(row["relevance_score"]), "rerank")
            for row in data["data"]
        ]
