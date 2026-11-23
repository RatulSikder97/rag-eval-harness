"""HTTP embedding providers: any OpenAI-compatible ``/embeddings`` endpoint, and Voyage AI."""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Sequence
from typing import Any

import httpx

from rag_eval.http import post_json


def _normalise(vec: Sequence[float]) -> list[float]:
    norm = sum(v * v for v in vec) ** 0.5
    return [v / norm for v in vec] if norm else list(vec)


class _BatchedHTTPEmbedder(ABC):
    batch_size = 64

    def __init__(
        self, *, model: str, dim: int, client: httpx.Client, url: str, max_retries: int
    ) -> None:
        self.model = model
        self._dim = dim
        self._client = client
        self._url = url
        self._max_retries = max_retries

    @property
    def dim(self) -> int:
        return self._dim

    @property
    @abstractmethod
    def name(self) -> str: ...

    @abstractmethod
    def _payload(self, texts: list[str], input_type: str) -> dict[str, Any]: ...

    def _embed(self, texts: Sequence[str], input_type: str) -> list[list[float]]:
        out: list[list[float]] = []
        for start in range(0, len(texts), self.batch_size):
            batch = list(texts[start : start + self.batch_size])
            data = post_json(
                self._client,
                self._url,
                self._payload(batch, input_type),
                max_retries=self._max_retries,
            )
            rows = sorted(data["data"], key=lambda row: row["index"])
            for row in rows:
                vec: list[float] = row["embedding"]
                if len(vec) != self._dim:
                    raise ValueError(
                        f"{self.model} returned dim {len(vec)}, configured RAG_EMBEDDING_DIM "
                        f"is {self._dim}"
                    )
                out.append(_normalise(vec))
        return out

    def embed_documents(self, texts: Sequence[str]) -> list[list[float]]:
        return self._embed(texts, "document")

    def embed_query(self, text: str) -> list[float]:
        return self._embed([text], "query")[0]


class OpenAICompatibleEmbedder(_BatchedHTTPEmbedder):
    """Works with OpenAI, Azure-compatible gateways, vLLM, Ollama, LiteLLM, etc."""

    def __init__(
        self,
        *,
        api_key: str,
        model: str,
        dim: int,
        base_url: str = "https://api.openai.com/v1",
        timeout_s: float = 60.0,
        max_retries: int = 4,
        client: httpx.Client | None = None,
    ) -> None:
        client = client or httpx.Client(
            timeout=timeout_s, headers={"Authorization": f"Bearer {api_key}"}
        )
        super().__init__(
            model=model,
            dim=dim,
            client=client,
            url=f"{base_url.rstrip('/')}/embeddings",
            max_retries=max_retries,
        )

    @property
    def name(self) -> str:
        return f"openai:{self.model}"

    def _payload(self, texts: list[str], input_type: str) -> dict[str, Any]:
        payload: dict[str, Any] = {"model": self.model, "input": texts}
        if self.model.startswith("text-embedding-3"):
            payload["dimensions"] = self._dim
        return payload


class VoyageEmbedder(_BatchedHTTPEmbedder):
    """Voyage AI embeddings (asymmetric: distinct document / query input types)."""

    def __init__(
        self,
        *,
        api_key: str,
        model: str,
        dim: int,
        timeout_s: float = 60.0,
        max_retries: int = 4,
        client: httpx.Client | None = None,
    ) -> None:
        client = client or httpx.Client(
            timeout=timeout_s, headers={"Authorization": f"Bearer {api_key}"}
        )
        super().__init__(
            model=model,
            dim=dim,
            client=client,
            url="https://api.voyageai.com/v1/embeddings",
            max_retries=max_retries,
        )

    @property
    def name(self) -> str:
        return f"voyage:{self.model}"

    def _payload(self, texts: list[str], input_type: str) -> dict[str, Any]:
        return {
            "model": self.model,
            "input": texts,
            "input_type": input_type,
            "output_dimension": self._dim,
        }
