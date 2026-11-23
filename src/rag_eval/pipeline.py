"""Factories that turn ``Settings`` into concrete components, plus the ingest step."""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path

from rag_eval.config import Settings
from rag_eval.embeddings import Embedder, HashingEmbedder, OpenAICompatibleEmbedder, VoyageEmbedder
from rag_eval.evals.judge import ClaudeJudge, Judge, OverlapJudge
from rag_eval.generation import ClaudeClient, ClaudeGenerator, ExtractiveGenerator, Generator
from rag_eval.ingest.chunking import Chunker, RecursiveChunker, SemanticChunker
from rag_eval.ingest.loader import iter_documents
from rag_eval.retrieval import HybridRetriever, Reranker, RetrievalMode, VoyageReranker
from rag_eval.store.base import VectorStore
from rag_eval.store.memory import MemoryStore
from rag_eval.types import Chunk, Document


class ConfigError(ValueError):
    """Raised for missing credentials or inconsistent settings."""


def build_embedder(settings: Settings) -> Embedder:
    match settings.embedder:
        case "hashing":
            return HashingEmbedder(settings.embedding_dim)
        case "openai":
            if settings.openai_api_key is None:
                raise ConfigError("RAG_OPENAI_API_KEY is required for the openai embedder")
            return OpenAICompatibleEmbedder(
                api_key=settings.openai_api_key.get_secret_value(),
                model=settings.openai_embedding_model,
                dim=settings.embedding_dim,
                base_url=settings.openai_base_url,
                max_retries=settings.max_retries,
            )
        case "voyage":
            if settings.voyage_api_key is None:
                raise ConfigError("RAG_VOYAGE_API_KEY is required for the voyage embedder")
            return VoyageEmbedder(
                api_key=settings.voyage_api_key.get_secret_value(),
                model=settings.voyage_embedding_model,
                dim=settings.embedding_dim,
                max_retries=settings.max_retries,
            )


def build_chunker(settings: Settings, embedder: Embedder) -> Chunker:
    if settings.chunker == "semantic":
        return SemanticChunker(
            embedder, chunk_size=settings.chunk_size, threshold=settings.semantic_threshold
        )
    return RecursiveChunker(settings.chunk_size, settings.chunk_overlap)


def build_store(settings: Settings, *, fresh: bool = False) -> VectorStore:
    """Return the configured store. ``fresh`` ignores any persisted in-memory index."""
    if settings.store == "pgvector":
        from rag_eval.store.pgvector import PgVectorStore

        return PgVectorStore(settings.pg_dsn, settings.embedding_dim, table=settings.pg_table)
    index = Path(settings.index_path)
    if not fresh and index.exists():
        return MemoryStore.load(index, embedder=build_embedder(settings).name)
    return MemoryStore(settings.embedding_dim)


def build_reranker(settings: Settings, name: str | None) -> Reranker | None:
    if name in (None, "none"):
        return None
    if name == "voyage":
        if settings.voyage_api_key is None:
            raise ConfigError("RAG_VOYAGE_API_KEY is required for the voyage reranker")
        return VoyageReranker(
            api_key=settings.voyage_api_key.get_secret_value(), max_retries=settings.max_retries
        )
    raise ConfigError(f"unknown reranker: {name}")


def build_retriever(
    settings: Settings,
    embedder: Embedder,
    store: VectorStore,
    *,
    mode: RetrievalMode = "hybrid",
    reranker: Reranker | None = None,
) -> HybridRetriever:
    return HybridRetriever(
        embedder,
        store,
        mode=mode,
        candidate_k=settings.candidate_k,
        rrf_k=settings.rrf_k,
        reranker=reranker,
    )


def _claude_client(settings: Settings) -> ClaudeClient:
    return ClaudeClient(
        timeout_s=settings.request_timeout_s,
        max_retries=settings.max_retries,
        max_tokens=settings.max_tokens,
    )


def build_generator(settings: Settings) -> Generator:
    if settings.llm == "anthropic":
        return ClaudeGenerator(
            _claude_client(settings),
            model=settings.generation_model,
            effort=settings.generation_effort,
        )
    return ExtractiveGenerator()


def build_judge(settings: Settings) -> Judge:
    if settings.llm == "anthropic":
        return ClaudeJudge(
            _claude_client(settings), model=settings.judge_model, effort=settings.judge_effort
        )
    return OverlapJudge()


@dataclass(frozen=True, slots=True)
class IngestStats:
    documents: int
    chunks: int


def ingest(
    documents: Iterable[Document],
    chunker: Chunker,
    embedder: Embedder,
    store: VectorStore,
    *,
    batch_size: int = 64,
) -> IngestStats:
    n_docs = n_chunks = 0
    pending: list[Chunk] = []
    for doc in documents:
        n_docs += 1
        pending.extend(chunker.chunk(doc))
        while len(pending) >= batch_size:
            batch, pending = pending[:batch_size], pending[batch_size:]
            store.add(batch, embedder.embed_documents([c.text for c in batch]))
            n_chunks += len(batch)
    if pending:
        store.add(pending, embedder.embed_documents([c.text for c in pending]))
        n_chunks += len(pending)
    return IngestStats(documents=n_docs, chunks=n_chunks)


def ingest_corpus(settings: Settings, corpus: Path, store: VectorStore) -> IngestStats:
    embedder = build_embedder(settings)
    return ingest(iter_documents(corpus), build_chunker(settings, embedder), embedder, store)
