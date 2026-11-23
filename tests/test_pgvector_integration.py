"""Integration test against a real Postgres + pgvector.

Run with:  docker compose up -d --wait && uv run pytest -m integration
"""

from __future__ import annotations

import os
from collections.abc import Iterator

import pytest

from rag_eval.embeddings import HashingEmbedder
from rag_eval.evals.golden import load_golden
from rag_eval.evals.runner import run_eval
from rag_eval.ingest.chunking import RecursiveChunker
from rag_eval.ingest.loader import iter_documents
from rag_eval.pipeline import ingest
from rag_eval.retrieval import HybridRetriever
from rag_eval.store.memory import MemoryStore

from .conftest import CORPUS, GOLDEN

pytestmark = pytest.mark.integration

DSN = os.environ.get("RAG_TEST_PG_DSN", "postgresql://rag:rag@localhost:5433/rag")


@pytest.fixture
def pg_store() -> Iterator[object]:
    psycopg = pytest.importorskip("psycopg")
    from rag_eval.store.pgvector import PgVectorStore

    try:
        store = PgVectorStore(DSN, 256, table="rag_chunks_test")
    except psycopg.OperationalError as exc:  # pragma: no cover - environment dependent
        pytest.skip(f"Postgres not reachable at {DSN}: {exc}")
    store.clear()
    yield store
    store.clear()
    store.close()


def test_pgvector_hybrid_matches_memory_quality(pg_store: object) -> None:
    from rag_eval.store.pgvector import PgVectorStore

    assert isinstance(pg_store, PgVectorStore)
    embedder = HashingEmbedder(256)
    chunker = RecursiveChunker(800, 120)
    stats = ingest(iter_documents(CORPUS), chunker, embedder, pg_store)
    assert pg_store.count() == stats.chunks

    # Re-ingesting is an idempotent upsert.
    ingest(iter_documents(CORPUS), chunker, embedder, pg_store)
    assert pg_store.count() == stats.chunks

    kw = pg_store.keyword_search("Retry-After header", 3)
    assert kw[0].chunk.doc_id == "api-rate-limits.md"
    vec = pg_store.vector_search(embedder.embed_query("mTLS certificate lifetime"), 3)
    assert vec[0].chunk.doc_id == "authentication.md"
    assert vec[0].chunk.metadata["section"]

    golden = load_golden(GOLDEN)
    pg = run_eval(golden, HybridRetriever(embedder, pg_store), sanity_cases=0)
    mem_store = MemoryStore(256)
    ingest(iter_documents(CORPUS), chunker, embedder, mem_store)
    mem = run_eval(golden, HybridRetriever(embedder, mem_store), sanity_cases=0)
    # Different lexical rankers (ts_rank_cd vs BM25) and ANN vs exact search: close, not equal.
    for name in ("retrieval.recall@5", "retrieval.mrr"):
        assert pg["metrics"][name] >= mem["metrics"][name] - 0.1, name


def test_pgvector_rejects_dimension_mismatch(pg_store: object) -> None:
    from rag_eval.store.pgvector import PgVectorStore

    with pytest.raises(ValueError, match="vector\\(256\\)"):
        PgVectorStore(DSN, 128, table="rag_chunks_test")
