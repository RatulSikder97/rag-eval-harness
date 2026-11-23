from __future__ import annotations

from pathlib import Path

import pytest

from rag_eval.embeddings import HashingEmbedder
from rag_eval.ingest.chunking import RecursiveChunker
from rag_eval.ingest.loader import iter_documents
from rag_eval.pipeline import ingest
from rag_eval.retrieval import HybridRetriever
from rag_eval.store.memory import MemoryStore

ROOT = Path(__file__).resolve().parent.parent
CORPUS = ROOT / "corpus"
GOLDEN = ROOT / "evals" / "golden.yaml"


@pytest.fixture(autouse=True)
def _isolated_env(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """Never let a developer's RAG_* env or .env leak into tests."""
    import os

    for key in list(os.environ):
        if key.startswith("RAG_"):
            monkeypatch.delenv(key)
    monkeypatch.chdir(tmp_path)


@pytest.fixture
def embedder() -> HashingEmbedder:
    return HashingEmbedder(256)


@pytest.fixture
def store(embedder: HashingEmbedder) -> MemoryStore:
    s = MemoryStore(embedder.dim)
    ingest(iter_documents(CORPUS), RecursiveChunker(800, 120), embedder, s)
    return s


@pytest.fixture
def retriever(embedder: HashingEmbedder, store: MemoryStore) -> HybridRetriever:
    return HybridRetriever(embedder, store)
