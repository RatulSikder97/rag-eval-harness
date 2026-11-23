from __future__ import annotations

import json
from collections.abc import Sequence
from pathlib import Path

import httpx
import pytest
import respx

from rag_eval.embeddings import HashingEmbedder
from rag_eval.retrieval import HybridRetriever, VoyageReranker, reciprocal_rank_fusion
from rag_eval.store.memory import MemoryStore
from rag_eval.types import Chunk, ScoredChunk


def _chunk(cid: str, text: str = "") -> Chunk:
    return Chunk(id=cid, doc_id=cid.split("#")[0], text=text or cid, ordinal=0)


def test_memory_store_vector_and_keyword_search(
    store: MemoryStore, embedder: HashingEmbedder
) -> None:
    hits = store.vector_search(embedder.embed_query("canary traffic percentage"), 3)
    assert hits[0].chunk.doc_id == "deployments.md"
    kw = store.keyword_search("Retry-After header 429", 3)
    assert kw[0].chunk.doc_id == "api-rate-limits.md"
    assert all(h.source == "keyword" for h in kw)
    assert store.keyword_search("the and of", 3) == []


def test_memory_store_upserts_by_id(embedder: HashingEmbedder) -> None:
    s = MemoryStore(embedder.dim)
    s.add([_chunk("a#0", "alpha"), _chunk("a#0", "beta")], embedder.embed_documents(["x", "y"]))
    assert s.count() == 1
    s.add([_chunk("a#0", "gamma")], embedder.embed_documents(["gamma"]))
    assert s.count() == 1
    assert s.keyword_search("gamma", 1)[0].chunk.text == "gamma"
    assert s.keyword_search("alpha", 1) == []


def test_memory_store_validates_inputs(embedder: HashingEmbedder) -> None:
    s = MemoryStore(embedder.dim)
    with pytest.raises(ValueError, match="same length"):
        s.add([_chunk("a#0")], [])
    with pytest.raises(ValueError, match="dim"):
        s.add([_chunk("a#0")], [[1.0, 0.0]])
    assert s.vector_search([0.0] * embedder.dim, 3) == []


def test_memory_store_round_trips_and_checks_embedder(
    store: MemoryStore, embedder: HashingEmbedder, tmp_path: Path
) -> None:
    path = tmp_path / "index.json"
    store.save(path, embedder=embedder.name)
    loaded = MemoryStore.load(path, embedder=embedder.name)
    assert loaded.count() == store.count()
    q = embedder.embed_query("restore drill")
    assert [h.chunk.id for h in loaded.vector_search(q, 5)] == [
        h.chunk.id for h in store.vector_search(q, 5)
    ]
    with pytest.raises(ValueError, match="re-run"):
        MemoryStore.load(path, embedder="voyage:other")
    store.clear()
    assert store.count() == 0


def test_rrf_rewards_agreement_and_is_deterministic() -> None:
    a = [ScoredChunk(_chunk(c), 1.0) for c in ["x#0", "y#0", "z#0"]]
    b = [ScoredChunk(_chunk(c), 1.0) for c in ["y#0", "w#0"]]
    fused = reciprocal_rank_fusion([a, b], k=60)
    assert [h.chunk.id for h in fused] == ["y#0", "x#0", "w#0", "z#0"]
    assert fused[0].score == pytest.approx(1 / 62 + 1 / 61)
    tie = reciprocal_rank_fusion([[ScoredChunk(_chunk("b#0"), 1)], [ScoredChunk(_chunk("a#0"), 1)]])
    assert [h.chunk.id for h in tie] == ["a#0", "b#0"]


@pytest.mark.parametrize("mode", ["hybrid", "vector", "keyword"])
def test_retriever_modes_find_the_right_doc(
    mode: str, embedder: HashingEmbedder, store: MemoryStore
) -> None:
    retriever = HybridRetriever(embedder, store, mode=mode)  # type: ignore[arg-type]
    hits = retriever.retrieve("How long are mTLS certificates valid?", 3)
    assert 1 <= len(hits) <= 3
    assert hits[0].chunk.doc_id == "authentication.md"


class ReverseReranker:
    def rerank(self, query: str, candidates: Sequence[ScoredChunk], k: int) -> list[ScoredChunk]:
        return [ScoredChunk(c.chunk, float(i), "rerank") for i, c in enumerate(candidates)][::-1][
            :k
        ]


def test_reranker_hook_sees_full_candidate_pool(
    embedder: HashingEmbedder, store: MemoryStore
) -> None:
    retriever = HybridRetriever(embedder, store, candidate_k=10, reranker=ReverseReranker())
    hits = retriever.retrieve("deploy freeze", 2)
    assert len(hits) == 2
    assert all(h.source == "rerank" for h in hits)
    assert hits[0].score > hits[1].score


@respx.mock
def test_voyage_reranker_maps_indices_back() -> None:
    route = respx.post("https://api.voyageai.com/v1/rerank").mock(
        return_value=httpx.Response(
            200,
            json={
                "data": [{"index": 1, "relevance_score": 0.9}, {"index": 0, "relevance_score": 0.1}]
            },
        )
    )
    candidates = [ScoredChunk(_chunk("a#0", "A"), 0.5), ScoredChunk(_chunk("b#0", "B"), 0.4)]
    out = VoyageReranker(api_key="k").rerank("q", candidates, 2)
    assert [h.chunk.id for h in out] == ["b#0", "a#0"]
    assert json.loads(route.calls.last.request.content)["documents"] == ["A", "B"]
    assert VoyageReranker(api_key="k").rerank("q", [], 2) == []
