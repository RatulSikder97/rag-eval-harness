from __future__ import annotations

import json
import math

import httpx
import pytest
import respx

from rag_eval.embeddings import HashingEmbedder, OpenAICompatibleEmbedder, VoyageEmbedder
from rag_eval.http import ProviderError, post_json


def _cos(a: list[float], b: list[float]) -> float:
    return sum(x * y for x, y in zip(a, b, strict=True))


@pytest.fixture(autouse=True)
def _no_sleep(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("rag_eval.http.time.sleep", lambda _s: None)


def test_hashing_embedder_is_deterministic_and_normalised() -> None:
    emb = HashingEmbedder(128)
    a, b = emb.embed_query("rotate api keys"), HashingEmbedder(128).embed_query("rotate api keys")
    assert a == b
    assert math.isclose(math.sqrt(sum(v * v for v in a)), 1.0, rel_tol=1e-9)
    assert emb.name == "hashing-128"


def test_hashing_embedder_ranks_lexical_neighbours_higher() -> None:
    emb = HashingEmbedder(512)
    q = emb.embed_query("how often should api keys be rotated")
    near, far = emb.embed_documents(
        ["Rotate API keys at least every 180 days.", "Canary deploys get 5% of traffic."]
    )
    assert _cos(q, near) > _cos(q, far)


def test_hashing_embedder_empty_text_is_zero_vector() -> None:
    assert HashingEmbedder(8).embed_query("") == [0.0] * 8
    with pytest.raises(ValueError, match="dim"):
        HashingEmbedder(0)


@respx.mock
def test_openai_compatible_embedder_orders_by_index_and_normalises() -> None:
    route = respx.post("https://gw.example/v1/embeddings").mock(
        return_value=httpx.Response(
            200,
            json={
                "data": [
                    {"index": 1, "embedding": [0.0, 2.0]},
                    {"index": 0, "embedding": [3.0, 0.0]},
                ]
            },
        )
    )
    emb = OpenAICompatibleEmbedder(
        api_key="test", model="text-embedding-3-small", dim=2, base_url="https://gw.example/v1/"
    )
    assert emb.embed_documents(["a", "b"]) == [[1.0, 0.0], [0.0, 1.0]]
    sent = json.loads(route.calls.last.request.content)
    assert sent == {"model": "text-embedding-3-small", "input": ["a", "b"], "dimensions": 2}
    assert route.calls.last.request.headers["authorization"] == "Bearer test"


@respx.mock
def test_embedder_rejects_dimension_mismatch() -> None:
    respx.post("https://gw.example/v1/embeddings").mock(
        return_value=httpx.Response(200, json={"data": [{"index": 0, "embedding": [1.0]}]})
    )
    emb = OpenAICompatibleEmbedder(api_key="k", model="m", dim=4, base_url="https://gw.example/v1")
    with pytest.raises(ValueError, match="RAG_EMBEDDING_DIM"):
        emb.embed_query("x")


@respx.mock
def test_voyage_embedder_sends_input_type() -> None:
    route = respx.post("https://api.voyageai.com/v1/embeddings").mock(
        return_value=httpx.Response(200, json={"data": [{"index": 0, "embedding": [1.0, 0.0]}]})
    )
    emb = VoyageEmbedder(api_key="k", model="voyage-3.5", dim=2)
    emb.embed_query("q")
    assert json.loads(route.calls.last.request.content)["input_type"] == "query"
    emb.embed_documents(["d"])
    assert json.loads(route.calls.last.request.content)["input_type"] == "document"
    assert emb.name == "voyage:voyage-3.5"


@respx.mock
def test_post_json_retries_retryable_status_then_succeeds() -> None:
    route = respx.post("https://p.example/x").mock(
        side_effect=[
            httpx.Response(429, headers={"retry-after": "1"}),
            httpx.Response(503),
            httpx.Response(200, json={"ok": True}),
        ]
    )
    with httpx.Client() as client:
        assert post_json(client, "https://p.example/x", {}, max_retries=3) == {"ok": True}
    assert route.call_count == 3


@respx.mock
def test_post_json_does_not_retry_client_errors() -> None:
    route = respx.post("https://p.example/x").mock(return_value=httpx.Response(401, text="no"))
    with httpx.Client() as client, pytest.raises(ProviderError, match="401"):
        post_json(client, "https://p.example/x", {}, max_retries=3)
    assert route.call_count == 1


@respx.mock
def test_post_json_gives_up_after_max_retries() -> None:
    respx.post("https://p.example/x").mock(side_effect=httpx.ConnectError("down"))
    with httpx.Client() as client, pytest.raises(ProviderError, match="down"):
        post_json(client, "https://p.example/x", {}, max_retries=2)
