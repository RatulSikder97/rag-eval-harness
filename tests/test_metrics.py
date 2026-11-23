from __future__ import annotations

import math

import pytest

from rag_eval.evals import generation_metrics as gm
from rag_eval.evals import retrieval_metrics as rm
from rag_eval.evals.golden import GoldenItem, RelevantSpec
from rag_eval.types import Answer, Chunk, ScoredChunk


def _c(cid: str, text: str = "x") -> Chunk:
    return Chunk(id=cid, doc_id=cid.split("#")[0], text=text, ordinal=int(cid.split("#")[1]))


SPECS = [RelevantSpec(doc="a.md", grade=2), RelevantSpec(doc="b.md", contains="needle")]


def test_spec_matching_uses_doc_and_case_insensitive_evidence() -> None:
    assert SPECS[0].matches(_c("a.md#3"))
    assert not SPECS[1].matches(_c("b.md#0", "hay"))
    assert SPECS[1].matches(_c("b.md#1", "a NEEDLE here"))


def test_each_spec_is_credited_once() -> None:
    ranked = [_c("a.md#0"), _c("a.md#1"), _c("z.md#0"), _c("b.md#0", "needle")]
    grades = rm.credited_grades(ranked, SPECS)
    assert grades == [2, 0, 0, 1]
    assert rm.recall_at_k(grades, 2, 1) == 0.5
    assert rm.recall_at_k(grades, 2, 4) == 1.0
    assert rm.recall_at_k(grades, 0, 4) == 0.0


def test_reciprocal_rank() -> None:
    assert rm.reciprocal_rank([_c("z.md#0"), _c("a.md#0")], SPECS) == 0.5
    assert rm.reciprocal_rank([_c("z.md#0")], SPECS) == 0.0


def test_ndcg_matches_hand_computation() -> None:
    grades = [0, 1, 2]
    dcg = 1 / math.log2(3) + 3 / math.log2(4)
    idcg = 3 / math.log2(2) + 1 / math.log2(3)
    assert rm.ndcg_at_k(grades, SPECS, 3) == pytest.approx(dcg / idcg)
    assert rm.ndcg_at_k([2, 1], SPECS, 2) == pytest.approx(1.0)
    assert rm.ndcg_at_k([], [], 3) == 0.0


ITEM = GoldenItem(
    id="q", question="?", relevant=[RelevantSpec(doc="a.md")], must_include=["65 days", "backups"]
)
CTX = [ScoredChunk(_c("a.md#0"), 1.0), ScoredChunk(_c("z.md#0"), 0.5)]


def test_citation_metrics() -> None:
    ans = Answer("x", ("a.md#0", "z.md#0", "made-up#9"), "m")
    assert gm.citation_validity(ans, CTX) == pytest.approx(2 / 3)
    assert gm.citation_precision(ans, CTX, ITEM) == pytest.approx(1 / 3)
    assert gm.cited_when_answering(ans) == 1.0
    uncited = Answer("x", (), "m")
    assert gm.citation_validity(uncited, CTX) is None
    assert gm.cited_when_answering(uncited) == 0.0
    assert gm.cited_when_answering(Answer("x", (), "m", abstained=True)) is None


def test_fact_recall_and_abstention() -> None:
    assert gm.fact_recall(Answer("Erasure takes 65 DAYS.", (), "m"), ITEM) == 0.5
    unanswerable = GoldenItem(id="u", question="?")
    assert gm.fact_recall(Answer("x", (), "m"), unanswerable) is None
    assert gm.abstention_correct(Answer("x", (), "m", abstained=True), unanswerable) == 1.0
    assert gm.abstention_correct(Answer("x", (), "m", abstained=True), ITEM) == 0.0
    assert gm.abstention_correct(Answer("x", (), "m"), ITEM) == 1.0
