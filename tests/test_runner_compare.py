from __future__ import annotations

import copy
from collections.abc import Sequence

import pytest

from rag_eval.evals.compare import compare_results, render_comparison
from rag_eval.evals.golden import GoldenSet, load_golden
from rag_eval.evals.judge import OverlapJudge
from rag_eval.evals.report import render_report
from rag_eval.evals.runner import run_eval
from rag_eval.generation import ExtractiveGenerator, LLMError
from rag_eval.retrieval import HybridRetriever
from rag_eval.types import Answer, ScoredChunk

from .conftest import GOLDEN


def test_golden_set_loads_and_rejects_duplicates() -> None:
    golden = load_golden(GOLDEN)
    assert len(golden.items) >= 15
    assert any(not i.answerable for i in golden.items)
    raw = golden.model_dump()
    raw["items"].append(raw["items"][0])
    with pytest.raises(ValueError, match="duplicate"):
        GoldenSet.model_validate(raw)


def test_full_offline_run_produces_sane_metrics(retriever: HybridRetriever) -> None:
    result = run_eval(
        load_golden(GOLDEN),
        retriever,
        generator=ExtractiveGenerator(),
        judge=OverlapJudge(),
        top_k=5,
    )
    m = result["metrics"]
    for name in (
        "retrieval.recall@1",
        "retrieval.recall@5",
        "retrieval.mrr",
        "retrieval.ndcg@5",
        "generation.citation_validity",
        "judge.pass_rate",
        "judge.sanity",
    ):
        assert 0.0 <= m[name] <= 1.0, name
    assert m["retrieval.recall@5"] >= m["retrieval.recall@1"]
    assert m["retrieval.recall@5"] >= 0.9
    assert m["generation.citation_validity"] == 1.0
    assert m["judge.sanity"] >= 0.9
    assert result["run"]["n_errors"] == 0
    report = render_report(result)
    assert "## Metrics" in report
    assert "`retrieval.mrr`" in report


class FlakyGenerator:
    name = "flaky"

    def generate(self, question: str, contexts: Sequence[ScoredChunk]) -> Answer:
        if "SEV1" in question:
            raise LLMError("boom")
        return Answer("text", (contexts[0].chunk.id,), "flaky")


def test_generation_errors_are_recorded_not_fatal(retriever: HybridRetriever) -> None:
    result = run_eval(load_golden(GOLDEN), retriever, generator=FlakyGenerator(), sanity_cases=0)
    assert result["run"]["n_errors"] == 1
    errored = next(i for i in result["items"] if "error" in i)
    assert errored["id"] == "sev1-definition"
    assert "retrieval.mrr" in errored["metrics"]
    assert "## Failures to look at" in render_report(result)


BASE = {"metrics": {"retrieval.mrr": 0.90, "retrieval.recall@5": 0.95, "judge.pass_rate": 0.70}}


def test_compare_flags_regressions_beyond_threshold() -> None:
    current = copy.deepcopy(BASE)
    current["metrics"]["retrieval.mrr"] = 0.89  # within threshold
    current["metrics"]["retrieval.recall@5"] = 0.90  # regression
    current["metrics"]["judge.pass_rate"] = 0.80  # improvement
    cmp = compare_results(BASE, current, threshold=0.02)
    status = {d.name: d.status for d in cmp.deltas}
    assert status == {
        "retrieval.mrr": "ok",
        "retrieval.recall@5": "regressed",
        "judge.pass_rate": "improved",
    }
    assert [d.name for d in cmp.regressions] == ["retrieval.recall@5"]
    assert "**1 regression(s)**" in render_comparison(cmp)


def test_compare_treats_missing_metrics_as_regressions_and_filters_prefixes() -> None:
    current = {"metrics": {"retrieval.mrr": 0.9, "retrieval.recall@5": 0.95, "new.metric": 1.0}}
    cmp = compare_results(BASE, current, threshold=0.02)
    assert {d.name: d.status for d in cmp.deltas}["judge.pass_rate"] == "missing"
    assert {d.name: d.status for d in cmp.deltas}["new.metric"] == "new"
    only_retrieval = compare_results(BASE, current, threshold=0.02, prefixes=("retrieval.",))
    assert only_retrieval.regressions == []
    assert "No regressions." in render_comparison(only_retrieval)
