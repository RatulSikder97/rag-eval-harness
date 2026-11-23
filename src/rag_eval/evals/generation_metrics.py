"""Deterministic generation checks: citation faithfulness, fact recall, abstention."""

from __future__ import annotations

from collections.abc import Sequence

from rag_eval.evals.golden import GoldenItem
from rag_eval.evals.retrieval_metrics import is_relevant
from rag_eval.types import Answer, ScoredChunk


def citation_validity(answer: Answer, contexts: Sequence[ScoredChunk]) -> float | None:
    """Share of cited ids that were actually in the prompt. Catches fabricated citations."""
    if not answer.citations:
        return None
    shown = {c.chunk.id for c in contexts}
    return sum(1 for cid in answer.citations if cid in shown) / len(answer.citations)


def citation_precision(
    answer: Answer, contexts: Sequence[ScoredChunk], item: GoldenItem
) -> float | None:
    """Share of citations that point at gold-relevant chunks."""
    if not answer.citations or not item.answerable:
        return None
    by_id = {c.chunk.id: c.chunk for c in contexts}
    cited = [by_id[cid] for cid in answer.citations if cid in by_id]
    return sum(1 for c in cited if is_relevant(c, item.relevant)) / len(answer.citations)


def cited_when_answering(answer: Answer) -> float | None:
    """1.0 if a non-abstaining answer cites at least one chunk, else 0.0."""
    if answer.abstained:
        return None
    return 1.0 if answer.citations else 0.0


def fact_recall(answer: Answer, item: GoldenItem) -> float | None:
    if not item.must_include:
        return None
    text = answer.text.lower()
    return sum(1 for fact in item.must_include if fact.lower() in text) / len(item.must_include)


def abstention_correct(answer: Answer, item: GoldenItem) -> float:
    """Answerable items should be answered; unanswerable ones should be declined."""
    return 1.0 if answer.abstained != item.answerable else 0.0
