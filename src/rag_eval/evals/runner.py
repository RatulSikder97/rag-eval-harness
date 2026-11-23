"""Run a golden set through retrieval -> generation -> judging and aggregate the metrics."""

from __future__ import annotations

import logging
import statistics
import subprocess
import time
from collections import defaultdict
from collections.abc import Callable, Sequence
from datetime import UTC, datetime
from typing import Any

from rag_eval import __version__
from rag_eval.evals import generation_metrics as gm
from rag_eval.evals import retrieval_metrics as rm
from rag_eval.evals.golden import GoldenItem, GoldenSet
from rag_eval.evals.judge import Judge, SanityCase, judge_sanity
from rag_eval.generation.base import Generator
from rag_eval.generation.claude import LLMError
from rag_eval.retrieval.hybrid import HybridRetriever

SCHEMA_VERSION = 1
DEFAULT_KS: tuple[int, ...] = (1, 3, 5)
log = logging.getLogger(__name__)


def _git_sha() -> str | None:
    try:
        out = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"], capture_output=True, text=True, check=True
        )
    except (OSError, subprocess.CalledProcessError):
        return None
    return out.stdout.strip() or None


def _likert01(score: int) -> float:
    return (score - 1) / 4


def _evaluate_item(
    item: GoldenItem,
    retriever: HybridRetriever,
    generator: Generator | None,
    judge: Judge | None,
    *,
    top_k: int,
    ks: Sequence[int],
) -> dict[str, Any]:
    started = time.perf_counter()
    hits = retriever.retrieve(item.question, max(top_k, *ks))
    retrieval_ms = (time.perf_counter() - started) * 1000
    contexts = hits[:top_k]
    ranked = [h.chunk for h in hits]

    metrics: dict[str, float | None] = {}
    if item.answerable:
        grades = rm.credited_grades(ranked, item.relevant)
        for k in ks:
            metrics[f"retrieval.recall@{k}"] = rm.recall_at_k(grades, len(item.relevant), k)
        metrics["retrieval.mrr"] = rm.reciprocal_rank(ranked, item.relevant)
        metrics[f"retrieval.ndcg@{max(ks)}"] = rm.ndcg_at_k(grades, item.relevant, max(ks))

    record: dict[str, Any] = {
        "id": item.id,
        "question": item.question,
        "tags": item.tags,
        "retrieved": [{"id": h.chunk.id, "score": round(h.score, 6)} for h in hits],
        "timings_ms": {"retrieval": round(retrieval_ms, 3)},
    }

    if generator is not None:
        try:
            started = time.perf_counter()
            answer = generator.generate(item.question, contexts)
            record["timings_ms"]["generation"] = round((time.perf_counter() - started) * 1000, 3)
        except LLMError as exc:
            log.warning("generation failed for %s: %s", item.id, exc)
            record["error"] = f"generation: {exc}"
            record["metrics"] = metrics
            return record
        record["answer"] = {
            "text": answer.text,
            "citations": list(answer.citations),
            "abstained": answer.abstained,
            "model": answer.model,
        }
        metrics["generation.citation_validity"] = gm.citation_validity(answer, contexts)
        metrics["generation.citation_precision"] = gm.citation_precision(answer, contexts, item)
        metrics["generation.cited_when_answering"] = gm.cited_when_answering(answer)
        metrics["generation.fact_recall"] = gm.fact_recall(answer, item)
        metrics["generation.abstention_accuracy"] = gm.abstention_correct(answer, item)

        if judge is not None and item.reference_answer:
            try:
                started = time.perf_counter()
                verdict = judge.judge(item.question, answer.text, item.reference_answer, contexts)
                record["timings_ms"]["judge"] = round((time.perf_counter() - started) * 1000, 3)
            except LLMError as exc:
                log.warning("judge failed for %s: %s", item.id, exc)
                record["error"] = f"judge: {exc}"
            else:
                record["judge"] = verdict.to_dict()
                metrics["judge.correctness"] = _likert01(verdict.correctness)
                metrics["judge.groundedness"] = _likert01(verdict.groundedness)
                metrics["judge.completeness"] = _likert01(verdict.completeness)
                metrics["judge.pass_rate"] = float(verdict.passed)

    record["metrics"] = {k: (round(v, 6) if v is not None else None) for k, v in metrics.items()}
    return record


def _aggregate(records: Sequence[dict[str, Any]]) -> dict[str, float]:
    values: dict[str, list[float]] = defaultdict(list)
    for record in records:
        for name, value in record["metrics"].items():
            if value is not None:
                values[name].append(value)
    return {name: round(statistics.fmean(vs), 6) for name, vs in sorted(values.items())}


def _timing_summary(records: Sequence[dict[str, Any]]) -> dict[str, float]:
    stages: dict[str, list[float]] = defaultdict(list)
    for record in records:
        for stage, ms in record["timings_ms"].items():
            stages[stage].append(ms)
    summary: dict[str, float] = {}
    for stage, ms in sorted(stages.items()):
        ordered = sorted(ms)
        summary[f"{stage}.p50_ms"] = round(statistics.median(ordered), 3)
        summary[f"{stage}.p95_ms"] = round(
            ordered[min(len(ordered) - 1, int(0.95 * len(ordered)))], 3
        )
    return summary


def run_eval(
    golden: GoldenSet,
    retriever: HybridRetriever,
    *,
    generator: Generator | None = None,
    judge: Judge | None = None,
    top_k: int = 5,
    ks: Sequence[int] = DEFAULT_KS,
    sanity_cases: int = 10,
    config: dict[str, Any] | None = None,
    on_item: Callable[[dict[str, Any]], None] | None = None,
) -> dict[str, Any]:
    records: list[dict[str, Any]] = []
    for item in golden.items:
        record = _evaluate_item(item, retriever, generator, judge, top_k=top_k, ks=ks)
        records.append(record)
        if on_item is not None:
            on_item(record)

    metrics = _aggregate(records)
    if judge is not None and sanity_cases > 0:
        with_ref = [i for i in golden.items if i.reference_answer][:sanity_cases]
        cases = [
            SanityCase(
                question=item.question,
                reference=item.reference_answer or "",
                control=with_ref[(n + 1) % len(with_ref)].reference_answer or "",
                contexts=retriever.retrieve(item.question, top_k),
            )
            for n, item in enumerate(with_ref)
            if len(with_ref) > 1
        ]
        try:
            sanity = judge_sanity(judge, cases)
        except LLMError as exc:
            log.warning("judge sanity check failed: %s", exc)
            sanity = None
        if sanity is not None:
            metrics["judge.sanity"] = round(sanity, 6)

    return {
        "schema_version": SCHEMA_VERSION,
        "run": {
            "created_at": datetime.now(UTC).isoformat(timespec="seconds"),
            "harness_version": __version__,
            "git_sha": _git_sha(),
            "golden_set": golden.name,
            "n_items": len(golden.items),
            "n_answerable": sum(1 for i in golden.items if i.answerable),
            "n_errors": sum(1 for r in records if "error" in r),
            "config": config or {},
        },
        "metrics": dict(sorted(metrics.items())),
        "timings": _timing_summary(records),
        "items": records,
    }
