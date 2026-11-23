"""LLM-as-judge with a fixed rubric and structured JSON verdicts, plus a sanity check.

A judge is itself a model that can drift or be gamed, so the harness measures it too:
``judge_sanity`` scores each item's reference answer against a *negative control* (another
item's reference answer). A judge worth trusting must rank the real reference higher nearly
every time; if this number drops, the judge scores are not evidence of anything.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import asdict, dataclass
from typing import Any, Protocol

from rag_eval.generation.claude import ClaudeClient, Effort, LLMError
from rag_eval.generation.prompts import JUDGE_SCHEMA, JUDGE_SYSTEM, render_judge_prompt
from rag_eval.text import tokenize
from rag_eval.types import ScoredChunk


@dataclass(frozen=True, slots=True)
class JudgeVerdict:
    correctness: int
    groundedness: int
    completeness: int
    passed: bool
    rationale: str

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class Judge(Protocol):
    @property
    def name(self) -> str: ...

    def judge(
        self, question: str, candidate: str, reference: str, contexts: Sequence[ScoredChunk]
    ) -> JudgeVerdict: ...


def _score(value: Any, field: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or not 1 <= value <= 5:
        raise LLMError(f"judge returned invalid {field}: {value!r}")
    return value


class ClaudeJudge:
    def __init__(
        self, client: ClaudeClient, *, model: str = "claude-sonnet-5-5", effort: Effort = "low"
    ) -> None:
        self.client = client
        self.model = model
        self.effort: Effort = effort

    @property
    def name(self) -> str:
        return f"anthropic:{self.model}"

    def judge(
        self, question: str, candidate: str, reference: str, contexts: Sequence[ScoredChunk]
    ) -> JudgeVerdict:
        data, _ = self.client.structured(
            model=self.model,
            system=JUDGE_SYSTEM,
            prompt=render_judge_prompt(question, candidate, reference, contexts),
            schema=JUDGE_SCHEMA,
            effort=self.effort,
        )
        correctness = _score(data.get("correctness"), "correctness")
        groundedness = _score(data.get("groundedness"), "groundedness")
        return JudgeVerdict(
            correctness=correctness,
            groundedness=groundedness,
            completeness=_score(data.get("completeness"), "completeness"),
            # Recompute `passed` from the rubric instead of trusting the model's boolean.
            passed=correctness >= 4 and groundedness >= 4,
            rationale=str(data.get("rationale", "")),
        )


def _to_likert(x: float) -> int:
    return 1 + round(4 * max(0.0, min(1.0, x)))


class OverlapJudge:
    """Deterministic offline judge: lexical overlap mapped onto the same 1-5 rubric."""

    @property
    def name(self) -> str:
        return "fake:overlap"

    def judge(
        self, question: str, candidate: str, reference: str, contexts: Sequence[ScoredChunk]
    ) -> JudgeVerdict:
        cand, ref = set(tokenize(candidate)), set(tokenize(reference))
        support = set(tokenize(" ".join(c.chunk.text for c in contexts)))
        overlap = len(cand & ref)
        precision = overlap / len(cand) if cand else 0.0
        recall = overlap / len(ref) if ref else 0.0
        f1 = 2 * precision * recall / (precision + recall) if overlap else 0.0
        grounded = len(cand & support) / len(cand) if cand else 0.0
        correctness, groundedness = _to_likert(f1 * 1.6), _to_likert(grounded)
        return JudgeVerdict(
            correctness=correctness,
            groundedness=groundedness,
            completeness=_to_likert(recall * 1.4),
            passed=correctness >= 4 and groundedness >= 4,
            rationale=f"token f1={f1:.2f}, grounded={grounded:.2f}",
        )


@dataclass(frozen=True, slots=True)
class SanityCase:
    question: str
    reference: str
    control: str
    contexts: Sequence[ScoredChunk]


def judge_sanity(judge: Judge, cases: Sequence[SanityCase]) -> float | None:
    """Fraction of cases where the judge scores the reference above the negative control."""
    if not cases:
        return None
    wins = 0
    for case in cases:
        real = judge.judge(case.question, case.reference, case.reference, case.contexts)
        fake = judge.judge(case.question, case.control, case.reference, case.contexts)
        wins += int(real.correctness > fake.correctness)
    return wins / len(cases)
