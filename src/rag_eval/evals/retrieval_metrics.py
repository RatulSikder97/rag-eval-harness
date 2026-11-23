"""Ranking metrics over a retrieved list, given chunk-invariant relevance specs.

Each relevance spec is credited at most once (the first chunk that satisfies it), so a
document split into many chunks cannot inflate recall or push nDCG above 1.
"""

from __future__ import annotations

import math
from collections.abc import Sequence

from rag_eval.evals.golden import RelevantSpec
from rag_eval.types import Chunk


def credited_grades(ranked: Sequence[Chunk], specs: Sequence[RelevantSpec]) -> list[int]:
    """Grade per rank position; 0 if the chunk satisfies no not-yet-credited spec."""
    remaining = list(specs)
    grades: list[int] = []
    for chunk in ranked:
        hit = next((s for s in remaining if s.matches(chunk)), None)
        if hit is None:
            grades.append(0)
        else:
            remaining.remove(hit)
            grades.append(hit.grade)
    return grades


def is_relevant(chunk: Chunk, specs: Sequence[RelevantSpec]) -> bool:
    return any(s.matches(chunk) for s in specs)


def recall_at_k(grades: Sequence[int], n_relevant: int, k: int) -> float:
    if n_relevant == 0:
        return 0.0
    return sum(1 for g in grades[:k] if g > 0) / n_relevant


def reciprocal_rank(ranked: Sequence[Chunk], specs: Sequence[RelevantSpec]) -> float:
    for rank, chunk in enumerate(ranked, start=1):
        if is_relevant(chunk, specs):
            return 1.0 / rank
    return 0.0


def _dcg(grades: Sequence[int]) -> float:
    return float(sum((2**g - 1) / math.log2(i + 2) for i, g in enumerate(grades)))


def ndcg_at_k(grades: Sequence[int], specs: Sequence[RelevantSpec], k: int) -> float:
    ideal = _dcg(sorted((s.grade for s in specs), reverse=True)[:k])
    return _dcg(grades[:k]) / ideal if ideal > 0 else 0.0
