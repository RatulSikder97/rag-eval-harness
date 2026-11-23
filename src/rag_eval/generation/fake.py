"""Deterministic extractive generator used in tests and offline CI.

It is a *real* baseline, not a stub: it picks the sentences with the highest query-term
overlap from the retrieved chunks and cites where they came from. If retrieval gets worse,
its answers get worse, so the generation metrics in CI still move with retrieval quality.
"""

from __future__ import annotations

from collections.abc import Sequence

from rag_eval.text import split_sentences, tokenize
from rag_eval.types import Answer, ScoredChunk


class ExtractiveGenerator:
    def __init__(self, *, max_sentences: int = 2, min_overlap: int = 1) -> None:
        self.max_sentences = max_sentences
        self.min_overlap = min_overlap

    @property
    def name(self) -> str:
        return "fake:extractive"

    def generate(self, question: str, contexts: Sequence[ScoredChunk]) -> Answer:
        terms = set(tokenize(question))
        candidates: list[tuple[int, int, int, str, str]] = []
        for rank, hit in enumerate(contexts):
            for pos, sentence in enumerate(split_sentences(hit.chunk.text)):
                if sentence.startswith("#"):
                    continue
                overlap = len(terms & set(tokenize(sentence)))
                if overlap >= self.min_overlap:
                    candidates.append((-overlap, rank, pos, sentence, hit.chunk.id))
        if not candidates:
            return Answer(
                text="The retrieved documents do not answer this question.",
                citations=(),
                model=self.name,
                abstained=True,
            )
        best = sorted(candidates)[: self.max_sentences]
        best.sort(key=lambda c: (c[1], c[2]))
        text = " ".join(c[3].lstrip("-* ").strip() for c in best)
        citations = tuple(dict.fromkeys(c[4] for c in best))
        return Answer(text=text, citations=citations, model=self.name)
