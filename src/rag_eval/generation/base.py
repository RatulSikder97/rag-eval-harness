"""Answer-generation interface."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Protocol

from rag_eval.types import Answer, ScoredChunk


class Generator(Protocol):
    @property
    def name(self) -> str: ...

    def generate(self, question: str, contexts: Sequence[ScoredChunk]) -> Answer: ...
