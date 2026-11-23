"""Golden-set schema.

Relevance is labelled at the *document + evidence* level, not by chunk id: a chunk is
relevant if it comes from ``doc`` and (optionally) contains the ``contains`` evidence string.
That keeps the golden set valid when the chunker or chunk size changes — which is exactly
the kind of change you want this harness to evaluate.
"""

from __future__ import annotations

from pathlib import Path

import yaml
from pydantic import BaseModel, ConfigDict, Field, model_validator

from rag_eval.types import Chunk


class RelevantSpec(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    doc: str
    contains: str | None = None
    grade: int = Field(default=1, ge=1, le=3)

    def matches(self, chunk: Chunk) -> bool:
        if chunk.doc_id != self.doc:
            return False
        return self.contains is None or self.contains.lower() in chunk.text.lower()


class GoldenItem(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    id: str
    question: str
    relevant: list[RelevantSpec] = Field(default_factory=list)
    reference_answer: str | None = None
    must_include: list[str] = Field(default_factory=list)
    tags: list[str] = Field(default_factory=list)

    @property
    def answerable(self) -> bool:
        return bool(self.relevant)


class GoldenSet(BaseModel):
    model_config = ConfigDict(extra="forbid")

    name: str
    description: str = ""
    items: list[GoldenItem]

    @model_validator(mode="after")
    def _unique_ids(self) -> GoldenSet:
        seen: set[str] = set()
        for item in self.items:
            if item.id in seen:
                raise ValueError(f"duplicate golden item id: {item.id}")
            seen.add(item.id)
        return self


def load_golden(path: Path) -> GoldenSet:
    with path.open(encoding="utf-8") as fh:
        return GoldenSet.model_validate(yaml.safe_load(fh))
