"""Core value types shared across ingestion, retrieval, generation and evaluation."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass(frozen=True, slots=True)
class Document:
    """A source document. ``id`` is the corpus-relative path and is stable across runs."""

    id: str
    text: str
    title: str = ""
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class Chunk:
    """A retrievable unit of text.

    ``id`` is ``"<doc_id>#<ordinal>"`` so citations stay human-readable and deterministic.
    """

    id: str
    doc_id: str
    text: str
    ordinal: int
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class ScoredChunk:
    chunk: Chunk
    score: float
    source: str = ""


@dataclass(frozen=True, slots=True)
class Answer:
    text: str
    citations: tuple[str, ...]
    model: str
    abstained: bool = False
