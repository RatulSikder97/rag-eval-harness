"""Chunking strategies.

Both chunkers are *section-aware*: markdown headings are hard boundaries and the section
heading is repeated at the top of every continuation chunk ("contextual chunk headers"), so
a chunk retrieved in isolation still says what it is about.
"""

from __future__ import annotations

import re
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Protocol

import numpy as np

from rag_eval.embeddings.base import Embedder
from rag_eval.text import split_sentences
from rag_eval.types import Chunk, Document

_HEADING_RE = re.compile(r"^(#{1,6})\s+(.+?)\s*$")

DEFAULT_SEPARATORS: tuple[str, ...] = ("\n\n", "\n", ". ", "; ", ", ", " ")


class Chunker(Protocol):
    def chunk(self, doc: Document) -> list[Chunk]: ...


@dataclass(frozen=True, slots=True)
class Section:
    heading: str
    path: tuple[str, ...]
    body: str


def split_sections(text: str) -> list[Section]:
    """Split markdown into heading-delimited sections, tracking the heading path."""
    sections: list[Section] = []
    stack: list[tuple[int, str]] = []
    heading = ""
    buf: list[str] = []

    def flush() -> None:
        body = "\n".join(buf).strip()
        if body:
            sections.append(Section(heading, tuple(h for _, h in stack), body))
        buf.clear()

    for line in text.splitlines():
        match = _HEADING_RE.match(line)
        if match:
            flush()
            level, title = len(match.group(1)), match.group(2)
            while stack and stack[-1][0] >= level:
                stack.pop()
            stack.append((level, title))
            heading = line.strip()
            continue
        buf.append(line)
    flush()
    return sections


def _build_chunks(doc: Document, pieces: Sequence[tuple[Section, str]]) -> list[Chunk]:
    chunks: list[Chunk] = []
    for ordinal, (section, body) in enumerate(pieces):
        text = f"{section.heading}\n{body}" if section.heading else body
        chunks.append(
            Chunk(
                id=f"{doc.id}#{ordinal}",
                doc_id=doc.id,
                text=text,
                ordinal=ordinal,
                metadata={"title": doc.title, "section": " > ".join(section.path)},
            )
        )
    return chunks


class RecursiveChunker:
    """Recursive character splitter: try coarse separators first, fall back to finer ones."""

    def __init__(
        self,
        chunk_size: int = 800,
        chunk_overlap: int = 120,
        separators: Sequence[str] = DEFAULT_SEPARATORS,
    ) -> None:
        if chunk_overlap >= chunk_size:
            raise ValueError("chunk_overlap must be smaller than chunk_size")
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.separators = tuple(separators)

    def chunk(self, doc: Document) -> list[Chunk]:
        pieces = [(s, part) for s in split_sections(doc.text) for part in self.split_text(s.body)]
        return _build_chunks(doc, pieces)

    def split_text(self, text: str) -> list[str]:
        return [c.strip() for c in self._split(text, self.separators) if c.strip()]

    def _split(self, text: str, separators: tuple[str, ...]) -> list[str]:
        if len(text) <= self.chunk_size:
            return [text]
        if not separators:
            step = self.chunk_size - self.chunk_overlap
            return [text[i : i + self.chunk_size] for i in range(0, len(text), step)]

        sep, rest = separators[0], separators[1:]
        if sep not in text:
            return self._split(text, rest)

        parts = [p + sep for p in text.split(sep)]
        parts[-1] = parts[-1][: -len(sep)]
        units: list[str] = []
        for part in parts:
            units.extend([part] if len(part) <= self.chunk_size else self._split(part, rest))
        return self._merge(units)

    def _merge(self, units: list[str]) -> list[str]:
        """Greedily pack units up to ``chunk_size``, carrying a tail of ``chunk_overlap``."""
        chunks: list[str] = []
        window: list[str] = []
        length = 0
        for unit in units:
            if window and length + len(unit) > self.chunk_size:
                chunks.append("".join(window))
                while window and (
                    length > self.chunk_overlap or length + len(unit) > self.chunk_size
                ):
                    length -= len(window.pop(0))
            window.append(unit)
            length += len(unit)
        if window:
            chunks.append("".join(window))
        return chunks


class SemanticChunker:
    """Sentence-level chunker that breaks where meaning shifts.

    Sentences are embedded once; a new chunk starts when the next sentence's cosine
    similarity to the running chunk centroid drops below ``threshold`` or the chunk would
    exceed ``chunk_size`` characters. Section headings remain hard boundaries.
    """

    def __init__(self, embedder: Embedder, chunk_size: int = 800, threshold: float = 0.25) -> None:
        self.embedder = embedder
        self.chunk_size = chunk_size
        self.threshold = threshold

    def chunk(self, doc: Document) -> list[Chunk]:
        pieces: list[tuple[Section, str]] = []
        for section in split_sections(doc.text):
            sentences = split_sentences(section.body)
            if not sentences:
                continue
            vectors = np.asarray(self.embedder.embed_documents(sentences), dtype=np.float64)
            pieces.extend((section, group) for group in self._group(sentences, vectors))
        return _build_chunks(doc, pieces)

    def _group(self, sentences: list[str], vectors: np.ndarray) -> list[str]:
        groups: list[str] = []
        current: list[str] = [sentences[0]]
        centroid = vectors[0].copy()
        for sentence, vec in zip(sentences[1:], vectors[1:], strict=True):
            norm = float(np.linalg.norm(centroid) * np.linalg.norm(vec)) or 1.0
            similarity = float(centroid @ vec) / norm
            size = sum(len(s) + 1 for s in current) + len(sentence)
            if similarity < self.threshold or size > self.chunk_size:
                groups.append(" ".join(current))
                current, centroid = [sentence], vec.copy()
            else:
                current.append(sentence)
                centroid = centroid + vec
        groups.append(" ".join(current))
        return groups
