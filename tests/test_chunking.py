from __future__ import annotations

import itertools

import pytest

from rag_eval.embeddings import HashingEmbedder
from rag_eval.ingest.chunking import RecursiveChunker, SemanticChunker, split_sections
from rag_eval.ingest.loader import iter_documents
from rag_eval.types import Document

from .conftest import CORPUS

DOC = Document(
    id="guide.md",
    title="Guide",
    text=(
        "# Guide\n\nIntro paragraph.\n\n## Install\n\n"
        + " ".join(f"Install step {i} requires care." for i in range(40))
        + "\n\n## Usage\n\nRun the tool. Read the output.\n"
    ),
)


def test_split_sections_tracks_heading_path() -> None:
    sections = split_sections(DOC.text)
    assert [s.path for s in sections] == [("Guide",), ("Guide", "Install"), ("Guide", "Usage")]
    assert sections[1].heading == "## Install"


def test_recursive_chunker_respects_size_and_repeats_heading() -> None:
    chunker = RecursiveChunker(chunk_size=200, chunk_overlap=40)
    chunks = chunker.chunk(DOC)
    install = [c for c in chunks if c.metadata["section"] == "Guide > Install"]
    assert len(install) > 1
    for c in install:
        assert c.text.startswith("## Install\n")
        assert len(c.text) <= 200 + len("## Install\n")
    assert [c.ordinal for c in chunks] == list(range(len(chunks)))
    assert chunks[0].id == "guide.md#0"


def test_recursive_chunker_overlaps_consecutive_chunks() -> None:
    chunker = RecursiveChunker(chunk_size=120, chunk_overlap=60)
    parts = chunker.split_text(" ".join(f"word{i}" for i in range(100)))
    assert len(parts) > 2
    for left, right in itertools.pairwise(parts):
        assert left.split()[-1] in right.split()


def test_recursive_chunker_hard_splits_unbroken_text() -> None:
    parts = RecursiveChunker(chunk_size=100, chunk_overlap=10).split_text("x" * 450)
    assert all(len(p) <= 100 for p in parts)
    assert "".join(p[10:] if i else p for i, p in enumerate(parts)).startswith("x" * 400)


def test_overlap_must_be_smaller_than_size() -> None:
    with pytest.raises(ValueError, match="chunk_overlap"):
        RecursiveChunker(chunk_size=100, chunk_overlap=100)


def test_chunk_ids_are_deterministic() -> None:
    chunker = RecursiveChunker(300, 50)
    first = [c.id for d in iter_documents(CORPUS) for c in chunker.chunk(d)]
    second = [c.id for d in iter_documents(CORPUS) for c in chunker.chunk(d)]
    assert first == second
    assert len(first) == len(set(first))


def test_semantic_chunker_splits_on_topic_shift() -> None:
    doc = Document(
        id="mixed.md",
        text=(
            "Postgres backups run nightly. Postgres backups are copied to another region. "
            "Postgres backups are verified quarterly. "
            "Feature flags ramp from one percent. Feature flags expire after thirty days."
        ),
    )
    chunks = SemanticChunker(HashingEmbedder(256), chunk_size=1000, threshold=0.2).chunk(doc)
    assert len(chunks) == 2
    assert "backups" in chunks[0].text
    assert chunks[1].text.startswith("Feature flags")


def test_semantic_chunker_caps_chunk_size() -> None:
    doc = Document(id="d.md", text=" ".join(f"Backups run at hour {i}." for i in range(50)))
    chunks = SemanticChunker(HashingEmbedder(256), chunk_size=150, threshold=-1.0).chunk(doc)
    assert all(len(c.text) <= 150 for c in chunks)
