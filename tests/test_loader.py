from __future__ import annotations

from pathlib import Path

import pytest

from rag_eval.ingest.loader import iter_documents

from .conftest import CORPUS


def test_loads_bundled_corpus_in_sorted_order() -> None:
    docs = list(iter_documents(CORPUS))
    ids = [d.id for d in docs]
    assert ids == sorted(ids)
    assert "incident-response.md" in ids
    assert next(d for d in docs if d.id == "incident-response.md").title == "Incident Response"


def test_strips_front_matter_and_skips_unsupported(tmp_path: Path) -> None:
    (tmp_path / "a.md").write_text("---\nowner: x\n---\nBody text", encoding="utf-8")
    (tmp_path / "sub").mkdir()
    (tmp_path / "sub" / "b_notes.txt").write_text("Plain notes", encoding="utf-8")
    (tmp_path / "c.json").write_text("{}", encoding="utf-8")
    (tmp_path / "empty.md").write_text("   ", encoding="utf-8")
    docs = list(iter_documents(tmp_path))
    assert [d.id for d in docs] == ["a.md", "sub/b_notes.txt"]
    assert docs[0].text == "Body text"
    assert docs[1].title == "B Notes"


def test_missing_corpus_raises(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError):
        list(iter_documents(tmp_path / "nope"))
