"""Load markdown / plain-text documents from a directory tree."""

from __future__ import annotations

import re
from collections.abc import Iterator
from pathlib import Path

from rag_eval.types import Document

SUPPORTED_SUFFIXES = frozenset({".md", ".markdown", ".txt"})
_FRONT_MATTER_RE = re.compile(r"\A---\n.*?\n---\n", re.DOTALL)
_H1_RE = re.compile(r"^#\s+(.+?)\s*$", re.MULTILINE)


def _title_for(path: Path, text: str) -> str:
    match = _H1_RE.search(text)
    if match:
        return match.group(1)
    return path.stem.replace("-", " ").replace("_", " ").title()


def load_document(path: Path, root: Path) -> Document:
    raw = path.read_text(encoding="utf-8")
    text = _FRONT_MATTER_RE.sub("", raw).strip()
    doc_id = path.relative_to(root).as_posix()
    return Document(id=doc_id, text=text, title=_title_for(path, text), metadata={"path": doc_id})


def iter_documents(root: Path) -> Iterator[Document]:
    """Yield documents in a stable (sorted) order so chunk ids are reproducible."""
    if root.is_file():
        yield load_document(root, root.parent)
        return
    if not root.is_dir():
        raise FileNotFoundError(f"corpus path does not exist: {root}")
    for path in sorted(root.rglob("*")):
        if path.is_file() and path.suffix.lower() in SUPPORTED_SUFFIXES:
            doc = load_document(path, root)
            if doc.text:
                yield doc
