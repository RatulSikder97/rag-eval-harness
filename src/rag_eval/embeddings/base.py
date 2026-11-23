"""Embedding provider interface."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Protocol, runtime_checkable


@runtime_checkable
class Embedder(Protocol):
    """Maps text to fixed-size, L2-normalised vectors.

    Documents and queries are separate calls because asymmetric models (e.g. Voyage)
    embed them differently.
    """

    @property
    def name(self) -> str: ...

    @property
    def dim(self) -> int: ...

    def embed_documents(self, texts: Sequence[str]) -> list[list[float]]: ...

    def embed_query(self, text: str) -> list[float]: ...
