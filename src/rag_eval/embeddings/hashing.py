"""Deterministic, zero-dependency embedder for tests and CI.

Signed feature hashing over word unigrams, word bigrams and character trigrams. It has no
semantics beyond lexical overlap, but it is stable across machines and Python versions
(blake2b, not ``hash()``), which is exactly what a regression baseline needs.
"""

from __future__ import annotations

import hashlib
import itertools
import math
from collections.abc import Sequence

from rag_eval.text import tokenize


class HashingEmbedder:
    def __init__(self, dim: int = 512) -> None:
        if dim <= 0:
            raise ValueError("dim must be positive")
        self._dim = dim

    @property
    def name(self) -> str:
        return f"hashing-{self._dim}"

    @property
    def dim(self) -> int:
        return self._dim

    def _features(self, text: str) -> list[tuple[str, float]]:
        words = tokenize(text)
        feats: list[tuple[str, float]] = [(f"w:{w}", 1.0) for w in words]
        feats += [(f"b:{a}_{b}", 0.5) for a, b in itertools.pairwise(words)]
        for w in words:
            padded = f"^{w}$"
            feats += [(f"c:{padded[i : i + 3]}", 0.25) for i in range(len(padded) - 2)]
        return feats

    def _embed(self, text: str) -> list[float]:
        vec = [0.0] * self._dim
        for feature, weight in self._features(text):
            digest = hashlib.blake2b(feature.encode(), digest_size=8).digest()
            value = int.from_bytes(digest, "little")
            index = value % self._dim
            sign = 1.0 if (value >> 63) & 1 else -1.0
            vec[index] += sign * weight
        norm = math.sqrt(sum(v * v for v in vec))
        return [v / norm for v in vec] if norm else vec

    def embed_documents(self, texts: Sequence[str]) -> list[list[float]]:
        return [self._embed(t) for t in texts]

    def embed_query(self, text: str) -> list[float]:
        return self._embed(text)
