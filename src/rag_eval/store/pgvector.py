"""Postgres + pgvector store.

* Dense search: ``vector`` column with an HNSW index on cosine distance.
* Lexical search: a generated ``tsvector`` column with a GIN index, queried as an OR of the
  query terms and ranked with ``ts_rank_cd`` (Postgres' stand-in for BM25).

Fusion happens in the retriever, so the store stays a thin, testable adapter.
"""

from __future__ import annotations

import json
import re
from collections.abc import Sequence
from typing import Any

import numpy as np
import psycopg
from pgvector.psycopg import register_vector
from psycopg import sql

from rag_eval.text import tokenize
from rag_eval.types import Chunk, ScoredChunk

_IDENT_RE = re.compile(r"^[a-z_][a-z0-9_]{0,62}$")


class PgVectorStore:
    def __init__(
        self,
        dsn: str,
        dim: int,
        *,
        table: str = "rag_chunks",
        hnsw_m: int = 16,
        hnsw_ef_construction: int = 64,
        connect_timeout_s: int = 5,
    ) -> None:
        if not _IDENT_RE.match(table):
            raise ValueError(f"invalid table name: {table!r}")
        self.dim = dim
        self.table = table
        self._conn = psycopg.connect(dsn, autocommit=True, connect_timeout=connect_timeout_s)
        self._conn.execute("CREATE EXTENSION IF NOT EXISTS vector")
        register_vector(self._conn)
        self._migrate(hnsw_m, hnsw_ef_construction)

    # -- schema ---------------------------------------------------------------------------

    def _migrate(self, m: int, ef_construction: int) -> None:
        t = sql.Identifier(self.table)
        self._conn.execute(
            sql.SQL(
                """
                CREATE TABLE IF NOT EXISTS {t} (
                    id         text PRIMARY KEY,
                    doc_id     text NOT NULL,
                    ordinal    integer NOT NULL,
                    text       text NOT NULL,
                    metadata   jsonb NOT NULL DEFAULT '{{}}'::jsonb,
                    embedding  vector({dim}) NOT NULL,
                    tsv        tsvector GENERATED ALWAYS AS (to_tsvector('english', text)) STORED
                )
                """
            ).format(t=t, dim=sql.Literal(self.dim))
        )
        existing = self._conn.execute(
            """
            SELECT a.atttypmod FROM pg_attribute a
            WHERE a.attrelid = %s::regclass AND a.attname = 'embedding'
            """,
            (self.table,),
        ).fetchone()
        if existing and existing[0] != self.dim:
            raise ValueError(
                f"table {self.table} stores vector({existing[0]}) but the embedder produces "
                f"{self.dim} dims; use a different RAG_PG_TABLE or drop the table"
            )
        self._conn.execute(
            sql.SQL(
                "CREATE INDEX IF NOT EXISTS {idx} ON {t} USING hnsw (embedding vector_cosine_ops) "
                "WITH (m = {m}, ef_construction = {ef})"
            ).format(
                idx=sql.Identifier(f"{self.table}_embedding_hnsw"),
                t=t,
                m=sql.Literal(m),
                ef=sql.Literal(ef_construction),
            )
        )
        self._conn.execute(
            sql.SQL("CREATE INDEX IF NOT EXISTS {idx} ON {t} USING gin (tsv)").format(
                idx=sql.Identifier(f"{self.table}_tsv_gin"), t=t
            )
        )
        self._conn.execute(
            sql.SQL("CREATE INDEX IF NOT EXISTS {idx} ON {t} (doc_id)").format(
                idx=sql.Identifier(f"{self.table}_doc_id"), t=t
            )
        )

    # -- writes ---------------------------------------------------------------------------

    def add(self, chunks: Sequence[Chunk], embeddings: Sequence[Sequence[float]]) -> None:
        if len(chunks) != len(embeddings):
            raise ValueError("chunks and embeddings must have the same length")
        if not chunks:
            return
        query = sql.SQL(
            """
            INSERT INTO {t} (id, doc_id, ordinal, text, metadata, embedding)
            VALUES (%s, %s, %s, %s, %s::jsonb, %s)
            ON CONFLICT (id) DO UPDATE SET
                doc_id = EXCLUDED.doc_id, ordinal = EXCLUDED.ordinal, text = EXCLUDED.text,
                metadata = EXCLUDED.metadata, embedding = EXCLUDED.embedding
            """
        ).format(t=sql.Identifier(self.table))
        rows = [
            (c.id, c.doc_id, c.ordinal, c.text, json.dumps(c.metadata), np.asarray(e, np.float32))
            for c, e in zip(chunks, embeddings, strict=True)
        ]
        with self._conn.transaction(), self._conn.cursor() as cur:
            cur.executemany(query, rows)

    def clear(self) -> None:
        self._conn.execute(sql.SQL("TRUNCATE {t}").format(t=sql.Identifier(self.table)))

    # -- reads ----------------------------------------------------------------------------

    @staticmethod
    def _to_chunk(row: tuple[Any, ...]) -> Chunk:
        return Chunk(id=row[0], doc_id=row[1], ordinal=row[2], text=row[3], metadata=row[4])

    def vector_search(self, embedding: Sequence[float], k: int) -> list[ScoredChunk]:
        with self._conn.transaction(), self._conn.cursor() as cur:
            # ef_search must be >= k or HNSW silently returns fewer rows.
            cur.execute(sql.SQL("SET LOCAL hnsw.ef_search = {n}").format(n=sql.Literal(max(40, k))))
            cur.execute(
                sql.SQL(
                    """
                    SELECT id, doc_id, ordinal, text, metadata, 1 - (embedding <=> %s) AS score
                    FROM {t} ORDER BY embedding <=> %s LIMIT %s
                    """
                ).format(t=sql.Identifier(self.table)),
                (np.asarray(embedding, np.float32), np.asarray(embedding, np.float32), k),
            )
            rows = cur.fetchall()
        return [ScoredChunk(self._to_chunk(r), float(r[5]), "vector") for r in rows]

    def keyword_search(self, query: str, k: int) -> list[ScoredChunk]:
        # OR the terms: websearch_to_tsquery ANDs everything, which is far too strict for
        # natural-language questions. Terms are [a-z0-9] after splitting, so this is safe.
        terms = sorted({part for tok in tokenize(query) for part in re.split(r"[-_]", tok) if part})
        if not terms:
            return []
        rows = self._conn.execute(
            sql.SQL(
                """
                SELECT id, doc_id, ordinal, text, metadata, ts_rank_cd(tsv, q) AS score
                FROM {t}, to_tsquery('english', %s) AS q
                WHERE tsv @@ q
                ORDER BY score DESC, id
                LIMIT %s
                """
            ).format(t=sql.Identifier(self.table)),
            (" | ".join(terms), k),
        ).fetchall()
        return [ScoredChunk(self._to_chunk(r), float(r[5]), "keyword") for r in rows]

    def count(self) -> int:
        row = self._conn.execute(
            sql.SQL("SELECT count(*) FROM {t}").format(t=sql.Identifier(self.table))
        ).fetchone()
        return int(row[0]) if row else 0

    def close(self) -> None:
        self._conn.close()
