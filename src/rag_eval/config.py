"""Runtime configuration, resolved from ``RAG_*`` environment variables (and ``.env``)."""

from __future__ import annotations

from typing import Literal

from pydantic import Field, SecretStr, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

EmbedderName = Literal["hashing", "openai", "voyage"]
StoreName = Literal["memory", "pgvector"]
LLMName = Literal["fake", "anthropic"]
ChunkerName = Literal["recursive", "semantic"]
Effort = Literal["low", "medium", "high", "xhigh", "max"]


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_prefix="RAG_", env_file=".env", env_file_encoding="utf-8", extra="ignore"
    )

    # Ingestion
    chunker: ChunkerName = "recursive"
    chunk_size: int = Field(default=800, gt=50, description="Max characters per chunk.")
    chunk_overlap: int = Field(default=120, ge=0)
    semantic_threshold: float = Field(default=0.25, ge=-1.0, le=1.0)

    # Embeddings
    embedder: EmbedderName = "hashing"
    embedding_dim: int = Field(default=512, gt=0)
    openai_base_url: str = "https://api.openai.com/v1"
    openai_embedding_model: str = "text-embedding-3-small"
    openai_api_key: SecretStr | None = None
    voyage_embedding_model: str = "voyage-3.5"
    voyage_api_key: SecretStr | None = None

    # Store
    store: StoreName = "memory"
    index_path: str = ".rag-index.json"
    pg_dsn: str = "postgresql://rag:rag@localhost:5433/rag"
    pg_table: str = "rag_chunks"

    # Retrieval
    top_k: int = Field(default=5, gt=0)
    candidate_k: int = Field(default=20, gt=0, description="Per-retriever depth before fusion.")
    rrf_k: int = Field(default=60, gt=0)

    # Generation / judging
    llm: LLMName = "fake"
    generation_model: str = "claude-sonnet-5-5"
    judge_model: str = "claude-sonnet-5-5"
    generation_effort: Effort = "medium"
    judge_effort: Effort = "low"
    max_tokens: int = Field(default=4096, gt=0)
    request_timeout_s: float = Field(default=120.0, gt=0)
    max_retries: int = Field(default=4, ge=0)

    @model_validator(mode="after")
    def _check_chunking(self) -> Settings:
        if self.chunk_overlap >= self.chunk_size:
            raise ValueError("chunk_overlap must be smaller than chunk_size")
        return self
