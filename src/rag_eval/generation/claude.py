"""Claude-backed generation via the official ``anthropic`` SDK, using structured outputs."""

from __future__ import annotations

import json
from collections.abc import Sequence
from typing import Any, Literal

import anthropic
from anthropic.types.beta import BetaTextBlock

from rag_eval.generation.prompts import ANSWER_SCHEMA, ANSWER_SYSTEM, render_answer_prompt
from rag_eval.types import Answer, ScoredChunk

Effort = Literal["low", "medium", "high", "xhigh", "max"]
FALLBACK_BETA = "server-side-fallback-2026-07-01"


class LLMError(RuntimeError):
    """A Claude call failed in a way the caller should surface (not silently score as 0)."""


class ClaudeClient:
    """Thin wrapper: one place for retries, timeouts, refusal handling and JSON parsing.

    Retries for 408/409/429/5xx and connection errors are delegated to the SDK
    (``max_retries`` with exponential backoff and ``retry-after`` support).
    """

    def __init__(
        self,
        *,
        timeout_s: float = 120.0,
        max_retries: int = 4,
        max_tokens: int = 4096,
        server_side_fallback: bool = True,
        client: anthropic.Anthropic | None = None,
    ) -> None:
        self._client = client or anthropic.Anthropic(timeout=timeout_s, max_retries=max_retries)
        self.max_tokens = max_tokens
        self.server_side_fallback = server_side_fallback

    def structured(
        self,
        *,
        model: str,
        system: str,
        prompt: str,
        schema: dict[str, Any],
        effort: Effort = "medium",
    ) -> tuple[dict[str, Any], str]:
        """Return ``(parsed_json, served_model)`` for a JSON-schema-constrained response."""
        extra: dict[str, Any] = {}
        if self.server_side_fallback:
            # On a safety decline the API re-runs the request on a fallback model in-call.
            extra = {"betas": [FALLBACK_BETA], "fallbacks": "default"}
        try:
            response = self._client.beta.messages.create(
                model=model,
                max_tokens=self.max_tokens,
                system=system,
                messages=[{"role": "user", "content": prompt}],
                output_config={
                    "effort": effort,
                    "format": {"type": "json_schema", "schema": schema},
                },
                **extra,
            )
        except anthropic.RateLimitError as exc:
            raise LLMError(f"rate limited after retries (request {exc.request_id})") from exc
        except anthropic.APIStatusError as exc:
            raise LLMError(
                f"Claude API error {exc.status_code} (request {exc.request_id}): {exc.message}"
            ) from exc
        except anthropic.APIConnectionError as exc:
            raise LLMError(f"could not reach the Claude API: {exc}") from exc
        except anthropic.AnthropicError as exc:
            raise LLMError(f"Claude client error: {exc}") from exc
        except TypeError as exc:
            # The SDK raises TypeError when no credential source resolves at request time.
            if "authentication" not in str(exc):
                raise
            raise LLMError(
                "no Anthropic credentials found; set ANTHROPIC_API_KEY (see .env.example)"
            ) from exc

        if response.stop_reason == "refusal":
            category = response.stop_details.category if response.stop_details else None
            raise LLMError(f"request declined by {response.model} (category={category})")
        if response.stop_reason == "max_tokens":
            raise LLMError(f"response truncated at max_tokens={self.max_tokens}")
        text = next((b.text for b in response.content if isinstance(b, BetaTextBlock)), None)
        if text is None:
            raise LLMError(f"no text block in response (stop_reason={response.stop_reason})")
        try:
            data: dict[str, Any] = json.loads(text)
        except json.JSONDecodeError as exc:
            raise LLMError(f"structured output was not valid JSON: {text[:200]!r}") from exc
        return data, response.model


class ClaudeGenerator:
    def __init__(
        self, client: ClaudeClient, *, model: str = "claude-sonnet-5-5", effort: Effort = "medium"
    ) -> None:
        self.client = client
        self.model = model
        self.effort: Effort = effort

    @property
    def name(self) -> str:
        return f"anthropic:{self.model}"

    def generate(self, question: str, contexts: Sequence[ScoredChunk]) -> Answer:
        data, served_by = self.client.structured(
            model=self.model,
            system=ANSWER_SYSTEM,
            prompt=render_answer_prompt(question, contexts),
            schema=ANSWER_SCHEMA,
            effort=self.effort,
        )
        return Answer(
            text=str(data["answer"]).strip(),
            citations=tuple(dict.fromkeys(str(c) for c in data["citations"])),
            model=served_by,
            abstained=bool(data["abstained"]),
        )
