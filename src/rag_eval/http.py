"""Shared HTTP helper with bounded exponential backoff for provider APIs."""

from __future__ import annotations

import random
import time
from typing import Any

import httpx

RETRYABLE_STATUS = frozenset({408, 409, 429, 500, 502, 503, 504, 529})


class ProviderError(RuntimeError):
    """A provider call failed after retries (or with a non-retryable status)."""


def post_json(
    client: httpx.Client,
    url: str,
    payload: dict[str, Any],
    *,
    max_retries: int = 4,
    base_delay_s: float = 0.5,
) -> dict[str, Any]:
    for attempt in range(max_retries + 1):
        try:
            response = client.post(url, json=payload)
        except httpx.TransportError as exc:
            if attempt == max_retries:
                raise ProviderError(f"POST {url} failed: {exc}") from exc
        else:
            if response.status_code < 400:
                data: dict[str, Any] = response.json()
                return data
            if response.status_code not in RETRYABLE_STATUS or attempt == max_retries:
                raise ProviderError(f"POST {url} -> {response.status_code}: {response.text[:300]}")
            retry_after = response.headers.get("retry-after")
            if retry_after and retry_after.replace(".", "", 1).isdigit():
                time.sleep(min(float(retry_after), 30.0))
                continue
        time.sleep(min(base_delay_s * 2**attempt, 20.0) * (0.5 + random.random() / 2))
    raise ProviderError(f"POST {url} exhausted retries")  # pragma: no cover
