from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol, runtime_checkable


@dataclass(frozen=True)
class Message:
    # role is "user" or "assistant"; the system prompt is passed separately to
    # LLMProvider.complete and prepended by the provider.
    role: str
    content: str


@dataclass(frozen=True)
class Completion:
    text: str
    prompt_tokens: int
    completion_tokens: int
    raw: dict


@runtime_checkable
class LLMProvider(Protocol):
    async def complete(
        self,
        *,
        system: str,
        messages: list[Message],
        temperature: float,
        max_tokens: int,
    ) -> Completion: ...

    async def aclose(self) -> None: ...


class ProviderError(Exception):
    """Base for all provider-layer failures."""


class ProviderHTTPError(ProviderError):
    """Non-retryable HTTP error (4xx other than 429)."""

    def __init__(self, status_code: int, body: str):
        super().__init__(f"HTTP {status_code}: {body[:500]}")
        self.status_code = status_code
        self.body = body


class ProviderUnavailable(ProviderError):
    """Retryable failure that persisted after exhausting retries (429/5xx/network/timeout)."""


class ProviderParseError(ProviderError):
    """Response arrived but could not be parsed into a Completion."""
