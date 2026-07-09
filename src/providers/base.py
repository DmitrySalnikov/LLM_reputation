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
class HttpAttempt:
    """A single HTTP request to the provider (for the L2 log: we record EVERY one, including network retries).

    The final successful attempt carries `response` (the extracted text) and token counts;
    intermediate retries have `response=None` and 0 tokens. `response_raw` is the verbatim
    body (`resp.text`), including a 5xx body; None on a network error (no response).

    Attributes:
        status: ok | parse_error | bad_json | bad_shape | http_error | server_error | network.
        status_code: HTTP code of the attempt (None on a network error).
        request: The verbatim payload that was sent.
        response: The extracted text (only on the final ok attempt).
        response_raw: The verbatim response body as a string (resp.text).
        error: Failure message (None on success).
        prompt_tokens: Prompt tokens (on the final ok attempt).
        completion_tokens: Completion tokens (on the final ok attempt).
    """

    status: str
    status_code: int | None
    request: dict
    response: str | None
    response_raw: str | None
    error: str | None
    prompt_tokens: int = 0
    completion_tokens: int = 0


@dataclass(frozen=True)
class Completion:
    text: str
    prompt_tokens: int
    completion_tokens: int
    raw: dict
    request: dict | None = None     # the exact payload sent to the provider (for the L2 log)
    attempts: tuple[HttpAttempt, ...] = ()   # all HTTP attempts of this complete() (incl. retries)


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
    """Base for all provider-layer failures.

    Carries the raw material of the failed call for the L2 log: what was sent (`request`) and
    all HTTP attempts (`attempts`, including retries and the terminal one). Populated in place
    at the `raise` site in the provider; upper layers add context (`agent_id`/`phase`/`attempt`)
    and unpack the attempts into `calls` (LLMCall). Attributes have defaults so they can always
    be read.
    """

    request: dict | None = None
    attempts: tuple = ()            # HttpAttempts (retries + the terminal one)
    # filled in by the upper layers (Agent.act / the game) before re-raising:
    agent_id: str | None = None
    phase: str | None = None
    attempt: int | None = None
    calls: tuple = ()               # LLMCalls of this act() (for the L2 log)


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
