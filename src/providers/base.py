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
    """Один HTTP-запрос к провайдеру (для L2-лога: пишем КАЖДЫЙ, включая сетевые ретраи).

    Финальная удачная попытка несёт `response` (извлечённый текст) и токены; у
    промежуточных ретраев `response=None`, токены 0. `response_raw` — дословное тело
    (`resp.text`), включая тело 5xx; None при сетевой ошибке (ответа нет).

    Attributes:
        status: ok | parse_error | bad_json | bad_shape | http_error | server_error | network.
        status_code: HTTP-код попытки (None при сетевой ошибке).
        request: Дословный отправленный payload.
        response: Извлечённый текст (только на финальной ok-попытке).
        response_raw: Дословное тело ответа строкой (resp.text).
        error: Сообщение сбоя (None при успехе).
        prompt_tokens: Токены промпта (на финальной ok-попытке).
        completion_tokens: Токены ответа (на финальной ok-попытке).
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
    request: dict | None = None     # дословный payload, отправленный провайдеру (для L2-лога)
    attempts: tuple[HttpAttempt, ...] = ()   # все HTTP-попытки этого complete() (вкл. ретраи)


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

    Несёт сырьё сбойного вызова для L2-лога: что отправили (`request`) и все HTTP-попытки
    (`attempts`, включая ретраи и терминальную). Заполняется на месте `raise` в провайдере;
    верхние слои дополняют контекст (`agent_id`/`phase`/`attempt`) и разворачивают попытки
    в `calls` (LLMCall). Атрибуты-дефолты, чтобы их всегда можно было прочитать.
    """

    request: dict | None = None
    attempts: tuple = ()            # HttpAttempt'ы (ретраи + терминальная)
    # дополняются верхними слоями (Agent.act / игра) перед повторным raise:
    agent_id: str | None = None
    phase: str | None = None
    attempt: int | None = None
    calls: tuple = ()               # LLMCall'ы этого act() (для L2-лога)


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
