from __future__ import annotations

import asyncio
import os
import random

import httpx

from src.core.config import ProviderCfg
from src.providers.base import (
    Completion,
    LLMProvider,
    Message,
    ProviderHTTPError,
    ProviderParseError,
    ProviderUnavailable,
)

_RETRY_BASE_S = 1.0
_RETRY_CAP_S = 30.0
_MAX_ATTEMPTS = 5


class OpenAICompatibleProvider:
    """Talks to any OpenAI-compatible /chat/completions endpoint (Ollama, OpenAI,
    Cerebras, Gemini, ...). Retries 429/5xx/network with exponential backoff."""

    def __init__(
        self,
        base_url: str,
        api_key: str,
        model: str,
        *,
        timeout_s: float = 120.0,
        client: httpx.AsyncClient | None = None,
    ):
        self._url = base_url.rstrip("/") + "/chat/completions"
        self._headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        }
        self._model = model
        self._owns_client = client is None
        self._client = client or httpx.AsyncClient(
            timeout=httpx.Timeout(connect=10.0, read=timeout_s, write=30.0, pool=10.0)
        )

    async def complete(
        self,
        *,
        system: str,
        messages: list[Message],
        temperature: float,
        max_tokens: int,
    ) -> Completion:
        payload = {
            "model": self._model,
            "messages": [
                {"role": "system", "content": system},
                *({"role": m.role, "content": m.content} for m in messages),
            ],
            "temperature": temperature,
            "max_tokens": max_tokens,
        }
        data = await self._post_with_retries(payload)
        try:
            content = data["choices"][0]["message"].get("content")
        except (KeyError, IndexError, TypeError, AttributeError) as e:
            raise ProviderParseError(f"unexpected response shape: {data!r}") from e
        usage = data.get("usage") or {}
        return Completion(
            text=content or "",
            prompt_tokens=int(usage.get("prompt_tokens", 0)),
            completion_tokens=int(usage.get("completion_tokens", 0)),
            raw=data,
        )

    async def aclose(self) -> None:
        if self._owns_client:
            await self._client.aclose()

    async def _post_with_retries(self, payload: dict) -> dict:
        last_exc: Exception | None = None
        for attempt in range(_MAX_ATTEMPTS):
            retry_after: float | None = None
            try:
                resp = await self._client.post(
                    self._url, json=payload, headers=self._headers
                )
            except (httpx.TransportError, httpx.TimeoutException) as e:
                last_exc = e
            else:
                code = resp.status_code
                if code < 400:
                    try:
                        return resp.json()
                    except ValueError as e:
                        raise ProviderParseError("response was not valid JSON") from e
                if code == 429 or 500 <= code < 600:
                    last_exc = ProviderUnavailable(f"HTTP {code} from {self._url}")
                    retry_after = _parse_retry_after(resp.headers.get("Retry-After"))
                else:
                    raise ProviderHTTPError(code, resp.text)

            if attempt == _MAX_ATTEMPTS - 1:
                break
            delay = retry_after if retry_after is not None else _backoff_delay(attempt)
            await asyncio.sleep(delay)

        raise ProviderUnavailable(
            f"exhausted {_MAX_ATTEMPTS} attempts to {self._url}"
        ) from last_exc


def make_provider(
    cfg: ProviderCfg, *, client: httpx.AsyncClient | None = None
) -> LLMProvider:
    api_key = (os.environ.get(cfg.api_key_env) if cfg.api_key_env else None) or "sk-noauth"
    return OpenAICompatibleProvider(
        cfg.base_url, api_key, cfg.model, timeout_s=cfg.timeout_s, client=client
    )


def _backoff_delay(attempt: int) -> float:
    capped = min(_RETRY_CAP_S, _RETRY_BASE_S * (2**attempt))
    return capped + random.random() * _RETRY_BASE_S


def _parse_retry_after(value: str | None) -> float | None:
    if not value:
        return None
    try:
        return float(value)
    except ValueError:
        # HTTP-date form is not handled; caller falls back to exponential backoff.
        return None
