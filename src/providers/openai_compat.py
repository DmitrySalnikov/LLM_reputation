from __future__ import annotations

import asyncio
import os
import random

import httpx

from src.core.config import ProviderCfg
from src.providers.base import (
    Completion,
    HttpAttempt,
    LLMProvider,
    Message,
    ProviderHTTPError,
    ProviderParseError,
    ProviderUnavailable,
)

_RETRY_BASE_S = 1.0
_RETRY_CAP_S = 30.0
_MAX_ATTEMPTS = 2


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
        # _post_with_retries гарантирует resp с извлекаемым ответом (битый конверт и кривую
        # форму он ретраит, как 5xx). Здесь парсим ещё раз — уже для извлечения контента.
        resp, attempts = await self._post_with_retries(payload)   # raises уже несут request+attempts
        raw_text = resp.text
        data = resp.json()
        content = data["choices"][0]["message"].get("content")    # извлекаемость гарантирована гейтом
        usage = data.get("usage") or {}
        pt = int(usage.get("prompt_tokens", 0))
        ct = int(usage.get("completion_tokens", 0))
        final = HttpAttempt(
            status="ok", status_code=resp.status_code, request=payload,
            response=content or "", response_raw=raw_text, error=None,
            prompt_tokens=pt, completion_tokens=ct,
        )
        return Completion(
            text=content or "", prompt_tokens=pt, completion_tokens=ct,
            raw=data, request=payload, attempts=attempts + (final,),
        )

    async def aclose(self) -> None:
        if self._owns_client:
            await self._client.aclose()

    async def _post_with_retries(self, payload: dict) -> tuple[httpx.Response, tuple[HttpAttempt, ...]]:
        """Сделать запрос с ретраями. Вернуть (resp с валидным JSON-телом, попытки-ретраи).

        Накапливает КАЖДУЮ HTTP-попытку (для L2-лога). Ретраит как транзиентное: сеть,
        429/5xx и **битый JSON-конверт** (`resp.json()` не распарсился — обычно прокси/обрыв).
        Терминально (raise с `request`+`attempts`): 4xx и исчерпание ретраев. Возвращает
        сырой `resp` (тело уже проверено как JSON); извлечение контента — в `complete`.
        """
        attempts: list[HttpAttempt] = []
        last_exc: Exception | None = None
        for attempt in range(_MAX_ATTEMPTS):
            retry_after: float | None = None
            try:
                resp = await self._client.post(
                    self._url, json=payload, headers=self._headers
                )
            except (httpx.TransportError, httpx.TimeoutException) as e:
                last_exc = e
                attempts.append(HttpAttempt(
                    status="network", status_code=None, request=payload,
                    response=None, response_raw=None, error=str(e)))
            else:
                code = resp.status_code
                if code < 400:
                    try:
                        data = resp.json()                       # гейт: тело извлекаемо?
                        data["choices"][0]["message"].get("content")
                    except ValueError:
                        last_exc = ProviderParseError("response was not valid JSON")
                        attempts.append(HttpAttempt(   # битый конверт — ретраим, как 5xx
                            status="bad_json", status_code=code, request=payload,
                            response=None, response_raw=resp.text, error="not valid JSON"))
                    except (KeyError, IndexError, TypeError, AttributeError):
                        last_exc = ProviderParseError(f"unexpected response shape: {data!r}")
                        attempts.append(HttpAttempt(   # валидный JSON, но не извлекается — тоже ретраим
                            status="bad_shape", status_code=code, request=payload,
                            response=None, response_raw=resp.text, error="unexpected shape"))
                    else:
                        return resp, tuple(attempts)
                elif code == 429 or 500 <= code < 600:
                    last_exc = ProviderUnavailable(f"HTTP {code} from {self._url}")
                    attempts.append(HttpAttempt(           # тело 5xx/429 тоже пишем
                        status="server_error", status_code=code, request=payload,
                        response=None, response_raw=resp.text, error=f"HTTP {code}"))
                    retry_after = _parse_retry_after(resp.headers.get("Retry-After"))
                else:
                    err = ProviderHTTPError(code, resp.text)
                    err.request = payload
                    err.attempts = tuple(attempts) + (HttpAttempt(
                        status="http_error", status_code=code, request=payload,
                        response=None, response_raw=resp.text, error=f"HTTP {code}"),)
                    raise err

            if attempt == _MAX_ATTEMPTS - 1:
                break
            delay = retry_after if retry_after is not None else _backoff_delay(attempt)
            await asyncio.sleep(delay)

        # исчерпали ретраи: каждая попытка уже в attempts (с телом 5xx / network)
        err = ProviderUnavailable(f"exhausted {_MAX_ATTEMPTS} attempts to {self._url}")
        err.request = payload
        err.attempts = tuple(attempts)
        raise err from last_exc


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
