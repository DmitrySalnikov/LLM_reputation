from __future__ import annotations

import json

from src.providers.base import Completion, HttpAttempt


def _scripted_completion(*, system, messages, temperature, max_tokens, text, pt, ct) -> Completion:
    """Собрать Completion как реальный провайдер: с request (отправленный payload),
    raw (распарсенное тело) и одной успешной HttpAttempt — чтобы L2-лог был осмысленным."""
    request = {
        "model": "scripted",
        "messages": [{"role": "system", "content": system},
                     *({"role": m.role, "content": m.content} for m in messages)],
        "temperature": temperature,
        "max_tokens": max_tokens,
    }
    raw = {"choices": [{"message": {"content": text}}],
           "usage": {"prompt_tokens": pt, "completion_tokens": ct}}
    raw_text = json.dumps(raw)
    attempt = HttpAttempt(status="ok", status_code=200, request=request,
                          response=text, response_raw=raw_text, error=None,
                          prompt_tokens=pt, completion_tokens=ct)
    return Completion(text=text, prompt_tokens=pt, completion_tokens=ct,
                      raw=raw, request=request, attempts=(attempt,))


class ScriptedProvider:
    """Test double implementing the LLMProvider protocol without any network.

    Returns the queued reply texts in order and records the (system, messages)
    of each call so tests can assert prompt assembly.
    """

    def __init__(self, replies: list[str], *, prompt_tokens: int = 2, completion_tokens: int = 3):
        self._queue = list(replies)
        self._pt = prompt_tokens
        self._ct = completion_tokens
        self.calls: list[tuple[str, list]] = []

    async def complete(self, *, system, messages, temperature, max_tokens) -> Completion:
        self.calls.append((system, messages))
        text = self._queue.pop(0)
        return _scripted_completion(system=system, messages=messages, temperature=temperature,
                                    max_tokens=max_tokens, text=text, pt=self._pt, ct=self._ct)

    async def aclose(self) -> None:
        pass
