from __future__ import annotations

import json

from src.providers.base import Completion, HttpAttempt


class ScriptedProvider:
    """Test double for LLMProvider without network: returns replies from a queue in order."""

    def __init__(self, replies: list[str], *, prompt_tokens: int = 2, completion_tokens: int = 3):
        self._queue = list(replies)
        self._pt = prompt_tokens
        self._ct = completion_tokens
        self.calls: list[tuple[str, list]] = []

    async def complete(self, *, system, messages, temperature, max_tokens) -> Completion:
        self.calls.append((system, messages))
        text = self._queue.pop(0)
        request = {"model": "scripted",
                   "messages": [{"role": "system", "content": system},
                                *({"role": m.role, "content": m.content} for m in messages)],
                   "temperature": temperature, "max_tokens": max_tokens}
        raw = {"choices": [{"message": {"content": text}}]}
        attempt = HttpAttempt(status="ok", status_code=200, request=request, response=text,
                              response_raw=json.dumps(raw), error=None,
                              prompt_tokens=self._pt, completion_tokens=self._ct)
        return Completion(text=text, prompt_tokens=self._pt, completion_tokens=self._ct,
                          raw=raw, request=request, attempts=(attempt,))

    async def aclose(self) -> None:
        pass
