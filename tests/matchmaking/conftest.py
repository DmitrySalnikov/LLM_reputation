from __future__ import annotations

from src.providers.base import Completion


class ScriptedProvider:
    """Test double implementing LLMProvider without network. Returns queued reply
    texts in order and records (system, messages) of each call."""

    def __init__(self, replies: list[str], *, prompt_tokens: int = 2, completion_tokens: int = 3):
        self._queue = list(replies)
        self._pt = prompt_tokens
        self._ct = completion_tokens
        self.calls: list[tuple[str, list]] = []

    async def complete(self, *, system, messages, temperature, max_tokens) -> Completion:
        self.calls.append((system, messages))
        text = self._queue.pop(0)
        return Completion(text=text, prompt_tokens=self._pt, completion_tokens=self._ct, raw={})

    async def aclose(self) -> None:
        pass
