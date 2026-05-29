from __future__ import annotations

import httpx
import pytest

from src.providers import Message, OpenAICompatibleProvider

OLLAMA_URL = "http://localhost:11434/v1"
OLLAMA_MODEL = "qwen3:8b"


def _ollama_up() -> bool:
    try:
        httpx.get("http://localhost:11434/api/tags", timeout=1.0)
        return True
    except Exception:
        return False


@pytest.mark.skipif(not _ollama_up(), reason="local Ollama not reachable")
async def test_ollama_roundtrip():
    p = OpenAICompatibleProvider(OLLAMA_URL, "sk-noauth", OLLAMA_MODEL)
    try:
        c = await p.complete(
            system="Reply with exactly one word.",
            messages=[Message("user", "Say hello.")],
            temperature=0.0,
            # qwen3 is a reasoning model: it needs room to finish <think> (returned
            # in message.reasoning) before it emits the answer into message.content.
            max_tokens=512,
        )
    finally:
        await p.aclose()
    # Provider plumbing: HTTP roundtrip + usage parsing worked.
    assert c.prompt_tokens > 0
    assert c.completion_tokens > 0
    # content is empty only if the budget ran out mid-reasoning (finish_reason="length").
    if c.raw["choices"][0].get("finish_reason") == "stop":
        assert c.text.strip() != ""
