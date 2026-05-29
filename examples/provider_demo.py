"""Demo of the provider layer against a local Ollama.

Run from the repo root so `src` is importable:

    PYTHONPATH=. .venv/bin/python examples/provider_demo.py
"""

import asyncio

from src.core.config import ProviderCfg
from src.providers import Message, make_provider


async def main():
    # 1. Provider config (as it would come from YAML) + build via the factory.
    cfg = ProviderCfg(
        base_url="http://localhost:11434/v1",
        model="llama3:8b",  # non-reasoning model: answers directly, no <think>
        temperature=0.0,
        max_tokens=64,
    )
    provider = make_provider(cfg)

    try:
        # 2. Single-shot: system is separate, one user message.
        c1 = await provider.complete(
            system="You are terse. Answer in one short sentence.",
            messages=[Message("user", "What is the capital of France?")],
            temperature=cfg.temperature,
            max_tokens=cfg.max_tokens,
        )
        print("=== single-shot ===")
        print("text  :", repr(c1.text))
        print("tokens:", c1.prompt_tokens, "prompt /", c1.completion_tokens, "completion")

        # 3. Multi-turn: shows why Message carries a role.
        #    "7" is the model's OWN earlier reply (assistant); then we ask again (user).
        c2 = await provider.complete(
            system="Reply with only a single digit, nothing else.",
            messages=[
                Message("user", "Pick a number from 0 to 9."),
                Message("assistant", "7"),
                Message("user", "Now add one to your number."),
            ],
            temperature=0.0,
            max_tokens=cfg.max_tokens,
        )
        print("=== multi-turn (roles) ===")
        print("text  :", repr(c2.text))
    finally:
        # 4. Close the HTTP client (the provider owns its own).
        await provider.aclose()


if __name__ == "__main__":
    asyncio.run(main())
