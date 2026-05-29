"""Demo of the game layer (layer 3): two rounds between the same two agents,
narrated step by step.

Run from the repo root so `src` is importable:

    PYTHONPATH=. .venv/bin/python examples/game_demo.py
"""

import asyncio
import random

from src.core.agent import Agent, AgentSetup
from src.core.config import GameCfg, ProviderCfg
from src.games.reputation_pd import ReputationPD
from src.providers import make_provider

RULES = (
    "Each round you and one partner secretly pick a number 0-9 at the same time "
    "(neither sees the other's until both have chosen).\n"
    "Scores (your x vs partner's y; numbers wrap, so 0 follows 9):\n"
    "- x == y: you both score 3.\n"
    "- x is one above y (e.g. you 5 & partner 4, or you 0 & partner 9): you 5, partner 0.\n"
    "- y is one above x: partner 5, you 0.\n"
    "- two or more apart: you both score 1.\n"
    "You chat briefly first; messages are not binding. The choice is secret and "
    "simultaneous. Maximize your own total score."
)


def _agent(id, persona, cfg):
    return Agent(id, AgentSetup(persona, cfg), make_provider(cfg))


def _print_round(n, rec, a, b):
    bar = "=" * 64
    print(f"\n{bar}\n  ROUND {n}\n{bar}")

    print("--- Negotiation ---")
    if not rec.transcript:
        print("(no messages exchanged)")
    for i, t in enumerate(rec.transcript, 1):
        print(f'{i}. {t["speaker"]}: {t["text"]}   [ready={t["ready"]}]')

    print("\n--- Secret choices revealed ---")
    print(f"A1 chose {rec.a_number}   (private rationale: {rec.a_rationale})")
    print(f"A2 chose {rec.b_number}   (private rationale: {rec.b_rationale})")

    print("\n--- Outcome ---")
    print(f"outcome (A1's view): {rec.outcome}   payoffs: A1={rec.a_payoff}, A2={rec.b_payoff}")
    print(f"cumulative scores:   A1={a.score}, A2={b.score}")
    print(f"token usage:         {rec.usage}")

    # Each agent's diary so far (this is what gets fed into its NEXT round).
    for agent in (a, b):
        diary = agent.memory.render(None)
        print(f"\n--- {agent.id}'s memory after round {n} ---")
        print(diary[0].content if diary else "(empty)")


async def main():
    cfg = ProviderCfg(
        base_url="http://localhost:11434/v1",
        model="llama3:8b",  # non-reasoning -> fast; for qwen3 raise max_tokens (>=512)
        temperature=0.7,
        max_tokens=1000,
    )
    a = _agent("A1", "You are a pragmatic, self-interested player who tries to win.", cfg)
    b = _agent("A2", "You are a cautious player who values trust.", cfg)
    game = ReputationPD(GameCfg(max_talk_turns=10), rules=RULES)

    # The system prompt each agent actually sends (identity + persona + rules).
    print("=== A1 system prompt ===")
    print(a.system_prompt(RULES))
    print("\n=== A2 system prompt ===")
    print(b.system_prompt(RULES))
    print()

    # Same two agents play two rounds on a shared rng. After round 1 each agent's
    # memory holds the round-1 diary, which is fed into its round-2 prompts
    # (history / reputation carry-over).
    try:
        rng = random.Random(7)
        for n in (1, 2):
            rec = await game.play_pairing(a, b, round=n, rng=rng)
            _print_round(n, rec, a, b)
    finally:
        await a.provider.aclose()
        await b.provider.aclose()


if __name__ == "__main__":
    asyncio.run(main())
