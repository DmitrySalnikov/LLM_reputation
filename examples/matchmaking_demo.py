"""Demo of the matchmaking layer (layer 4) tying layers 1-4 together: a small
population is paired each round by the matchmaker, every pair plays one game, and
scores accumulate. This is a hand-rolled preview of what the orchestrator (layer 5)
will formalize (population + round loop + storage + resume).

Run from the repo root so `src` is importable (needs a local Ollama):

    uv run python examples/matchmaking_demo.py
"""

import asyncio
import random

from src.core.agent import Agent, AgentSetup
from src.core.config import GameCfg, ProviderCfg
from src.games.reputation_pd import ReputationPD
from src.matchmaking import make_matchmaker

SEED = 7
ROUNDS = 3

# Roster-style: an explicit list of personas, cycled to fill the population (the
# orchestrator's `roster` PopulationGenerator will do exactly this).
PERSONAS = [
    "You are a pragmatic, self-interested player who tries to win.",
    "You are a cautious player who values trust.",
    "You are a bold opportunist who probes for an edge.",
]
N_AGENTS = 5  # odd on purpose -> one agent sits idle each round


def _build_population(cfg) -> dict[str, Agent]:
    # Stable ids A1..An, persona by cycling the roster (like RosterGenerator.build).
    agents = {}
    from src.providers import make_provider

    for i in range(1, N_AGENTS + 1):
        persona = PERSONAS[(i - 1) % len(PERSONAS)]
        agents[f"A{i}"] = Agent(f"A{i}", AgentSetup(persona, cfg), make_provider(cfg))
    return agents


async def main():
    cfg = ProviderCfg(
        base_url="http://localhost:11434/v1",
        model="llama3:8b",  # non-reasoning -> fast; for qwen3 raise max_tokens (>=512)
        temperature=0.7,
        max_tokens=1000,
    )
    pop = _build_population(cfg)
    ids = list(pop)
    game = ReputationPD(GameCfg(max_talk_turns=4))
    idle_payoff = GameCfg().payoffs.P  # C3 decision: idle pays P (a neutral floor)

    # The matchmaker gets its OWN derived rng (M1 decision) so the *partition*
    # sequence is reproducible by SEED regardless of the (nondeterministic) games.
    mm = make_matchmaker("random")
    mm.setup(ids, random.Random(f"{SEED}:matchmaker"), cfg)

    bar = "=" * 64
    try:
        for r in range(1, ROUNDS + 1):
            plan = await mm.plan_round(ids, r, actor=None)
            print(f"\n{bar}\n  ROUND {r}\n{bar}")
            pretty = ", ".join(f"({a}->{b})" for a, b in plan.pairings)
            print(f"plan: pairings {pretty or '(none)'}   idle: {plan.idle or '(none)'}")

            # Orchestrator plays disjoint pairs in parallel; here we go sequentially
            # for a readable narration (the pairs share no agent, so order is moot).
            for a_id, b_id in plan.pairings:
                rec = await game.play_pairing(pop[a_id], pop[b_id], round=r)
                print(
                    f"  ({a_id}->{b_id}): {len(rec.transcript)} msgs -> "
                    f"{a_id}={rec.a_number}, {b_id}={rec.b_number} -> {rec.outcome}  "
                    f"payoffs {a_id}={rec.a_payoff}, {b_id}={rec.b_payoff}"
                )
            for c in plan.idle:
                pop[c].score += idle_payoff
                print(f"  idle {c}: +{idle_payoff} (sat out)")

        print(f"\n{bar}\n  FINAL SCOREBOARD (after {ROUNDS} rounds)\n{bar}")
        for agent in sorted(pop.values(), key=lambda a: a.score, reverse=True):
            print(f"  {agent.id}: {agent.score:g}   ({len(agent.memory.entries)} games played)")
    finally:
        for agent in pop.values():
            await agent.provider.aclose()


if __name__ == "__main__":
    asyncio.run(main())
