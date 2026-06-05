"""Demo / entry point for the orchestrator (layer 5): build a population from a config,
run one episode over it (population -> matchmaker -> game, all rounds), narrating each
round's negotiations via the observer seam, then printing the scoreboard.

run_episode returns nothing: the caller owns the population (reads scores from it,
closes it) and collects per-round data via the observer — the same seam the Logger
layer will use to persist rounds.

There is no CLI — this script IS the entry point. Run from the repo root (needs a local
Ollama serving the model in config/example.yaml):

    PYTHONPATH=. .venv/bin/python examples/orchestrator_demo.py
"""

import asyncio
import random

from src.core.config import load_episode
from src.core.orchestrator import run_episode
from src.population import make_population

CONFIG = "config/example.yaml"


def narrate_round(r, plan, recs):
    print(f"\n{'─' * 60}\n  ROUND {r}")
    if plan.idle:
        print(f"  idle (sat out): {', '.join(plan.idle)}")
    for rec in recs:
        print(f"\n  {rec.a_id} vs {rec.b_id}  ({rec.a_id} opens):")
        if rec.transcript:
            for i, t in enumerate(rec.transcript, 1):
                print(f"    {i}. {t['speaker']}: {t['text']}   [ready={t['ready']}]")
        else:
            print("    (no messages exchanged)")
        print(
            f"    choices: {rec.a_id}={rec.a_number}, {rec.b_id}={rec.b_number}"
            f"  ->  {rec.outcome}   (payoffs {rec.a_id}={rec.a_payoff:g}, {rec.b_id}={rec.b_payoff:g})"
        )
        print(f"      {rec.a_id} reason: {rec.a_rationale}")
        print(f"      {rec.b_id} reason: {rec.b_rationale}")


async def main():
    cfg = load_episode(CONFIG)
    print(
        f"Running {cfg.rounds} rounds, {cfg.population.n_agents} agents, "
        f"matchmaker={cfg.matchmaker}, seed={cfg.seed}"
    )
    pop = make_population(cfg.population, context_window=cfg.context_window).build(
        random.Random(cfg.seed)
    )
    records = []

    def observer(r, plan, recs):
        narrate_round(r, plan, recs)
        records.extend(recs)        # collect per-round facts via the observer channel

    try:
        await run_episode(cfg, pop, observer=observer)
    finally:
        await pop.aclose()

    bar = "=" * 60
    print(f"\n{bar}\n  SCOREBOARD ({len(records)} games over {cfg.rounds} rounds)\n{bar}")
    scores = {a.id: a.score for a in pop}      # final state read from the caller-owned pop
    for agent_id, score in sorted(scores.items(), key=lambda kv: kv[1], reverse=True):
        games = sum(1 for rec in records if agent_id in (rec.a_id, rec.b_id))
        print(f"  {agent_id}: {score:g}   ({games} games)")

    outcomes = {}
    for rec in records:
        outcomes[rec.outcome] = outcomes.get(rec.outcome, 0) + 1
    print("\noutcomes:", outcomes)


if __name__ == "__main__":
    asyncio.run(main())
