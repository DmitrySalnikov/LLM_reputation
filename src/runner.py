"""Experiment runner — drive one episode, persist it, narrate it live.

This is the reusable engine behind `experiment.py`: it owns no config of its own.
`experiment.py` defines the config and calls `run(cfg, db_path, name)`; tests or other
scripts can do the same. Persistence (Storage) and the live console narration are wired
together into one observer so the DB and the printed transcript never diverge.
"""

from __future__ import annotations

import os
import random
import sqlite3

from src.core.config import EpisodeCfg
from src.core.orchestrator import run_episode
from src.population import make_population
from src.storage import Storage


def narrate_round(r, plan, recs) -> None:
    """Print one round to console as soon as it's played (the live observer)."""
    print(f"\n{'─' * 60}\n  ROUND {r}")
    if plan.idle:
        print(f"  idle (sat out): {', '.join(plan.idle)}")
    for rec in recs:
        print(f"\n  {rec.a_id} vs {rec.b_id}:")
        if rec.transcript:
            for i, t in enumerate(rec.transcript, 1):
                print(f"    {i}. {t['speaker']}: {t['text']}   [ready={t['ready']}]")
        else:
            print("    (no messages exchanged)")
        print(f"    choices: {rec.a_id}={rec.a_number}, {rec.b_id}={rec.b_number}"
              f"  ->  {rec.outcome}   (payoffs {rec.a_id}={rec.a_payoff:g}, {rec.b_id}={rec.b_payoff:g})")
        print(f"      {rec.a_id} reason: {rec.a_rationale}")
        print(f"      {rec.b_id} reason: {rec.b_rationale}")


async def run_experiment(cfg: EpisodeCfg, db_path: str, name: str | None = None) -> str | None:
    """Build the population, run the episode, persist + narrate each round, score it.
    Returns the run_id, or None if this exact config is already stored (de-dup)."""
    pop = make_population(cfg.population, context_window=cfg.context_window).build(
        random.Random(cfg.seed)
    )
    st = Storage(db_path)
    try:
        try:
            run_id = st.begin(cfg, pop, name)    # INSERT runs+agents; fails if already stored
        except sqlite3.IntegrityError:
            print("identical config already in DB — nothing to do "
                  "(change seed or config to re-run)")
            return None

        def observer(r, plan, recs):             # persist AND narrate each round live
            st.observe(r, plan, recs)
            narrate_round(r, plan, recs)

        await run_episode(cfg, pop, observer=observer)
        st.finish(pop)

        bar = "=" * 60
        print(f"\n{bar}\n  FINAL SCOREBOARD\n{bar}")
        for a in sorted(pop, key=lambda a: a.score, reverse=True):
            print(f"  {a.id}: {a.score:g}")
    finally:
        st.close()
        await pop.aclose()
    return run_id


async def run(cfg: EpisodeCfg, db_path: str, name: str | None = None) -> str | None:
    """Top-level entry: print a header, run the experiment, print the replay hint."""
    os.makedirs(os.path.dirname(db_path) or ".", exist_ok=True)
    n_agents = sum(a.count for a in cfg.population.agents)
    print(f"Running experiment{f' {name!r}' if name else ''} into {db_path}: "
          f"{n_agents} agents, {cfg.rounds} rounds, seed={cfg.seed}")

    run_id = await run_experiment(cfg, db_path, name)
    if run_id is not None:
        print(f"\nrun_id={run_id}   "
              f"(replay: PYTHONPATH=. uv run python replay.py {run_id})")
    return run_id
