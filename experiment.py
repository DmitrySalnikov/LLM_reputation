"""Run a reputation experiment end to end and persist it to SQLite.

Edit the CONFIG block below, then run from the repo root (needs a local Ollama).
An optional first argument names the run (stored in runs.name as a human label):

    PYTHONPATH=. .venv/bin/python experiment.py ["run name"]

Every run is appended to one DB ("one DB, many runs"). A run is keyed by a hash of
its config (run_id): re-running an identical config is skipped, so the DB de-dups.
To force a fresh run, change the seed (or delete the DB file).

Read the stored run back, round by round, with:

    PYTHONPATH=. .venv/bin/python replay.py <run_id>
"""

import asyncio
import os
import random
import sqlite3
import sys

from src.core.config import AgentSpec, EpisodeCfg, GameCfg, PopulationCfg, ProviderCfg
from src.core.orchestrator import run_episode
from src.population import make_population
from src.storage import Storage

# ============================== CONFIG — EDIT HERE ==============================
DB = "experiment.db"

PROVIDER = ProviderCfg(
    base_url="http://localhost:11434/v1",
    model="llama3.1:8b",
    temperature=0.7,
    max_tokens=256,
)

PRAGMATIC = "You are a pragmatic, self-interested player who tries to win."
CAUTIOUS = "You are a cautious player who values trust."

# The single experiment to run. `agents` personas are cycled to fill `n_agents`.
CONFIG = EpisodeCfg(
    seed=1,
    rounds=2,
    matchmaker="random",
    population=PopulationCfg(
        kind="roster",
        n_agents=3,
        agents=[
            AgentSpec(persona=PRAGMATIC, provider=PROVIDER),
            AgentSpec(persona=CAUTIOUS, provider=PROVIDER),
        ],
    ),
    game=GameCfg(max_talk_turns=2),
)
# ===============================================================================


async def run_experiment(cfg, db_path, name=None):
    pop = make_population(cfg.population, context_window=cfg.context_window).build(
        random.Random(cfg.seed)
    )
    st = Storage(db_path)
    try:
        try:
            run_id = st.begin(cfg, pop, name)    # INSERT runs+agents; fails if already stored
        except sqlite3.IntegrityError:
            print("identical config already in DB — nothing to do "
                  "(change seed or delete the DB to re-run)")
            return None
        await run_episode(cfg, pop, observer=st.observe)
        st.finish(pop)
    finally:
        st.close()
        await pop.aclose()
    return run_id


async def main():
    name = sys.argv[1] if len(sys.argv) > 1 else None    # optional human label for the run
    os.makedirs(os.path.dirname(DB) or ".", exist_ok=True)
    print(f"Running experiment{f' {name!r}' if name else ''} into {DB}: "
          f"{CONFIG.population.n_agents} agents, {CONFIG.rounds} rounds, "
          f"matchmaker={CONFIG.matchmaker}, seed={CONFIG.seed}\n")

    run_id = await run_experiment(CONFIG, DB, name)
    if run_id is None:
        return
    print(f"run_id={run_id}")

    # --- quick summary read-back (full history: `replay.py <run_id>`) ---
    conn = sqlite3.connect(DB)
    dist = dict(conn.execute(
        "SELECT a_outcome, COUNT(*) FROM pairings WHERE run_id=? GROUP BY a_outcome", (run_id,)
    ).fetchall())
    total = sum(dist.values())
    cc = f"{dist.get('CC', 0) / total * 100:.0f}%" if total else "n/a"
    scores = conn.execute(
        "SELECT agent_id, final_score FROM agents WHERE run_id=? ORDER BY final_score DESC", (run_id,)
    ).fetchall()
    conn.close()

    bar = "=" * 60
    print(f"\n{bar}\n  SUMMARY (read back from SQLite)\n{bar}")
    print(f"outcomes={dist}  CC={cc}")
    print("scores: " + ", ".join(f"{a}={s:g}" for a, s in scores))
    print(f"\nReplay in full:  PYTHONPATH=. .venv/bin/python replay.py {run_id}")


if __name__ == "__main__":
    asyncio.run(main())
