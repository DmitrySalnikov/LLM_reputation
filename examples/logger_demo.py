"""Demo / integration for the Logger (storage) layer.

A small real sweep: run a fixed LIST of configs (varying the population composition —
all-pragmatic vs all-cautious vs mixed) into ONE SQLite DB, then read the results back
via SQL. Shows the whole stack end to end (config -> population -> orchestrator ->
Storage -> SQLite) and the "one DB, many runs" design.

Run from the repo root (needs a local Ollama serving the model below):

    PYTHONPATH=. .venv/bin/python examples/logger_demo.py
"""

import asyncio
import os
import random
import sqlite3

from src.core.config import AgentSpec, EpisodeCfg, GameCfg, PopulationCfg, ProviderCfg
from src.core.orchestrator import run_episode
from src.population import make_population
from src.storage import Storage

DB = "runs/demo.db"
PROVIDER = ProviderCfg(base_url="http://localhost:11434/v1", model="llama3:8b",
                       temperature=0.7, max_tokens=256)

PRAGMATIC = "You are a pragmatic, self-interested player who tries to win."
CAUTIOUS = "You are a cautious player who values trust."


def _cfg(personas):
    return EpisodeCfg(
        seed=1, rounds=2, matchmaker="random",
        population=PopulationCfg(
            kind="roster", n_agents=4,
            agents=[AgentSpec(persona=p, provider=PROVIDER) for p in personas],
        ),
        game=GameCfg(max_talk_turns=2),
    )


# The full list of configs to run (a tiny sweep over population composition).
CONFIGS = [
    ("all-pragmatic", _cfg([PRAGMATIC])),
    ("all-cautious", _cfg([CAUTIOUS])),
    ("mixed", _cfg([PRAGMATIC, CAUTIOUS])),
]


async def run_one(label, cfg, db_path):
    pop = make_population(cfg.population, context_window=cfg.context_window).build(
        random.Random(cfg.seed)
    )
    st = Storage(db_path)
    run_id = st.begin(cfg, pop)
    try:
        await run_episode(cfg, pop, observer=st.observe)
        st.finish(pop)
    finally:
        st.close()
        await pop.aclose()
    print(f"  ran [{label}] -> run_id={run_id}")
    return run_id


async def main():
    os.makedirs("runs", exist_ok=True)
    for suffix in ("", "-wal", "-shm"):          # fresh DB for the demo
        if os.path.exists(DB + suffix):
            os.remove(DB + suffix)

    print(f"Sweeping {len(CONFIGS)} configs into {DB} (one DB, many runs):")
    for label, cfg in CONFIGS:
        print(f"  - {label}: {cfg.population.n_agents} agents, {cfg.rounds} rounds")
    print()
    ids = {label: await run_one(label, cfg, DB) for label, cfg in CONFIGS}

    # --- read everything back via SQL ---
    conn = sqlite3.connect(DB)
    bar = "=" * 60
    print(f"\n{bar}\n  READ-BACK FROM SQLITE\n{bar}")
    print(f"runs in DB: {conn.execute('SELECT COUNT(*) FROM runs').fetchone()[0]}")
    for label, run_id in ids.items():
        dist = dict(conn.execute(
            "SELECT a_outcome, COUNT(*) FROM pairings WHERE run_id=? GROUP BY a_outcome", (run_id,)
        ).fetchall())
        total = sum(dist.values())
        msgs = conn.execute("SELECT COUNT(*) FROM messages WHERE run_id=?", (run_id,)).fetchone()[0]
        scores = conn.execute(
            "SELECT agent_id, final_score FROM agents WHERE run_id=? ORDER BY final_score DESC", (run_id,)
        ).fetchall()
        cc_pct = f"{dist.get('CC', 0) / total * 100:.0f}%" if total else "n/a"
        print(f"\n[{label}]  run_id={run_id}")
        print(f"  games={total}  outcomes={dist}  CC={cc_pct}  messages={msgs}")
        print("  scores: " + ", ".join(f"{a}={s:g}" for a, s in scores))
    conn.close()


if __name__ == "__main__":
    asyncio.run(main())
