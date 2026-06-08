"""Replay a stored episode from the Logger DB — the read side of the storage layer.

Given a run_id, pull the WHOLE history back out of SQLite and narrate it exactly like
matchmaking_demo / orchestrator_demo did live: round by round, every pairing's dialogue,
choices, outcome and payoffs, who sat idle, then the final scoreboard. This touches no
engine and no LLM — it proves the normalized schema is enough to reconstruct an episode.

Run from the repo root:

    PYTHONPATH=. .venv/bin/python replay.py <run_id> [db_path]

With no run_id it lists the runs in the DB and exits.
"""

import json
import sqlite3
import sys

DB_DEFAULT = "experiment.db"


def list_runs(conn):
    rows = conn.execute(
        "SELECT run_id, name, seed, created_at, finished_at FROM runs ORDER BY created_at"
    ).fetchall()
    if not rows:
        print("(no runs in this DB)")
        return
    print(f"{len(rows)} run(s):")
    for run_id, name, seed, created, finished in rows:
        state = "done" if finished else "unfinished"
        label = f"  {name!r}" if name else ""
        print(f"  {run_id}{label}  seed={seed}  {created}  [{state}]")


def replay(conn, run_id):
    run = conn.execute(
        "SELECT name, config, seed, created_at, finished_at FROM runs WHERE run_id=?", (run_id,)
    ).fetchone()
    if run is None:
        print(f"run_id {run_id!r} not found in this DB.")
        list_runs(conn)
        return
    name, config, seed, created, finished = run
    cfg = json.loads(config)

    bar = "=" * 64
    title = f"{run_id}  ({name})" if name else run_id
    print(f"{bar}\n  REPLAY run_id={title}\n{bar}")
    print(
        f"seed={seed}  rounds={cfg['rounds']}  matchmaker={cfg['matchmaker']}  "
        f"agents={cfg['population']['n_agents']}"
    )
    print(f"created={created}  finished={finished or '(unfinished)'}")

    print("\nconfig (full):")
    print("  " + json.dumps(cfg, indent=2, sort_keys=True).replace("\n", "\n  "))

    agents = conn.execute(
        "SELECT agent_id, persona, provider, final_score FROM agents WHERE run_id=? ORDER BY agent_id",
        (run_id,),
    ).fetchall()
    print("\nroster:")
    for aid, persona, provider, score in agents:
        p = json.loads(provider)
        print(f"  {aid}: {persona}")
        print(f"       provider: model={p['model']} base_url={p['base_url']} "
              f"temp={p['temperature']} max_tokens={p['max_tokens']}")

    rounds = [
        r for (r,) in conn.execute(
            "SELECT round_idx FROM rounds WHERE run_id=? ORDER BY round_idx", (run_id,)
        )
    ]
    for r in rounds:
        print(f"\n{'─' * 60}\n  ROUND {r}")
        idle = [a for (a,) in conn.execute(
            "SELECT agent_id FROM idle WHERE run_id=? AND round_idx=? ORDER BY agent_id", (run_id, r)
        )]
        if idle:
            print(f"  idle (sat out): {', '.join(idle)}")

        pairings = conn.execute(
            """SELECT pair_idx, a_id, b_id, a_number, b_number, a_rationale, b_rationale,
                      a_outcome, a_payoff, b_payoff, usage_calls
               FROM pairings WHERE run_id=? AND round_idx=? ORDER BY pair_idx""",
            (run_id, r),
        ).fetchall()
        for (pi, a_id, b_id, a_num, b_num, a_rat, b_rat,
             outcome, a_pay, b_pay, calls) in pairings:
            print(f"\n  {a_id} vs {b_id}  ({a_id} opens):")
            msgs = conn.execute(
                """SELECT speaker, text, ready FROM messages
                   WHERE run_id=? AND round_idx=? AND pair_idx=? ORDER BY turn_idx""",
                (run_id, r, pi),
            ).fetchall()
            if msgs:
                for i, (speaker, text, ready) in enumerate(msgs, 1):
                    print(f"    {i}. {speaker}: {text}   [ready={bool(ready)}]")
            else:
                print("    (no messages exchanged)")
            print(
                f"    choices: {a_id}={a_num}, {b_id}={b_num}  ->  {outcome}   "
                f"(payoffs {a_id}={a_pay:g}, {b_id}={b_pay:g})  [{calls} llm calls]"
            )
            print(f"      {a_id} reason: {a_rat}")
            print(f"      {b_id} reason: {b_rat}")

    # final scoreboard + games-played, reconstructed from pairings
    scores = dict(conn.execute(
        "SELECT agent_id, final_score FROM agents WHERE run_id=?", (run_id,)
    ))
    games = {aid: 0 for aid in scores}
    for (a_id, b_id) in conn.execute(
        "SELECT a_id, b_id FROM pairings WHERE run_id=?", (run_id,)
    ):
        games[a_id] += 1
        games[b_id] += 1

    print(f"\n{bar}\n  FINAL SCOREBOARD\n{bar}")
    for aid, score in sorted(scores.items(), key=lambda kv: (kv[1] or 0), reverse=True):
        s = "?" if score is None else f"{score:g}"
        print(f"  {aid}: {s}   ({games[aid]} games)")

    dist = dict(conn.execute(
        "SELECT a_outcome, COUNT(*) FROM pairings WHERE run_id=? GROUP BY a_outcome", (run_id,)
    ))
    total = sum(dist.values())
    cc = f"{dist.get('CC', 0) / total * 100:.0f}%" if total else "n/a"
    print(f"\noutcomes: {dist}   CC={cc}   games={total}")


def main():
    db = sys.argv[2] if len(sys.argv) > 2 else DB_DEFAULT
    conn = sqlite3.connect(db)
    try:
        if len(sys.argv) < 2:
            print(f"usage: replay_run.py <run_id> [db_path={DB_DEFAULT}]\n")
            list_runs(conn)
        else:
            replay(conn, sys.argv[1])
    finally:
        conn.close()


if __name__ == "__main__":
    main()
