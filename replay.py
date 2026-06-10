"""Replay a stored episode from the Logger DB — the read side of the storage layer.

Given a run_id, pull the WHOLE history back out of SQLite and narrate it exactly like
matchmaking_demo / orchestrator_demo did live: round by round, every pairing's dialogue,
choices, outcome and payoffs, who sat idle, then the final scoreboard. This touches no
engine and no LLM — it proves the normalized schema is enough to reconstruct an episode.

Run from the repo root:

    PYTHONPATH=. .venv/bin/python replay.py <run_id> [--config]

With no run_id it lists the runs in the DB and exits. Pass --config (or -c) to also show
the episode config: the prompts in full, then the roster, then the remaining scalar knobs
(prompts / agents / name pools are stripped from that last dump to avoid repetition).
"""

import json
import re
import sqlite3
import sys
from datetime import datetime

DB_DEFAULT = "experiment.db"

# Все настраиваемые промпты живут в cfg["game"]; в --config их печатаем отдельной
# секцией после шапки, а из дампа конфига убираем, чтобы не дублировать простыни текста.
_PROMPT_KEYS = ("rules", "talk_prompt", "decide_prompt", "predict_prompt", "reflect_prompt")
# Из дампа конфига выкидываем и эти ключи population — ростер печатается отдельной секцией.
_POP_DROP = ("agents", "first_name_pool", "last_name_pool")


def _trim_ms(ts):
    """Drop fractional seconds and the tz offset from an ISO timestamp."""
    if not ts:
        return ts
    ts = re.sub(r"\.\d+", "", ts)                  # strip fractional seconds
    return re.sub(r"[+-]\d{2}:\d{2}$", "", ts)     # strip tz offset


def _duration(created, finished):
    """Wall-clock length of a run, computed from its two timestamps. '—' if unfinished."""
    if not (created and finished):
        return "—"
    try:
        secs = int((datetime.fromisoformat(finished) - datetime.fromisoformat(created)).total_seconds())
    except ValueError:
        return "?"
    h, rem = divmod(secs, 3600)
    m, s = divmod(rem, 60)
    return f"{h}h{m}m{s}s" if h else f"{m}m{s}s" if m else f"{s}s"


def list_runs(conn):
    rows = conn.execute(
        "SELECT run_id, name, config, created_at, finished_at FROM runs ORDER BY created_at"
    ).fetchall()
    if not rows:
        print("(no runs in this DB)")
        return
    print(f"{len(rows)} run(s):")
    for run_id, name, config, created, finished in rows:
        cfg = json.loads(config)
        n_agents = sum(a.get("count", 1) for a in cfg["population"]["agents"])
        state = "done" if finished else "unfinished"
        label = f"  {name!r}" if name else ""
        print(f"  {run_id}  {_trim_ms(created)}  "
              f"{n_agents} agents, {cfg['rounds']} rounds, {_duration(created, finished)}  "
              f"[{state}]{label}")


def replay(conn, run_id, show_config=False):
    run = conn.execute(
        "SELECT name, config, seed, created_at, finished_at FROM runs WHERE run_id=?", (run_id,)
    ).fetchone()
    if run is None:
        print(f"run_id {run_id!r} not found in this DB.")
        list_runs(conn)
        return
    name, config, seed, created, finished = run
    cfg = json.loads(config)

    n_agents = sum(a.get("count", 1) for a in cfg["population"]["agents"])  # derived from counts

    bar = "=" * 64
    title = f"{run_id}  ({name})" if name else run_id
    print(f"{bar}\n  REPLAY run_id={title}\n{bar}")
    print(f"{n_agents} agents, {cfg['rounds']} rounds, "
          f"max_talk_turns={cfg['game']['max_talk_turns']}")
    print(f"created={_trim_ms(created)}  finished={_trim_ms(finished) or '(unfinished)'}")

    if show_config:
        game = cfg.get("game", {})
        present = [k for k in _PROMPT_KEYS if k in game]
        print("\nprompts:")
        if not present:
            print("  (not recorded — run predates configurable prompts)")
        for key in present:
            text = game[key]
            print(f"  [{key}]")
            print("    " + (text.replace("\n", "\n    ") if text else "(empty)"))

    print(f"\nroster ({n_agents} agents):")
    for spec in cfg["population"]["agents"]:        # one line per type, as in the config
        p = spec["provider"]
        print(f"  {spec.get('count', 1)}x {spec['persona']}")
        print(f"       provider: model={p['model']} "
              f"temp={p['temperature']} max_tokens={p['max_tokens']}")

    if show_config:
        # config dump AFTER prompts + roster (both shown above) and WITHOUT them: drop the
        # prompt strings, the agents list and the name pools — keep it to the scalar knobs.
        game = cfg.get("game", {})
        pop = cfg.get("population", {})
        slim = dict(cfg)
        slim["game"] = {k: v for k, v in game.items() if k not in _PROMPT_KEYS}
        slim["population"] = {k: v for k, v in pop.items() if k not in _POP_DROP}
        print("\nconfig (prompts + roster shown above):")
        print("  " + json.dumps(slim, indent=2, sort_keys=True).replace("\n", "\n  "))

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
                      a_outcome, a_payoff, b_payoff, a_predicted, b_predicted, usage_calls
               FROM pairings WHERE run_id=? AND round_idx=? ORDER BY pair_idx""",
            (run_id, r),
        ).fetchall()
        for (pi, a_id, b_id, a_num, b_num, a_rat, b_rat,
             outcome, a_pay, b_pay, a_pred, b_pred, calls) in pairings:
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
            if a_pred is not None or b_pred is not None:   # prediction strategy: guess of partner's number
                print(f"    predictions: {a_id} guessed {b_id}={a_pred}, {b_id} guessed {a_id}={b_pred}")
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
    args = sys.argv[1:]
    show_config = any(a in ("--config", "-c") for a in args)
    pos = [a for a in args if a not in ("--config", "-c")]   # positional args only
    conn = sqlite3.connect(DB_DEFAULT)
    try:
        if not pos:
            print(f"usage: replay.py <run_id> [--config]   (db: {DB_DEFAULT})\n")
            list_runs(conn)
        else:
            replay(conn, pos[0], show_config=show_config)
    finally:
        conn.close()


if __name__ == "__main__":
    main()
