"""Find "gossip" — mentions of third-party players in cheap-talk messages.

Gossip = a message in which the speaker names a PLAYER outside their current pair
(i.e. a real agent id from this run, different from both the speaker and the
round partner). Direct address to the partner and self-mentions are excluded —
that's dyadic memory, not talk about a third party.

    uv run python find_gossip.py [--db research.db] [--model SUBSTR] \
                                 [--bare] [--context N] [--summary]

  --db       path to the database (default research.db)
  --model    keep only runs whose name contains the substring (e.g. deepseek)
  --bare     also catch "bare" numeric ids (without the word Player); risks false
             positives on arithmetic, so it's off by default
  --context  length of the displayed message snippet (default 160)
  --summary  print only the per-model summary, without individual messages
"""

from __future__ import annotations

import re
import sqlite3
import sys

DB = "research.db"

_PLAYER_RE = re.compile(r"Player\s*(\d+)", re.IGNORECASE)
_BARE_RE = re.compile(r"\b(\d{2,4})\b")


def _norm(agent_id: str) -> str:
    """Normalize an agent id to a plain number: 'Player 167' -> '167'."""
    return agent_id.replace("Player", "").replace("player", "").strip()


def _model_of(name: str) -> str:
    """Model name = the full run name up to the first space ('deepseek-v4-pro 43')."""
    return name.split(" ", 1)[0] if name else name


def _agent_ids_by_run(conn: sqlite3.Connection) -> dict[int, set[str]]:
    """Set of real agent ids in each run (to filter out non-players)."""
    out: dict[int, set[str]] = {}
    for run_id, aid in conn.execute("SELECT run_id, agent_id FROM agents"):
        out.setdefault(run_id, set()).add(_norm(aid))
    return out


def _pair_members(conn: sqlite3.Connection) -> dict[tuple[int, int, int], set[str]]:
    """The two members of each pair by (run, round, pair) from the pairings table."""
    out: dict[tuple[int, int, int], set[str]] = {}
    for run_id, rnd, pair, a_id, b_id in conn.execute(
        "SELECT run_id, round_idx, pair_idx, a_id, b_id FROM pairings"
    ):
        members = {_norm(a_id), _norm(b_id)} - {""}
        out[(run_id, rnd, pair)] = members
    return out


def _mentioned(text: str, valid: set[str], bare: bool) -> set[str]:
    """Ids of real players mentioned in the text (via 'Player N', optionally bare numbers)."""
    ids = {m for m in _PLAYER_RE.findall(text)}
    if bare:
        ids |= {m for m in _BARE_RE.findall(text)}
    return {m for m in ids if m in valid}


def find_gossip(
    conn: sqlite3.Connection, *, model: str | None, bare: bool, context: int
) -> list[tuple]:
    """Collect all messages referencing a third player (pure read).

    Returns:
        List of tuples (run_id, model, round, pair, turn, speaker, third_ids, snippet),
        ordered by (run_id, round, pair, turn).
    """
    run_name = dict(conn.execute("SELECT run_id, name FROM runs"))
    ids_by_run = _agent_ids_by_run(conn)
    members = _pair_members(conn)

    hits: list[tuple] = []
    for run_id, rnd, pair, turn, speaker, text in conn.execute(
        "SELECT run_id, round_idx, pair_idx, turn_idx, speaker, text FROM messages "
        "ORDER BY run_id, round_idx, pair_idx, turn_idx"
    ):
        name = run_name.get(run_id, "")
        if model and model not in name:
            continue
        valid = ids_by_run.get(run_id, set())
        # pair members: from pairings, otherwise fallback — both speakers in this pair
        pair_ids = members.get((run_id, rnd, pair)) or {_norm(speaker)}
        third = sorted(m for m in _mentioned(text, valid, bare) if m not in pair_ids)
        if third:
            snippet = " ".join(text.split())[:context]
            hits.append((run_id, _model_of(name), rnd, pair, turn, speaker, third, snippet))
    return hits


def print_hits(hits: list[tuple]) -> None:
    """Print found gossip lines, one per line."""
    for run_id, model, rnd, pair, turn, speaker, third, snippet in hits:
        print(f"run {run_id} [{model}] r{rnd} p{pair} t{turn}  {speaker} -> "
              f"{','.join(third)}: {snippet}")


def print_summary(hits: list[tuple]) -> None:
    """Summary: how much gossip and in how many runs, per model."""
    from collections import defaultdict
    lines: dict[str, int] = defaultdict(int)
    runs: dict[str, set[int]] = defaultdict(set)
    for run_id, model, *_ in hits:
        lines[model] += 1
        runs[model].add(run_id)
    print(f"\n{'model':18} {'gossip-lines':>12} {'runs':>6}")
    for model in sorted(lines, key=lambda m: -lines[m]):
        print(f"{model:18} {lines[model]:>12} {len(runs[model]):>6}")
    print(f"{'TOTAL':18} {sum(lines.values()):>12} "
          f"{len({h[0] for h in hits}):>6}")


def _has_flag(args: list[str], name: str) -> bool:
    """Whether a boolean flag is present in argv."""
    return name in args


def _flag(args: list[str], name: str, default: str) -> str:
    """Value of a single flag in argv."""
    return args[args.index(name) + 1] if name in args else default


def main() -> None:
    """CLI entry point: find gossip and print it (+ summary)."""
    args = sys.argv[1:]
    db_path = _flag(args, "--db", DB)
    model = _flag(args, "--model", "") or None
    bare = _has_flag(args, "--bare")
    context = int(_flag(args, "--context", "160"))
    summary_only = _has_flag(args, "--summary")

    conn = sqlite3.connect(db_path)
    try:
        hits = find_gossip(conn, model=model, bare=bare, context=context)
    finally:
        conn.close()

    if not summary_only:
        if hits:
            print_hits(hits)
        else:
            print("No gossip found for the current filter.")
    print_summary(hits)


if __name__ == "__main__":
    main()
