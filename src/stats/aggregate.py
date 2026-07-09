from __future__ import annotations

import sqlite3
from dataclasses import dataclass

from src.stats.wilson import wilson_interval


@dataclass(frozen=True)
class RunRow:
    """A single judge-scored run: emerged outcome + which design it belongs to."""

    run_id: int
    config_hash: str
    name: str | None
    emerged: bool


@dataclass(frozen=True)
class DesignStat:
    """Statistics for a single design (config_hash): rate + Wilson interval."""

    config_hash: str
    name: str | None
    n: int
    n_emerged: int
    rate: float
    ci_lo: float
    ci_hi: float
    run_ids: tuple[int, ...]


def load_judged_runs(conn: sqlite3.Connection, run_ids: list[int]) -> list[RunRow]:
    """Read (config_hash, name, emerged) for runs that have a verdict.

    Runs without a verdict are silently dropped (INNER JOIN). Result order matches run_ids."""
    if not run_ids:
        return []
    found = {
        rid: (ch, name, bool(em))
        for rid, ch, name, em in conn.execute(
            "SELECT r.run_id, r.config_hash, r.name, v.emerged "
            "FROM runs r JOIN judge_verdicts v ON v.run_id = r.run_id "
            f"WHERE r.run_id IN ({','.join('?' * len(run_ids))})",
            run_ids,
        )
    }
    return [RunRow(rid, *found[rid]) for rid in run_ids if rid in found]


def aggregate_by_design(rows: list[RunRow]) -> list[DesignStat]:
    """Group runs by config_hash; rate + 95% Wilson interval per group.

    Group order follows the first occurrence of config_hash in rows. The group's name is
    the name of its first run (used as the chart label)."""
    order: list[str] = []
    groups: dict[str, list[RunRow]] = {}
    for r in rows:
        if r.config_hash not in groups:
            groups[r.config_hash] = []
            order.append(r.config_hash)
        groups[r.config_hash].append(r)
    out: list[DesignStat] = []
    for ch in order:
        grp = groups[ch]
        n = len(grp)
        k = sum(1 for r in grp if r.emerged)
        lo, hi = wilson_interval(k, n)
        out.append(DesignStat(
            config_hash=ch, name=grp[0].name, n=n, n_emerged=k,
            rate=k / n, ci_lo=lo, ci_hi=hi,
            run_ids=tuple(r.run_id for r in grp),
        ))
    return out
