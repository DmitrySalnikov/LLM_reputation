"""Collect emergence rate statistics by design from judged runs.

Takes runs with a judge verdict, groups them by config_hash, computes the share of
"reputation institution emerged" with a 95% Wilson interval, prints a table, and
writes stats.json (+ stats.csv).

    uv run python collect_stats.py [--design HASH ...] [--exclude-design HASH ...] \
                                   [--name LABEL ...] [--exclude-name LABEL ...] \
                                   [--out stats.json] [--csv stats.csv]
"""

from __future__ import annotations

import csv
import json
import sqlite3
import sys
from dataclasses import asdict

from src.stats.aggregate import DesignStat, aggregate_by_design, load_judged_runs
from src.stats.selection import RunFilter, filter_from_argv, selected_run_ids

DB = "experiment.db"


def collect(db_path: str, flt: RunFilter) -> list[DesignStat]:
    """Read the DB and compute statistics by design (pure read)."""
    conn = sqlite3.connect(db_path)
    try:
        run_ids = selected_run_ids(conn, flt)
        rows = load_judged_runs(conn, run_ids)
        return aggregate_by_design(rows)
    finally:
        conn.close()


def stats_to_json(stats: list[DesignStat], flt: RunFilter) -> dict:
    """Serializable artifact object: applied filters + list of designs.

    RunFilter tuples are converted to lists so the JSON has the usual shape."""
    filters = {k: (list(v) if isinstance(v, tuple) else v)
               for k, v in asdict(flt).items()}
    return {
        "filters": filters,
        "designs": [{**asdict(s), "run_ids": list(s.run_ids)} for s in stats],
    }


def write_json(path: str, obj: dict) -> None:
    """Write the statistics object to a JSON file."""
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)


def write_csv(path: str, stats: list[DesignStat]) -> None:
    """Write statistics to a CSV file (without run_ids)."""
    with open(path, "w", encoding="utf-8", newline="") as f:
        w = csv.writer(f)
        w.writerow(["config_hash", "name", "n", "n_emerged", "rate", "ci_lo", "ci_hi"])
        for s in stats:
            w.writerow([s.config_hash, s.name or "", s.n, s.n_emerged,
                        f"{s.rate:.4f}", f"{s.ci_lo:.4f}", f"{s.ci_hi:.4f}"])


def print_table(stats: list[DesignStat]) -> None:
    """Human-readable table printed to the console."""
    if not stats:
        print("No evaluated runs match the filter — no statistics.")
        return
    print(f"{'design':16} {'name':12} {'n':>4} {'emrg':>5} {'rate':>6}  95% CI")
    for s in stats:
        print(f"{s.config_hash[:16]:16} {(s.name or '—'):12} {s.n:>4} {s.n_emerged:>5} "
              f"{s.rate:>6.2f}  [{s.ci_lo:.2f}, {s.ci_hi:.2f}]")


def _flag(args: list[str], name: str, default: str) -> str:
    """Find the value of a single flag in argv."""
    return args[args.index(name) + 1] if name in args else default


def main() -> None:
    """CLI entry point: collect statistics and write artifacts."""
    args = sys.argv[1:]
    flt = filter_from_argv(args)
    out = _flag(args, "--out", "stats.json")
    csv_path = _flag(args, "--csv", "stats.csv")
    stats = collect(DB, flt)
    print_table(stats)
    if not stats:
        sys.exit(1)
    write_json(out, stats_to_json(stats, flt))
    write_csv(csv_path, stats)
    print(f"\nWritten: {out}, {csv_path}")


if __name__ == "__main__":
    main()
