"""Split a DB of runs into separate files — one run per file.

Each file is a standalone SQLite database with the full schema of the source DB, but
rows for exactly one run. The filename is taken from the number at the end of the run
name ('qwen3-FP8 16' -> 16.db); if there is no number — from run_id. Existing files are
not touched by default, so the function can be called after every run (see research.py)
— it reports only what's new.

    uv run python export_runs.py [--db qwen3.db] [--out DIR] [--overwrite]

  --db         source DB (default qwen3.db)
  --out        folder for the files (default: DB name without extension, e.g. qwen3)
  --overwrite  recreate files that already exist
"""

from __future__ import annotations

import os
import re
import sqlite3
import sys

DB = "qwen3.db"

_NUM_RE = re.compile(r"(\d+)\s*$")


def _out_dir_for(db_path: str) -> str:
    """Default folder: DB name without extension ('qwen3.db' -> 'qwen3')."""
    return os.path.splitext(os.path.basename(db_path))[0]


def _run_filename(name: str | None, run_id: int) -> str:
    """Run filename: number at the end of the name ('… 16' -> '16'), otherwise run_id."""
    m = _NUM_RE.search(name or "")
    return f"{m.group(1) if m else run_id}.db"


def _schema_ddl(conn: sqlite3.Connection) -> list[str]:
    """CREATE statements of the source schema: tables first, then indexes.

    Skip sqlite_sequence and auto-indexes (sql IS NULL)."""
    return [sql for (sql,) in conn.execute(
        "SELECT sql FROM sqlite_master WHERE sql IS NOT NULL AND name<>'sqlite_sequence' "
        "ORDER BY (type='index')"
    )]


def _keyed_tables(conn: sqlite3.Connection) -> tuple[list[str], set[str]]:
    """All user tables and the subset of those that have a run_id column."""
    tables = [t for (t,) in conn.execute(
        "SELECT name FROM sqlite_master WHERE type='table' AND name<>'sqlite_sequence'"
    )]
    keyed = {t for t in tables
             if any(col[1] == "run_id" for col in conn.execute(f"PRAGMA table_info({t})"))}
    return tables, keyed


def export_run(conn: sqlite3.Connection, run_id: int, out_dir: str,
               *, overwrite: bool = False) -> str | None:
    """Export one run into a separate file out_dir/<name>.db (a clean read of the source).

    Returns:
        Path to the created file, or None if the file already existed and overwrite=False.
    """
    row = conn.execute("SELECT name FROM runs WHERE run_id=?", (run_id,)).fetchone()
    if row is None:
        raise ValueError(f"run run_id={run_id} not found in the DB")
    os.makedirs(out_dir, exist_ok=True)
    path = os.path.join(out_dir, _run_filename(row[0], run_id))
    if os.path.exists(path):
        if not overwrite:
            return None
        os.remove(path)

    out = sqlite3.connect(path)
    try:
        for sql in _schema_ddl(conn):
            out.execute(sql)
        out.commit()
    finally:
        out.close()

    tables, keyed = _keyed_tables(conn)
    conn.execute("ATTACH DATABASE ? AS out", (path,))
    try:
        for t in tables:
            if t in keyed:
                conn.execute(f"INSERT INTO out.{t} SELECT * FROM main.{t} WHERE run_id=?", (run_id,))
            else:
                conn.execute(f"INSERT INTO out.{t} SELECT * FROM main.{t}")
        conn.commit()
    finally:
        conn.execute("DETACH DATABASE out")
    return path


def export_all(db_path: str, out_dir: str, *, overwrite: bool = False) -> list[str]:
    """Export every run of the DB into its own file; return paths of the files actually created."""
    conn = sqlite3.connect(db_path)
    try:
        run_ids = [r for (r,) in conn.execute("SELECT run_id FROM runs ORDER BY run_id")]
        made = [p for rid in run_ids
                if (p := export_run(conn, rid, out_dir, overwrite=overwrite)) is not None]
    finally:
        conn.close()
    return made


def _has_flag(args: list[str], name: str) -> bool:
    """Whether a boolean flag is present in argv."""
    return name in args


def _flag(args: list[str], name: str, default: str) -> str:
    """Value of a single flag in argv."""
    return args[args.index(name) + 1] if name in args else default


def main() -> None:
    """CLI entry point: split the DB into per-run files (idempotent)."""
    args = sys.argv[1:]
    db_path = _flag(args, "--db", DB)
    out_dir = _flag(args, "--out", _out_dir_for(db_path))
    overwrite = _has_flag(args, "--overwrite")
    made = export_all(db_path, out_dir, overwrite=overwrite)
    print(f"{db_path} -> {out_dir}/: created files: {len(made)}")
    for p in made:
        print(f"  {p}")


if __name__ == "__main__":
    main()
