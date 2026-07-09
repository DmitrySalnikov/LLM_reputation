from __future__ import annotations

import sqlite3

import pytest

from src.storage.schema import init_schema


@pytest.fixture
def conn():
    """Clean in-memory DB with the real L1 schema."""
    c = sqlite3.connect(":memory:")
    init_schema(c)
    yield c
    c.close()


def add_run(conn, run_id, *, config_hash="d1", name=None, finished=True,
            seed=0, emerged=None):
    """Insert a run (+ optionally a judge verdict) into the test DB.

    created_at is made unique per run_id so that the sample order is deterministic."""
    conn.execute(
        "INSERT INTO runs(run_id, name, config, config_hash, seed, created_at, finished_at) "
        "VALUES (?,?,?,?,?,?,?)",
        (run_id, name, "{}", config_hash, seed, f"2026-01-01T00:00:{run_id:02d}",
         "2026-01-01T01:00:00" if finished else None),
    )
    if emerged is not None:
        conn.execute(
            "INSERT INTO judge_verdicts(run_id, emerged, explanation, evidence, model, created_at) "
            "VALUES (?,?,?,?,?,?)",
            (run_id, int(emerged), "expl", "[]", "judge-m", "2026-01-01T01:00:00"),
        )
    conn.commit()
