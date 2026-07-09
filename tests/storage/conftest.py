from __future__ import annotations

import sqlite3

import pytest

from src.storage.schema import init_schema


@pytest.fixture
def conn():
    """Clean in-memory DB with the real L1 schema (FK disabled — convenient for targeted inserts)."""
    c = sqlite3.connect(":memory:")
    init_schema(c)
    yield c
    c.close()


def insert_run_row(conn, run_id, *, config_hash="d1", name=None):
    """Minimal runs row (for tests that need a parent row for judge_verdicts)."""
    conn.execute(
        "INSERT INTO runs(run_id, name, config, config_hash, seed, created_at, finished_at) "
        "VALUES (?,?,?,?,?,?,?)",
        (run_id, name, "{}", config_hash, 0, "2026-01-01T00:00:00", "2026-01-01T01:00:00"),
    )
    conn.commit()
