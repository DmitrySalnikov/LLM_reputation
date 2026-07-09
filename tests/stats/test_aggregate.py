from __future__ import annotations

import pytest
from conftest import add_run

from src.stats.aggregate import aggregate_by_design, load_judged_runs


def test_load_skips_runs_without_verdict(conn):
    add_run(conn, 1, config_hash="d1", emerged=True)
    add_run(conn, 2, config_hash="d1", emerged=None)   # no verdict
    rows = load_judged_runs(conn, [1, 2])
    assert [r.run_id for r in rows] == [1]
    assert rows[0].emerged is True


def test_aggregate_groups_and_computes_rate(conn):
    add_run(conn, 1, config_hash="d1", name="base", emerged=True)
    add_run(conn, 2, config_hash="d1", name="base", emerged=False)
    add_run(conn, 3, config_hash="d2", name="alt", emerged=True)
    stats = aggregate_by_design(load_judged_runs(conn, [1, 2, 3]))
    assert [s.config_hash for s in stats] == ["d1", "d2"]
    d1 = stats[0]
    assert d1.n == 2 and d1.n_emerged == 1 and d1.rate == pytest.approx(0.5)
    assert d1.name == "base"
    assert d1.run_ids == (1, 2)
    assert d1.ci_lo == pytest.approx(0.0945, abs=1e-3)
    assert d1.ci_hi == pytest.approx(0.9055, abs=1e-3)


def test_aggregate_empty_input():
    assert aggregate_by_design([]) == []
