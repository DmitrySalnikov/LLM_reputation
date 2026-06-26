from __future__ import annotations

from conftest import add_run

from src.stats.selection import RunFilter, filter_from_argv, selected_run_ids


def test_no_filter_returns_all_finished(conn):
    add_run(conn, 1, finished=True)
    add_run(conn, 2, finished=False)
    assert selected_run_ids(conn, RunFilter()) == [1]


def test_finished_only_can_be_disabled(conn):
    add_run(conn, 1, finished=True)
    add_run(conn, 2, finished=False)
    assert selected_run_ids(conn, RunFilter(finished_only=False)) == [1, 2]


def test_include_designs_is_a_whitelist(conn):
    add_run(conn, 1, config_hash="da")
    add_run(conn, 2, config_hash="db")
    assert selected_run_ids(conn, RunFilter(include_designs=("db",))) == [2]


def test_exclude_designs_and_names(conn):
    add_run(conn, 1, config_hash="da", name="keep")
    add_run(conn, 2, config_hash="db", name="drop")
    flt = RunFilter(exclude_designs=("db",), exclude_names=("drop",))
    assert selected_run_ids(conn, flt) == [1]


def test_filter_from_argv_collects_repeated_flags():
    argv = ["--design", "d1", "--design", "d2", "--exclude-name", "bad", "--out", "x.json"]
    flt = filter_from_argv(argv)
    assert flt.include_designs == ("d1", "d2")
    assert flt.exclude_names == ("bad",)
    assert flt.include_names == ()
