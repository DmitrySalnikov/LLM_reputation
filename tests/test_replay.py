from __future__ import annotations

import json
import sqlite3

from replay import cited_set, highlight, load_verdict


def test_cited_set_parses_evidence_json():
    evidence = json.dumps([{"round": 0, "pair": 1, "turn": 2}, {"round": 3, "pair": 0, "turn": 0}])
    assert cited_set(evidence) == {(0, 1, 2), (3, 0, 0)}


def test_highlight_wraps_in_yellow_when_on():
    assert highlight("msg", on=True) == "\033[93mmsg\033[0m"


def test_highlight_passthrough_when_off():
    assert highlight("msg", on=False) == "msg"


def test_load_verdict_none_for_old_db_without_table():
    conn = sqlite3.connect(":memory:")               # БД без таблицы judge_verdicts
    try:
        assert load_verdict(conn, "whatever") is None
    finally:
        conn.close()


def test_load_verdict_returns_row_when_present():
    conn = sqlite3.connect(":memory:")
    try:
        conn.execute(
            "CREATE TABLE judge_verdicts (run_id TEXT PRIMARY KEY, emerged INTEGER, "
            "explanation TEXT, evidence TEXT, model TEXT, created_at TEXT)"
        )
        conn.execute(
            "INSERT INTO judge_verdicts VALUES (?,?,?,?,?,?)",
            ("rid1", 1, "gossip", json.dumps([{"round": 0, "pair": 0, "turn": 1}]), "judge-m", "t"),
        )
        assert load_verdict(conn, "rid1") == (1, "gossip", '[{"round": 0, "pair": 0, "turn": 1}]')
        assert load_verdict(conn, "missing") is None
    finally:
        conn.close()
