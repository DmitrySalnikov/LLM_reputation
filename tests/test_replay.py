from __future__ import annotations

import json
import sqlite3

from replay import (
    _expand_newlines, _preview, _provider_line, _readable, _roster_line,
    cited_set, highlight, load_verdict,
)


def test_readable_starts_content_body_on_new_line():
    out = _readable('"content": "Your memory\\nof rounds"')
    assert out == '"content": "\nYour memory\nof rounds"'   # тело начинается с новой строки


def test_readable_unescapes_quotes():
    assert _readable('Set \\"ready\\": true') == 'Set "ready": true'


def test_preview_collapses_whitespace_and_keeps_short_text():
    assert _preview("hello\n   there  world") == "hello there world"


def test_preview_truncates_long_text_with_ellipsis():
    p = _preview("x" * 100, n=10)
    assert len(p) == 10 and p.endswith("…")


def test_preview_empty_for_none_or_blank():
    assert _preview(None) == "" and _preview("") == ""


def test_expand_newlines_turns_escaped_into_real():
    assert _expand_newlines("a\\nb") == "a\nb"        # два символа \n -> реальный перевод строки


def test_expand_newlines_passes_through_none():
    assert _expand_newlines(None) is None


def test_provider_line_puts_model_last():
    prov = {"model": "llama3.1:8b", "temperature": 0.7, "max_tokens": 2000}
    assert _provider_line(prov) == "provider: temp=0.7 max_tokens=2000 model=llama3.1:8b"


def test_roster_line_shows_persona_and_count():
    assert _roster_line({"persona": "pragmatic", "count": 3}) == "  3x pragmatic"


def test_roster_line_renders_null_persona_as_placeholder():
    assert _roster_line({"persona": None, "count": 1}) == "  1x (no persona)"


def test_roster_line_count_defaults_to_one():
    assert _roster_line({"persona": "p"}) == "  1x p"


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
