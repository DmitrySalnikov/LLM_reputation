from __future__ import annotations

from src.judge import KeywordCount
from src.storage import Storage


def _insert_run(st, run_id=1):
    st.conn.execute(
        "INSERT INTO runs(run_id, name, config, config_hash, seed, created_at, finished_at) "
        "VALUES (?, 'demo', '{}', 'd1', 0, '2026-01-01T00:00:00', '2026-01-01T01:00:00')",
        (run_id,),
    )
    st.conn.commit()


def test_should_save_keyword_count_row():
    st = Storage(":memory:")
    try:
        _insert_run(st, 1)
        st.save_keyword_count(KeywordCount(term="123", count=2, speakers=("A1", "A3")), run_id=1)
        row = st.conn.execute(
            "SELECT term, count, speakers FROM keyword_counts WHERE run_id=1"
        ).fetchone()
        assert row == ("123", 2, '["A1", "A3"]')
    finally:
        st.close()


def test_should_upsert_keyword_count_on_repeat_term():
    # повторный (run_id, term) заменяет строку, не дублирует
    st = Storage(":memory:")
    try:
        _insert_run(st, 1)
        st.save_keyword_count(KeywordCount(term="123", count=1, speakers=("A1",)), run_id=1)
        st.save_keyword_count(KeywordCount(term="123", count=5, speakers=("A1", "A2")), run_id=1)
        rows = st.conn.execute(
            "SELECT count FROM keyword_counts WHERE run_id=1 AND term='123'"
        ).fetchall()
        assert rows == [(5,)]
    finally:
        st.close()
