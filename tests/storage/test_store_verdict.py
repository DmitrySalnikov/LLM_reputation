from __future__ import annotations

from src.judge import JudgeVerdict, MessageRef
from src.storage import Storage


def _verdict():
    return JudgeVerdict(emerged=True, explanation="gossip",
                        evidence=[MessageRef(round=0, pair=0, turn=0)])


def test_has_verdict_false_then_true_after_save():
    st = Storage(":memory:")
    try:
        st.conn.execute(
            "INSERT INTO runs(run_id, name, config, config_hash, seed, created_at, finished_at) "
            "VALUES (1, NULL, '{}', 'd1', 0, '2026-01-01T00:00:00', '2026-01-01T01:00:00')"
        )
        st.conn.commit()
        assert st.has_verdict(1) is False
        st.save_verdict(_verdict(), model="judge-m", run_id=1)
        assert st.has_verdict(1) is True
        row = st.conn.execute(
            "SELECT emerged, model FROM judge_verdicts WHERE run_id=1"
        ).fetchone()
        assert row == (1, "judge-m")
    finally:
        st.close()
