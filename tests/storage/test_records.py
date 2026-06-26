from __future__ import annotations

from conftest import insert_run_row

from src.storage.records import reconstruct_records


def _add_pairing(conn, run_id, round_idx, pair_idx, *, finished=1, a="A1", b="A2"):
    conn.execute(
        "INSERT INTO pairings(run_id, round_idx, pair_idx, a_id, b_id, finished, "
        "a_number, b_number, a_outcome, a_payoff, b_payoff) "
        "VALUES (?,?,?,?,?,?,?,?,?,?,?)",
        (run_id, round_idx, pair_idx, a, b, finished,
         7 if finished else None, 7 if finished else None,
         "CC" if finished else None, 3.0 if finished else None, 3.0 if finished else None),
    )


def _add_msg(conn, run_id, round_idx, pair_idx, turn_idx, speaker, text, ready=0):
    conn.execute(
        "INSERT INTO messages(run_id, round_idx, pair_idx, turn_idx, speaker, text, ready) "
        "VALUES (?,?,?,?,?,?,?)",
        (run_id, round_idx, pair_idx, turn_idx, speaker, text, ready),
    )


def test_reconstructs_finished_pairings_with_true_pair_idx(conn):
    insert_run_row(conn, 1)
    _add_pairing(conn, 1, 0, 0)
    _add_pairing(conn, 1, 0, 1)
    _add_msg(conn, 1, 0, 1, 0, "A1", "hi")
    _add_msg(conn, 1, 0, 1, 1, "A2", "pick 7", ready=1)
    conn.commit()

    recs = reconstruct_records(conn, 1)
    assert [(r.round, r.pair) for r in recs] == [(0, 0), (0, 1)]
    second = recs[1]
    assert second.pair == 1                        # истинный pair_idx сохранён
    assert [m["text"] for m in second.transcript] == ["hi", "pick 7"]
    assert second.transcript[1]["ready"] is True


def test_skips_aborted_pairings(conn):
    insert_run_row(conn, 1)
    _add_pairing(conn, 1, 0, 0, finished=0)        # сорвана — без результата
    conn.commit()
    assert reconstruct_records(conn, 1) == []
