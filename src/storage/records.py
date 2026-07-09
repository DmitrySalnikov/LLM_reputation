from __future__ import annotations

import sqlite3
from dataclasses import dataclass, field


@dataclass(frozen=True)
class ReplayRecord:
    """Minimal record of a finished pair, reconstructed from the DB for the LLM judge.

    Carries the true pair_idx (the pair field) so the judge's evidence references match
    lookups against the messages table in replay.py. transcript is the pair's public
    cheap-talk."""

    round: int
    pair: int
    a_id: str
    b_id: str
    transcript: list[dict] = field(default_factory=list)
    finished: bool = True


def reconstruct_records(conn: sqlite3.Connection, run_id: int) -> list[ReplayRecord]:
    """Reconstruct a run's finished pairs (round, true pair_idx, messages).

    Only finished=1 (the judge sees only pairs that were played to completion). Order —
    round_idx, pair_idx; messages — by turn_idx. Aborted pairs are skipped."""
    pairings = conn.execute(
        "SELECT round_idx, pair_idx, a_id, b_id FROM pairings "
        "WHERE run_id=? AND finished=1 ORDER BY round_idx, pair_idx",
        (run_id,),
    ).fetchall()
    out: list[ReplayRecord] = []
    for round_idx, pair_idx, a_id, b_id in pairings:
        transcript = [
            {"speaker": s, "text": t, "ready": bool(r)}
            for s, t, r in conn.execute(
                "SELECT speaker, text, ready FROM messages "
                "WHERE run_id=? AND round_idx=? AND pair_idx=? ORDER BY turn_idx",
                (run_id, round_idx, pair_idx),
            )
        ]
        out.append(ReplayRecord(round=round_idx, pair=pair_idx,
                                a_id=a_id, b_id=b_id, transcript=transcript))
    return out
