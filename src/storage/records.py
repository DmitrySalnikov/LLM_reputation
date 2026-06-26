from __future__ import annotations

import sqlite3
from dataclasses import dataclass, field


@dataclass(frozen=True)
class ReplayRecord:
    """Минимальная запись завершённой пары, восстановленная из БД для LLM-судьи.

    Несёт истинный pair_idx (поле pair), чтобы evidence-ссылки судьи совпадали с поиском
    по таблице messages в replay.py. transcript — публичный cheap-talk пары."""

    round: int
    pair: int
    a_id: str
    b_id: str
    transcript: list[dict] = field(default_factory=list)
    finished: bool = True


def reconstruct_records(conn: sqlite3.Connection, run_id: int) -> list[ReplayRecord]:
    """Восстановить завершённые пары прогона (round, истинный pair_idx, сообщения).

    Только finished=1 (судья видит лишь доигранные пары). Порядок — round_idx, pair_idx;
    сообщения — по turn_idx. Сорванные пары пропускаются."""
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
