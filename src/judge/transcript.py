"""Рендеринг публичного cheap-talk эпизода для LLM-судьи.

Судья видит ТОЛЬКО публичные сообщения: ни rationale, ни рефлексий, ни выбранных
чисел, ни payoff'ов (решение пользователя в спеке). Каждое сообщение помечается
стабильным id [r<round>.p<pair>.t<turn>], совпадающим с ключами таблицы messages.
"""

from __future__ import annotations

from src.games.base import PairingRecord


def _pair_index(rec, fallback: int) -> int:
    """pair_idx записи: явный rec.pair (если задан) — иначе позиция в раунде.

    Живой судья передаёт PairingRecord без поля pair → нумерация позицией (как раньше);
    backfill передаёт ReplayRecord с истинным pair_idx из БД → ссылки совпадают с messages."""
    p = getattr(rec, "pair", None)
    return fallback if p is None else p


def _by_round(records: list[PairingRecord]) -> dict[int, list[PairingRecord]]:
    """Сгруппировать записи по раундам, сохранив порядок (он задаёт pair_idx)."""
    grouped: dict[int, list[PairingRecord]] = {}
    for rec in records:
        grouped.setdefault(rec.round, []).append(rec)
    return grouped


def render_transcript(records: list[PairingRecord]) -> str:
    """Отрендерить все публичные сообщения эпизода с id-тегами для цитирования.

    Args:
        records: Записи всех пар эпизода в порядке наблюдения (как их собирает
            observer: раунд за раундом, пары в порядке plan.pairings).

    Returns:
        Многострочный текст: раунды, пары, помеченные сообщения.
    """
    lines: list[str] = []
    grouped = _by_round(records)
    for rnd in sorted(grouped):
        lines.append(f"ROUND {rnd}")
        for i, rec in enumerate(grouped[rnd]):
            p = _pair_index(rec, i)
            lines.append(f"  Pairing r{rnd}.p{p}: {rec.a_id} vs {rec.b_id}")
            if not rec.transcript:
                lines.append("    (no messages exchanged)")
            for t, msg in enumerate(rec.transcript):
                lines.append(f"    [r{rnd}.p{p}.t{t}] {msg['speaker']}: {msg['text']}")
    return "\n".join(lines)


def valid_refs(records: list[PairingRecord]) -> set[tuple[int, int, int]]:
    """Построить множество существующих (round, pair, turn) для проверки цитат судьи."""
    refs: set[tuple[int, int, int]] = set()
    grouped = _by_round(records)
    for rnd, recs in grouped.items():
        for i, rec in enumerate(recs):
            p = _pair_index(rec, i)
            for t in range(len(rec.transcript)):
                refs.add((rnd, p, t))
    return refs
