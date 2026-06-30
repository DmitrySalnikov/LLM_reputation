"""Детерминированный судья: подсчёт упоминаний термина в публичном cheap-talk.

Альтернатива LLM-судье. Никакого LLM — ищем подстроку (с учётом регистра) в ТЕКСТЕ
реплик. Имена говорящих (player names) НЕ учитываются: поле speaker используется лишь
как ключ множества, но никогда не сопоставляется с термином. Результат эпизода —
число РАЗНЫХ говорящих, чьи реплики содержат термин хотя бы раз.
"""

from __future__ import annotations

from dataclasses import dataclass

from src.games.base import PairingRecord


@dataclass(frozen=True)
class KeywordCount:
    """Результат поиска термина в публичном cheap-talk одного эпизода.

    Attributes:
        term: Искомая подстрока (число или слово).
        count: Число разных говорящих, чьи реплики содержат термин.
        speakers: Их id, отсортированные (для трассировки).
    """

    term: str
    count: int
    speakers: tuple[str, ...]


def count_mentions(records: list[PairingRecord], term: str) -> KeywordCount:
    """Подсчитать разных говорящих, упомянувших термин в своих репликах.

    Поиск подстроки с учётом регистра (`term in text`). Проверяется только text
    реплики; speaker не сопоставляется с термином (имена не учитываются).

    Args:
        records: Записи пар эпизода; берётся только публичный transcript каждой
            ({speaker, text, ready}). Подходят и PairingRecord, и ReplayRecord.
        term: Искомая подстрока.

    Returns:
        KeywordCount с числом разных говорящих и их отсортированными id.
    """
    speakers: set[str] = set()
    for rec in records:
        for msg in rec.transcript:
            if term in msg["text"]:
                speakers.add(msg["speaker"])
    return KeywordCount(term=term, count=len(speakers), speakers=tuple(sorted(speakers)))
