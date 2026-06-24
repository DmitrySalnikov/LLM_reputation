from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class MessageRef:
    """Ссылка на одно публичное сообщение эпизода (ключи совпадают со схемой БД)."""

    round: int   # round_idx
    pair: int    # pair_idx — позиция пары внутри раунда (как в Storage.observe)
    turn: int    # turn_idx


@dataclass(frozen=True)
class JudgeVerdict:
    """Вердикт LLM-судьи по одному эпизоду."""

    emerged: bool                 # возник ли институт репутации
    explanation: str              # краткое объяснение судьи
    evidence: list[MessageRef]    # проверенные ссылки на сообщения-доказательства


class JudgeError(Exception):
    """Судья не смог вынести вердикт (неразборчивый ответ после повтора)."""
