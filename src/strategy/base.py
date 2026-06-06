"""Протокол стратегии игры и результат решения."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

from src.core.agent import Agent


@dataclass(frozen=True)
class Decision:
    """Результат решения агента в одной игре.

    Attributes:
        number: Итоговый выбор 0..9, который идёт в подсчёт очков.
        rationale: Обоснование, сохраняемое в память и запись.
        predicted: Предсказанное число партнёра (None для стратегии direct).
        predicted_rationale: Обоснование предсказания (None для direct).
        usage: (prompt_tokens, completion_tokens) для агрегирования в игре.
    """

    number: int
    rationale: str
    predicted: int | None = None
    predicted_rationale: str | None = None
    usage: tuple[int, int] = (0, 0)


class PlayStrategy(Protocol):
    """Протокол стратегии игры: превращает состояние раунда в решение агента."""

    async def decide(self, agent: Agent, partner_id: str, round: int,
                     feed: str, rules: str) -> Decision: ...


def make_strategy(cfg) -> PlayStrategy:
    """Собрать стратегию из конфигурации эпизода (play_strategy/prediction_mapping).

    Args:
        cfg: Конфигурация эпизода с полями play_strategy и prediction_mapping.

    Returns:
        Экземпляр стратегии игры, соответствующий конфигурации.

    Raises:
        ValueError: Если имя стратегии не распознано.
    """
    from src.strategy.direct import DirectStrategy
    from src.strategy.mappings import get_mapping
    from src.strategy.prediction import PredictionStrategy

    if cfg.play_strategy == "direct":
        return DirectStrategy()
    if cfg.play_strategy == "prediction":
        return PredictionStrategy(get_mapping(cfg.prediction_mapping))
    raise ValueError(f"неизвестная стратегия игры: {cfg.play_strategy!r}")
