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
        calls: Сырые LLMCall'ы фаз стратегии (decide/predict) для L2-лога.
    """

    number: int
    rationale: str
    predicted: int | None = None
    predicted_rationale: str | None = None
    usage: tuple[int, int] = (0, 0)
    calls: tuple = ()


class PlayStrategy(Protocol):
    """Протокол стратегии игры: превращает состояние раунда в решение агента."""

    async def decide(self, agent: Agent, partner_id: str, round: int,
                     feed: str, reason: str = "") -> Decision: ...


def make_strategy(play_strategy: str, prediction_mapping: str, game_cfg) -> PlayStrategy:
    """Собрать стратегию по её имени (стратегия живёт на агенте, см. AgentSpec).

    Args:
        play_strategy: "direct" | "prediction".
        prediction_mapping: имя отображения predict->выбор (нужно только для prediction).
        game_cfg: GameCfg — шаблоны промптов фаз decide/predict.

    Returns:
        Экземпляр стратегии игры, соответствующий имени.

    Raises:
        ValueError: Если имя стратегии не распознано.
    """
    from src.strategy.direct import DirectStrategy
    from src.strategy.mappings import get_mapping
    from src.strategy.prediction import PredictionStrategy

    if play_strategy == "direct":
        return DirectStrategy(game_cfg)
    if play_strategy == "prediction":
        return PredictionStrategy(get_mapping(prediction_mapping), game_cfg)
    raise ValueError(f"неизвестная стратегия игры: {play_strategy!r}")
