"""Стратегия предсказания: предсказать число партнёра, затем отобразить его в выбор."""

from __future__ import annotations

from src.core.agent import Agent, Phase, PhaseKind
from src.core.config import GameCfg
from src.games.prompts import predict_context
from src.strategy.base import Decision
from src.strategy.mappings import PredictionMapping


class PredictionStrategy:
    """Стратегия предсказания: агент предсказывает число партнёра, отображение даёт выбор."""

    def __init__(self, mapping: PredictionMapping, game_cfg: GameCfg):
        """Инициализировать стратегию отображением предсказания в выбор.

        Args:
            mapping: Чистая функция предсказанное число -> собственный выбор (0..9).
            game_cfg: Конфигурация игры (шаблон predict_prompt и флаг rationale).
        """
        self._mapping = mapping
        self._game = game_cfg
        self._rationale = game_cfg.rationale

    async def decide(self, agent: Agent, partner_id: str, round: int,
                     feed: str, rules: str) -> Decision:
        """Запросить предсказание числа партнёра и отобразить его в финальный выбор.

        Args:
            agent: Агент, принимающий решение.
            partner_id: Идентификатор партнёра в текущем раунде.
            round: Номер раунда.
            feed: Отрендеренная история переговоров.
            rules: Текст правил игры для системного промпта.

        Returns:
            Решение с итоговым числом (после отображения), предсказанием и обоснованием.
        """
        res = await agent.act(
            Phase(PhaseKind.PREDICT, predict_context(self._game, partner_id, round, feed), rules=rules)
        )
        predicted = res.data["number"]
        rationale = res.data["rationale"] if self._rationale else ""
        return Decision(
            number=self._mapping(predicted),
            rationale=rationale,
            predicted=predicted,
            predicted_rationale=rationale if self._rationale else None,
            usage=res.usage,
            calls=res.calls,
        )
