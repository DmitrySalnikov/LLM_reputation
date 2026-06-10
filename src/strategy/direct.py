"""Прямая стратегия: агент сразу выбирает число через фазу DECIDE."""

from __future__ import annotations

from src.core.agent import Agent, Phase, PhaseKind
from src.games.prompts import decide_context
from src.strategy.base import Decision


class DirectStrategy:
    """Стратегия прямого выбора числа без шага предсказания."""

    def __init__(self, *, rationale: bool = True):
        """Инициализировать стратегию.

        Args:
            rationale: Просить ли обоснование перед числом (game.rationale).
        """
        self._rationale = rationale

    async def decide(self, agent: Agent, partner_id: str, round: int,
                     feed: str, rules: str) -> Decision:
        """Запросить у агента финальный выбор числа в фазе DECIDE.

        Args:
            agent: Агент, принимающий решение.
            partner_id: Идентификатор партнёра в текущем раунде.
            round: Номер раунда.
            feed: Отрендеренная история переговоров.
            rules: Текст правил игры для системного промпта.

        Returns:
            Решение с выбранным числом и обоснованием (без предсказания).
        """
        ctx = decide_context(partner_id, round, feed, rationale=self._rationale)
        res = await agent.act(Phase(PhaseKind.DECIDE, ctx, rules=rules))
        return Decision(
            number=res.data["number"],
            rationale=res.data["rationale"] if self._rationale else "",
            usage=res.usage,
        )
