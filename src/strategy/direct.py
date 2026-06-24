"""Прямая стратегия: агент сразу выбирает число через фазу DECIDE."""

from __future__ import annotations

from src.core.agent import Agent, Phase, PhaseKind
from src.core.config import GameCfg
from src.games.prompts import decide_context
from src.strategy.base import Decision


class DirectStrategy:
    """Стратегия прямого выбора числа без шага предсказания."""

    def __init__(self, game_cfg: GameCfg):
        """Инициализировать стратегию конфигурацией игры.

        Args:
            game_cfg: Конфигурация игры (шаблон decide_prompt и флаг rationale).
        """
        self._game = game_cfg
        self._rationale = game_cfg.rationale

    async def decide(self, agent: Agent, partner_id: str, round: int,
                     feed: str, rules: str, reason: str = "") -> Decision:
        """Запросить у агента финальный выбор числа в фазе DECIDE.

        Args:
            agent: Агент, принимающий решение.
            partner_id: Идентификатор партнёра в текущем раунде.
            round: Номер раунда.
            feed: Отрендеренная история переговоров.
            rules: Текст правил игры для системного промпта.
            reason: Почему закрылся чат (для строки закрытия в промпте).

        Returns:
            Решение с выбранным числом и обоснованием (без предсказания).
        """
        res = await agent.act(
            Phase(PhaseKind.DECIDE,
                  decide_context(self._game, partner_id, round, feed, agent.score, reason),
                  rules=rules, game_cfg=self._game)
        )
        return Decision(
            number=res.data["number"],
            rationale=res.data["rationale"] if self._rationale else "",
            usage=res.usage,
            calls=res.calls,
        )
