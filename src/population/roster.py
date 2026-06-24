from __future__ import annotations

from src.core.agent import AgentSetup
from src.population.base import Population


class RosterGenerator:
    """MVP population generator: an explicit list of specs, each expanded by its `count`."""

    def __init__(self, pop_cfg, *, context_window: int | None = None):
        self._cfg = pop_cfg
        self._window = context_window

    def build(self, rng) -> Population:
        """Собрать популяцию; имена сэмплируются из пулов конфигурации по rng.

        При пустых пулах имён id назначаются как A1, A2, … (резервный режим).

        Args:
            rng: Генератор случайных чисел для детерминированного сэмплирования имён.

        Returns:
            Заполненную популяцию агентов.
        """
        pop = Population(context_window=self._window)
        names = _sample_names(self._cfg, rng)        # length = sum(count); names or A{n} fallback
        i = 0
        for spec in self._cfg.agents:                # build `count` agents of each type, in order
            for _ in range(spec.count):
                pop.add(AgentSetup(spec.persona, self._cfg.provider, self._cfg.identity_prompt,
                                   spec.play_strategy, spec.prediction_mapping),
                        agent_id=names[i])
                i += 1
        return pop


def _sample_names(cfg, rng) -> list[str | None]:
    """Сэмплировать уникальные имена 'Имя Фамилия' без повторов.

    Args:
        cfg: Конфигурация популяции с пулами имён и фамилий.
        rng: Генератор случайных чисел.

    Returns:
        Список строк 'Имя Фамилия' длиной sum(count); список None при пустых пулах.
    """
    total = sum(spec.count for spec in cfg.agents)
    if not cfg.first_name_pool or not cfg.last_name_pool:
        return [None] * total
    firsts = rng.sample(cfg.first_name_pool, total)
    lasts = rng.sample(cfg.last_name_pool, total)
    return [f"{f} {l}" for f, l in zip(firsts, lasts)]
