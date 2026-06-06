from __future__ import annotations

from src.core.agent import AgentSetup
from src.population.base import Population


class RosterGenerator:
    """MVP population generator: an explicit list of specs, cycled up to n_agents."""

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
        specs = self._cfg.agents
        names = _sample_names(self._cfg, rng)
        for i in range(self._cfg.n_agents):
            spec = specs[i % len(specs)]              # shorter than n_agents -> cycle
            pop.add(AgentSetup(spec.persona, spec.provider), agent_id=names[i])
        return pop


def _sample_names(cfg, rng) -> list[str | None]:
    """Сэмплировать уникальные имена 'Имя Фамилия' без повторов.

    Args:
        cfg: Конфигурация популяции с пулами имён и фамилий.
        rng: Генератор случайных чисел.

    Returns:
        Список строк 'Имя Фамилия' длиной n_agents; список None при пустых пулах.
    """
    if not cfg.first_name_pool or not cfg.last_name_pool:
        return [None] * cfg.n_agents
    firsts = rng.sample(cfg.first_name_pool, cfg.n_agents)
    lasts = rng.sample(cfg.last_name_pool, cfg.n_agents)
    return [f"{f} {l}" for f, l in zip(firsts, lasts)]
