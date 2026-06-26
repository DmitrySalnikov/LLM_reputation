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
                pop.add(AgentSetup(spec.system_prompt, self._cfg.provider,
                                   spec.play_strategy, spec.prediction_mapping),
                        agent_id=names[i])
                i += 1
        return pop


def _sample_names(cfg, rng) -> list[str | None]:
    """Сэмплировать уникальные id из пулов имён без повторов.

    Три режима в зависимости от того, какие пулы заданы:
      * оба пула  -> 'Имя Фамилия' (по одному уникальному имени и фамилии на агента);
      * один пул  -> id = сам элемент пула, без фамилии (напр. 'Player 348' одним полем);
      * ни одного -> None (роутер назначит резервные A1..An).

    Args:
        cfg: Конфигурация популяции с (опциональными) пулами имён и фамилий.
        rng: Генератор случайных чисел.

    Returns:
        Список id длиной sum(count); список None при пустых пулах. Числовые элементы пула
        (YAML-числа) приводятся к строке.
    """
    total = sum(spec.count for spec in cfg.agents)
    firsts, lasts = cfg.first_name_pool, cfg.last_name_pool
    if firsts and lasts:
        f = rng.sample(firsts, total)
        l = rng.sample(lasts, total)
        return [f"{a} {b}" for a, b in zip(f, l)]
    if firsts or lasts:                          # ровно один пул -> id = сам элемент (без фамилии)
        return [str(x) for x in rng.sample(firsts or lasts, total)]
    return [None] * total                        # пулов нет -> A1..An
