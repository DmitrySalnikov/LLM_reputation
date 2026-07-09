from __future__ import annotations

from src.core.agent import AgentSetup
from src.population.base import Population


class RosterGenerator:
    """MVP population generator: an explicit list of specs, each expanded by its `count`."""

    def __init__(self, pop_cfg, *, context_window: int | None = None):
        self._cfg = pop_cfg
        self._window = context_window

    def build(self, rng) -> Population:
        """Build the population; names are sampled from the config pools via rng.

        With empty name pools, ids are assigned as A1, A2, … (fallback mode).

        Args:
            rng: Random number generator for deterministic name sampling.

        Returns:
            The populated roster of agents.
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
    """Sample unique ids from the name pools without repeats.

    Three modes depending on which pools are set:
      * both pools -> 'First Last' (one unique first and last name per agent);
      * one pool   -> id = the pool element itself, without a last name (e.g. 'Player 348' as a single field);
      * neither    -> None (the caller assigns fallback A1..An).

    Args:
        cfg: Population config with (optional) first- and last-name pools.
        rng: Random number generator.

    Returns:
        A list of ids of length sum(count); a list of None if the pools are empty. Numeric
        pool elements (YAML numbers) are coerced to strings.
    """
    total = sum(spec.count for spec in cfg.agents)
    firsts, lasts = cfg.first_name_pool, cfg.last_name_pool
    if firsts and lasts:
        f = rng.sample(firsts, total)
        l = rng.sample(lasts, total)
        return [f"{a} {b}" for a, b in zip(f, l)]
    if firsts or lasts:                          # exactly one pool -> id = the element itself (no last name)
        return [str(x) for x in rng.sample(firsts or lasts, total)]
    return [None] * total                        # no pools -> A1..An
