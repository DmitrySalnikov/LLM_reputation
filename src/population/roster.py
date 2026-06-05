from __future__ import annotations

from src.core.agent import AgentSetup
from src.population.base import Population


class RosterGenerator:
    """MVP population generator: an explicit list of specs, cycled up to n_agents."""

    def __init__(self, pop_cfg, *, context_window: int | None = None):
        self._cfg = pop_cfg
        self._window = context_window

    def build(self, rng) -> Population:
        # rng unused here (roster is deterministic); kept for the PopulationGenerator
        # contract — future generators (mixed/homogeneous) use it for random composition.
        pop = Population(context_window=self._window)
        specs = self._cfg.agents
        for i in range(self._cfg.n_agents):
            spec = specs[i % len(specs)]              # shorter than n_agents -> cycle
            pop.add(AgentSetup(spec.persona, spec.provider))
        return pop
