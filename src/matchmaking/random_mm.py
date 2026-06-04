from __future__ import annotations

from src.matchmaking.base import RoundPlan


class RandomMatchmaker:
    """Shuffle the live ids and split into disjoint pairs; an odd one out sits idle.

    The shuffle is driven by the rng handed in at setup. That rng is the matcher's
    *own* stream (the orchestrator derives a dedicated one, e.g.
    Random(f"{seed}:matchmaker")), so the sequence of partitions is reproducible by
    seed regardless of what the games do — see matching-plan.md §1.3.
    """

    def __init__(self):
        self._rng = None

    def setup(self, agent_ids, rng, cfg=None) -> None:
        self._rng = rng

    async def plan_round(self, agent_ids, round, actor=None) -> RoundPlan:
        ids = list(agent_ids)               # copy: never mutate the caller's list
        self._rng.shuffle(ids)
        idle = [ids.pop()] if len(ids) % 2 else []
        pairings = [(ids[i], ids[i + 1]) for i in range(0, len(ids), 2)]
        return RoundPlan(pairings=pairings, idle=idle, events=[])
