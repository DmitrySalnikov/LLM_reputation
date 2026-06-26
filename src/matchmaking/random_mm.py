from __future__ import annotations

from src.matchmaking.base import RoundPlan


class RandomMatchmaker:
    """Shuffle the live ids and split into disjoint pairs; an odd one out sits idle.

    The shuffle is driven by the per-round rng handed to `plan_round`. The caller
    derives a dedicated stream per round (Random(f"{seed}:matchmaker:{round}")), so
    round r's partition depends only on (ids, r) — reproducible by seed and resumable
    without replaying earlier rounds. See matching-plan.md §1.3.
    """

    def setup(self, agent_ids, cfg=None) -> None:
        pass                                # random matcher keeps no static state

    async def plan_round(self, agent_ids, round, rng, actor=None) -> RoundPlan:
        ids = list(agent_ids)               # copy: never mutate the caller's list
        rng.shuffle(ids)
        idle = [ids.pop()] if len(ids) % 2 else []
        pairings = [(ids[i], ids[i + 1]) for i in range(0, len(ids), 2)]
        return RoundPlan(pairings=pairings, idle=idle, events=[])
