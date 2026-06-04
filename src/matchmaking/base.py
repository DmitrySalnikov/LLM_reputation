from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol


@dataclass
class RoundPlan:
    pairings: list[tuple[str, str]]   # disjoint pairs (a, b); a opens cheap-talk
    idle: list[str]                   # who sits out (odd N) — 0 or 1 in the MVP
    events: list[dict]                # seam for interactive matchmakers; random -> []


class Matchmaker(Protocol):
    def setup(self, agent_ids: list[str], rng, cfg) -> None: ...
    async def plan_round(self, agent_ids: list[str], round: int, actor) -> RoundPlan: ...
    #   actor: callback to query an agent (interactive matchmakers); random ignores it


def make_matchmaker(kind: str) -> Matchmaker:
    if kind == "random":
        from src.matchmaking.random_mm import RandomMatchmaker

        return RandomMatchmaker()
    raise ValueError(f"unknown matchmaker kind: {kind!r}")
