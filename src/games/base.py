from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

from src.core.agent import Agent


@dataclass
class PairingRecord:
    round: int
    a_id: str
    b_id: str
    transcript: list[dict]          # public cheap-talk: [{speaker, text, ready}]
    a_number: int
    b_number: int
    a_rationale: str                # private; never shown to the partner, kept for analysis
    b_rationale: str
    outcome: str                    # from A's perspective: CC / DC / CD / DD
    a_payoff: float
    b_payoff: float
    usage: dict                     # {"prompt_tokens", "completion_tokens", "calls"}


class Game(Protocol):
    async def play_pairing(self, a: Agent, b: Agent, round: int, rng) -> PairingRecord: ...
