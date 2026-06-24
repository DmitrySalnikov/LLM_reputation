from __future__ import annotations

from dataclasses import dataclass, field
from typing import Protocol

from src.core.agent import Agent


@dataclass
class PairingRecord:
    round: int
    a_id: str
    b_id: str
    transcript: list[dict]          # public cheap-talk: [{speaker, text, ready}]
    # Результаты пары. NULL/пустые, если пара сорвалась (finished=False, LLM-сбой):
    a_number: int | None = None
    b_number: int | None = None
    a_rationale: str | None = None  # private; never shown to the partner, kept for analysis
    b_rationale: str | None = None
    outcome: str | None = None      # from A's perspective: CC / DC / CD / DD
    a_payoff: float | None = None
    b_payoff: float | None = None
    usage: dict = field(default_factory=dict)   # {"prompt_tokens", "completion_tokens", "calls"}
    a_predicted: int | None = None  # стратегия prediction; None для direct
    b_predicted: int | None = None
    a_reflection: str | None = None  # пост-игровая рефлексия; None, если выключена
    b_reflection: str | None = None
    a_notes: str | None = None      # memory notes после раунда; None, если в этом раунде не свёртывали
    b_notes: str | None = None
    finished: bool = True           # False = пара сорвана LLM-сбоем (результатов нет)
    llm_calls: list = field(default_factory=list)   # сырые LLMCall'ы пары (L2-лог)


class Game(Protocol):
    async def play_pairing(self, a: Agent, b: Agent, round: int) -> PairingRecord: ...
