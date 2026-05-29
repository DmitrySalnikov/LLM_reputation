from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(frozen=True)
class ProviderCfg:
    base_url: str
    model: str
    api_key_env: str = ""
    temperature: float = 0.7
    max_tokens: int = 512
    timeout_s: float = 120.0


@dataclass(frozen=True)
class Payoffs:
    R: float = 3.0  # both cooperate
    T: float = 5.0  # successful betrayal (off-by-one)
    P: float = 1.0  # both defect / miscoordinate
    S: float = 0.0  # betrayed
    # invariants: T > R > P > S and 2R > T + S (strict PD)


@dataclass(frozen=True)
class GameCfg:
    payoffs: Payoffs = field(default_factory=Payoffs)
    max_talk_turns: int = 6          # hard ceiling on total cheap-talk turns in a pairing
    talk_stop_rule: str = "both_ready_latch"  # MVP: only this rule
