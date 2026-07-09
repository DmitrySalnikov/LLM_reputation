from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class MessageRef:
    """Reference to a single public message of the episode (keys match the DB schema)."""

    round: int   # round_idx
    pair: int    # pair_idx — position of the pair within the round (as in Storage.observe)
    turn: int    # turn_idx


@dataclass(frozen=True)
class JudgeVerdict:
    """Verdict of the LLM judge for a single episode."""

    emerged: bool                 # whether the reputation institution emerged
    explanation: str              # brief explanation from the judge
    evidence: list[MessageRef]    # verified references to evidence messages


class JudgeError(Exception):
    """The judge failed to produce a verdict (unparsable response after a retry)."""
