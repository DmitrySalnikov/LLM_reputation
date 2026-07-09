from __future__ import annotations

from typing import Protocol


class TalkStopRule(Protocol):
    """Cheap-talk stop rule: how the message exchange in a pair ends.

    The negotiation loop takes turns (a, b, a, b…) up to the hard cap max_talk_turns.
    The rule governs three micro-decisions on each turn — this factors behavior into two
    INDEPENDENT traits (whether to stay silent after finish; whether finish is sticky) and
    leaves room for future variants (`either`, `fixed_k`):

      skip_turn  — should an already-ready speaker stay silent this turn (silent / keeps talking);
      next_ready — how to update the readiness flag from a new message (sticky / revocable finish);
      is_over    — is it time to end the negotiation (after the latest message).

    `ready` — a map {agent_id: is the agent ready to finish}; next_ready decides whether to
    overwrite it with the signal (revocable) or latch it (`prev or signal`).
    """

    def skip_turn(self, speaker_id: str, ready: dict[str, bool]) -> bool: ...

    def next_ready(self, prev: bool, signal: bool) -> bool: ...

    def is_over(self, ready: dict[str, bool]) -> bool: ...


class BothReadyLatch:
    """Latch: signal finish — go silent and wait for the other to mature.

    Traits: silent after finish + finish is sticky. Negotiation ends when BOTH are finished;
    a ready agent no longer gets a turn (its finish can no longer change)."""

    def skip_turn(self, speaker_id: str, ready: dict[str, bool]) -> bool:
        return ready[speaker_id]

    def next_ready(self, prev: bool, signal: bool) -> bool:
        return prev or signal

    def is_over(self, ready: dict[str, bool]) -> bool:
        return all(ready.values())


class BothReadyRevocable:
    """Revocable finish: keep talking even after signaling finish, and can retract it.

    Traits: keeps talking + finish is revocable. The ready flag is overwritten by every
    message, so finish=false retracts the earlier agreement. Stop — when finish is true for
    BOTH at once."""

    def skip_turn(self, speaker_id: str, ready: dict[str, bool]) -> bool:
        return False

    def next_ready(self, prev: bool, signal: bool) -> bool:
        return signal

    def is_over(self, ready: dict[str, bool]) -> bool:
        return all(ready.values())


class BothReadyCommitted:
    """Sticky finish + continued dialogue: signal finish — can't take it back, but keep talking.

    Traits: keeps talking (like revocable) + finish is sticky (like latch). The agent keeps
    taking turns even after finish, but ready is latched (`prev or signal`) — cannot be
    retracted. Stop — when EACH has signaled finish at least once."""

    def skip_turn(self, speaker_id: str, ready: dict[str, bool]) -> bool:
        return False

    def next_ready(self, prev: bool, signal: bool) -> bool:
        return prev or signal

    def is_over(self, ready: dict[str, bool]) -> bool:
        return all(ready.values())


_RULES = {
    "both_ready_latch": BothReadyLatch,
    "both_ready_revocable": BothReadyRevocable,
    "both_ready_committed": BothReadyCommitted,
}


def make_talk_rule(name: str) -> TalkStopRule:
    """Build a cheap-talk stop rule by name (single registry; see also _validate in config).

    Args:
        name: Rule name ("both_ready_latch" | "both_ready_revocable").

    Returns:
        An instance of the stop rule.

    Raises:
        ValueError: If the rule name is not recognized.
    """
    try:
        return _RULES[name]()
    except KeyError:
        raise ValueError(f"unknown talk_stop_rule: {name!r}")
