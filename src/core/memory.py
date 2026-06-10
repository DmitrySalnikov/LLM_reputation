from __future__ import annotations

from dataclasses import dataclass

from src.providers.base import Message


@dataclass
class MemoryEntry:
    round: int
    partner_id: str
    transcript: list[dict]  # [{speaker, text, ready}]
    my_number: int
    my_rationale: str
    partner_number: int
    outcome: str
    payoff: float
    my_predicted: int | None = None  # стратегия prediction; None для direct


class Memory:
    def __init__(self) -> None:
        self.entries: list[MemoryEntry] = []

    def add(self, entry: MemoryEntry) -> None:
        self.entries.append(entry)

    def render(self, window: int | None) -> list[Message]:
        if window is None:
            entries = self.entries
        elif window <= 0:
            entries = []
        else:
            entries = self.entries[-window:]
        if not entries:
            return []
        diary = "Past rounds (oldest first):\n\n" + "\n\n".join(
            _render_entry(e) for e in entries
        )
        return [Message("user", diary)]


def _render_entry(e: MemoryEntry) -> str:
    lines = [f"[Round {e.round} · partner {e.partner_id}]"]
    if e.transcript:
        lines.append("Talk:")
        for turn in e.transcript:
            # the transcript holds only the two players; relabel my own lines as "me".
            speaker = e.partner_id if turn.get("speaker") == e.partner_id else "me"
            ready = str(bool(turn.get("ready"))).lower()
            lines.append(f"  {speaker}: {turn.get('text', '')} (ready={ready})")
    if e.my_predicted is not None:
        lines.append(f"I predicted {e.partner_id} would pick {e.my_predicted}.")
    reason = f" (reason: {e.my_rationale})" if e.my_rationale else ""
    lines.append(
        f"Choices: me={e.my_number}{reason}, {e.partner_id}={e.partner_number}. "
        f"Outcome: {e.outcome}. Payoff to me: {e.payoff:g}."
    )
    return "\n".join(lines)
