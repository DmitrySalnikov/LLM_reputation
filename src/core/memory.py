from __future__ import annotations

from dataclasses import dataclass

from src.providers.base import Message


@dataclass
class MemoryEntry:
    round: int
    my_id: str              # id самого агента (для метки "<my_id> (you)" в дневнике)
    partner_id: str
    transcript: list[dict]  # [{speaker, text, ready}]
    my_number: int
    my_rationale: str
    partner_number: int
    outcome: str
    payoff: float            # выигрыш самого агента в этом раунде
    partner_payoff: float    # выигрыш партнёра (для симметричной строки "Payoffs: ...")
    my_predicted: int | None = None  # стратегия prediction; None для direct
    my_reflection: str | None = None  # пост-игровая рефлексия; None, если выключена


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
        diary = "Your memory of past rounds:\n" + "\n".join(
            _render_entry(e) for e in entries
        )
        return [Message("user", diary)]


def _render_entry(e: MemoryEntry) -> str:
    # Дневник едет в user-сообщении, адресованном самому агенту: чужие реплики — по имени,
    # свои — "<имя> (you)" (одна метка на всю запись, симметрично оппоненту).
    me = f"{e.my_id} (you)"
    lines = [f"[Round {e.round} · opponent {e.partner_id}]"]
    if e.transcript:
        lines.append("Talk:")
        for turn in e.transcript:
            label = e.partner_id if turn.get("speaker") == e.partner_id else me
            mark = " (ready=true)" if turn.get("ready") else ""   # ready=false не выводим
            lines.append(f"  {label}: {turn.get('text', '')}{mark}")
    if e.my_predicted is not None:
        lines.append(f"{me} predicted {e.partner_id} would pick {e.my_predicted}.")
    reason = f" (reason: {e.my_rationale})" if e.my_rationale else ""
    lines.append(
        f"Choices: {me}={e.my_number}{reason}, {e.partner_id}={e.partner_number}. "
        f"Payoffs: {me}={e.payoff:g}, {e.partner_id}={e.partner_payoff:g}."
    )
    if e.my_reflection:
        lines.append(f"Takeaway of {me} after that round: {e.my_reflection}")
    return "\n".join(lines)
