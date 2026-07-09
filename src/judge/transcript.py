"""Render the public cheap-talk of an episode for the LLM judge.

The judge sees ONLY public messages: no rationale, no reflections, no chosen
numbers, no payoffs (a decision made by the user in the spec). Each message is
tagged with a stable id [r<round>.p<pair>.t<turn>] matching the keys of the
messages table.
"""

from __future__ import annotations

from src.games.base import PairingRecord


def _pair_index(rec, fallback: int) -> int:
    """The record's pair_idx: explicit rec.pair (if set) — otherwise the position in the round.

    The live judge is passed a PairingRecord with no pair field → numbering falls back
    to position (as before); backfill is passed a ReplayRecord with the true pair_idx
    from the DB → references match the messages table."""
    p = getattr(rec, "pair", None)
    return fallback if p is None else p


def _by_round(records: list[PairingRecord]) -> dict[int, list[PairingRecord]]:
    """Group records by round, preserving order (it defines pair_idx)."""
    grouped: dict[int, list[PairingRecord]] = {}
    for rec in records:
        grouped.setdefault(rec.round, []).append(rec)
    return grouped


def render_transcript(records: list[PairingRecord]) -> str:
    """Render all public messages of the episode with id tags for citation.

    Args:
        records: All pairing records of the episode in observation order (as
            collected by the observer: round by round, pairs in plan.pairings order).

    Returns:
        Multi-line text: rounds, pairs, tagged messages.
    """
    lines: list[str] = []
    grouped = _by_round(records)
    for rnd in sorted(grouped):
        lines.append(f"ROUND {rnd}")
        for i, rec in enumerate(grouped[rnd]):
            p = _pair_index(rec, i)
            lines.append(f"  Pairing r{rnd}.p{p}: {rec.a_id} vs {rec.b_id}")
            if not rec.transcript:
                lines.append("    (no messages exchanged)")
            for t, msg in enumerate(rec.transcript):
                lines.append(f"    [r{rnd}.p{p}.t{t}] {msg['speaker']}: {msg['text']}")
    return "\n".join(lines)


def valid_refs(records: list[PairingRecord]) -> set[tuple[int, int, int]]:
    """Build the set of existing (round, pair, turn) to validate the judge's citations."""
    refs: set[tuple[int, int, int]] = set()
    grouped = _by_round(records)
    for rnd, recs in grouped.items():
        for i, rec in enumerate(recs):
            p = _pair_index(rec, i)
            for t in range(len(rec.transcript)):
                refs.add((rnd, p, t))
    return refs
