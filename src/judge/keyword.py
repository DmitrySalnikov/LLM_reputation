"""Deterministic judge: count mentions of a term in the public cheap-talk.

An alternative to the LLM judge. No LLM — we search for a substring (case-sensitive)
in the message TEXT. Speaker names (player names) are NOT taken into account: the
speaker field is used only as a set key and is never compared against the term. The
episode result is the number of DISTINCT speakers whose messages contain the term at
least once.
"""

from __future__ import annotations

from dataclasses import dataclass

from src.games.base import PairingRecord


@dataclass(frozen=True)
class KeywordCount:
    """Result of searching for a term in the public cheap-talk of a single episode.

    Attributes:
        term: The substring to search for (a number or a word).
        count: Number of distinct speakers whose messages contain the term.
        speakers: Their ids, sorted (for tracing).
    """

    term: str
    count: int
    speakers: tuple[str, ...]


def count_mentions(records: list[PairingRecord], term: str) -> KeywordCount:
    """Count distinct speakers who mentioned the term in their messages.

    Case-sensitive substring search (`term in text`). Only the message text is
    checked; speaker is never compared against the term (names are not taken into
    account).

    Args:
        records: Pairing records of the episode; only the public transcript of each
            ({speaker, text, ready}) is used. Both PairingRecord and ReplayRecord work.
        term: The substring to search for.

    Returns:
        KeywordCount with the number of distinct speakers and their sorted ids.
    """
    speakers: set[str] = set()
    for rec in records:
        for msg in rec.transcript:
            if term in msg["text"]:
                speakers.add(msg["speaker"])
    return KeywordCount(term=term, count=len(speakers), speakers=tuple(sorted(speakers)))
