from __future__ import annotations

from src.games.base import PairingRecord
from src.judge.transcript import render_transcript, valid_refs


def _rec(round=0, a="A1", b="A2", transcript=None):
    return PairingRecord(
        round=round, a_id=a, b_id=b,
        transcript=[
            {"speaker": a, "text": "hi there", "ready": False},
            {"speaker": b, "text": "let us both pick 7", "ready": True},
        ] if transcript is None else transcript,
        a_number=4, b_number=4, a_rationale="PRIVATE-RA", b_rationale="PRIVATE-RB",
        outcome="CC", a_payoff=3.0, b_payoff=3.0,
        usage={"prompt_tokens": 1, "completion_tokens": 1, "calls": 2},
        a_reflection="PRIVATE-REFL-A", b_reflection="PRIVATE-REFL-B",
    )


def test_messages_tagged_with_round_pair_turn_ids():
    out = render_transcript([_rec(round=0), _rec(round=0, a="A3", b="A4"), _rec(round=1)])
    assert "[r0.p0.t0] A1: hi there" in out
    assert "[r0.p0.t1] A2: let us both pick 7" in out
    assert "[r0.p1.t0] A3: hi there" in out          # second pairing of round 0
    assert "[r1.p0.t0] A1: hi there" in out          # pair index restarts each round
    assert "ROUND 0" in out and "ROUND 1" in out


def test_only_public_messages_are_rendered():
    out = render_transcript([_rec()])
    assert "PRIVATE-RA" not in out                   # no rationales
    assert "PRIVATE-REFL-A" not in out               # no reflections
    assert "payoff" not in out.lower()               # no payoffs
    assert "CC" not in out                           # no outcomes


def test_empty_transcript_notes_silence():
    out = render_transcript([_rec(transcript=[])])
    assert "(no messages exchanged)" in out


def test_valid_refs_lists_every_message():
    refs = valid_refs([_rec(round=0), _rec(round=1, transcript=[])])
    assert refs == {(0, 0, 0), (0, 0, 1)}            # round 1 has no messages
