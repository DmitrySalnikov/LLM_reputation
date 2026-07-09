from __future__ import annotations

from src.games.base import PairingRecord
from src.judge.keyword import KeywordCount, count_mentions


def _rec(round=0, a="A1", b="A2", transcript=None):
    """Pairing record with only a public transcript (the rest doesn't matter for counting)."""
    return PairingRecord(round=round, a_id=a, b_id=b, transcript=transcript or [])


def test_should_count_distinct_speakers_not_occurrences():
    # one speaker repeats the term twice in a single message -> 1
    rec = _rec(transcript=[{"speaker": "A1", "text": "123 and again 123", "ready": False}])
    result = count_mentions([rec], "123")
    assert result.count == 1
    assert result.speakers == ("A1",)


def test_should_ignore_speaker_name_and_match_only_reply_text():
    # the speaker is named after the term, but the term isn't in their text -> 0 (names aren't counted)
    rec = _rec(transcript=[{"speaker": "123", "text": "hello there", "ready": False}])
    result = count_mentions([rec], "123")
    assert result.count == 0
    assert result.speakers == ()


def test_should_match_case_sensitively():
    rec = _rec(transcript=[{"speaker": "A1", "text": "Trust me", "ready": False}])
    result = count_mentions([rec], "trust")
    assert result.count == 0


def test_should_match_substring():
    rec = _rec(transcript=[{"speaker": "A1", "text": "pick 1234 now", "ready": False}])
    result = count_mentions([rec], "123")
    assert result.count == 1


def test_should_return_zero_for_no_records():
    result = count_mentions([], "123")
    assert result == KeywordCount(term="123", count=0, speakers=())


def test_should_count_each_distinct_speaker_once_across_records():
    recs = [
        _rec(round=0, transcript=[{"speaker": "A1", "text": "trust 7", "ready": False},
                                  {"speaker": "A2", "text": "no", "ready": False}]),
        _rec(round=1, transcript=[{"speaker": "A1", "text": "trust again", "ready": False},
                                  {"speaker": "A3", "text": "i trust you", "ready": False}]),
    ]
    result = count_mentions(recs, "trust")
    assert result.count == 2
    assert result.speakers == ("A1", "A3")   # sorted, A1 not duplicated
