from __future__ import annotations

from src.games.base import PairingRecord
from src.judge.keyword import KeywordCount, count_mentions


def _rec(round=0, a="A1", b="A2", transcript=None):
    """Запись пары только с публичным transcript (остальное не важно для подсчёта)."""
    return PairingRecord(round=round, a_id=a, b_id=b, transcript=transcript or [])


def test_should_count_distinct_speakers_not_occurrences():
    # один говорящий повторяет термин дважды в одной реплике -> 1
    rec = _rec(transcript=[{"speaker": "A1", "text": "123 and again 123", "ready": False}])
    result = count_mentions([rec], "123")
    assert result.count == 1
    assert result.speakers == ("A1",)


def test_should_ignore_speaker_name_and_match_only_reply_text():
    # говорящий назван термином, но в его тексте термина нет -> 0 (имена не учитываются)
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
    assert result.speakers == ("A1", "A3")   # отсортированы, A1 не задвоен
