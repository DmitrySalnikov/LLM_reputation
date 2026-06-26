from __future__ import annotations

import pytest

from src.games.talk_rules import (
    BothReadyCommitted,
    BothReadyLatch,
    BothReadyRevocable,
    make_talk_rule,
)


# ---- factory ----

def test_make_talk_rule_returns_latch():
    assert isinstance(make_talk_rule("both_ready_latch"), BothReadyLatch)


def test_make_talk_rule_returns_revocable():
    assert isinstance(make_talk_rule("both_ready_revocable"), BothReadyRevocable)


def test_make_talk_rule_returns_committed():
    assert isinstance(make_talk_rule("both_ready_committed"), BothReadyCommitted)


def test_make_talk_rule_unknown_raises():
    with pytest.raises(ValueError):
        make_talk_rule("bogus")


# ---- next_ready: липкий (latch/committed) vs отзываемый (revocable) флаг ----

def test_latch_next_ready_is_sticky():
    rule = BothReadyLatch()
    assert rule.next_ready(False, True) is True     # выставил finish
    assert rule.next_ready(True, False) is True     # липкий: не сбрасывается


def test_revocable_next_ready_overwrites():
    rule = BothReadyRevocable()
    assert rule.next_ready(False, True) is True
    assert rule.next_ready(True, False) is False    # отзываемый: сигнал перезаписывает


def test_committed_next_ready_is_sticky():
    rule = BothReadyCommitted()
    assert rule.next_ready(False, True) is True
    assert rule.next_ready(True, False) is True     # липкий: finish нельзя отозвать


# ---- latch: готовый агент молчит; стоп — когда оба готовы ----

def test_latch_ready_speaker_skips_turn():
    rule = BothReadyLatch()
    assert rule.skip_turn("A", {"A": True, "B": False}) is True    # A защёлкнулся -> молчит
    assert rule.skip_turn("A", {"A": False, "B": True}) is False   # A ещё не готов -> говорит


def test_latch_is_over_only_when_both_ready():
    rule = BothReadyLatch()
    assert rule.is_over({"A": True, "B": True}) is True
    assert rule.is_over({"A": True, "B": False}) is False


# ---- revocable: агент всегда говорит (finish отзываемый); стоп — когда оба готовы ----

def test_revocable_never_skips_turn():
    rule = BothReadyRevocable()
    assert rule.skip_turn("A", {"A": True, "B": False}) is False   # даже выставив finish — говорит
    assert rule.skip_turn("A", {"A": False, "B": True}) is False


def test_revocable_is_over_only_when_both_ready():
    rule = BothReadyRevocable()
    assert rule.is_over({"A": True, "B": True}) is True
    assert rule.is_over({"A": False, "B": True}) is False


# ---- committed: говорит дальше (как revocable), но finish липкий (как latch) ----

def test_committed_never_skips_turn():
    rule = BothReadyCommitted()
    assert rule.skip_turn("A", {"A": True, "B": False}) is False
    assert rule.skip_turn("A", {"A": False, "B": True}) is False
