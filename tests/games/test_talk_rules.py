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


# ---- next_ready: sticky (latch/committed) vs revocable flag ----

def test_latch_next_ready_is_sticky():
    rule = BothReadyLatch()
    assert rule.next_ready(False, True) is True     # set finish
    assert rule.next_ready(True, False) is True     # sticky: does not reset


def test_revocable_next_ready_overwrites():
    rule = BothReadyRevocable()
    assert rule.next_ready(False, True) is True
    assert rule.next_ready(True, False) is False    # revocable: the new signal overwrites


def test_committed_next_ready_is_sticky():
    rule = BothReadyCommitted()
    assert rule.next_ready(False, True) is True
    assert rule.next_ready(True, False) is True     # sticky: finish cannot be revoked


# ---- latch: a ready agent falls silent; stop — once both are ready ----

def test_latch_ready_speaker_skips_turn():
    rule = BothReadyLatch()
    assert rule.skip_turn("A", {"A": True, "B": False}) is True    # A latched -> falls silent
    assert rule.skip_turn("A", {"A": False, "B": True}) is False   # A not ready yet -> speaks


def test_latch_is_over_only_when_both_ready():
    rule = BothReadyLatch()
    assert rule.is_over({"A": True, "B": True}) is True
    assert rule.is_over({"A": True, "B": False}) is False


# ---- revocable: the agent always speaks (finish is revocable); stop — once both are ready ----

def test_revocable_never_skips_turn():
    rule = BothReadyRevocable()
    assert rule.skip_turn("A", {"A": True, "B": False}) is False   # speaks even after setting finish
    assert rule.skip_turn("A", {"A": False, "B": True}) is False


def test_revocable_is_over_only_when_both_ready():
    rule = BothReadyRevocable()
    assert rule.is_over({"A": True, "B": True}) is True
    assert rule.is_over({"A": False, "B": True}) is False


# ---- committed: keeps speaking (like revocable), but finish is sticky (like latch) ----

def test_committed_never_skips_turn():
    rule = BothReadyCommitted()
    assert rule.skip_turn("A", {"A": True, "B": False}) is False
    assert rule.skip_turn("A", {"A": False, "B": True}) is False
