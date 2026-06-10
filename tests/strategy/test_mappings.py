from __future__ import annotations

import pytest

from src.strategy.mappings import get_mapping


def test_match_returns_predicted_unchanged():
    f = get_mapping("match")
    assert [f(p) for p in range(10)] == list(range(10))


def test_one_above_increments_mod_10():
    f = get_mapping("one_above")
    assert [f(p) for p in range(10)] == [1, 2, 3, 4, 5, 6, 7, 8, 9, 0]


def test_one_above_wraps_nine_to_zero():
    assert get_mapping("one_above")(9) == 0


def test_unknown_mapping_raises():
    with pytest.raises(ValueError):
        get_mapping("nope")
