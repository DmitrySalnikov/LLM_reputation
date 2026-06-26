from __future__ import annotations

import pytest

from src.stats.wilson import wilson_interval


def test_zero_successes_lower_bound_is_zero():
    lo, hi = wilson_interval(0, 10)
    assert lo == 0.0
    assert hi == pytest.approx(0.2775, abs=1e-3)


def test_all_successes_upper_bound_is_one():
    lo, hi = wilson_interval(10, 10)
    assert hi == 1.0
    assert lo == pytest.approx(0.7225, abs=1e-3)


def test_midpoint_is_symmetric():
    lo, hi = wilson_interval(5, 10)
    assert lo == pytest.approx(0.2366, abs=1e-3)
    assert hi == pytest.approx(0.7634, abs=1e-3)


def test_interval_narrows_as_n_grows():
    _, hi_small = wilson_interval(5, 10)
    _, hi_big = wilson_interval(50, 100)
    assert (hi_big - 0.5) < (hi_small - 0.5)


def test_rejects_bad_inputs():
    with pytest.raises(ValueError):
        wilson_interval(0, 0)
    with pytest.raises(ValueError):
        wilson_interval(11, 10)
