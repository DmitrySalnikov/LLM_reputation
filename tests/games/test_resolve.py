from __future__ import annotations

from src.core.config import GameCfg, Payoffs
from src.games.reputation_pd import ReputationPD


def _game(**kw):
    return ReputationPD(GameCfg(**kw))


def test_equal_is_cc():
    g = _game()
    for n in range(10):
        assert g.resolve(n, n) == ("CC", 3.0, 3.0)


def test_x_off_by_one_is_dc():
    g = _game()
    for y in range(10):
        x = (y + 1) % 10
        assert g.resolve(x, y) == ("DC", 5.0, 0.0)


def test_y_off_by_one_is_cd():
    g = _game()
    for x in range(10):
        y = (x + 1) % 10
        assert g.resolve(x, y) == ("CD", 0.0, 5.0)


def test_wraparound_zero_beats_nine():
    g = _game()
    assert g.resolve(0, 9) == ("DC", 5.0, 0.0)
    assert g.resolve(9, 0) == ("CD", 0.0, 5.0)


def test_exhaustive_classification():
    g = _game()
    R, T, P, S = 3.0, 5.0, 1.0, 0.0
    for x in range(10):
        for y in range(10):
            got = g.resolve(x, y)
            if x == y:
                assert got == ("CC", R, R)
            elif x == (y + 1) % 10:
                assert got == ("DC", T, S)
            elif y == (x + 1) % 10:
                assert got == ("CD", S, T)
            else:
                assert got == ("DD", P, P)


def test_no_mutual_betrayal():
    g = _game()
    for x in range(10):
        for y in range(10):
            _, px, py = g.resolve(x, y)
            assert not (px == 5.0 and py == 5.0)  # off-by-one is never mutual


def test_payoff_invariants():
    p = Payoffs()
    assert p.T > p.R > p.P > p.S
    assert 2 * p.R > p.T + p.S


def test_custom_payoffs():
    g = _game(payoffs=Payoffs(R=2, T=4, P=1, S=0))
    assert g.resolve(3, 3) == ("CC", 2, 2)
    assert g.resolve(4, 3) == ("DC", 4, 0)
    assert g.resolve(2, 5) == ("DD", 1, 1)
