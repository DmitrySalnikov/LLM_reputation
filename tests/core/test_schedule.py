from __future__ import annotations

from src.core.config import (
    AgentSpec, ChangePoint, EpisodeCfg, GameCfg, PopulationCfg, ProviderCfg, cfg_for_round,
)


def _base(**kw):
    spec = AgentSpec(count=2)
    return EpisodeCfg(
        seed=0, rounds=10, matchmaker="random",
        population=PopulationCfg(kind="roster", agents=[spec],
                                 provider=ProviderCfg(base_url="http://x/v1", model="m")),
        game=GameCfg(max_talk_turns=0), **kw,
    )


def test_no_schedule_returns_base_unchanged():
    base = _base()
    assert base.schedule == ()
    assert cfg_for_round(base, 5) is base          # no schedule — same object, no rebuild


def test_scalar_patch_applies_from_its_round_onward():
    base = _base(schedule=(ChangePoint(from_round=4, patch={"game": {"payoffs": {"T": 6}}}),))
    assert cfg_for_round(base, 3).game.payoffs.T == base.game.payoffs.T   # before the change point — old
    assert cfg_for_round(base, 4).game.payoffs.T == 6                      # from the change point — new
    assert cfg_for_round(base, 9).game.payoffs.T == 6                      # and onward (sticky)


def test_later_change_point_overrides_earlier():
    base = _base(schedule=(ChangePoint(from_round=2, patch={"game": {"max_talk_turns": 4}}),
                           ChangePoint(from_round=6, patch={"game": {"max_talk_turns": 0}})))
    assert cfg_for_round(base, 1).game.max_talk_turns == 0     # before the first change point — base
    assert cfg_for_round(base, 5).game.max_talk_turns == 4     # first change point
    assert cfg_for_round(base, 6).game.max_talk_turns == 0     # second overrides


def test_change_points_unordered_in_schedule_still_fold_by_round():
    # order in the tuple doesn't matter — we fold by ascending from_round
    base = _base(schedule=(ChangePoint(from_round=6, patch={"game": {"max_talk_turns": 0}}),
                           ChangePoint(from_round=2, patch={"game": {"max_talk_turns": 4}})))
    assert cfg_for_round(base, 6).game.max_talk_turns == 0


def test_deep_merge_keeps_sibling_payoffs():
    # patching one payoff must not wipe out the others
    base = _base(schedule=(ChangePoint(from_round=1, patch={"game": {"payoffs": {"T": 9}}}),))
    r = cfg_for_round(base, 1).game.payoffs
    assert r.T == 9 and r.R == base.game.payoffs.R and r.S == base.game.payoffs.S and r.P == base.game.payoffs.P


def test_resolved_round_cfg_carries_no_schedule():
    # a materialized round config is the config for ONE round, not a schedule
    base = _base(schedule=(ChangePoint(from_round=1, patch={"idle_payoff": 2.0}),))
    assert cfg_for_round(base, 1).schedule == ()
    assert cfg_for_round(base, 1).idle_payoff == 2.0
