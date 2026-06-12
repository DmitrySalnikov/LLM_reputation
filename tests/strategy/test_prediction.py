from __future__ import annotations

from conftest import ScriptedProvider

from src.core.agent import Agent, AgentSetup
from src.core.config import GameCfg, ProviderCfg
from src.strategy.mappings import get_mapping
from src.strategy.prediction import PredictionStrategy


def _agent(replies):
    cfg = ProviderCfg(base_url="http://x/v1", model="m")
    return Agent("A1", AgentSetup("You are A1.", cfg, "You are AI agent {id}."), ScriptedProvider(replies))


async def test_prediction_maps_predicted_to_final_choice():
    agent = _agent(['{"number": 4, "rationale": "mid"}'])
    d = await PredictionStrategy(get_mapping("one_above"), GameCfg()).decide(agent, "A2", 1, "", "R")
    assert d.predicted == 4          # predict-step output
    assert d.number == 5             # one_above mapping applied (4 -> 5)
    assert d.predicted_rationale == "mid"
    assert d.rationale == "mid"      # reasoning carried into the recorded rationale


async def test_prediction_match_mapping_is_identity():
    agent = _agent(['{"number": 8, "rationale": "high"}'])
    d = await PredictionStrategy(get_mapping("match"), GameCfg()).decide(agent, "A2", 1, "", "R")
    assert d.predicted == 8 and d.number == 8


async def test_prediction_rationale_off_asks_bare_number_and_drops_text():
    agent = _agent(['{"number": 4, "rationale": "volunteered anyway"}'])
    d = await PredictionStrategy(get_mapping("one_above"), GameCfg(rationale=False)).decide(
        agent, "A2", 1, "", "R")
    assert d.predicted == 4 and d.number == 5
    assert d.rationale == "" and d.predicted_rationale is None
    _, messages = agent.provider.calls[0]
    assert "rationale" not in messages[-1].content.lower()


def test_make_strategy_passes_rationale_flag():
    from src.core.config import EpisodeCfg, GameCfg, PopulationCfg, ProviderCfg
    from src.strategy.base import make_strategy

    pop = PopulationCfg(kind="roster", agents=[],
                        provider=ProviderCfg(base_url="http://x/v1", model="m"))
    cfg = EpisodeCfg(seed=1, rounds=1, matchmaker="random", population=pop,
                     game=GameCfg(rationale=False), play_strategy="prediction",
                     prediction_mapping="one_above")
    s = make_strategy(cfg)
    assert s._rationale is False

    cfg = EpisodeCfg(seed=1, rounds=1, matchmaker="random", population=pop,
                     game=GameCfg(), play_strategy="direct")
    assert make_strategy(cfg)._rationale is True
