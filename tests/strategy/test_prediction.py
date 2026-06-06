from __future__ import annotations

from conftest import ScriptedProvider

from src.core.agent import Agent, AgentSetup
from src.core.config import ProviderCfg
from src.strategy.mappings import get_mapping
from src.strategy.prediction import PredictionStrategy


def _agent(replies):
    cfg = ProviderCfg(base_url="http://x/v1", model="m")
    return Agent("A1", AgentSetup("You are A1.", cfg), ScriptedProvider(replies))


async def test_prediction_maps_predicted_to_final_choice():
    agent = _agent(['{"number": 4, "rationale": "mid"}'])
    d = await PredictionStrategy(get_mapping("one_above")).decide(agent, "A2", 1, "", "R")
    assert d.predicted == 4          # predict-step output
    assert d.number == 5             # one_above mapping applied (4 -> 5)
    assert d.predicted_rationale == "mid"
    assert d.rationale == "mid"      # reasoning carried into the recorded rationale


async def test_prediction_match_mapping_is_identity():
    agent = _agent(['{"number": 8, "rationale": "high"}'])
    d = await PredictionStrategy(get_mapping("match")).decide(agent, "A2", 1, "", "R")
    assert d.predicted == 8 and d.number == 8
