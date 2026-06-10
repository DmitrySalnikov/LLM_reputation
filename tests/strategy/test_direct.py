from __future__ import annotations

from conftest import ScriptedProvider

from src.core.agent import Agent, AgentSetup
from src.core.config import ProviderCfg
from src.strategy.direct import DirectStrategy


def _agent(replies):
    cfg = ProviderCfg(base_url="http://x/v1", model="m")
    return Agent("A1", AgentSetup("You are A1.", cfg), ScriptedProvider(replies))


async def test_direct_returns_parsed_number_no_prediction():
    agent = _agent(['{"number": 6, "rationale": "because"}'])
    d = await DirectStrategy().decide(agent, "A2", round=1, feed="", rules="R")
    assert d.number == 6
    assert d.rationale == "because"
    assert d.predicted is None
    assert d.predicted_rationale is None


async def test_direct_rationale_off_asks_bare_number_and_drops_text():
    agent = _agent(['{"number": 6, "rationale": "volunteered anyway"}'])
    d = await DirectStrategy(rationale=False).decide(agent, "A2", round=1, feed="", rules="R")
    assert d.number == 6
    assert d.rationale == ""                      # volunteered text never reaches memory
    _, messages = agent.provider.calls[0]
    assert "rationale" not in messages[-1].content.lower()
