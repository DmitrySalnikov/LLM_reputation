from __future__ import annotations

from conftest import ScriptedProvider

from src.core.agent import Agent, AgentSetup
from src.core.config import GameCfg, ProviderCfg
from src.games.reputation_pd import ReputationPD


def _agent(id, replies):
    cfg = ProviderCfg(base_url="http://x/v1", model="m")
    return Agent(id, AgentSetup(f"You are {id}.", cfg), ScriptedProvider(replies))


def _decide(n, rationale="r"):
    return '{"number": %d, "rationale": "%s"}' % (n, rationale)


def _talk(msg, ready):
    return '{"message": "%s", "ready": %s}' % (msg, "true" if ready else "false")


# ---- Slice 2: decision only (talk off) ----

async def test_decision_only_cc():
    g = ReputationPD(GameCfg(max_talk_turns=0))
    a = _agent("A1", [_decide(4, "ra")])
    b = _agent("A2", [_decide(4, "rb")])
    rec = await g.play_pairing(a, b, round=1)
    assert rec.transcript == []
    assert (rec.a_number, rec.b_number, rec.outcome) == (4, 4, "CC")
    assert (rec.a_payoff, rec.b_payoff) == (3.0, 3.0)
    assert a.score == 3.0 and b.score == 3.0
    ea, eb = a.memory.entries[0], b.memory.entries[0]
    assert (ea.my_number, ea.partner_number, ea.my_rationale, ea.outcome) == (4, 4, "ra", "CC")
    assert (eb.my_number, eb.partner_number, eb.my_rationale, eb.outcome) == (4, 4, "rb", "CC")


async def test_decision_dc_mirrored_outcome():
    g = ReputationPD(GameCfg(max_talk_turns=0))
    a = _agent("A1", [_decide(5)])  # a == b+1 -> a betrayed b
    b = _agent("A2", [_decide(4)])
    rec = await g.play_pairing(a, b, 1)
    assert rec.outcome == "DC"
    assert (rec.a_payoff, rec.b_payoff) == (5.0, 0.0)
    assert a.memory.entries[0].outcome == "DC"   # I betrayed
    assert b.memory.entries[0].outcome == "CD"   # I was betrayed (mirrored)
    assert b.memory.entries[0].my_number == 4 and b.memory.entries[0].partner_number == 5


async def test_rationale_privacy():
    g = ReputationPD(GameCfg(max_talk_turns=0))
    a = _agent("A1", [_decide(1, "secret-a")])
    b = _agent("A2", [_decide(7, "secret-b")])
    rec = await g.play_pairing(a, b, 1)
    assert rec.a_rationale == "secret-a" and rec.b_rationale == "secret-b"
    assert a.memory.entries[0].my_rationale == "secret-a"
    # b's private rationale must not leak into a's memory entry
    assert "secret-b" not in str(a.memory.entries[0])


# ---- Slice 3: cheap-talk loop ----

async def test_talk_until_both_ready_latch():
    g = ReputationPD(GameCfg(max_talk_turns=6))
    a = _agent("A1", [_talk("hi", False), _talk("ok 4", True), _decide(4)])
    b = _agent("A2", [_talk("4?", True), _decide(4)])
    rec = await g.play_pairing(a, b, 1)  # a first
    assert [t["speaker"] for t in rec.transcript] == ["A1", "A2", "A1"]
    assert rec.transcript[-1]["ready"] is True
    assert rec.outcome == "CC"


async def test_talk_ceiling_caps_turns():
    g = ReputationPD(GameCfg(max_talk_turns=2))
    a = _agent("A1", [_talk("a1", False), _decide(3)])
    b = _agent("A2", [_talk("b1", False), _decide(5)])
    rec = await g.play_pairing(a, b, 1)
    assert [t["speaker"] for t in rec.transcript] == ["A1", "A2"]  # capped at 2


async def test_min_one_each_even_if_first_ready():
    g = ReputationPD(GameCfg(max_talk_turns=6))
    a = _agent("A1", [_talk("ready now", True), _decide(2)])
    b = _agent("A2", [_talk("ok", True), _decide(2)])
    rec = await g.play_pairing(a, b, 1)
    assert [t["speaker"] for t in rec.transcript] == ["A1", "A2"]  # b still gets a turn


async def test_latched_agent_stays_silent():
    g = ReputationPD(GameCfg(max_talk_turns=6))
    a = _agent("A1", [_talk("done", True), _decide(0)])
    b = _agent("A2", [_talk("hmm", False), _talk("ok", True), _decide(0)])
    rec = await g.play_pairing(a, b, 1)
    assert [t["speaker"] for t in rec.transcript] == ["A1", "A2", "A2"]


async def test_first_speaker_is_first_arg():
    # The matcher fixes who opens via pairing order: the first positional agent speaks first.
    g = ReputationPD(GameCfg(max_talk_turns=6))
    a = _agent("A1", [_talk("a", True), _decide(0)])
    b = _agent("A2", [_talk("b", True), _decide(0)])
    rec = await g.play_pairing(a, b, 1)  # a passed first -> A1 opens
    assert rec.transcript[0]["speaker"] == "A1"

    # swap the arguments -> A2 opens (matcher returned (A2, A1))
    a = _agent("A1", [_talk("a", True), _decide(0)])
    b = _agent("A2", [_talk("b", True), _decide(0)])
    rec = await g.play_pairing(b, a, 1)  # b passed first -> A2 opens
    assert rec.transcript[0]["speaker"] == "A2"


async def test_usage_aggregated():
    g = ReputationPD(GameCfg(max_talk_turns=2))
    a = _agent("A1", [_talk("a", False), _decide(1)])
    b = _agent("A2", [_talk("b", False), _decide(2)])
    rec = await g.play_pairing(a, b, 1)
    assert rec.usage["calls"] == 4  # 2 talk + 2 decide acts
    assert rec.usage["prompt_tokens"] == 8 and rec.usage["completion_tokens"] == 12


async def test_decide_context_contains_feed_and_ids():
    g = ReputationPD(GameCfg(max_talk_turns=2))
    a = _agent("A1", [_talk("take 4 plz", True), _decide(4)])
    b = _agent("A2", [_talk("ok", True), _decide(4)])
    await g.play_pairing(a, b, 7)
    system, messages = a.provider.calls[-1]  # a's DECIDE call
    ctx = messages[-1].content
    assert "A2" in ctx and "Round 7" in ctx and "take 4 plz" in ctx
    assert "0 to 9" in system  # rules went into the system prompt


# ---- Task 6: strategy delegation + prediction persistence ----

async def test_direct_strategy_leaves_predicted_none():
    g = ReputationPD(GameCfg(max_talk_turns=0))   # default DirectStrategy
    a = _agent("A1", [_decide(4)])
    b = _agent("A2", [_decide(4)])
    rec = await g.play_pairing(a, b, 1)
    assert rec.a_predicted is None and rec.b_predicted is None
    assert a.memory.entries[0].my_predicted is None


async def test_prediction_strategy_records_and_remembers_predictions():
    from src.strategy.mappings import get_mapping
    from src.strategy.prediction import PredictionStrategy

    g = ReputationPD(GameCfg(max_talk_turns=0),
                     strategy=PredictionStrategy(get_mapping("one_above")))
    a = _agent("A1", ['{"number": 4, "rationale": "pa"}'])   # predicts 4 -> chooses 5
    b = _agent("A2", ['{"number": 4, "rationale": "pb"}'])   # predicts 4 -> chooses 5
    rec = await g.play_pairing(a, b, 1)
    assert (rec.a_predicted, rec.b_predicted) == (4, 4)
    assert (rec.a_number, rec.b_number) == (5, 5)
    assert rec.outcome == "CC"
    # private scratchpad: the predicted value lives in the acting agent's memory
    assert a.memory.entries[0].my_predicted == 4
    assert "pa" in str(a.memory.entries[0])
    # the partner's prediction reasoning never leaks into a's memory entry
    assert "pb" not in str(a.memory.entries[0])
