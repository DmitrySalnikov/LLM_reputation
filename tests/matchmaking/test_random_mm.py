from __future__ import annotations

import random

import pytest
from conftest import ScriptedProvider

from src.core.agent import Agent, AgentSetup
from src.core.config import GameCfg, ProviderCfg
from src.games.reputation_pd import ReputationPD
from src.matchmaking.base import RoundPlan, make_matchmaker
from src.matchmaking.random_mm import RandomMatchmaker


def _ids(n):
    return [f"A{i}" for i in range(1, n + 1)]


def _mm(ids, seed=0):
    mm = RandomMatchmaker()
    mm.setup(ids, random.Random(seed), None)
    return mm


# ---- Slice 1: types + factory ----

def test_factory_returns_random():
    assert isinstance(make_matchmaker("random"), RandomMatchmaker)


def test_factory_unknown_kind_raises():
    with pytest.raises(ValueError):
        make_matchmaker("nope")


# ---- Slice 2: plan_round + partition invariant ----

@pytest.mark.parametrize("n", [0, 1, 2, 3, 4, 5])
async def test_partition_invariant(n):
    ids = _ids(n)
    plan = await _mm(ids).plan_round(ids, 0)
    paired = [x for pair in plan.pairings for x in pair]
    # every id appears exactly once across pairings ∪ idle
    assert sorted(paired + plan.idle) == sorted(ids)
    assert len(set(paired)) == len(paired)            # no id twice in pairings
    assert not (set(paired) & set(plan.idle))         # pairings ∩ idle == ∅
    for x, y in plan.pairings:
        assert x != y                                 # distinct ids within a pair
    # idle exactly when N is odd; pairings count is N // 2
    assert len(plan.idle) == n % 2
    assert len(plan.pairings) == n // 2


async def test_input_not_mutated():
    ids = _ids(4)
    snapshot = list(ids)
    await _mm(ids).plan_round(ids, 0)
    assert ids == snapshot


# ---- Slice 3: determinism + seams + integration with the game ----

async def test_deterministic_same_seed():
    ids = _ids(6)
    mm_a, mm_b = _mm(ids, 42), _mm(ids, 42)
    for r in range(5):
        assert await mm_a.plan_round(ids, r) == await mm_b.plan_round(ids, r)


async def test_different_seed_differs():
    ids = _ids(6)
    mm_a, mm_b = _mm(ids, 1), _mm(ids, 2)
    plans_a = [await mm_a.plan_round(ids, r) for r in range(5)]
    plans_b = [await mm_b.plan_round(ids, r) for r in range(5)]
    assert plans_a != plans_b


async def test_seams_actor_ignored_and_events_empty():
    ids = _ids(2)
    mm = _mm(ids)
    actor_calls = []

    async def actor(*a, **k):
        actor_calls.append(1)

    plan = await mm.plan_round(ids, 0, actor)
    assert plan.events == []
    assert actor_calls == []                          # random matchmaker never calls actor


def _decide(n, rationale="r"):
    return '{"number": %d, "rationale": "%s"}' % (n, rationale)


def _agent(id):
    cfg = ProviderCfg(base_url="http://x/v1", model="m")
    return Agent(id, AgentSetup(f"You are {id}.", cfg), ScriptedProvider([_decide(4)]))


async def test_integration_matching_to_game():
    # 3 agents -> 1 pair + 1 idle; play the pair, idle agent stays untouched.
    agents = {i: _agent(i) for i in _ids(3)}
    ids = list(agents)
    plan = await _mm(ids).plan_round(ids, 0)
    assert len(plan.pairings) == 1 and len(plan.idle) == 1

    game = ReputationPD(GameCfg(max_talk_turns=0))
    for x, y in plan.pairings:
        rec = await game.play_pairing(agents[x], agents[y], 0)
        assert rec.outcome == "CC"                    # both scripted to pick 4
        assert len(agents[x].memory.entries) == 1
        assert len(agents[y].memory.entries) == 1
        assert agents[x].score == 3.0 and agents[y].score == 3.0

    idle = plan.idle[0]
    assert agents[idle].score == 0.0                  # matcher never touched the idle agent
    assert agents[idle].memory.entries == []


def test_roundplan_is_dataclass_equal():
    assert RoundPlan([("A1", "A2")], [], []) == RoundPlan([("A1", "A2")], [], [])
