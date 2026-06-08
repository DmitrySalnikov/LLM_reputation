from __future__ import annotations

import random

import pytest

from src.core import orchestrator as orch
from src.core.config import AgentSpec, EpisodeCfg, GameCfg, PopulationCfg, ProviderCfg
from src.population import base as popbase
from src.population import make_population
from src.providers.base import Completion, ProviderUnavailable


class FixedProvider:
    """Returns a constant DECIDE reply for every call — order-independent, so it is
    safe under gather even when shared by all agents via the provider cache."""

    def __init__(self, cfg, number: int = 4):
        self.cfg = cfg
        self.closed = 0
        self._reply = '{"number": %d, "rationale": "r"}' % number

    async def complete(self, **kw):
        return Completion(text=self._reply, prompt_tokens=2, completion_tokens=3, raw={})

    async def aclose(self):
        self.closed += 1


@pytest.fixture
def providers(monkeypatch):
    made = []
    monkeypatch.setattr(popbase, "make_provider", lambda cfg: made.append(FixedProvider(cfg)) or made[-1])
    return made


def _cfg(n=3, rounds=2, seed=0):
    spec = AgentSpec(persona="p", provider=ProviderCfg(base_url="http://x/v1", model="m"), count=n)
    return EpisodeCfg(
        seed=seed,
        rounds=rounds,
        matchmaker="random",
        population=PopulationCfg(kind="roster", agents=[spec]),
        game=GameCfg(max_talk_turns=0),     # decision-only -> deterministic CC for all
    )


async def _run(cfg, observer=None):
    """Caller owns the population: build it, run, close it; return it for inspection."""
    pop = make_population(cfg.population, context_window=cfg.context_window).build(
        random.Random(cfg.seed)
    )
    try:
        await orch.run_episode(cfg, pop, observer=observer)
    finally:
        await pop.aclose()
    return pop


async def test_run_episode_drives_rounds(providers):
    records = []
    pop = await _run(_cfg(n=3, rounds=2), observer=lambda r, p, recs: records.extend(recs))
    # 3 agents -> 1 pair + 1 idle per round; 2 rounds -> 2 pairings via the observer
    assert len(records) == 2
    assert all(rec.outcome == "CC" for rec in records)
    # final scores live on the (caller-owned) population
    scores = {a.id: a.score for a in pop}
    assert set(scores) == {"A1", "A2", "A3"}
    # per round: 2 paired +3 (CC) + 1 idle +1 = 7; 2 rounds -> 14
    assert sum(scores.values()) == pytest.approx(14.0)
    # one shared provider (same base_url/model), closed exactly once by the caller
    assert len(providers) == 1 and providers[0].closed == 1


async def test_observer_gets_each_round(providers):
    seen = []

    async def observer(r, plan, recs):
        seen.append((r, plan, recs))

    await _run(_cfg(n=4, rounds=3), observer=observer)
    assert [r for r, _, _ in seen] == [0, 1, 2]
    for r, plan, recs in seen:
        assert len(recs) == len(plan.pairings)      # recs align with the round's pairings
        assert len(plan.pairings) == 2 and plan.idle == []   # N=4 -> 2 pairs, no idle


async def test_sync_observer_supported(providers):
    seen = []
    await _run(_cfg(n=2, rounds=2), observer=lambda r, plan, recs: seen.append(r))
    assert seen == [0, 1]


async def test_fail_fast_propagates(monkeypatch):
    # provider raises -> fail-fast -> run_episode raises -> caller still closes providers
    class BoomProvider:
        def __init__(self, cfg):
            self.closed = 0

        async def complete(self, **kw):
            raise ProviderUnavailable("boom")

        async def aclose(self):
            self.closed += 1

    made = []
    monkeypatch.setattr(popbase, "make_provider", lambda cfg: made.append(BoomProvider(cfg)) or made[-1])
    with pytest.raises(ProviderUnavailable):
        await _run(_cfg(n=2, rounds=1))
    assert made and all(p.closed == 1 for p in made)
