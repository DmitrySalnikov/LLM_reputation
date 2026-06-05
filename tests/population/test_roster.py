from __future__ import annotations

import random

import pytest

from src.core.config import AgentSpec, PopulationCfg, ProviderCfg
from src.population import Population, make_population
from src.population import base as popbase


class FakeProvider:
    def __init__(self, cfg):
        self.cfg = cfg
        self.closed = 0

    async def complete(self, **kw):
        raise NotImplementedError

    async def aclose(self):
        self.closed += 1


@pytest.fixture
def created(monkeypatch):
    """Patch the provider factory so building a population creates FakeProviders;
    return the list of created providers for caching / aclose assertions."""
    made = []

    def factory(cfg):
        p = FakeProvider(cfg)
        made.append(p)
        return p

    monkeypatch.setattr(popbase, "make_provider", factory)
    return made


def _spec(persona, base_url="http://x/v1", model="m"):
    return AgentSpec(persona=persona, provider=ProviderCfg(base_url=base_url, model=model))


def _pop_cfg(n, specs):
    return PopulationCfg(kind="roster", n_agents=n, agents=specs)


def test_roster_cycles_personas_and_ids(created):
    specs = [_spec("p0"), _spec("p1")]
    pop = make_population(_pop_cfg(5, specs)).build(random.Random(0))
    assert isinstance(pop, Population)
    assert pop.ids() == ["A1", "A2", "A3", "A4", "A5"]
    assert [a.setup.persona for a in pop] == ["p0", "p1", "p0", "p1", "p0"]
    assert len(pop) == 5


def test_provider_cached_by_base_url_model(created):
    specs = [_spec("p0", model="m"), _spec("p1", model="m"), _spec("p2", model="other")]
    pop = make_population(_pop_cfg(3, specs)).build(random.Random(0))
    a1, a2, a3 = pop.get("A1"), pop.get("A2"), pop.get("A3")
    assert a1.provider is a2.provider          # same (base_url, model) -> shared client
    assert a1.provider is not a3.provider      # different model -> different client
    assert len(created) == 2                    # only two providers ever created


def test_context_window_threaded_to_agents(created):
    pop = make_population(_pop_cfg(2, [_spec("p")]), context_window=3).build(random.Random(0))
    assert pop.get("A1")._window == 3 and pop.get("A2")._window == 3


async def test_aclose_closes_each_unique_provider_once(created):
    specs = [_spec("p0", model="m"), _spec("p1", model="m"), _spec("p2", model="other")]
    pop = make_population(_pop_cfg(3, specs)).build(random.Random(0))
    await pop.aclose()
    assert len(created) == 2
    assert all(p.closed == 1 for p in created)   # each unique client closed exactly once


def test_make_population_unknown_kind_raises():
    cfg = PopulationCfg(kind="nope", n_agents=1, agents=[_spec("p")])
    with pytest.raises(ValueError):
        make_population(cfg)
