from __future__ import annotations

import random

import httpx
import pytest

from src.core.config import AgentSpec, EpisodeCfg, GameCfg, PopulationCfg, ProviderCfg
from src.core.orchestrator import run_episode
from src.population import make_population

OLLAMA_URL = "http://localhost:11434/v1"
MODEL = "llama3:8b"


def _ollama_up() -> bool:
    try:
        httpx.get("http://localhost:11434/api/tags", timeout=1.0)
        return True
    except Exception:
        return False


@pytest.mark.skipif(not _ollama_up(), reason="local Ollama not reachable")
async def test_run_episode_against_ollama():
    spec = AgentSpec(count=2, system_prompt="You are a pragmatic player.")
    cfg = EpisodeCfg(
        seed=1,
        rounds=1,
        matchmaker="random",
        population=PopulationCfg(
            kind="roster",
            agents=[spec],
            provider=ProviderCfg(base_url=OLLAMA_URL, model=MODEL, temperature=0.7, max_tokens=256),
        ),
        game=GameCfg(max_talk_turns=2),
    )
    pop = make_population(cfg.population, context_window=cfg.context_window).build(
        random.Random(cfg.seed)
    )
    records = []
    try:
        await run_episode(cfg, pop, observer=lambda r, p, recs: records.extend(recs))
    finally:
        await pop.aclose()

    assert {a.id for a in pop} == {"A1", "A2"}
    assert len(records) == 1
    rec = records[0]
    assert rec.outcome in {"CC", "DC", "CD", "DD"}
    assert 0 <= rec.a_number <= 9 and 0 <= rec.b_number <= 9
