from __future__ import annotations

import httpx
import pytest

from src.core.agent import Agent, AgentSetup
from src.core.config import GameCfg, ProviderCfg
from src.games.reputation_pd import ReputationPD
from src.providers import make_provider

OLLAMA_URL = "http://localhost:11434/v1"
MODEL = "llama3:8b"


def _ollama_up() -> bool:
    try:
        httpx.get("http://localhost:11434/api/tags", timeout=1.0)
        return True
    except Exception:
        return False


@pytest.mark.skipif(not _ollama_up(), reason="local Ollama not reachable")
async def test_pairing_against_ollama():
    cfg = ProviderCfg(base_url=OLLAMA_URL, model=MODEL, temperature=0.7, max_tokens=256)
    a = Agent("A1", AgentSetup("You are a pragmatic player.", cfg, "You are AI agent {id}."), make_provider(cfg))
    b = Agent("A2", AgentSetup("You are a cautious player.", cfg, "You are AI agent {id}."), make_provider(cfg))
    game = ReputationPD(GameCfg(max_talk_turns=4))
    try:
        rec = await game.play_pairing(a, b, 1)
    finally:
        await a.provider.aclose()
        await b.provider.aclose()
    assert 0 <= rec.a_number <= 9 and 0 <= rec.b_number <= 9
    assert rec.outcome in {"CC", "DC", "CD", "DD"}
    assert len(a.memory.entries) == 1 and len(b.memory.entries) == 1
    assert rec.usage["calls"] >= 2  # at least the two decisions
