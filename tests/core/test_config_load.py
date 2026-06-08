from __future__ import annotations

import textwrap

import pytest

from src.core.config import EpisodeCfg, GameCfg, load_episode

EXAMPLE = "config/example.yaml"


def test_load_example():
    cfg = load_episode(EXAMPLE)
    assert isinstance(cfg, EpisodeCfg)
    assert cfg.seed == 42 and cfg.rounds == 6
    assert cfg.matchmaker == "random"
    assert cfg.context_window is None
    assert cfg.idle_payoff == 1
    assert cfg.max_concurrency == 4
    assert cfg.population.kind == "roster"
    assert len(cfg.population.agents) == 2          # two agent types
    assert [a.count for a in cfg.population.agents] == [2, 2]
    assert sum(a.count for a in cfg.population.agents) == 4   # derived population size
    assert isinstance(cfg.game, GameCfg)
    assert cfg.game.payoffs.T == 5
    assert cfg.game.max_talk_turns == 8
    assert not hasattr(cfg, "db_path")              # persistence belongs to the Logger layer


def test_yaml_anchor_shared_provider():
    cfg = load_episode(EXAMPLE)
    p0 = cfg.population.agents[0].provider
    p1 = cfg.population.agents[1].provider
    assert p0 == p1                                  # &default / *default -> identical provider cfg
    assert p0.model == "llama3:8b"
    assert p0.base_url.endswith("/v1")
    assert p0.api_key_env == "OLLAMA_KEY"


def test_defaults_applied(tmp_path):
    f = tmp_path / "min.yaml"
    f.write_text(textwrap.dedent(
        """
        seed: 1
        rounds: 3
        matchmaker: random
        population:
          kind: roster
          agents:
            - persona: "p"
              provider: {base_url: "http://x/v1", model: "m"}
        """
    ))
    cfg = load_episode(str(f))
    assert cfg.idle_payoff == 1.0                    # default
    assert cfg.max_concurrency == 4                  # default
    assert cfg.context_window is None               # default
    assert cfg.population.agents[0].count == 1       # default count when omitted
    assert isinstance(cfg.game, GameCfg)            # default GameCfg when omitted
    assert cfg.game.payoffs.R == 3.0


def test_missing_required_raises(tmp_path):
    f = tmp_path / "bad.yaml"
    f.write_text("rounds: 3\nmatchmaker: random\n")  # no seed, no population
    with pytest.raises(KeyError):
        load_episode(str(f))
