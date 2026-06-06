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
    assert cfg.population.n_agents == 4
    assert len(cfg.population.agents) == 2          # cycled up to n_agents at build time
    assert isinstance(cfg.game, GameCfg)
    assert cfg.game.payoffs.T == 5
    assert cfg.game.max_talk_turns == 3
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
          n_agents: 2
          first_name_pool: [Kurisu, Mayuri]
          last_name_pool: [Makise, Shiina]
          agents:
            - persona: "p"
              provider: {base_url: "http://x/v1", model: "m"}
        """
    ))
    cfg = load_episode(str(f))
    assert cfg.idle_payoff == 1.0                    # default
    assert cfg.max_concurrency == 4                  # default
    assert cfg.context_window is None               # default
    assert isinstance(cfg.game, GameCfg)            # default GameCfg when omitted
    assert cfg.game.payoffs.R == 3.0


def test_missing_required_raises(tmp_path):
    f = tmp_path / "bad.yaml"
    f.write_text("rounds: 3\nmatchmaker: random\n")  # no seed, no population
    with pytest.raises(KeyError):
        load_episode(str(f))


def test_load_example_has_name_pools():
    cfg = load_episode(EXAMPLE)
    assert len(cfg.population.first_name_pool) >= cfg.population.n_agents
    assert len(cfg.population.last_name_pool) >= cfg.population.n_agents


def test_default_play_strategy_is_direct():
    cfg = load_episode(EXAMPLE)
    assert cfg.play_strategy == "direct"
    assert cfg.prediction_mapping == "match"


def test_prediction_config_loads(tmp_path):
    f = tmp_path / "pred.yaml"
    f.write_text(textwrap.dedent(
        """
        seed: 1
        rounds: 2
        matchmaker: random
        play_strategy: prediction
        prediction_mapping: one_above
        population:
          kind: roster
          n_agents: 2
          first_name_pool: [Kurisu, Mayuri, Itaru]
          last_name_pool: [Makise, Shiina, Hashida]
          agents:
            - persona: "p"
              provider: {base_url: "http://x/v1", model: "m"}
        """
    ))
    cfg = load_episode(str(f))
    assert cfg.play_strategy == "prediction"
    assert cfg.prediction_mapping == "one_above"


def _write_pop_yaml(tmp_path, *, strategy="direct", mapping="match",
                    firsts="[Kurisu, Mayuri]", lasts="[Makise, Shiina]", n=2):
    f = tmp_path / "c.yaml"
    f.write_text(textwrap.dedent(
        f"""
        seed: 1
        rounds: 2
        matchmaker: random
        play_strategy: {strategy}
        prediction_mapping: {mapping}
        population:
          kind: roster
          n_agents: {n}
          first_name_pool: {firsts}
          last_name_pool: {lasts}
          agents:
            - persona: "p"
              provider: {{base_url: "http://x/v1", model: "m"}}
        """
    ))
    return str(f)


def test_unknown_play_strategy_raises(tmp_path):
    with pytest.raises(ValueError):
        load_episode(_write_pop_yaml(tmp_path, strategy="bogus"))


def test_unknown_prediction_mapping_raises(tmp_path):
    with pytest.raises(ValueError):
        load_episode(_write_pop_yaml(tmp_path, strategy="prediction", mapping="bogus"))


def test_pool_smaller_than_n_agents_raises(tmp_path):
    with pytest.raises(ValueError):
        load_episode(_write_pop_yaml(tmp_path, firsts="[Only]", lasts="[Makise, Shiina]"))


def test_duplicate_pool_entries_raise(tmp_path):
    with pytest.raises(ValueError):
        load_episode(_write_pop_yaml(tmp_path, firsts="[Kurisu, Kurisu]"))


def test_missing_pools_raise(tmp_path):
    f = tmp_path / "nopools.yaml"
    f.write_text(textwrap.dedent(
        """
        seed: 1
        rounds: 2
        matchmaker: random
        population:
          kind: roster
          n_agents: 2
          agents:
            - persona: "p"
              provider: {base_url: "http://x/v1", model: "m"}
        """
    ))
    with pytest.raises(ValueError):
        load_episode(str(f))
