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
    assert cfg.game.max_talk_turns == 3
    assert not hasattr(cfg, "db_path")              # persistence belongs to the Logger layer


def test_reflection_and_rationale_defaults(tmp_path):
    f = tmp_path / "min.yaml"
    f.write_text(textwrap.dedent(
        """
        seed: 1
        rounds: 3
        matchmaker: random
        population:
          kind: roster
          provider: {base_url: "http://x/v1", model: "m"}
          first_name_pool: [Kurisu, Mayuri]
          last_name_pool: [Makise, Shiina]
          agents:
            - persona: "p"
        """
    ))
    cfg = load_episode(str(f))
    assert cfg.game.reflection is False
    assert cfg.game.rationale is True            # думать перед числом — по умолчанию да


def test_rationale_loaded_from_game_block(tmp_path):
    f = tmp_path / "no_reason.yaml"
    f.write_text(textwrap.dedent(
        """
        seed: 1
        rounds: 3
        matchmaker: random
        game: {rationale: false}
        population:
          kind: roster
          provider: {base_url: "http://x/v1", model: "m"}
          first_name_pool: [Kurisu, Mayuri]
          last_name_pool: [Makise, Shiina]
          agents:
            - persona: "p"
        """
    ))
    assert load_episode(str(f)).game.rationale is False


def test_reflection_loaded_from_game_block():
    cfg = load_episode(EXAMPLE)
    assert cfg.game.reflection is True   # включено в примере конфигурации


def test_population_provider_loaded():
    cfg = load_episode(EXAMPLE)
    p = cfg.population.provider                       # один провайдер на популяцию (&default / *default)
    assert p.model == "Qwen/Qwen2.5-7B-Instruct-Turbo"
    assert p.base_url.endswith("/v1")
    assert p.api_key_env == "TOGETHER_API_KEY"


def test_defaults_applied(tmp_path):
    f = tmp_path / "min.yaml"
    f.write_text(textwrap.dedent(
        """
        seed: 1
        rounds: 3
        matchmaker: random
        population:
          kind: roster
          provider: {base_url: "http://x/v1", model: "m"}
          agents:
            - persona: "p"
        """
    ))
    cfg = load_episode(str(f))
    assert cfg.idle_payoff == 1.0                    # default
    assert cfg.max_concurrency == 4                  # default
    assert cfg.context_window is None               # default
    assert cfg.population.agents[0].count == 1       # default count when omitted
    assert cfg.population.first_name_pool == []      # pools optional -> empty by default
    assert isinstance(cfg.game, GameCfg)            # default GameCfg when omitted
    assert cfg.game.payoffs.R == 3.0


def test_persona_optional_defaults_to_none(tmp_path):
    f = tmp_path / "no_persona.yaml"
    f.write_text(textwrap.dedent(
        """
        seed: 1
        rounds: 3
        matchmaker: random
        population:
          kind: roster
          provider: {base_url: "http://x/v1", model: "m"}
          agents:
            - {}
        """
    ))
    cfg = load_episode(str(f))
    assert cfg.population.agents[0].persona is None


def test_provider_required_at_population_level(tmp_path):
    f = tmp_path / "no_provider.yaml"
    f.write_text(textwrap.dedent(
        """
        seed: 1
        rounds: 3
        matchmaker: random
        population:
          kind: roster
          agents:
            - {persona: "p"}
        """
    ))
    with pytest.raises(KeyError):                     # provider обязателен на уровне population, дефолта нет
        load_episode(str(f))


def test_identity_prompt_defaults_when_omitted(tmp_path):
    f = tmp_path / "no_identity.yaml"
    f.write_text(textwrap.dedent(
        """
        seed: 1
        rounds: 3
        matchmaker: random
        population:
          kind: roster
          provider: {base_url: "http://x/v1", model: "m"}
          agents:
            - {persona: "p"}
        """
    ))
    cfg = load_episode(str(f))                        # identity_prompt — общий на популяцию, с дефолтом
    assert cfg.population.identity_prompt == "You are AI agent {id}."


def test_identity_prompt_loaded_from_population_block(tmp_path):
    f = tmp_path / "human.yaml"
    f.write_text(textwrap.dedent(
        """
        seed: 1
        rounds: 3
        matchmaker: random
        population:
          kind: roster
          provider: {base_url: "http://x/v1", model: "m"}
          identity_prompt: "You are a human player named {id}."
          agents:
            - {persona: "p"}
        """
    ))
    cfg = load_episode(str(f))
    assert cfg.population.identity_prompt == "You are a human player named {id}."


def test_missing_required_raises(tmp_path):
    f = tmp_path / "bad.yaml"
    f.write_text("rounds: 3\nmatchmaker: random\n")  # no seed, no population
    with pytest.raises(KeyError):
        load_episode(str(f))


def test_load_example_has_name_pools():
    cfg = load_episode(EXAMPLE)
    total = sum(a.count for a in cfg.population.agents)
    assert len(cfg.population.first_name_pool) >= total
    assert len(cfg.population.last_name_pool) >= total


def test_default_play_strategy_is_direct():
    cfg = load_episode(EXAMPLE)                       # стратегия теперь на агенте (per-spec)
    assert all(a.play_strategy == "direct" for a in cfg.population.agents)
    assert all(a.prediction_mapping == "match" for a in cfg.population.agents)


def test_prediction_config_loads(tmp_path):
    f = tmp_path / "pred.yaml"
    f.write_text(textwrap.dedent(
        """
        seed: 1
        rounds: 2
        matchmaker: random
        population:
          kind: roster
          provider: {base_url: "http://x/v1", model: "m"}
          first_name_pool: [Kurisu, Mayuri, Itaru]
          last_name_pool: [Makise, Shiina, Hashida]
          agents:
            - persona: "p"
              count: 2
              play_strategy: prediction
              prediction_mapping: one_above
        """
    ))
    spec = load_episode(str(f)).population.agents[0]
    assert spec.play_strategy == "prediction"
    assert spec.prediction_mapping == "one_above"


def test_heterogeneous_strategies_per_spec(tmp_path):
    f = tmp_path / "mixed.yaml"
    f.write_text(textwrap.dedent(
        """
        seed: 1
        rounds: 2
        matchmaker: random
        population:
          kind: roster
          provider: {base_url: "http://x/v1", model: "m"}
          first_name_pool: [Kurisu, Mayuri]
          last_name_pool: [Makise, Shiina]
          agents:
            - {persona: "a", count: 1, play_strategy: direct}
            - {persona: "b", count: 1, play_strategy: prediction, prediction_mapping: one_above}
        """
    ))
    a, b = load_episode(str(f)).population.agents
    assert a.play_strategy == "direct"
    assert b.play_strategy == "prediction" and b.prediction_mapping == "one_above"


def _write_pop_yaml(tmp_path, *, strategy="direct", mapping="match",
                    firsts="[Kurisu, Mayuri]", lasts="[Makise, Shiina]", count=2):
    f = tmp_path / "c.yaml"
    f.write_text(textwrap.dedent(
        f"""
        seed: 1
        rounds: 2
        matchmaker: random
        population:
          kind: roster
          provider: {{base_url: "http://x/v1", model: "m"}}
          first_name_pool: {firsts}
          last_name_pool: {lasts}
          agents:
            - persona: "p"
              count: {count}
              play_strategy: {strategy}
              prediction_mapping: {mapping}
        """
    ))
    return str(f)


def test_unknown_play_strategy_raises(tmp_path):
    with pytest.raises(ValueError):
        load_episode(_write_pop_yaml(tmp_path, strategy="bogus"))


def test_unknown_prediction_mapping_raises(tmp_path):
    with pytest.raises(ValueError):
        load_episode(_write_pop_yaml(tmp_path, strategy="prediction", mapping="bogus"))


def test_pool_smaller_than_agent_count_raises(tmp_path):
    # firsts has 1 name but the population has 2 agents -> invalid
    with pytest.raises(ValueError):
        load_episode(_write_pop_yaml(tmp_path, firsts="[Only]", lasts="[Makise, Shiina]"))


def test_duplicate_pool_entries_raise(tmp_path):
    with pytest.raises(ValueError):
        load_episode(_write_pop_yaml(tmp_path, firsts="[Kurisu, Kurisu]"))


def test_missing_pools_fall_back(tmp_path):
    # pools are OPTIONAL: omitting them is valid and the roster falls back to A1..An ids
    f = tmp_path / "nopools.yaml"
    f.write_text(textwrap.dedent(
        """
        seed: 1
        rounds: 2
        matchmaker: random
        population:
          kind: roster
          provider: {base_url: "http://x/v1", model: "m"}
          agents:
            - persona: "p"
              count: 2
        """
    ))
    cfg = load_episode(str(f))                       # must NOT raise
    assert cfg.population.first_name_pool == []
    assert cfg.population.last_name_pool == []


# ---- LLM judge config (optional block, separate model) ----

def _judge_yaml(tmp_path, judge_block):
    f = tmp_path / "judge.yaml"
    f.write_text(textwrap.dedent(
        f"""
        seed: 1
        rounds: 2
        matchmaker: random
        {judge_block}
        population:
          kind: roster
          provider: {{base_url: "http://x/v1", model: "m"}}
          agents:
            - persona: "p"
        """
    ))
    return str(f)


def test_judge_absent_by_default():
    cfg = load_episode(EXAMPLE)
    assert cfg.judge is None


def test_judge_block_loads(tmp_path):
    path = _judge_yaml(tmp_path, 'judge: {provider: {base_url: "http://j/v1", model: "judge-m"}}')
    cfg = load_episode(path)
    assert cfg.judge is not None
    assert cfg.judge.provider.model == "judge-m"
    assert cfg.judge.provider.base_url == "http://j/v1"
    assert "{transcript}" in cfg.judge.prompt        # default prompt has the placeholder


def test_judge_custom_prompt_loads(tmp_path):
    path = _judge_yaml(
        tmp_path,
        'judge: {provider: {base_url: "http://j/v1", model: "judge-m"}, prompt: "Judge this: {transcript}"}',
    )
    assert load_episode(path).judge.prompt == "Judge this: {transcript}"


def test_judge_without_provider_raises(tmp_path):
    path = _judge_yaml(tmp_path, 'judge: {prompt: "no provider here"}')
    with pytest.raises(ValueError):
        load_episode(path)
