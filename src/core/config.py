from __future__ import annotations

from dataclasses import dataclass, field

import yaml


@dataclass(frozen=True)
class ProviderCfg:
    base_url: str
    model: str
    api_key_env: str = ""
    temperature: float = 0.7
    max_tokens: int = 512
    timeout_s: float = 120.0


@dataclass(frozen=True)
class Payoffs:
    R: float = 3.0  # both cooperate
    T: float = 5.0  # successful betrayal (off-by-one)
    P: float = 1.0  # both defect / miscoordinate
    S: float = 0.0  # betrayed
    # invariants: T > R > P > S and 2R > T + S (strict PD)


@dataclass(frozen=True)
class GameCfg:
    payoffs: Payoffs = field(default_factory=Payoffs)
    max_talk_turns: int = 6          # hard ceiling on total cheap-talk turns in a pairing
    talk_stop_rule: str = "both_ready_latch"  # MVP: only this rule
    reflection: bool = False         # пост-игровая рефлексия: доп. LLM-вызов после исхода


@dataclass(frozen=True)
class AgentSpec:
    persona: str
    provider: ProviderCfg


@dataclass(frozen=True)
class PopulationCfg:
    kind: str
    n_agents: int
    agents: list[AgentSpec]          # shorter than n_agents -> cycled at build time
    first_name_pool: list[str] = field(default_factory=list)
    last_name_pool: list[str] = field(default_factory=list)


@dataclass(frozen=True)
class EpisodeCfg:
    seed: int
    rounds: int
    matchmaker: str
    population: PopulationCfg
    game: GameCfg
    context_window: int | None = None
    idle_payoff: float = 1.0         # C3: idle pays P by default
    max_concurrency: int = 4
    play_strategy: str = "direct"          # "direct" | "prediction"
    prediction_mapping: str = "match"      # используется только при play_strategy="prediction"
    # NB: no db_path here — persistence lives in the separate Logger layer, not the orchestrator.


def _provider_cfg(d: dict) -> ProviderCfg:
    return ProviderCfg(**d)


def _game_cfg(d: dict) -> GameCfg:
    d = dict(d)
    payoffs = Payoffs(**d.pop("payoffs")) if "payoffs" in d else Payoffs()
    return GameCfg(payoffs=payoffs, **d)


def _population_cfg(d: dict) -> PopulationCfg:
    agents = [
        AgentSpec(persona=a["persona"], provider=_provider_cfg(a["provider"]))
        for a in d["agents"]
    ]
    return PopulationCfg(
        kind=d["kind"],
        n_agents=d["n_agents"],
        agents=agents,
        first_name_pool=d.get("first_name_pool", []),
        last_name_pool=d.get("last_name_pool", []),
    )


def _validate(d: dict) -> None:
    """Проверить конфигурацию эпизода один раз при загрузке; падать быстро.

    Args:
        d: Словарь конфигурации эпизода, полученный из YAML.

    Raises:
        ValueError: При недопустимой стратегии, отображении или пулах имён.
    """
    strategy = d.get("play_strategy", "direct")
    if strategy not in ("direct", "prediction"):
        raise ValueError(
            f"play_strategy должен быть 'direct' или 'prediction', получено: {strategy!r}"
        )
    if strategy == "prediction":
        from src.strategy.mappings import get_mapping

        get_mapping(d.get("prediction_mapping", "match"))  # бросит ValueError при неизвестном имени

    pop = d["population"]
    n = pop["n_agents"]
    for key in ("first_name_pool", "last_name_pool"):
        if key not in pop:
            raise ValueError(f"в конфигурации population отсутствует обязательное поле {key!r}")
        pool = pop[key]
        if len(set(pool)) != len(pool):
            raise ValueError(f"{key} содержит повторяющиеся имена")
        if len(pool) < n:
            raise ValueError(f"{key} (размер {len(pool)}) меньше n_agents ({n})")


def load_episode(path: str) -> EpisodeCfg:
    """Load one episode config from YAML. pyyaml resolves &anchors / *aliases itself,
    so a provider shared via *default arrives as the same dict for every agent."""
    with open(path) as f:
        d = yaml.safe_load(f)
    _validate(d)
    return EpisodeCfg(
        seed=d["seed"],
        rounds=d["rounds"],
        matchmaker=d["matchmaker"],
        population=_population_cfg(d["population"]),
        game=_game_cfg(d.get("game", {})),
        context_window=d.get("context_window"),
        idle_payoff=d.get("idle_payoff", 1.0),
        max_concurrency=d.get("max_concurrency", 4),
        play_strategy=d.get("play_strategy", "direct"),
        prediction_mapping=d.get("prediction_mapping", "match"),
    )
