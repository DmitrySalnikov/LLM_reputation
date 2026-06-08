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


# Prompt templates. These were hardcoded in src/games/reputation_pd.py; they now live
# here as editable config defaults so an experiment can swap them and so the exact text
# is persisted into the run's stored config. Placeholders are substituted by literal
# string replacement (NOT str.format — the JSON examples contain real braces):
#   rules:        {R} {T} {P} {S}        <- payoff values
#   talk_prompt:  {partner} {round} {feed}
#   decide_prompt:{partner} {round} {feed}
DEFAULT_RULES = (
    "You are playing a repeated game. Each round you face one partner.\n"
    "Both of you secretly pick an integer from 0 to 9 at the same time.\n"
    "- Equal numbers: you each score {R}.\n"
    "- Your number exactly one higher than your partner's (0 counts as following 9): "
    "you score {T}, your partner {S}.\n"
    "- Your partner's number exactly one higher than yours: they score {T}, you {S}.\n"
    "- Any other case: you each score {P}.\n"
    "Before choosing you may exchange short messages. Messages are not binding; "
    "the final choice is secret and simultaneous. Maximize your own total score."
)

DEFAULT_TALK_PROMPT = (
    "Your partner this round is {partner}. Round {round}.\n"
    "Negotiation so far:\n{feed}\n\n"
    'Send a short message to your partner. Set "ready": true when you have nothing more to say.\n'
    'Respond ONLY as JSON: {"message": "<your message>", "ready": <true|false>}'
)

DEFAULT_DECIDE_PROMPT = (
    "Your partner this round is {partner}. Round {round}.\n"
    "Negotiation:\n{feed}\n\n"
    "Now secretly choose your number from 0 to 9.\n"
    'Respond ONLY as JSON: {"number": <0-9>, "rationale": "<short reason>"}'
)


@dataclass(frozen=True)
class GameCfg:
    payoffs: Payoffs = field(default_factory=Payoffs)
    max_talk_turns: int = 6          # hard ceiling on total cheap-talk turns in a pairing
    talk_stop_rule: str = "both_ready_latch"  # MVP: only this rule
    rules: str = DEFAULT_RULES               # system-prompt game rules ({R}/{T}/{P}/{S})
    talk_prompt: str = DEFAULT_TALK_PROMPT    # cheap-talk turn ({partner}/{round}/{feed})
    decide_prompt: str = DEFAULT_DECIDE_PROMPT  # decision turn ({partner}/{round}/{feed})


@dataclass(frozen=True)
class AgentSpec:
    persona: str
    provider: ProviderCfg
    count: int = 1                   # how many agents of this type to build


@dataclass(frozen=True)
class PopulationCfg:
    kind: str
    agents: list[AgentSpec]          # each spec expanded by its `count`; total = sum(counts)


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
    # NB: no db_path here — persistence lives in the separate Logger layer, not the orchestrator.


def _provider_cfg(d: dict) -> ProviderCfg:
    return ProviderCfg(**d)


def _game_cfg(d: dict) -> GameCfg:
    d = dict(d)
    payoffs = Payoffs(**d.pop("payoffs")) if "payoffs" in d else Payoffs()
    return GameCfg(payoffs=payoffs, **d)


def _population_cfg(d: dict) -> PopulationCfg:
    agents = [
        AgentSpec(persona=a["persona"], provider=_provider_cfg(a["provider"]),
                  count=a.get("count", 1))
        for a in d["agents"]
    ]
    return PopulationCfg(kind=d["kind"], agents=agents)


def load_episode(path: str) -> EpisodeCfg:
    """Load one episode config from YAML. pyyaml resolves &anchors / *aliases itself,
    so a provider shared via *default arrives as the same dict for every agent."""
    with open(path) as f:
        d = yaml.safe_load(f)
    return EpisodeCfg(
        seed=d["seed"],
        rounds=d["rounds"],
        matchmaker=d["matchmaker"],
        population=_population_cfg(d["population"]),
        game=_game_cfg(d.get("game", {})),
        context_window=d.get("context_window"),
        idle_payoff=d.get("idle_payoff", 1.0),
        max_concurrency=d.get("max_concurrency", 4),
    )
