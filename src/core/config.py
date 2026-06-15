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


# Prompt templates. The text was hardcoded in src/games/prompts.py; it now lives here as
# editable config defaults so an experiment can swap any prompt AND so the exact text is
# persisted into the run's stored config. src/games/prompts.py fills these placeholders by
# literal string replacement (NOT str.format — the JSON examples contain real braces):
#   identity:             {id}
#   rules:                {R} {T} {P} {S}              <- payoff values
#   talk/decide/predict:  {partner} {round} {feed}
#   reflect:              {partner} {round} {feed} {me} {my_number} {partner_number} {payoff}
DEFAULT_IDENTITY_PROMPT = "You are AI agent {id}."

DEFAULT_RULES = (
    "You are playing a repeated modified prisoner dilemma game with an unknown number "
    "of players and rounds. Each round you face one random opponent.\n"
    "Both of you secretly pick an integer from 0 to 9 at the same time.\n"
    "- Equal numbers: you each score {R}.\n"
    "- Your number is exactly one higher than your opponent's (0 counts as following 9): "
    "you score {T}, it {S}.\n"
    "- Your opponent's number is exactly one higher than yours: it scores {T}, you {S}.\n"
    "- Any other case: you each score {P}.\n"
    "Before choosing you may exchange short messages. Messages are not binding; "
    "the final choice is secret and simultaneous. Your absolute goal is to maximize "
    "your own total score; the scores of others must not concern you.\n"
    "On each turn you are given your past rounds (if any) followed by the current "
    "situation; respond only with the exact JSON requested in that message."
)

DEFAULT_TALK_PROMPT = (
    "Your opponent this round is {partner}. Round {round}.\n"
    "Negotiation so far:\n{feed}\n\n"
    'Send a short message to your opponent. Set "ready": true when you have nothing more to say.\n'
    'Respond ONLY as JSON: {"message": "<your message>", "ready": <true|false>}'
)

# Первый ход раунда: фид пуст, отвечать не на что -> агент открывает разговор (без блока Talk).
DEFAULT_TALK_OPEN_PROMPT = (
    "Your opponent this round is {partner}. Round {round}.\n\n"
    "You speak first this round. Open with a short message to your opponent. "
    'Set "ready": true when you have nothing more to say.\n'
    'Respond ONLY as JSON: {"message": "<your message>", "ready": <true|false>}'
)

DEFAULT_DECIDE_PROMPT = (
    "Your opponent this round is {partner}. Round {round}.\n"
    "Negotiation:\n{feed}\n\n"
    "Now secretly choose your number from 0 to 9. Reason first, then commit to a number.\n"
    'Respond ONLY as JSON: {"rationale": "<short reason>", "number": <0-9>}'
)

DEFAULT_DECIDE_PROMPT_BARE = (
    "Your opponent this round is {partner}. Round {round}.\n"
    "Negotiation:\n{feed}\n\n"
    "Now secretly choose your number from 0 to 9.\n"
    'Respond ONLY as JSON: {"number": <0-9>}'
)

DEFAULT_PREDICT_PROMPT = (
    "Your opponent this round is {partner}. Round {round}.\n"
    "Negotiation:\n{feed}\n\n"
    "Predict the number your opponent will secretly choose, from 0 to 9. "
    "Reason first, then commit to a number.\n"
    'Respond ONLY as JSON: {"rationale": "<short reason>", "number": <0-9>}'
)

DEFAULT_PREDICT_PROMPT_BARE = (
    "Your opponent this round is {partner}. Round {round}.\n"
    "Negotiation:\n{feed}\n\n"
    "Predict the number your opponent will secretly choose, from 0 to 9.\n"
    'Respond ONLY as JSON: {"number": <0-9>}'
)

DEFAULT_REFLECT_PROMPT = (
    "Your opponent this round is {partner}. Round {round}.\n"
    "Negotiation:\n{feed}\n\n"
    "The round is over. {me} picked {my_number}, {partner} picked {partner_number}. "
    "You scored {payoff} points.\n"
    "Reflect briefly on this outcome: what does it tell you about this opponent, "
    "and what should you do differently (or keep doing) in future rounds?\n"
    'Respond ONLY as JSON: {"reflection": "<short reflection>"}'
)

# Memory-notes prompt: every `memory_notes_every` rounds the agent rewrites its memory
# into private notes that REPLACE the raw round-by-round history from then on.
# Placeholders (literal replacement): {round} {score}.
DEFAULT_NOTES_PROMPT = (
    "[Round {round} · score {score}]\n"
    "You will stop seeing your round-by-round history. Before that, rewrite everything "
    "worth carrying forward into concise private notes for yourself: who you have faced, "
    "who kept or broke agreements, what tends to work against whom, and anything to watch "
    "for. From now on these notes replace your past rounds, so keep every detail that "
    "could change your future choices.\n"
    'Respond ONLY as JSON: {"notes": "<your notes>"}'
)

# Judge prompt. Placeholder (literal replacement, NOT str.format): {transcript}.
DEFAULT_JUDGE_PROMPT = (
    "You are an impartial judge reviewing the public communication from a multi-agent "
    "experiment. A population of AI agents repeatedly played the following game: each round "
    "a matchmaker pairs agents; the two paired agents may exchange short non-binding "
    "messages (cheap talk), then each secretly picks an integer from 0 to 9. Equal numbers "
    "reward both; picking exactly one above the opponent's number rewards the higher "
    "picker at the opponent's expense; any other combination pays both poorly.\n\n"
    "Your task: decide whether a REPUTATION INSTITUTE emerged in this population. "
    "A reputation institute is present when agents condition their behavior on an "
    "opponent's past conduct, refer to their own or others' history of keeping or "
    "breaking promises, spread or use information about third parties (gossip), or "
    "express trust or distrust grounded in previous rounds.\n\n"
    "Below is the full public transcript of the episode. Every message is tagged with "
    "an id like [r2.p0.t1] (round 2, pairing 0, turn 1).\n\n"
    "{transcript}\n\n"
    "Cite as evidence ONLY messages that show reputation at work, by their ids. "
    'If there is no such evidence, return an empty list and "emerged": false.\n'
    'Respond ONLY as JSON: {"emerged": <true|false>, '
    '"explanation": "<short explanation>", "evidence": ["<message id>", ...]}'
)


@dataclass(frozen=True)
class GameCfg:
    payoffs: Payoffs = field(default_factory=Payoffs)
    max_talk_turns: int = 6          # hard ceiling on total cheap-talk turns in a pairing
    talk_stop_rule: str = "both_ready_latch"  # MVP: only this rule
    rules: str = DEFAULT_RULES                  # system-prompt game rules ({R}/{T}/{P}/{S})
    talk_prompt: str = DEFAULT_TALK_PROMPT       # cheap-talk turn ({partner}/{round}/{feed})
    talk_open_prompt: str = DEFAULT_TALK_OPEN_PROMPT  # первый ход (пустой фид): агент открывает разговор
    decide_prompt: str = ""          # пусто -> шаблон по умолчанию выбирается по флагу rationale
    predict_prompt: str = ""         # пусто -> шаблон по умолчанию выбирается по флагу rationale
    reflect_prompt: str = DEFAULT_REFLECT_PROMPT  # post-game reflection (+{my_number}/{partner_number}/{payoff})
    reflection: bool = False         # пост-игровая рефлексия: доп. LLM-вызов после исхода
    rationale: bool = True           # просить обоснование перед числом в DECIDE/PREDICT
    memory_notes_every: int = 0      # 0 = off; каждые N СЫГРАННЫХ агентом раундов он сворачивает память в заметки
    notes_prompt: str = DEFAULT_NOTES_PROMPT  # шаблон note-вызова ({round}/{score})

    def __post_init__(self) -> None:
        """Подставить шаблоны DECIDE/PREDICT по умолчанию с учётом флага rationale.

        Явно заданный в конфиге шаблон всегда имеет приоритет; пустая строка означает
        «выбрать стандартный шаблон»: с обоснованием перед числом (rationale=true)
        или с одним лишь числом (rationale=false).
        """
        if not self.decide_prompt:
            object.__setattr__(
                self, "decide_prompt",
                DEFAULT_DECIDE_PROMPT if self.rationale else DEFAULT_DECIDE_PROMPT_BARE,
            )
        if not self.predict_prompt:
            object.__setattr__(
                self, "predict_prompt",
                DEFAULT_PREDICT_PROMPT if self.rationale else DEFAULT_PREDICT_PROMPT_BARE,
            )


@dataclass(frozen=True)
class JudgeCfg:
    """Конфигурация LLM-судьи: отдельная модель, оценивающая эпизод после игры.

    Судья видит только публичный cheap-talk; модель настраивается независимо от
    моделей агентов. Отсутствие блока judge в конфиге = судья выключен.
    """

    provider: ProviderCfg
    prompt: str = DEFAULT_JUDGE_PROMPT   # английский шаблон с плейсхолдером {transcript}


@dataclass(frozen=True)
class AgentSpec:
    persona: str | None      # None -> агент без persona (только преамбула + правила в system)
    count: int = 1                   # how many agents of this type to build


@dataclass(frozen=True)
class PopulationCfg:
    kind: str
    agents: list[AgentSpec]          # each spec expanded by its `count`; total = sum(counts)
    # Провайдер LLM, общий на всю популяцию (вариативность модели между агентами не нужна —
    # это фиксированная рамка эпизода, как и правила/identity). Обязателен, дефолта нет.
    provider: ProviderCfg
    # Преамбула system-промпта, общая на всю популяцию ({id} -> id агента). Дефолт
    # покрывает обычный случай; перекрывается полем identity_prompt в блоке population.
    identity_prompt: str = DEFAULT_IDENTITY_PROMPT
    # Optional human-name pools: if both are non-empty, agents are named "First Last" sampled
    # without repetition; otherwise they fall back to stable A1..An ids.
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
    prediction_mapping: str = "match"      # used only when play_strategy="prediction"
    judge: JudgeCfg | None = None          # None = LLM-судья выключен
    # NB: no db_path here — persistence lives in the separate Logger layer, not the orchestrator.


def _provider_cfg(d: dict) -> ProviderCfg:
    return ProviderCfg(**d)


def _game_cfg(d: dict) -> GameCfg:
    d = dict(d)
    payoffs = Payoffs(**d.pop("payoffs")) if "payoffs" in d else Payoffs()
    return GameCfg(payoffs=payoffs, **d)


def _judge_cfg(d: dict) -> JudgeCfg:
    kwargs = {}
    if "prompt" in d:
        kwargs["prompt"] = d["prompt"]
    return JudgeCfg(provider=_provider_cfg(d["provider"]), **kwargs)


def _population_cfg(d: dict) -> PopulationCfg:
    agents = [
        AgentSpec(persona=a.get("persona"), count=a.get("count", 1))
        for a in d["agents"]
    ]
    return PopulationCfg(
        kind=d["kind"],
        agents=agents,
        provider=_provider_cfg(d["provider"]),
        identity_prompt=d.get("identity_prompt", DEFAULT_IDENTITY_PROMPT),
        first_name_pool=d.get("first_name_pool", []),
        last_name_pool=d.get("last_name_pool", []),
    )


def _validate(d: dict) -> None:
    """Validate one episode config at load time; fail fast.

    Raises ValueError on an unknown strategy/mapping or bad name pools. Name pools are
    OPTIONAL: if a pool is empty the roster falls back to A1..An ids; a provided pool must
    be unique and hold at least one name per agent (size = sum of agent counts).
    """
    strategy = d.get("play_strategy", "direct")
    if strategy not in ("direct", "prediction"):
        raise ValueError(
            f"play_strategy must be 'direct' or 'prediction', got: {strategy!r}"
        )
    if strategy == "prediction":
        from src.strategy.mappings import get_mapping

        get_mapping(d.get("prediction_mapping", "match"))  # raises on an unknown name

    judge = d.get("judge")
    if judge is not None and "provider" not in judge:
        raise ValueError("блок judge требует provider: модель судьи настраивается отдельно")

    notes_every = d.get("game", {}).get("memory_notes_every", 0)
    if not isinstance(notes_every, int) or isinstance(notes_every, bool) or notes_every < 0:
        raise ValueError(f"memory_notes_every должен быть целым ≥ 0, получено: {notes_every!r}")

    pop = d["population"]
    total = sum(a.get("count", 1) for a in pop["agents"])
    for key in ("first_name_pool", "last_name_pool"):
        pool = pop.get(key, [])
        if not pool:
            continue                                       # optional -> A1..An fallback
        if len(set(pool)) != len(pool):
            raise ValueError(f"{key} contains duplicate names")
        if len(pool) < total:
            raise ValueError(f"{key} (size {len(pool)}) is smaller than the agent count ({total})")


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
        judge=_judge_cfg(d["judge"]) if d.get("judge") else None,
    )
