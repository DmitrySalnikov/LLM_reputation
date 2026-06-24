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
# persisted into the run's stored config. src/games/prompts.py and src/core/memory.py fill
# these placeholders by literal string replacement (NOT str.format — the JSON examples
# contain real braces):
#   identity:             {id}
#   rules:                {R} {T} {P} {S}              <- payoff values
#   talk/decide/predict:  {partner} {round} {feed}
#   reflect:              {partner} {round} {feed} {me} {my_number} {partner_number} {payoff}
# History (a past round is replayed to the agent as a game transcript — see the `history_*`,
# `msg_*`, `opener_*`, `reason_*` fields below). The whole input is one flowing transcript:
# the system rules declare the tags <game>/<you>/<opponent name>; memory renders past rounds
# with them, and the live talk/decide prompts continue the same transcript for this round.
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
    "Messages from the game are tagged <game></game>, your own lines <you></you>, "
    "and any other player's lines with their name <Name></Name>. On each turn you are "
    "given the transcript of your past rounds (if any) followed by the current situation; "
    "respond only with the exact JSON requested in that message."
)

# Фраза «ты открываешь раунд» — общий текст для истории (прошлый раунд, где первым ходил
# сам агент) и для живого talk_open_prompt, чтобы обе фазы читались дословно одинаково.
_OPENER_SELF = (
    "You speak first this round — send a short message to your opponent. "
    'Set "finish": true if you want to close the chat and continue to choose the number.'
)

DEFAULT_TALK_PROMPT = (
    "<game>Round {round} · opponent {partner}\n"
    "The chat has been open. {opener}</game>\n"
    "{feed}\n"
    "<game>Your turn — reply to your opponent. "
    'Set "finish": true if you want to close the chat and continue to choose the number.\n'
    'Respond ONLY as JSON: {"message": "<your message>", "finish": <true|false>}</game>'
)

# Первый ход раунда: фид пуст, отвечать не на что -> агент открывает разговор (без блока Talk).
DEFAULT_TALK_OPEN_PROMPT = (
    "<game>Round {round} · opponent {partner}\n"
    "The chat has been open. " + _OPENER_SELF + "\n"
    "Please, write your first message in the following JSON format: "
    'Respond ONLY as JSON: {"message": "<your message>", "finish": <true|false>}</game>'
)

# DECIDE/PREDICT are fully static templates (only {round}/{partner}/{feed}/{reason} are
# substituted) — no text is assembled from chunks. The `rationale` flag picks ONE whole
# template: the rationale variant asks to reason first, the _BARE variant asks only for the
# number. Both are complete and readable on their own.
DEFAULT_DECIDE_PROMPT = (
    "<game>Round {round} · opponent {partner}\n"
    "The chat has been open.</game>\n"
    "{feed}\n"
    "<game>The chat has been closed as {reason}. Choose the number. "
    "Reason first, then commit to a number.\n"
    'Respond ONLY as JSON: {"rationale": "<short reason>", "number": <0-9>}</game>'
)

DEFAULT_DECIDE_PROMPT_BARE = (
    "<game>Round {round} · opponent {partner}\n"
    "The chat has been open.</game>\n"
    "{feed}\n"
    "<game>The chat has been closed as {reason}. Choose the number.\n"
    'Respond ONLY as JSON: {"number": <0-9>}</game>'
)

# ── History (past-round replay) templates ────────────────────────────────────
# A finished round is replayed to the agent as a game transcript: an opening <game> line,
# the cheap-talk messages (own = <you>, opponent = <name>), a close line, the agent's own
# secret number as a <you> line, and a revealing <game> result line. src/core/memory.py
# fills the placeholders; the live talk/decide prompts above reuse `msg_self`/`msg_partner`
# to render the current round's feed, so past and present read identically.
#   history_round_prompt:  {round} {partner} {opener}
#   opener_self / opener_partner:  the {opener} sentence (partner-form takes {partner});
#                          opener_self is the same text the live talk_open_prompt opens with
#   msg_self / msg_partner:  one cheap-talk line ({text}; partner-form also {partner})
#   history_close_prompt:  {reason}  (same wording as the live decide close line)
#   reason_limit / reason_agreed:  the {reason} phrase
#   history_result_prompt: {round} {partner} {partner_number} {payoff} {partner_payoff} {total}
#                          ({total} = score after the round; own number shown above as a <you> line)
DEFAULT_HISTORY_ROUND_PROMPT = (
    "<game>Round {round} · opponent {partner}\nThe chat has been open. {opener}</game>"
)
DEFAULT_OPENER_SELF = _OPENER_SELF
DEFAULT_OPENER_PARTNER = "{partner} starts first:"
DEFAULT_MSG_SELF = "<you>{text}</you>"
DEFAULT_MSG_PARTNER = "<{partner}>{text}</{partner}>"
DEFAULT_HISTORY_CLOSE_PROMPT = "<game>The chat has been closed as {reason}. Choose the number.</game>"
DEFAULT_REASON_LIMIT = "the messages number limit has been reached"
DEFAULT_REASON_AGREED = "both players agreed to stop"
DEFAULT_HISTORY_RESULT_PROMPT = (
    "<game>The choice has been accepted. {partner} chose {partner_number}. "
    "Payoffs: you = {payoff}, {partner} = {partner_payoff}.\n"
    "Your total score after round {round} is {total} points.</game>"
)

# Private trace lines appended to a past round's transcript — the agent's own scratch notes
# (prediction / reasoning / takeaway). Tagged <you>; each is rendered only when its field is
# present AND its own flag (show_predicted / show_rationale / show_reflection) is on.
# Placeholders: {partner} {my_predicted} (predicted), {my_rationale}, {my_reflection}.
DEFAULT_HISTORY_PREDICTED_PROMPT = "<you>(I predicted {partner} would pick {my_predicted})</you>"
DEFAULT_HISTORY_RATIONALE_PROMPT = "<you>(my reasoning: {my_rationale})</you>"
DEFAULT_HISTORY_REFLECTION_PROMPT = "<you>(my takeaway: {my_reflection})</you>"

# PREDICT mirrors DECIDE byte-for-byte (same transcript open/close lines, same {reason});
# only the directive differs — predict the opponent's number instead of choosing your own.
DEFAULT_PREDICT_PROMPT = (
    "<game>Round {round} · opponent {partner}\n"
    "The chat has been open.</game>\n"
    "{feed}\n"
    "<game>The chat has been closed as {reason}. "
    "Predict the number your opponent will secretly choose, from 0 to 9. "
    "Reason first, then commit to a number.\n"
    'Respond ONLY as JSON: {"rationale": "<short reason>", "number": <0-9>}</game>'
)

DEFAULT_PREDICT_PROMPT_BARE = (
    "<game>Round {round} · opponent {partner}\n"
    "The chat has been open.</game>\n"
    "{feed}\n"
    "<game>The chat has been closed as {reason}. "
    "Predict the number your opponent will secretly choose, from 0 to 9.\n"
    'Respond ONLY as JSON: {"number": <0-9>}</game>'
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
# into private notes that REPLACE the raw round-by-round history from then on. Wrapped in
# <game> like the other game instructions. Placeholders (literal replacement): {round} {score}.
DEFAULT_NOTES_PROMPT = (
    "<game>All that history is being compressed into your notes. "
    "Write down every detail that will help you in future rounds.\n"
    'Respond ONLY as JSON: {"notes": "<your notes>"}</game>'
)

# How consolidated memory is rendered back into the transcript: the notes block, tagged
# <you> — it is the agent's own private memo (the rules declare <you> as "your own lines").
# Placeholder (literal replacement): {notes}.
DEFAULT_NOTES_BLOCK_PROMPT = "<you>{notes}</you>"
# Section headers framing the two parts of memory when notes are on: the consolidated notes
# and the raw buffer of rounds played since the last consolidation. Tagged <game> (system
# framing); the buffer header's <game> meets the first buffered round's <game> and the seam
# collapses (Agent._merge_game_blocks) so the header opens that round's block.
DEFAULT_NOTES_HEADER = "<game>Your notes from earlier rounds:</game>"
DEFAULT_BUFFER_HEADER = "<game>Your rounds since those notes:</game>"


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
    # rationale=True -> используется *_prompt (просит рассуждать перед числом),
    # rationale=False -> *_prompt_bare (только число). Это выбор ЦЕЛОГО статичного шаблона,
    # а не склейка текста по условию. Пусто -> соответствующий DEFAULT_*.
    rationale: bool = True           # просить обоснование перед числом в DECIDE/PREDICT
    decide_prompt: str = ""          # пусто -> DEFAULT_DECIDE_PROMPT (rationale-вариант, {round}/{partner}/{feed}/{reason})
    decide_prompt_bare: str = ""     # пусто -> DEFAULT_DECIDE_PROMPT_BARE (только число)
    predict_prompt: str = ""         # пусто -> DEFAULT_PREDICT_PROMPT (rationale-вариант)
    predict_prompt_bare: str = ""    # пусто -> DEFAULT_PREDICT_PROMPT_BARE (только число)
    reflect_prompt: str = DEFAULT_REFLECT_PROMPT  # post-game reflection (+{my_number}/{partner_number}/{payoff})
    reflection: bool = False         # пост-игровая рефлексия: доп. LLM-вызов после исхода
    memory_notes_every: int = 0      # 0 = off; каждые N СЫГРАННЫХ агентом раундов он сворачивает память в заметки
    notes_prompt: str = DEFAULT_NOTES_PROMPT  # шаблон note-вызова ({round}/{score})
    notes_block_prompt: str = DEFAULT_NOTES_BLOCK_PROMPT  # обёртка заметок в истории ({notes})
    notes_header: str = DEFAULT_NOTES_HEADER    # метка-заголовок над свёрнутыми заметками
    buffer_header: str = DEFAULT_BUFFER_HEADER  # метка-заголовок над буфером раундов после консолидации
    # История прошлого раунда отрисовывается агенту как игровой транскрипт (теги <game>/<you>/<имя>);
    # эти шаблоны живут в конфиге, чтобы текст промпта не был зашит в коде (см. src/core/memory.py).
    history_round_prompt: str = DEFAULT_HISTORY_ROUND_PROMPT   # {round} {partner} {opener}
    opener_self: str = DEFAULT_OPENER_SELF                     # фраза {opener}, когда первым говорил сам агент
    opener_partner: str = DEFAULT_OPENER_PARTNER               # фраза {opener}, когда первым говорил партнёр ({partner})
    msg_self: str = DEFAULT_MSG_SELF                           # строка реплики самого агента ({text})
    msg_partner: str = DEFAULT_MSG_PARTNER                     # строка реплики партнёра ({partner}/{text})
    history_close_prompt: str = DEFAULT_HISTORY_CLOSE_PROMPT   # {reason}
    reason_limit: str = DEFAULT_REASON_LIMIT                   # фраза {reason}: чат закрылся по лимиту реплик
    reason_agreed: str = DEFAULT_REASON_AGREED                 # фраза {reason}: оба согласились закрыть чат
    history_result_prompt: str = DEFAULT_HISTORY_RESULT_PROMPT  # {round} {partner} {partner_number} {payoff} {partner_payoff} {total}
    # Приватные следы в истории прошлого раунда (личный скрэтчпад агента) — каждый под СВОИМ
    # флагом; строка добавляется, только если флаг включён И её поле непусто.
    show_predicted: bool = True                                  # добавлять ли строку предсказания
    show_rationale: bool = True                                  # добавлять ли строку обоснования
    show_reflection: bool = True                                 # добавлять ли строку рефлексии
    history_predicted_prompt: str = DEFAULT_HISTORY_PREDICTED_PROMPT    # {partner} {my_predicted}
    history_rationale_prompt: str = DEFAULT_HISTORY_RATIONALE_PROMPT    # {my_rationale}
    history_reflection_prompt: str = DEFAULT_HISTORY_REFLECTION_PROMPT  # {my_reflection}

    def __post_init__(self) -> None:
        """Заполнить пустые шаблоны DECIDE/PREDICT (оба варианта) дефолтами.

        Каждый шаблон статичен: пустая строка означает «взять стандартный», иначе берётся
        ровно заданный текст. Какой из двух вариантов попадёт агенту, решает флаг rationale
        в decide_context/predict_context — это выбор целого шаблона, не склейка текста.
        """
        for name, default in (
            ("decide_prompt", DEFAULT_DECIDE_PROMPT),
            ("decide_prompt_bare", DEFAULT_DECIDE_PROMPT_BARE),
            ("predict_prompt", DEFAULT_PREDICT_PROMPT),
            ("predict_prompt_bare", DEFAULT_PREDICT_PROMPT_BARE),
        ):
            if not getattr(self, name):
                object.__setattr__(self, name, default)


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
    play_strategy: str = "direct"        # "direct" | "prediction" — стратегия игры этого спека
    prediction_mapping: str = "match"    # отображение predict->выбор (только при play_strategy="prediction")


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
    judge: JudgeCfg | None = None          # None = LLM-судья выключен
    # NB: стратегия (play_strategy/prediction_mapping) теперь живёт на агенте (AgentSpec),
    # а не на эпизоде — популяция может быть гетерогенной (direct + prediction в одном эпизоде).
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
        AgentSpec(persona=a.get("persona"), count=a.get("count", 1),
                  play_strategy=a.get("play_strategy", "direct"),
                  prediction_mapping=a.get("prediction_mapping", "match"))
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

    Raises ValueError on an unknown strategy/mapping or bad name pools. Strategy lives
    per-agent now (population.agents[*].play_strategy/prediction_mapping). Name pools are
    OPTIONAL: if a pool is empty the roster falls back to A1..An ids; a provided pool must
    be unique and hold at least one name per agent (size = sum of agent counts).
    """
    from src.strategy.mappings import get_mapping

    for spec in d["population"]["agents"]:
        strategy = spec.get("play_strategy", "direct")
        if strategy not in ("direct", "prediction"):
            raise ValueError(
                f"play_strategy must be 'direct' or 'prediction', got: {strategy!r}"
            )
        if strategy == "prediction":
            get_mapping(spec.get("prediction_mapping", "match"))  # raises on an unknown name

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
        judge=_judge_cfg(d["judge"]) if d.get("judge") else None,
    )
