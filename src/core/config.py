from __future__ import annotations

import random
from dataclasses import asdict, dataclass, field

import yaml


@dataclass(frozen=True)
class ProviderCfg:
    base_url: str
    model: str
    api_key_env: str = ""
    temperature: float = 0.7
    max_tokens: int = 512
    timeout_s: float = 120.0
    # Reasoning control for reasoning models (e.g. DeepSeek-V4-Pro on Together).
    # reasoning=False -> {"reasoning": {"enabled": false}} is sent in the payload (Non-think); True (default)
    # sends nothing — the provider decides on its own (the field is ignored for non-reasoning models).
    # reasoning_effort (if non-empty) -> {"reasoning_effort": "<val>"} ("high"/"max"; groundwork for the future).
    reasoning: bool = True
    reasoning_effort: str = ""
    # Arbitrary extra payload fields, sent as-is (provider-specific). E.g.,
    # disabling thinking for Qwen3 on vLLM: {"chat_template_kwargs": {"enable_thinking": false}}.
    # Merged into the payload last and can override the base fields.
    extra_body: dict = field(default_factory=dict)


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
# `msg_*`, `reason_*` fields below). The whole input is one flowing transcript:
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

# The agent's full system prompt — ONE template string. There is no longer the old
# identity+persona+rules assembly: the whole system prompt is given by one AgentSpec.system_prompt
# field. The engine substitutes into it only the {id} parameter and the payoffs
# {R}/{T}/{P}/{S}/{max_talk_turns} (Agent.system_prompt); everything else is taken verbatim.
# The default reproduces the old text (preamble + rules); it's convenient to set a shared
# prompt via a YAML anchor (&system_default) and reference it (*system_default).
DEFAULT_SYSTEM_PROMPT = DEFAULT_IDENTITY_PROMPT + "\n\n" + DEFAULT_RULES

DEFAULT_TALK_PROMPT = (
    "<game>Round {round} · opponent {partner}\n"
    "The chat has been opened.</game>\n"
    "{feed}\n"
    "<game>Your turn — reply to your opponent. "
    'Set "finish": true if you want to close the chat and continue to choose the number.\n'
    'Respond ONLY as JSON: {"message": "<your message>", "finish": <true|false>}</game>'
)

# The first turn of the round: the feed is empty, there is nothing to reply to -> the agent opens the conversation (no Talk block).
DEFAULT_TALK_OPEN_PROMPT = (
    "<game>Round {round} · opponent {partner}\n"
    "The chat has been opened. You speak first this round — send a short message to your opponent. "
    'Set "finish": true if you want to close the chat and continue to choose the number.\n'
    "Please write your first message in the following JSON format: "
    'Respond ONLY as JSON: {"message": "<your message>", "finish": <true|false>}</game>'
)

# DECIDE/PREDICT are fully static templates (only {round}/{partner}/{feed}/{reason} are
# substituted) — no text is assembled from chunks. The `rationale` flag picks ONE whole
# template: the rationale variant asks to reason first, the _BARE variant asks only for the
# number. Both are complete and readable on their own.
DEFAULT_DECIDE_PROMPT = (
    "<game>Round {round} · opponent {partner}\n"
    "The chat has been opened.</game>\n"
    "{feed}\n"
    "<game>The chat has been closed as {reason}. Choose the number. "
    "Reason first, then commit to a number.\n"
    'Respond ONLY as JSON: {"rationale": "<short reason>", "number": <0-9>}</game>'
)

DEFAULT_DECIDE_PROMPT_BARE = (
    "<game>Round {round} · opponent {partner}\n"
    "The chat has been opened.</game>\n"
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
#   history_round_prompt:  {round} {partner}
#   msg_self / msg_partner:  one cheap-talk line ({text}; partner-form also {partner})
#   history_close_prompt:  {reason}  (same wording as the live decide close line)
#   reason_limit / reason_agreed:  the {reason} phrase
#   history_result_prompt: {round} {partner} {partner_number} {payoff} {partner_payoff} {total}
#                          ({total} = score after the round; own number shown above as a <you> line)
DEFAULT_HISTORY_ROUND_PROMPT = (
    "<game>Round {round} · opponent {partner}\nThe chat has been opened.</game>"
)
DEFAULT_MSG_SELF = "<you>{text}</you>"
DEFAULT_MSG_PARTNER = "<{partner}>{text}</{partner}>"
DEFAULT_HISTORY_CLOSE_PROMPT = "<game>The chat has been closed as {reason}. Choose the number.</game>"
# The rationale variant of the history close line: mirrors the live decide_prompt (without the
# JSON tail) when rationale is enabled. The bare/rationale choice in _render_entry follows the
# cfg.rationale flag, same as in decide. Additive: history_close_prompt (no suffix) remains the
# bare variant, the old key is left untouched.
DEFAULT_HISTORY_CLOSE_PROMPT_RATIONALE = (
    "<game>The chat has been closed as {reason}. Give your rationale first, then choose the number.</game>"
)
DEFAULT_REASON_LIMIT = "the messages number limit has been reached"
DEFAULT_REASON_AGREED = "both players agreed to stop"
DEFAULT_HISTORY_RESULT_PROMPT = (
    "<game>The choice has been accepted. {partner} chose {partner_number}. "
    "Payoffs: you = {payoff}, {partner} = {partner_payoff}.\n"
    "Your total score after round {round} is {total} points.</game>"
)

# Private trace lines in a past round's transcript — the agent's own scratch notes
# (prediction / reasoning / takeaway). Tagged <you>; each is rendered only when its field is
# present AND its own flag (show_predicted / show_rationale / show_reflection) is on.
# Placeholders: {partner} {my_predicted} (predicted), {my_rationale} {my_number} (rationale),
# {my_reflection}.
DEFAULT_HISTORY_PREDICTED_PROMPT = "<you>(I predicted {partner} would pick {my_predicted})</you>"
# history_rationale_prompt with rationale enabled (show_rationale) is the agent's response block:
# a single <you> block repeating the JSON response {rationale, number} — rationale and number
# together, as one message (rationale before the number, the same order as in decide_prompt), in
# place of the choice, before the revealing result. If rationale is off, the number is rendered
# as a separate msg_self line and this template is not used. Placeholders: {my_rationale} {my_number}.
DEFAULT_HISTORY_RATIONALE_PROMPT = "<you>rationale: {my_rationale}\nnumber: {my_number}</you>"
DEFAULT_HISTORY_REFLECTION_PROMPT = "<you>(my takeaway: {my_reflection})</you>"

# PREDICT mirrors DECIDE byte-for-byte (same transcript open/close lines, same {reason});
# only the directive differs — predict the opponent's number instead of choosing your own.
DEFAULT_PREDICT_PROMPT = (
    "<game>Round {round} · opponent {partner}\n"
    "The chat has been opened.</game>\n"
    "{feed}\n"
    "<game>The chat has been closed as {reason}. "
    "Predict the number your opponent will secretly choose, from 0 to 9. "
    "Reason first, then commit to a number.\n"
    'Respond ONLY as JSON: {"rationale": "<short reason>", "number": <0-9>}</game>'
)

DEFAULT_PREDICT_PROMPT_BARE = (
    "<game>Round {round} · opponent {partner}\n"
    "The chat has been opened.</game>\n"
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


# Correction on a parse retry: appended to the user message WHEN a phase response fails to
# parse (Agent.act, max 2 retries). The text used to be hardcoded in src/core/agent.py as a
# single dict and for DECIDE/PREDICT always required the rationale schema — even in bare mode
# (rationale=false), which contradicted the prompt itself. Now these are config fields, one per
# phase plus a bare variant, and the engine picks bare/rationale the same way as for the prompt
# itself (by the rationale flag). No placeholders — the text goes out verbatim.
DEFAULT_TALK_CORRECTION = (
    "Respond with ONLY valid JSON, nothing else: "
    '{"message": "<your message>", "finish": <true|false>}'
)
DEFAULT_DECIDE_CORRECTION = (
    "Respond with ONLY valid JSON, nothing else: "
    '{"rationale": "<short reason>", "number": <integer 0-9>}'
)
DEFAULT_DECIDE_CORRECTION_BARE = (
    'Respond with ONLY valid JSON, nothing else: {"number": <integer 0-9>}'
)
DEFAULT_PREDICT_CORRECTION = DEFAULT_DECIDE_CORRECTION
DEFAULT_PREDICT_CORRECTION_BARE = DEFAULT_DECIDE_CORRECTION_BARE
DEFAULT_REFLECT_CORRECTION = (
    'Respond with ONLY valid JSON, nothing else: {"reflection": "<short reflection>"}'
)
DEFAULT_NOTE_CORRECTION = (
    'Respond with ONLY valid JSON, nothing else: {"notes": "<your notes>"}'
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

# The judge's correction on retry (used to be hardcoded in src/judge/judge.py). No placeholders.
DEFAULT_JUDGE_CORRECTION = (
    "Respond with ONLY valid JSON, nothing else: "
    '{"emerged": <true|false>, "explanation": "<short explanation>", '
    '"evidence": ["<message id>", ...]}'
)


@dataclass(frozen=True)
class GameCfg:
    payoffs: Payoffs = field(default_factory=Payoffs)
    max_talk_turns: int = 6          # hard ceiling on total cheap-talk turns in a pairing
    talk_stop_rule: str = "both_ready_latch"  # MVP: only this rule
    talk_prompt: str = DEFAULT_TALK_PROMPT       # cheap-talk turn ({partner}/{round}/{feed})
    talk_open_prompt: str = DEFAULT_TALK_OPEN_PROMPT  # first turn (empty feed): the agent opens the conversation
    # rationale=True -> *_prompt is used (asks to reason before the number),
    # rationale=False -> *_prompt_bare (number only). This is a choice of a WHOLE static template,
    # not a conditional text assembly. Empty -> the corresponding DEFAULT_*.
    rationale: bool = True           # ask for a rationale before the number in DECIDE/PREDICT
    decide_prompt: str = ""          # empty -> DEFAULT_DECIDE_PROMPT (rationale variant, {round}/{partner}/{feed}/{reason})
    decide_prompt_bare: str = ""     # empty -> DEFAULT_DECIDE_PROMPT_BARE (number only)
    predict_prompt: str = ""         # empty -> DEFAULT_PREDICT_PROMPT (rationale variant)
    predict_prompt_bare: str = ""    # empty -> DEFAULT_PREDICT_PROMPT_BARE (number only)
    reflect_prompt: str = DEFAULT_REFLECT_PROMPT  # post-game reflection (+{my_number}/{partner_number}/{payoff})
    reflection: bool = False         # post-game reflection: an extra LLM call after the outcome
    memory_notes_every: int = 0      # 0 = off; every N rounds PLAYED by the agent, it folds memory into notes
    notes_prompt: str = DEFAULT_NOTES_PROMPT  # note-call template ({round}/{score})
    notes_block_prompt: str = DEFAULT_NOTES_BLOCK_PROMPT  # notes wrapper in history ({notes})
    notes_header: str = DEFAULT_NOTES_HEADER    # header label above the consolidated notes
    buffer_header: str = DEFAULT_BUFFER_HEADER  # header label above the round buffer after consolidation
    # A past round's history is rendered to the agent as a game transcript (tags <game>/<you>/<name>);
    # these templates live in the config so the prompt text is not hardcoded in the code (see src/core/memory.py).
    history_round_prompt: str = DEFAULT_HISTORY_ROUND_PROMPT   # {round} {partner}
    msg_self: str = DEFAULT_MSG_SELF                           # the agent's own message line ({text})
    msg_partner: str = DEFAULT_MSG_PARTNER                     # the partner's message line ({partner}/{text})
    history_close_prompt: str = DEFAULT_HISTORY_CLOSE_PROMPT   # {reason} (bare / rationale=false)
    history_close_prompt_rationale: str = DEFAULT_HISTORY_CLOSE_PROMPT_RATIONALE  # {reason} (rationale=true, mirrors decide)
    reason_limit: str = DEFAULT_REASON_LIMIT                   # the {reason} phrase: chat closed due to the message limit
    reason_agreed: str = DEFAULT_REASON_AGREED                 # the {reason} phrase: both agreed to close the chat
    history_result_prompt: str = DEFAULT_HISTORY_RESULT_PROMPT  # {round} {partner} {partner_number} {payoff} {partner_payoff} {total}
    # Private traces in a past round's history (the agent's personal scratchpad) — each under ITS
    # OWN flag; the line is added only if the flag is on AND its field is non-empty.
    show_predicted: bool = True                                  # whether to add the prediction line
    show_rationale: bool = True                                  # whether to add the rationale line
    show_reflection: bool = True                                 # whether to add the reflection line
    history_predicted_prompt: str = DEFAULT_HISTORY_PREDICTED_PROMPT    # {partner} {my_predicted}
    history_rationale_prompt: str = DEFAULT_HISTORY_RATIONALE_PROMPT    # {my_rationale} {my_number} (rationale + number as one block)
    history_reflection_prompt: str = DEFAULT_HISTORY_REFLECTION_PROMPT  # {my_reflection}
    # Corrections on parse retry (per phase; DECIDE/PREDICT — bare/rationale same as the prompt itself).
    # No placeholders — appended verbatim to the user message when the response fails to parse.
    talk_correction: str = DEFAULT_TALK_CORRECTION
    decide_correction: str = DEFAULT_DECIDE_CORRECTION
    decide_correction_bare: str = DEFAULT_DECIDE_CORRECTION_BARE
    predict_correction: str = DEFAULT_PREDICT_CORRECTION
    predict_correction_bare: str = DEFAULT_PREDICT_CORRECTION_BARE
    reflect_correction: str = DEFAULT_REFLECT_CORRECTION
    note_correction: str = DEFAULT_NOTE_CORRECTION

    def __post_init__(self) -> None:
        """Fill empty DECIDE/PREDICT templates (both variants) with defaults.

        Each template is static: an empty string means "use the standard one", otherwise
        the exact given text is used. Which of the two variants reaches the agent is decided
        by the rationale flag in decide_context/predict_context — this is a choice of a whole
        template, not text assembly.
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
    """LLM judge configuration: a separate model that evaluates the episode after the game.

    The judge sees only the public cheap-talk; its model is configured independently of
    the agents' models. Absence of a judge block in the config = the judge is disabled.
    """

    provider: ProviderCfg
    prompt: str = DEFAULT_JUDGE_PROMPT   # English template with the {transcript} placeholder
    correction: str = DEFAULT_JUDGE_CORRECTION  # correction on retry when the response fails to parse


@dataclass(frozen=True)
class AgentSpec:
    count: int = 1                   # how many agents of this type to build
    play_strategy: str = "direct"        # "direct" | "prediction" — this spec's play strategy
    prediction_mapping: str = "match"    # predict->choice mapping (only when play_strategy="prediction")
    # The agent's full system prompt (ONE string). There's no longer a separate persona/identity_prompt/rules — it's all here.
    # {id} and the payoffs {R}/{T}/{P}/{S}/{max_talk_turns} are substituted; usually set via a YAML anchor.
    system_prompt: str = DEFAULT_SYSTEM_PROMPT


@dataclass(frozen=True)
class PopulationCfg:
    kind: str
    agents: list[AgentSpec]          # each spec expanded by its `count`; total = sum(counts)
    # The LLM provider, shared across the whole population (variation of the model between agents
    # is not needed — it's a fixed frame for the episode). Required, no default.
    provider: ProviderCfg
    # Optional human-name pools: if both are non-empty, agents are named "First Last" sampled
    # without repetition; otherwise they fall back to stable A1..An ids.
    first_name_pool: list[str] = field(default_factory=list)
    last_name_pool: list[str] = field(default_factory=list)


@dataclass(frozen=True)
class ChangePoint:
    """One episode schedule change point (takes effect from round `from_round` onward).

    Kinds of edits (can be combined in one point):
      patch   — partial override of a scalar config (game/payoffs/prompts/strategy etc.),
                **sticky**: applies from from_round onward (folded in via deep-merge).
      roster  — {"join": [...], "leave": [...]} — a roster mutation, an event (Phase 2).
      pairing — an explicit pairing split for THIS round, **one-off** (Phase 3).
      inject  — {agent_id: number} — force a number onto an agent for THIS round, one-off (Phase 4).

    Stored sparsely. The full round config is assembled by cfg_for_round (patch only);
    imperative directives (roster/pairing/inject) are handled by the controller (Phases 2-4)."""

    from_round: int
    patch: dict | None = None
    roster: dict | None = None
    pairing: tuple | None = None
    inject: dict | None = None


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
    judge: JudgeCfg | None = None          # None = the LLM judge is disabled
    schedule: tuple[ChangePoint, ...] = ()  # per-round schedule of edits (see cfg_for_round); empty = one config for the whole run
    # NB: the strategy (play_strategy/prediction_mapping) now lives on the agent (AgentSpec),
    # not on the episode — the population can be heterogeneous (direct + prediction in one episode).
    # NB: no db_path here — persistence lives in the separate Logger layer, not the orchestrator.


def _provider_cfg(d: dict) -> ProviderCfg:
    return ProviderCfg(**d)


def _game_cfg(d: dict) -> GameCfg:
    d = dict(d)
    d.pop("rules", None)             # legacy: rules are no longer a separate field — they travel inside system_prompt
    payoffs = Payoffs(**d.pop("payoffs")) if "payoffs" in d else Payoffs()
    return GameCfg(payoffs=payoffs, **d)


def _judge_cfg(d: dict) -> JudgeCfg:
    kwargs = {}
    if "prompt" in d:
        kwargs["prompt"] = d["prompt"]
    if "correction" in d:
        kwargs["correction"] = d["correction"]
    return JudgeCfg(provider=_provider_cfg(d["provider"]), **kwargs)


def _population_cfg(d: dict) -> PopulationCfg:
    # legacy persona/identity_prompt keys are simply ignored (a.get doesn't read them) — old
    # stored configs still load, just without the removed fields.
    agents = [
        AgentSpec(count=a.get("count", 1),
                  play_strategy=a.get("play_strategy", "direct"),
                  prediction_mapping=a.get("prediction_mapping", "match"),
                  system_prompt=a.get("system_prompt", DEFAULT_SYSTEM_PROMPT))
        for a in d["agents"]
    ]
    return PopulationCfg(
        kind=d["kind"],
        agents=agents,
        provider=_provider_cfg(d["provider"]),
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
        raise ValueError("judge block requires provider: the judge's model is configured separately")

    from src.games.talk_rules import make_talk_rule

    make_talk_rule(d.get("game", {}).get("talk_stop_rule", "both_ready_latch"))  # raises on unknown

    notes_every = d.get("game", {}).get("memory_notes_every", 0)
    if not isinstance(notes_every, int) or isinstance(notes_every, bool) or notes_every < 0:
        raise ValueError(f"memory_notes_every must be an integer >= 0, got: {notes_every!r}")

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

    # Early (fail-fast) validation of every schedule phase: sticky patch points are folded in
    # order, and EACH folded config must also be valid — otherwise the error would only surface
    # at the moment of the round. folded has no "schedule" key, so _validate on it only runs
    # the field checks (no repeated loop).
    schedule = d.get("schedule")
    if schedule:
        folded = {k: v for k, v in d.items() if k != "schedule"}
        patches = sorted((c for c in schedule if c.get("patch")), key=lambda c: c["from_round"])
        for cp in patches:
            folded = _deep_merge(folded, cp["patch"])
            _validate(folded)


def _change_point(c: dict) -> ChangePoint:
    """Build a ChangePoint from a dict (YAML or asdict). pairing — a list → a tuple of pairs."""
    pairing = c.get("pairing")
    return ChangePoint(
        from_round=c["from_round"],
        patch=c.get("patch"),
        roster=c.get("roster"),
        pairing=tuple(tuple(p) for p in pairing) if pairing is not None else None,
        inject=c.get("inject"),
    )


def _resolve_seed(seed):
    """Convert the config's `seed` field into a concrete int.

    `random` (a string, case-insensitive) means "pick a random seed at load time": every
    config load then produces a new seed. This is the ONLY intentional point of
    non-determinism in config assembly — a system entropy source (`SystemRandom`), not the
    simulation rng (that is still built from the already-resolved seed in runner). The
    chosen int is stored verbatim into the run (runs.seed/config), so the run itself stays
    reproducible by that number; on resume/extend the stored int is returned as-is (the
    `random` string is no longer there)."""
    if isinstance(seed, str) and seed.strip().lower() == "random":
        return random.SystemRandom().randrange(2 ** 31)
    return seed


def episode_from_dict(d: dict) -> EpisodeCfg:
    """Build an EpisodeCfg from a dict — the common path for YAML and for stored runs.config.

    Accepts both a YAML dict (load_episode) and asdict(cfg) from the DB (runner.resume_run
    when resuming/extending a run): both forms are structurally identical (game.payoffs is
    a nested dict, population.agents is a list of specs). Validation is one path for both."""
    _validate(d)
    return EpisodeCfg(
        seed=_resolve_seed(d["seed"]),
        rounds=d["rounds"],
        matchmaker=d["matchmaker"],
        population=_population_cfg(d["population"]),
        game=_game_cfg(d.get("game", {})),
        context_window=d.get("context_window"),
        idle_payoff=d.get("idle_payoff", 1.0),
        max_concurrency=d.get("max_concurrency", 4),
        judge=_judge_cfg(d["judge"]) if d.get("judge") else None,
        schedule=tuple(_change_point(c) for c in d.get("schedule") or ()),
    )


def _deep_merge(base: dict, patch: dict) -> dict:
    """Recursively apply patch onto base (a new dict).

    dict → recurse; everything else (scalars, lists) — replaced wholesale. Lists are NOT
    merged: this is deliberate — leaf fields are replaced, while composition (the list field
    population.agents) is changed by roster directives, not patch (Phase 2)."""
    out = dict(base)
    for k, v in patch.items():
        out[k] = _deep_merge(out[k], v) if isinstance(v, dict) and isinstance(out.get(k), dict) else v
    return out


def cfg_for_round(cfg: EpisodeCfg, r: int) -> EpisodeCfg:
    """Materialize the full EpisodeCfg for round r, folding in sticky patch points.

    A pure function: the same (cfg, r) → the same result. Imperative directives
    (roster/pairing/inject) are NOT applied here — they are handled by the controller
    (Phases 2-4). Without a schedule, the same object is returned (no rebuilding)."""
    if not cfg.schedule:
        return cfg
    d = asdict(cfg)
    d.pop("schedule", None)                              # the schedule is not part of a single round's config
    for cp in sorted(cfg.schedule, key=lambda c: c.from_round):
        if cp.from_round <= r and cp.patch:
            d = _deep_merge(d, cp.patch)
    return episode_from_dict(d)


def load_episode(path: str) -> EpisodeCfg:
    """Load one episode config from YAML. pyyaml resolves &anchors / *aliases itself,
    so a provider shared via *default arrives as the same dict for every agent."""
    with open(path) as f:
        d = yaml.safe_load(f)
    return episode_from_dict(d)
