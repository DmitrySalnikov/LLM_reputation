# LLM Reputation — Design Spec

**Date:** 2026-05-27
**Status:** Draft, pending user review

## Purpose

A research framework for studying whether a reputation system emerges spontaneously among LLM agents that play a repeated cyclic-undercut number game. Agents are never explicitly prompted to share information about other agents — any gossip during negotiation must be the agent's own initiative.

The deliverable is a configurable experimental harness that produces structured, queryable artifacts (SQLite + per-agent Markdown transcripts) for later analysis.

## Scope (v1)

In scope:
- Sequential execution of experiments with configurable population, schedule, payoffs, prompts, and memory mode
- OpenAI-compatible LLM client (works against OpenAI, Ollama, and vLLM via their `/v1` endpoints)
- Two play strategies (direct LLM choice; prediction-then-mapping) with pluggable prediction mappings
- Two memory modes (`round`, `episode`) with global-notes summarization at round boundaries
- SQLite persistence + per-agent Markdown transcripts
- Reproducibility via `RANDOM_SEED`

Out of scope (v1):
- Concurrency / parallelism within a round
- Resume-from-crash
- Per-agent strategy overrides (architecture supports it; not exposed in v1 config)
- Anthropic / native HF transformers adapters (architecture supports adding them; only OpenAI-compat ships)
- Analysis tooling — the framework produces data; analysis is downstream

## 1. Hierarchy & terminology

| Level | What it is |
|---|---|
| **Experiment** | One full run. Fixed agent population, fixed config. One process invocation. |
| **Round** | A batch where each agent meets K unique partners. The full pairing schedule for the round is announced to every agent at round start. |
| **Episode** | One encounter — two specific agents play `GAMES_PER_EPISODE` games together. The pair is **ordered**: the first agent in the pair always speaks first in every negotiation turn-0 of every game in this episode. |
| **Game** | The atomic interaction: fixed-turn negotiation, then a simultaneous cyclic choice ∈ {0..9}, then scoring. |

A Round contains `(AGENT_COUNT × K / 2)` episodes; each episode contains `GAMES_PER_EPISODE` games. Episodes within a round run sequentially.

### Payoff (cyclic) — default `ScoringRule`

Numbers form a cyclic ring of size 10 (0 follows 9). Given choices `a, b ∈ {0..9}`:

- `a == b` → both get **H2**
- `(a − b) mod 10 == 1` → **a** gets **H1**, **b** gets **H4**
- `(b − a) mod 10 == 1` → **b** gets **H1**, **a** gets **H4**
- otherwise → both get **H3**

Constraint: `H1 > H2 > H3 > H4`. Signs unrestricted (any of the four may be negative).

### Agent identity

Names are sampled at experiment startup from two pools in `.env`:
- `FIRST_NAME_POOL` — comma-separated; all entries unique; `|pool| ≥ AGENT_COUNT`
- `LAST_NAME_POOL` — comma-separated; all entries unique; `|pool| ≥ AGENT_COUNT`

For each of the `AGENT_COUNT` agents we sample one first name and one last name without replacement from each pool (seeded by `RANDOM_SEED`). Each agent ends up with a unique first AND unique last name (e.g., `"Alice Smith"`). Names are stable for the entire experiment and visible to partners.

### Critical prompt-design constraint

No built-in prompt asks an agent to describe, evaluate, or share information about other agents. The notes prompt says "reflect on the round/episode", not "tell us about each agent you met". Gossip during negotiation must be the agent's own initiative.

## 2. Architecture (Approach A — layered with Protocol extension points)

```
LLM_reputation/
├── .env.example                    # template the user copies to .env
├── pyproject.toml
├── main.py                         # CLI entry: load config → run experiment
├── src/llm_reputation/
│   ├── config.py                   # pydantic-settings; validates .env at startup
│   ├── domain.py                   # frozen dataclasses (Game, Episode, Round, GameResult)
│   ├── scoring.py                  # ScoringRule protocol + CyclicDiffRule
│   ├── scheduler.py                # PairingScheduler protocol + RandomKPartnerScheduler
│   ├── agent.py                    # Agent: identity, memory layers, prompt rendering
│   ├── strategy.py                 # PlayStrategy protocol + Direct/Prediction + PredictionMapping
│   ├── llm/
│   │   ├── client.py               # LLMClient protocol
│   │   └── openai_compat.py        # OpenAI-compatible client
│   ├── persistence/
│   │   ├── sink.py                 # EventSink protocol
│   │   ├── schema.sql              # SQLite DDL
│   │   ├── sqlite_sink.py          # writes events to SQLite
│   │   └── transcripts.py          # appends per-agent Markdown transcripts
│   └── orchestrator/
│       ├── experiment.py           # run_experiment(config) — top-level loop
│       ├── round.py                # run_round
│       ├── episode.py              # run_episode
│       └── game.py                 # run_game
├── tests/
│   ├── conftest.py                 # shared fixtures, FakeLLMClient, InMemorySink
│   ├── test_scoring.py
│   ├── test_scheduler.py
│   ├── test_name_generator.py
│   ├── test_config.py
│   ├── test_agent.py
│   ├── test_strategy.py
│   ├── test_persistence.py
│   └── test_orchestrator.py
└── docs/superpowers/specs/2026-05-27-llm-reputation-design.md
```

### Isolation properties

- `scoring.py`, `scheduler.py`, `domain.py` are pure (no I/O, no LLM, no DB) — unit-testable in microseconds.
- `agent.py` depends on `LLMClient` (injected, swappable for `FakeLLMClient` in tests) and renders prompts; it knows nothing about persistence or orchestration.
- `orchestrator/` is the only place that wires things together; each file is one loop level.
- `persistence/` is invoked by the orchestrator through `EventSink` — swap SQLite for any other backend without touching anything else.
- `llm/` holds adapters; today one (OpenAI-compat), future ones drop in as siblings.

### Extension points

| Add this | Implement | Register where |
|---|---|---|
| New scoring rule (different payoff math, different choice space) | `ScoringRule` Protocol in `scoring.py` | `config.py` factory |
| New pairing scheme | `PairingScheduler` Protocol in `scheduler.py` | `config.py` factory |
| New play strategy | `PlayStrategy` Protocol in `strategy.py` | `config.py` factory |
| New prediction mapping | `PredictionMapping` Protocol in `strategy.py` | `config.py` factory |
| New model backend (Anthropic, native Ollama, native vLLM) | `LLMClient` Protocol in `llm/client.py` | new file in `llm/`, `config.py` factory |
| New persistence sink (Postgres, JSONL, etc.) | `EventSink` Protocol in `persistence/sink.py` | `config.py` factory |

## 3. Configuration (`.env`)

Single `.env` file, loaded by `pydantic-settings` at startup. Validation runs once; bad config fails fast with a Russian error message. `.env.example` is committed as a template.

```ini
# === Population & schedule ===
AGENT_COUNT=4
FIRST_NAME_POOL="Alice,Bob,Carol,Dave,Eve,Frank"   # all unique, |pool| >= AGENT_COUNT
LAST_NAME_POOL="Smith,Jones,Brown,Davis,Wilson"    # all unique, |pool| >= AGENT_COUNT
ROUNDS_PER_EXPERIMENT=5
PARTNERS_PER_ROUND=3                                # K; each agent plays this many partners per round, no repeats
GAMES_PER_EPISODE=3
NEGOTIATION_TURNS_PER_AGENT=2                       # messages each agent sends per game before choosing

# === Payoffs (signs unrestricted; H1 > H2 > H3 > H4 enforced) ===
H1=10
H2=5
H3=0
H4=-5

# === Model (OpenAI-compatible; covers OpenAI / Ollama / vLLM via /v1) ===
MODEL_BASE_URL=https://api.openai.com/v1
MODEL_API_KEY=sk-...
MODEL_NAME=gpt-4o-mini
MODEL_TEMPERATURE=0.7
MODEL_MAX_TOKENS=512                                # per single response

# === Play strategy ===
PLAY_STRATEGY=direct                                # "direct" | "prediction"
PREDICTION_MAPPING=match                            # "match" | "one_above" — only used when PLAY_STRATEGY=prediction

# === Memory ===
MEMORY_MODE=round                                   # "round" | "episode"

# === Prompts (placeholders — fill in later) ===
AGENT_SYSTEM_PROMPT="<TODO: define persona for {name}>"
NEGOTIATION_PROMPT="<TODO>"
CHOICE_PROMPT="<TODO>"
PREDICTION_PROMPT="<TODO>"                          # only used when PLAY_STRATEGY=prediction
EPISODE_NOTES_PROMPT="<TODO>"                       # only used when MEMORY_MODE=episode
GLOBAL_NOTES_UPDATE_PROMPT="<TODO>"                 # used in both modes — given current global notes + recent events, produce new global notes

# === Runtime ===
RUN_DIR=runs
RANDOM_SEED=42
LOG_LEVEL=INFO
```

### Startup validation

1. `len(FIRST_NAME_POOL.split(",")) ≥ AGENT_COUNT`; all entries unique
2. `len(LAST_NAME_POOL.split(",")) ≥ AGENT_COUNT`; all entries unique
3. `0 < PARTNERS_PER_ROUND ≤ AGENT_COUNT − 1`
4. `(AGENT_COUNT × PARTNERS_PER_ROUND) mod 2 == 0` (otherwise no perfect schedule exists)
5. `H1 > H2 > H3 > H4`
6. `MODEL_BASE_URL` is a valid URL
7. If `PLAY_STRATEGY=prediction`, `PREDICTION_PROMPT` must be non-empty and `PREDICTION_MAPPING` must resolve to a registered mapping
8. If `MEMORY_MODE=episode`, `EPISODE_NOTES_PROMPT` must be non-empty
9. All required prompts (`AGENT_SYSTEM_PROMPT`, `NEGOTIATION_PROMPT`, `CHOICE_PROMPT`, `GLOBAL_NOTES_UPDATE_PROMPT`) must be non-empty

### Prompt template substitutions

Documented contract — only these variables are guaranteed; any other `{name}` in a template raises at startup.

- `AGENT_SYSTEM_PROMPT`: `{name}`, `{agent_names}`, `{h1}`, `{h2}`, `{h3}`, `{h4}`, `{rounds_per_experiment}`, `{partners_per_round}`, `{games_per_episode}`, `{negotiation_turns}`
- `NEGOTIATION_PROMPT`: `{partner_name}`, `{game_index}`, `{games_per_episode}`, `{turn_index}`, `{total_turns}`
- `CHOICE_PROMPT`: `{partner_name}`, `{game_index}`, `{games_per_episode}`
- `PREDICTION_PROMPT`: `{partner_name}`, `{game_index}`, `{games_per_episode}`
- `EPISODE_NOTES_PROMPT`: `{partner_name}`, `{round_index}`, `{episode_index}`
- `GLOBAL_NOTES_UPDATE_PROMPT`: `{round_index}`, `{place}`, `{round_income}`

Multi-line values in `.env` are supported (`python-dotenv` parses them inside double quotes).

## 4. Agent memory model

### Memory layers

| Layer | Lifetime | Visible to partner? |
|---|---|---|
| **System prompt** | Whole experiment | No |
| **Global notes** | Whole experiment (updated at round end) | No |
| **Round memory** | One round (reset at round end) | No |
| **Episode notes** *(mode `episode` only)* | One round (accumulated, then discarded at round end) | No |
| **Private game scratchpad** *(prediction strategy only)* | One episode | No |

**Round memory** holds the pairing-schedule announcement plus the agent's own episodes that have occurred so far in the current round (negotiations + outcomes). It is reset at round boundaries — only Global Notes survive across rounds.

### Context assembly for each LLM call

Each call to an agent is built fresh from the layers above (no hidden state in the LLM client):

```
[system]
  <rendered AGENT_SYSTEM_PROMPT>
  --- Current global notes ---
  <agent's current global_notes text>

[user/assistant alternating — round replay]
  Round 2 starts. Pairing schedule for this round: ...
  You are now paired with Bob Jones. You will play 3 games.
  Game 1 negotiation: <NEGOTIATION_PROMPT rendered>
  <assistant: agent's negotiation reply>
  <user: Bob's negotiation reply>
  ... (negotiation turns)
  Choice: <CHOICE_PROMPT rendered>
  <assistant: "5">
  Game 1 result: you chose 5, Bob chose 6. You earned -5, Bob earned 10.
  ... (next games / episodes)
  [If mode=episode and an episode has ended:]
    Episode 1 reflection: <EPISODE_NOTES_PROMPT rendered>
    <assistant: agent's episode note>

[user — the current question]
  <whichever prompt is being run: negotiate / predict / choose / write notes>
```

**Partner-privacy invariant:** the only `user` turns in this agent's context are the agent's own direct interactions with partners and orchestrator-injected events the agent is supposed to see (schedule, outcomes, standings). Events from *other* pairings happening in the same round never enter this agent's context. The Private game scratchpad (prediction reasoning) is in the agent's context but not in the partner's.

### Memory modes

**`MEMORY_MODE=round`**
- No episode notes are written.
- Within a round, agents have: system prompt + global notes + round memory of episodes played so far this round.
- At round end: agent is shown current global notes + the round summary, asked via `GLOBAL_NOTES_UPDATE_PROMPT` to produce an updated version. The new text replaces the old global notes.
- Round memory is cleared.

**`MEMORY_MODE=episode`**
- After each episode within a round: agent writes an episode note via `EPISODE_NOTES_PROMPT`. These notes are appended to the agent's round memory and visible during the rest of this round.
- At round end: agent is shown current global notes + all this round's episode notes, asked via `GLOBAL_NOTES_UPDATE_PROMPT` to produce updated global notes. The new text replaces the old global notes.
- All this round's episode notes are then discarded from agent memory.
- Round memory is cleared.

**Discarded notes are persisted.** Episode notes in mode `episode` (which leave the agent's working memory after round end) are stored permanently in SQLite and in transcripts. The agent forgets; the experiment record does not.

### Places announcement

After each round, total income for that round is announced to all agents as a ranking (`"Round 3 standings: 1. Alice Smith — 27; 2. Bob Jones — 14; …"`). The agent sees this *before* writing global notes — so the ranking can influence what gets remembered.

## 5. Play strategy

`PlayStrategy` is a Protocol with one method:

```python
class PlayStrategy(Protocol):
    async def decide(self, agent: Agent, partner: Agent, game_idx: int) -> int: ...
```

### `DirectLLMStrategy`

LLM is asked directly via `CHOICE_PROMPT`. Full agent context (system + global notes + round memory) is passed. The orchestrator parses an integer 0–9 from the response (see Section 6, parsing). The raw response is persisted alongside the parsed choice.

### `PredictionBasedStrategy`

Two-step decision:

1. **Predict:** LLM is called with `PREDICTION_PROMPT`. Full context provided. Response parsed as integer 0–9. The whole exchange (prompt + LLM's raw response + parsed prediction) is appended to the agent's *private game scratchpad* — visible in this agent's future context, never in the partner's.
2. **Map:** A pure function `PredictionMapping(predicted: int) -> int` produces the final choice.

### `PredictionMapping` implementations

| Name | Function |
|---|---|
| `match` | `choose = predicted` |
| `one_above` | `choose = (predicted + 1) mod 10` |

New mappings are added by implementing the Protocol and registering a name in `config.py`.

### Why prediction reasoning stays in agent memory

Even though the number drives the choice, the LLM's prediction-rationale is useful when it later writes notes — "I predicted Bob would pick 5 because he kept mentioning 'middle is safe'…" — which is the kind of reasoning the reputation research wants to surface.

### Per-agent strategy override

Not in v1 — all agents share the experiment's strategy. `PlayStrategy` is injected into `Agent` at construction, so per-agent strategies are a trivial later extension.

## 6. LLM client & output parsing

### `LLMClient` Protocol

```python
@dataclass(frozen=True)
class Message:
    role: Literal["system", "user", "assistant"]
    content: str

class LLMClient(Protocol):
    async def complete(self, messages: list[Message]) -> str: ...
```

### Default implementation: `OpenAICompatClient`

Wraps the `openai` async SDK with `MODEL_BASE_URL`, `MODEL_API_KEY`, `MODEL_NAME`, `MODEL_TEMPERATURE`, `MODEL_MAX_TOKENS` from config. Covers OpenAI itself, Ollama (`http://localhost:11434/v1`), and vLLM (`http://host:port/v1`) with no code changes — only `MODEL_BASE_URL` switches.

### Retry policy

One retry with exponential backoff on transient errors (HTTP 5xx, network timeout). After the second failure, raise and abort the experiment with a Russian error message. Research data integrity > graceful degradation.

### Output parsing (at the agent layer, not the client)

The LLM client is intentionally dumb (returns text). The agent owns parsing:
- `parse_number(text) -> int` — extracts the first standalone single digit in `[0..9]` using regex `(?<!\d)([0-9])(?!\d)` (rejects `"12"` and `"choice42"`). If none found → one retry with a stricter user message (`"Ответьте только одной цифрой от 0 до 9."`). Second failure → abort.
- Negotiation messages are returned verbatim (no parsing).

This keeps the LLM adapter portable — no provider-specific structured-output features required.

## 7. Persistence

Two artifacts per experiment, rooted at `runs/<experiment_id>/`:

```
runs/<experiment_id>/
├── experiment.sqlite               # structured, queryable
├── config_snapshot.json            # full config at start (for reproducibility)
└── transcripts/
    ├── Alice_Smith.md              # per-agent chronological view
    ├── Bob_Jones.md
    └── ...
```

### SQLite schema

```sql
CREATE TABLE experiments (
    id              INTEGER PRIMARY KEY,
    started_at      TEXT    NOT NULL,
    finished_at     TEXT,
    random_seed     INTEGER NOT NULL,
    memory_mode     TEXT    NOT NULL,   -- 'round' | 'episode'
    play_strategy   TEXT    NOT NULL,   -- 'direct' | 'prediction'
    prediction_mapping TEXT,            -- nullable
    config_json     TEXT    NOT NULL
);

CREATE TABLE agents (
    id              INTEGER PRIMARY KEY,
    experiment_id   INTEGER NOT NULL REFERENCES experiments(id),
    first_name      TEXT    NOT NULL,
    last_name       TEXT    NOT NULL,
    full_name       TEXT    NOT NULL,
    system_prompt_rendered TEXT NOT NULL,
    UNIQUE (experiment_id, full_name)
);

CREATE TABLE rounds (
    id              INTEGER PRIMARY KEY,
    experiment_id   INTEGER NOT NULL REFERENCES experiments(id),
    number          INTEGER NOT NULL,   -- 0-indexed
    schedule_json   TEXT    NOT NULL,
    started_at      TEXT    NOT NULL,
    finished_at     TEXT,
    UNIQUE (experiment_id, number)
);

CREATE TABLE episodes (
    id              INTEGER PRIMARY KEY,
    round_id        INTEGER NOT NULL REFERENCES rounds(id),
    number          INTEGER NOT NULL,
    agent_a_id      INTEGER NOT NULL REFERENCES agents(id),
    agent_b_id      INTEGER NOT NULL REFERENCES agents(id),
    started_at      TEXT    NOT NULL,
    finished_at     TEXT
);

CREATE TABLE games (
    id              INTEGER PRIMARY KEY,
    episode_id      INTEGER NOT NULL REFERENCES episodes(id),
    number          INTEGER NOT NULL,
    choice_a        INTEGER NOT NULL,
    choice_b        INTEGER NOT NULL,
    income_a        REAL    NOT NULL,
    income_b        REAL    NOT NULL,
    finished_at     TEXT    NOT NULL
);

CREATE TABLE negotiation_messages (
    id              INTEGER PRIMARY KEY,
    game_id         INTEGER NOT NULL REFERENCES games(id),
    turn            INTEGER NOT NULL,
    agent_id        INTEGER NOT NULL REFERENCES agents(id),
    content         TEXT    NOT NULL,
    created_at      TEXT    NOT NULL
);

CREATE TABLE decisions (
    id              INTEGER PRIMARY KEY,
    game_id         INTEGER NOT NULL REFERENCES games(id),
    agent_id        INTEGER NOT NULL REFERENCES agents(id),
    strategy        TEXT    NOT NULL,    -- 'direct' | 'prediction'
    predicted_value INTEGER,             -- nullable; only set in prediction strategy
    prediction_raw_response TEXT,        -- the LLM's full prediction text
    final_choice    INTEGER NOT NULL,
    choice_raw_response TEXT NOT NULL
);

CREATE TABLE round_standings (
    id              INTEGER PRIMARY KEY,
    round_id        INTEGER NOT NULL REFERENCES rounds(id),
    agent_id        INTEGER NOT NULL REFERENCES agents(id),
    round_income    REAL    NOT NULL,
    place           INTEGER NOT NULL    -- 1-indexed
);

CREATE TABLE notes (
    id              INTEGER PRIMARY KEY,
    experiment_id   INTEGER NOT NULL REFERENCES experiments(id),
    round_id        INTEGER REFERENCES rounds(id),
    episode_id      INTEGER REFERENCES episodes(id),
    agent_id        INTEGER NOT NULL REFERENCES agents(id),
    note_type       TEXT    NOT NULL,    -- 'episode' | 'global_snapshot'
    content         TEXT    NOT NULL,
    created_at      TEXT    NOT NULL
);
```

### `notes.note_type` values

- `episode` — per-episode reflection (only created when `MEMORY_MODE=episode`); `episode_id` set, `round_id` set.
- `global_snapshot` — agent's global notes captured after each round-end update (both modes); `round_id` set, `episode_id` null.

Snapshots let an analyst trace how an agent's worldview evolved across rounds.

### Transcript files

One Markdown file per agent. Append-only, written as events happen — a crashed run still leaves a partial transcript. Structure (excerpt):

```markdown
# Experiment 1 — Alice Smith

## System prompt
<rendered system prompt>

## Round 1

### Schedule announcement
You will play 3 episodes this round, with: Bob Jones, then Carol Brown, then Dave Wilson.

### Episode 1 of 3 — with Bob Jones

#### Game 1 of 3
**Negotiation:**
- **Alice Smith:** "..."
- **Bob Jones:** "..."

**[Private — prediction]:** _"I predict Bob will pick 7 because..."_ → predicted = 7
**Final choice:** 8  (PredictionBasedStrategy + one_above)

**Outcome:** Alice 8, Bob 7 → Alice +10, Bob −5

#### Episode 1 note
> Bob seems to favor middle numbers...

### Round 1 standings (announced)
1. Alice Smith — 27   2. Bob Jones — 14   3. ...

### Global notes (after this round)
> Bob tends toward middle numbers. Carol is unpredictable.
```

Private content (prediction reasoning) shows up only in that agent's own transcript, never in the partner's.

## 8. Orchestrator flow

Sequential, end-to-end. Each call into an agent (`observe_*`, `negotiate`, `write_*`, `update_global_notes`) builds the LLM context using the layers from Section 4.

```python
async def run_experiment(config: Config):
    sink = SqliteSink(config)
    sink.start_experiment()

    rule      = ScoringRuleFactory.build(config)
    scheduler = SchedulerFactory.build(config)
    strategy  = StrategyFactory.build(config)
    llm       = LLMClientFactory.build(config)
    rng       = random.Random(config.random_seed)

    agents = create_agents(config, llm, rng, sink)

    for r in range(config.rounds_per_experiment):
        schedule = scheduler.schedule(agents, k=config.partners_per_round, rng=rng)
        sink.start_round(r, schedule)
        for agent in agents:
            agent.observe_schedule_announcement(r, schedule)

        for e_idx, (a, b) in enumerate(schedule):
            sink.start_episode(r, e_idx, a, b)
            a.observe_episode_start(b); b.observe_episode_start(a)

            for g in range(config.games_per_episode):
                sink.start_game(g)
                # negotiation: alternating, fixed turns each
                for turn in range(config.negotiation_turns_per_agent * 2):
                    speaker, listener = (a, b) if turn % 2 == 0 else (b, a)
                    msg = await speaker.negotiate(listener, g)
                    listener.observe_partner_message(msg)
                    sink.record_message(turn, speaker, msg)
                # decisions: each agent decides independently
                choice_a = await strategy.decide(a, partner=b, game_idx=g)
                choice_b = await strategy.decide(b, partner=a, game_idx=g)
                income_a, income_b = rule.score(choice_a, choice_b)
                sink.record_game(choice_a, choice_b, income_a, income_b)
                a.observe_game_outcome(b, choice_a, choice_b, income_a, income_b)
                b.observe_game_outcome(a, choice_b, choice_a, income_b, income_a)

            sink.finish_episode()
            if config.memory_mode == "episode":
                for who, partner in [(a, b), (b, a)]:
                    note = await who.write_episode_note(partner)
                    sink.record_note(who, kind="episode", content=note)
                    who.commit_episode_note(note)

        # End of round
        standings = compute_round_standings(agents, r)
        sink.record_standings(r, standings)
        for agent in agents:
            agent.observe_round_standings(standings)

        for agent in agents:
            new_global = await agent.update_global_notes()
            sink.record_note(agent, kind="global_snapshot", content=new_global)
            agent.commit_global_notes(new_global)

        for agent in agents:
            agent.reset_round_memory()

        sink.finish_round(r)

    sink.finish_experiment()
```

### Invariants

- One agent never participates in two episodes simultaneously (sequential within round).
- Choices in a single game are computed without sharing — `decide(a, …)` and `decide(b, …)` see only their own context.
- `reset_round_memory()` is the *only* mechanism that drops episode notes; everything else is append-only.
- All `sink.*` calls happen before the next state transition — a crash mid-round leaves a complete record of everything up to that point.

## 9. Testing strategy

TDD per project conventions: failing tests before implementation. AAA pattern. One assertion per test where possible.

### Layers

**1. Pure unit (no LLM, no I/O):**
- `test_scoring.py` — exhaustive over the 100 input pairs in `[0..9]²`; assert cyclic mapping and tie/win/loss behaviour
- `test_scheduler.py` — no duplicate partners per agent within a round; all agents used; K respected; deterministic with seed
- `test_name_generator.py` — uniqueness of first names, uniqueness of last names, deterministic with seed
- `test_config.py` — validation rules (H ordering, pool sizes ≥ `AGENT_COUNT`, `AGENT_COUNT × K` even, prompt placeholders required when relevant mode is enabled)
- `test_strategy.py` — `MatchPrediction` and `OneAbovePredicted` mappings; with `FakeLLMClient`, prediction strategy populates the private scratchpad

**2. Test doubles:**
- `FakeLLMClient` — two modes:
  - Queue: `FakeLLMClient(responses=["go", "ok", "5"])` pops in order
  - Pattern: `FakeLLMClient(callback=lambda msgs: …)` for context-sensitive replies
- `InMemorySink` — implements `EventSink`, captures all events in lists for assertions

**3. Integration (real orchestrator + fakes):**
- Tiny experiment (2 agents, 1 round, 1 episode, 1 game) — verify all populated tables
- Multi-round: verify global_snapshot per round + round memory reset
- Mode `episode`: verify episode notes persisted to SQLite **and** dropped from agent context after round end
- Strategy variants: direct vs prediction (verify `decisions.predicted_value` populated only for prediction)
- Privacy: assert agent A's transcript never contains agent B's prediction reasoning

**4. Live smoke (not in CI):**
- One tiny run against a real OpenAI-compat endpoint behind `@pytest.mark.live` — verifies wiring only

## 10. Error handling — fail fast

| Failure mode | Response |
|---|---|
| Config invalid at startup | pydantic-settings → Russian error → exit before any LLM call |
| LLM transient error (HTTP 5xx, timeout) | One retry with exponential backoff; second failure → abort experiment |
| LLM response unparseable as number | One retry with stricter prompt; second failure → abort |
| SQLite write fails | Log Russian error and raise — never silently lose data |
| Schedule unschedulable | Caught at config validation, never reaches runtime |

Research data integrity > graceful degradation. A crash with a clear log beats a silently corrupted experiment.

## 11. Reproducibility

A given `RANDOM_SEED` deterministically fixes:
- Name assignment (which agent gets which first/last name)
- The pairing schedule produced by `RandomKPartnerScheduler` for every round

LLM responses themselves are not reproducible (depend on provider + temperature), but the experimental setup is. The full config snapshot is persisted as `config_snapshot.json` alongside the SQLite file.

## 12. Open questions / future work

- Per-agent strategy overrides (architecture supports; no v1 config)
- Concurrency within a round (architecture supports via async; not exposed)
- `--resume` from last successful round (state is in SQLite; resume logic not implemented)
- Native Anthropic / vLLM / Ollama adapters for provider-specific features (prompt caching, logprobs, structured generation)
- Analysis tooling — metrics on emergent reputation (e.g., does an agent that cooperated last round get cooperated with more this round?)
