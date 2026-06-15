# Architecture

One **episode** = one config file. The orchestrator drives a population of LLM
agents through repeated rounds of a coordination game, narrating each round via a
single observer callback.

> **Sources.** This page is an English overview synthesised from the authoritative
> design docs in this folder (the `agent-games-*.md` files, in Russian). For the full
> rationale read:
> [overall plan & research question](./agent-games-plan.md) ·
> [MVP code architecture](./agent-games-mvp-arch.md) ·
> [layers in plain words](./agent-games-mvp-explained.md) ·
> [one-round sequence diagram](./agent-games-mvp-sequence.md).

## Layered design (dependency flow is one-directional, bottom → top)

```
providers/      OpenAI-compatible HTTP client + retries        (no deps on us)
   ↑
core/           Agent, Memory, config dataclasses              (depends on providers)
   ↑
games/          ReputationPD game + prompt builders
strategy/       PlayStrategy: how an agent turns a round into a number
   ↑
population/     Population (live roster + provider cache)
matchmaking/    Matchmaker: who plays whom each round
   ↑
core/orchestrator.py   run_episode: glues all of the above
```

Each layer has its own deep-dive design doc (`agent-games-*-plan.md`):

| layer | code | design doc |
|-------|------|------------------------------|
| provider | `src/providers/` | [agent-games-provider-plan.md](./agent-games-provider-plan.md) |
| agent + memory | `src/core/agent.py`, `memory.py` | [agent-games-agent-plan.md](./agent-games-agent-plan.md) |
| game | `src/games/` | [agent-games-game-plan.md](./agent-games-game-plan.md) |
| matchmaking | `src/matchmaking/` | [agent-games-matching-plan.md](./agent-games-matching-plan.md) |
| orchestrator + population | `src/core/orchestrator.py`, `src/population/` | [agent-games-orchestrator-plan.md](./agent-games-orchestrator-plan.md) |

`games/` and `strategy/` are mutually referential by design — the cycle is broken
with **lazy imports** (see `src/games/reputation_pd.py:20` and
`src/strategy/base.py:49`). Prompt builders live in `src/games/prompts.py`, which
imports neither game nor strategy so both can share it without a cycle.

## The game: ReputationPD

`src/games/reputation_pd.py`. A repeated PD-flavoured coordination game.
Deep dive: [agent-games-game-plan.md](./agent-games-game-plan.md).

- Each round two agents secretly pick an integer 0–9.
- Equal numbers → both score `R` (mutual cooperation).
- One number exactly one higher than the other, **mod 10** (0 follows 9) → the
  higher scores `T` (betrayal), the lower scores `S`.
- Anything else → both score `P` (miscoordination).
- Outcome strings are from A's perspective: `CC`/`DC`/`CD`/`DD`; `_FLIP`
  (`reputation_pd.py:11`) mirrors them for B's memory.

Payoff invariants live next to `Payoffs` in `src/core/config.py:18` (`T > R > P > S`,
`2R > T + S`).

### Pairing flow (`ReputationPD.play_pairing`)

1. **Cheap talk** (`_cheap_talk`): agents alternate short messages. `a` always
   opens (the matcher fixes orientation via pairing order — no rng in the game).
   Stop rule `both_ready_latch`: a turn ends only when **both** agents have set
   `ready: true`; once ready, an agent latches silent. Hard ceiling
   `cfg.max_talk_turns`. Each agent necessarily speaks at least once.
2. **Decide**: the configured `PlayStrategy.decide` is called for each agent with
   the public talk feed. This produces a `Decision` (final number + rationale,
   plus optional prediction).
3. **Resolve**: scores mutate on the agents.
4. **Reflect** (optional, `game.reflection: true`): each agent makes one extra
   `REFLECT` LLM call over the revealed result (both numbers + own payoff) and
   returns a short reflection. It is private to its author, like the rationale.
5. **Record**: a `PairingRecord` (`src/games/base.py:9`) is returned and each
   agent's `Memory` gets an entry (including the reflection, which the diary
   feeds back into the agent's future LLM inputs).
6. **Memory notes** (optional, `game.memory_notes_every: N`): each agent makes one
   extra `NOTE` LLM call after every **N rounds it has actually played** (counted
   per-agent as `len(memory.entries)` after the memory writes — idle rounds don't
   count, and the two agents of a pairing decide independently). NOTE rewrites the
   agent's whole memory into private notes; from then on `Memory.render` sends those
   notes **instead of** the raw round history, plus the raw buffer of rounds played
   since the last consolidation (`noted_upto`). The notes ride the pairing: stored in
   `pairings.a_notes/b_notes`, their L2 calls in `llm_calls` with `phase='note'`.
   A failed NOTE call aborts the pairing like any other LLM failure.

## Strategies (`src/strategy/`)

`PlayStrategy` (Protocol, `strategy/base.py:30`) maps round state → `Decision`.

- **direct** (`strategy/direct.py`): one `DECIDE` phase; agent picks the number itself.
- **prediction** (`strategy/prediction.py`): one `PREDICT` phase (agent predicts the
  partner's number), then a pure `PredictionMapping` (`strategy/mappings.py`) turns
  that prediction into the agent's own choice. Mappings: `match` (copy →
  cooperate), `one_above` (rational best-response, off-by-one for `T`).

`make_strategy(cfg)` (`strategy/base.py:37`) selects from `cfg.play_strategy` /
`cfg.prediction_mapping`. Add a new strategy: implement the Protocol, register it
in `make_strategy`, and (if it needs validation) extend `_validate` in
`src/core/config.py:88`.

## Agent & phases (`src/core/agent.py`)

Deep dive: [agent-games-agent-plan.md](./agent-games-agent-plan.md).

An `Agent` owns its `Memory`, running `score`, and an `LLMProvider`. `Agent.act`
glues the memory diary and the phase context into a single user message, calls the provider, and parses the
reply as JSON with up to `_MAX_PARSE_RETRIES` correction retries; on total failure it
raises `ActParseError` (no substitution) — the pairing is aborted (`finished=0`) and the
episode stops, same as a provider error. It also bumps `parse_failures`.

Five `PhaseKind`s: `TALK`, `DECIDE`, `PREDICT`, `REFLECT`, `NOTE`. `NOTE` consolidates
memory (its `act` renders the full memory, ignoring the window). In DECIDE/PREDICT the
JSON answer puts `rationale` before `number`, so reasoning tokens are generated
before the choice is committed; with `game.rationale: false` the prompt asks for
a bare `{"number": ...}` instead. JSON extraction is lenient — raw, fenced, and
balanced-brace candidates are all tried (`_extract_json_obj`).

### LLM input trace

For DECIDE/PREDICT/REFLECT calls, `Agent.act` logs the exact LLM input (system prompt,
memory diary, phase context, retry corrections) at DEBUG level via the
`src.core.agent` logger — one record per provider attempt (`_render_trace`,
`src/core/agent.py`). Silent unless the caller configures logging:
`examples/orchestrator_demo.py` attaches a handler when `LLM_TRACE=1` is set.
TALK calls are not traced. Design: `docs/superpowers/specs/2026-06-10-llm-decide-trace-design.md`.

## Population & matchmaking

Deep dives: [matchmaking](./agent-games-matching-plan.md) ·
[population & orchestrator](./agent-games-orchestrator-plan.md).

- `Population` (`src/population/base.py`) is a mutable roster. It owns a **provider
  cache** keyed by `(base_url, model)`, so agents sharing a model share one httpx
  client (connection pooling); `aclose()` closes each unique provider once. Ids are
  never reused (`A1`, `A2`, … fallback) but real ids are sampled `First Last` names.
- `RosterGenerator` (`src/population/roster.py`) builds from an explicit spec list,
  cycled up to `n_agents`, sampling unique names from the config pools.
- `RandomMatchmaker` (`src/matchmaking/random_mm.py`) shuffles ids into disjoint
  pairs; an odd one out sits idle and earns `idle_payoff`. Uses its **own** rng
  stream (`Random(f"{seed}:matchmaker")`) so partitions are reproducible by seed
  regardless of what the games do.

## The orchestrator (`src/core/orchestrator.py`)

Deep dive: [agent-games-orchestrator-plan.md](./agent-games-orchestrator-plan.md).

`run_episode(cfg, pop, *, observer)` has **side effects only** — it mutates agents
(score/memory) and emits each round to `observer`. It returns nothing: the **caller
owns the population**, builds it, reads final scores from it, and `aclose()`s it.

- The `observer` callback `(round, RoundPlan, list[PairingRecord])` is the **only
  output channel**. The future Logger layer will plug in here to persist rounds —
  there is deliberately no `db_path` in the config.
- Pairings within a round run concurrently under an `asyncio.Semaphore`
  (`cfg.max_concurrency`).
- **LLM failures don't fail-fast mid-round.** `play_pairing` catches a `ProviderError`
  and returns the pairing as **unfinished** (`PairingRecord.finished=False`, results NULL,
  full raw L2 log kept). The round finishes and is emitted to `observer` (so the failure is
  persisted), then `run_episode` raises `EpisodeAborted` — stopping the episode at a round
  boundary, with `runs.finished_at` left NULL as a crash marker. Raw LLM I/O of every
  HTTP call (incl. retries, parse-retries, failures) is logged to `llm_calls`; see
  `claude_docs/agent-games-logger-plan.md` §9.

## LLM judge (`src/judge/`)

An optional post-episode component that evaluates whether a reputation institute
emerged from the agents' interactions. It is **invoked by the callers** (runner /
`orchestrator_demo.py`) after `run_episode` returns — the orchestrator itself is
untouched.

- **Input**: the public cheap-talk transcript only (agent messages from all pairings,
  all rounds). Private rationales, reflections, and payoffs are not shown.
- **LLM call**: one call via the judge's own `ProviderCfg`; one retry if the reply
  cannot be parsed. On persistent failure the error is logged and the episode result
  is unaffected.
- **Output**: a `JudgeVerdict` (emerged: bool, explanation, evidence — validated
  references to the cited messages). The verdict is printed immediately after the
  episode summary.
- **Persistence**: `run_experiment` (in `src/runner.py`) stores the verdict in the
  `judge_verdicts` SQLite table, linked by `run_id`. The judge block is
  **excluded from the `run_id` hash** so toggling it on/off does not create new run
  entries.
- **replay.py**: cited messages are highlighted in yellow (ANSI); a JUDGE VERDICT
  section is appended after the round-by-round replay.

Enable by adding a `judge:` block to the episode YAML (see `docs/configuration.md`)
or by constructing `JudgeCfg` directly in `experiment.py`.

## Key seams (intentionally not yet built)

- **Logger layer** — persistence via the observer; no DB in the orchestrator.
- **Selection/evolution** — `Population.add` is used today; remove/replace are
  documented seams for a future selection layer.
- **Interactive matchmakers** — `plan_round(..., actor=...)` and `RoundPlan.events`
  exist for matchmakers that query agents; `random` ignores them.

## Further reading

- [agent-games-plan.md](./agent-games-plan.md) — the master plan: research
  question, fixed contract, the full (beyond-MVP) layered design.
- [agent-games-mvp-arch.md](./agent-games-mvp-arch.md) — MVP code
  architecture: module tree, exact interfaces, round flow, the planned seams.
- [agent-games-mvp-explained.md](./agent-games-mvp-explained.md) — the
  five layers explained in plain language with examples.
- [agent-games-mvp-sequence.md](./agent-games-mvp-sequence.md) — Mermaid
  sequence diagram of one round.
- Per-layer plans:
  [provider](./agent-games-provider-plan.md) ·
  [agent](./agent-games-agent-plan.md) ·
  [game](./agent-games-game-plan.md) ·
  [matchmaking](./agent-games-matching-plan.md) ·
  [orchestrator](./agent-games-orchestrator-plan.md).
