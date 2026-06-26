# CLAUDE.md

Research simulation engine studying whether **reputation** emerges on its own in
populations of LLM agents that repeatedly play a coordination game with cheap talk.
One **episode** = one YAML config: the orchestrator builds a population, a matchmaker
pairs agents each round, each pair exchanges short messages (cheap-talk) and then
secretly picks a number 0–9; payoffs reward matching and punish being undercut.

## Stack

Python 3.12, asyncio, httpx (OpenAI-compatible LLM calls), PyYAML, python-dotenv.
Tooling: `uv`, `pytest` + `pytest-asyncio` (auto mode — no `@pytest.mark.asyncio`
needed). No web framework — everything is driven from `examples/`.

## Project map

```
src/
├── providers/      OpenAI-compatible HTTP client + retries (Together.ai, Ollama)
│   ├── base.py         LLMProvider Protocol, Message, Completion, error types
│   └── openai_compat.py  the real httpx client
├── core/
│   ├── agent.py        Agent.act: memory + phase context -> LLM -> parsed JSON;
│   │                   phases TALK/DECIDE/PREDICT/REFLECT; DEBUG trace of LLM input
│   ├── memory.py       per-agent diary of past rounds, rendered into the prompt
│   ├── config.py       frozen dataclasses + load_episode + _validate (fail fast)
│   └── orchestrator.py run_episode: rounds loop, semaphore, observer callback
├── games/
│   ├── reputation_pd.py  ReputationPD: cheap-talk loop, resolve, memory writes
│   ├── prompts.py        English prompt builders (talk/decide/predict/reflect/notes)
│   ├── talk_rules.py     TalkStopRule Protocol + make_talk_rule (latch | revocable)
│   └── base.py           PairingRecord, Game Protocol
├── strategy/       PlayStrategy: direct (pick a number) | prediction (predict
│                   partner, map via match/one_above) — base.py, mappings.py
├── judge/          LLM-судья: вердикт о возникновении института репутации
│                   (один вызов после эпизода, видит только публичный cheap-talk)
├── population/     Population (live roster, provider cache) + RosterGenerator
├── matchmaking/    Matchmaker Protocol + RandomMatchmaker (disjoint pairs, idle)
config/             one YAML = one episode (experiment.yaml, example.yaml, example_prediction.yaml)
examples/           runnable demos; orchestrator_demo.py is the main entry point
tests/              mirrors src/; unit tests stub the LLM, smoke tests hit Ollama
docs/               English overviews + Russian per-layer design docs (see index)
```

Dependency flow is one-directional bottom→top: providers → core → games/strategy →
population/matchmaking → orchestrator. `games` and `strategy` reference each other
via lazy imports (`src/games/reputation_pd.py:20`).

## Commands

```bash
uv sync --extra dev                                   # install deps (incl. pytest)
uv run pytest                                          # all tests
uv run pytest tests/strategy/test_prediction.py       # single test file
uv run pytest -k resolve                              # by name
uv run python examples/orchestrator_demo.py                          # run example.yaml episode
uv run python examples/orchestrator_demo.py config/example_prediction.yaml
LLM_TRACE=1 uv run python examples/orchestrator_demo.py             # + print exact LLM input per DECIDE/PREDICT call
uv run python judge_runs.py                            # backfill judge verdicts for stored runs
uv run python collect_stats.py                         # aggregate verdicts -> stats.json + stats.csv
uv run python plot_stats.py                            # stats.json -> stats.png (Wilson CI bars)
uv run python experiment.py [config.yaml] ["run name"]               # one run -> experiment.db
uv run python research.py                                            # model×game sweep -> research.db (idempotent/resumable)
```

`research.py` runs a grid (MODELS × `GAMES_PER_MODEL`) into `research.db`; runs are named
`"<model> <n>"`. It first resumes every unfinished run, then fills missing games by name —
re-running it just continues where it left off (prints each run's name + `resume`/`calculating`).

Running an episode needs a reachable provider; the API key is read from `.env`
(`TOGETHER_API_KEY`). `.env` is gitignored — never commit it.

## Critical conventions (apply to nearly every task)

- Docstrings (Google-style), `print`s, logging, and error messages are in
  **Russian**; established English terms (payoff, cheap-talk, matchmaker) stay
  untranslated. LLM-facing prompt text is **English** (`src/games/prompts.py`).
- TDD: write the failing test first (AAA, behavioural name), then minimal code.
- Manage dependencies only via `uv` (`uv add` / `uv remove` / `uv sync`) — never
  `pip` / `poetry`.
- No printing or persistence inside `src/` — output leaves the engine only through
  the orchestrator's `observer` callback (silent DEBUG logging is the one
  exception: handlers are configured by the caller, never in `src/`).
- `from __future__ import annotations` at the top of every module; rng objects are
  passed in, never created inside library code.

<important if="you are writing or modifying tests">
- Unit tests use the `ScriptedProvider` stub (per-package `tests/*/conftest.py`) —
  no network; queue exact JSON replies, assert on outcomes AND on the recorded
  `(system, messages)` calls.
- Smoke tests (`test_smoke_*_ollama.py`) hit a local Ollama and auto-skip if absent.
- `pythonpath = ["."]` is set — tests import `src.*` directly.
- Details and conventions: `docs/testing.md`.
</important>

<important if="you are adding a new provider, strategy, matchmaker, population generator, or talk-stop rule">
Implement the Protocol and register it through its `make_*` factory:

| seam | Protocol | factory |
|------|----------|---------|
| provider | `LLMProvider` (`src/providers/base.py`) | `make_provider` (`src/providers/openai_compat.py:115`) |
| strategy | `PlayStrategy` (`src/strategy/base.py`) | `make_strategy` (`src/strategy/base.py:37`) |
| matchmaker | `Matchmaker` (`src/matchmaking/base.py`) | `make_matchmaker` (`src/matchmaking/base.py:20`) |
| population | `PopulationGenerator` (`src/population/base.py`) | `make_population` (`src/population/base.py:67`) |
| talk-stop rule | `TalkStopRule` (`src/games/talk_rules.py`) | `make_talk_rule` (`src/games/talk_rules.py`) |

If the new kind needs config validation, extend `_validate` in `src/core/config.py`.
</important>

<important if="you are modifying the orchestrator or the episode run loop">
- The orchestrator's **only output channel is the `observer` callback**
  `(round, RoundPlan, list[PairingRecord])`.
- The **caller owns the `Population`**: it builds it and must `await pop.aclose()`
  (in a `finally`).
- Pairings in a round run concurrently under `asyncio.Semaphore(cfg.max_concurrency)`,
  fail-fast via `gather`.
- Deep dive: `docs/architecture.md`.
</important>

<important if="you are adding or changing configuration fields or episode YAML">
- Config objects are frozen dataclasses in `src/core/config.py`; wire new fields
  through the `_*_cfg` builders / `load_episode`.
- Validate input once at load (`_validate`), fail fast with a Russian message.
- Provider blocks are shared across agents via YAML anchors; `ProviderCfg` points
  at any OpenAI-compatible `/chat/completions` endpoint.
- Field reference, anchors, population pools: `docs/configuration.md`.
</important>

<important if="you are changing Agent.act, phases, prompts, or memory rendering">
- The LLM input is assembled in `Agent.act` (`src/core/agent.py`): system =
  the agent's single `system_prompt` template taken **verbatim** (no more identity+persona+rules
  assembly), with only `{id}` and the payoff placeholders `{R}/{T}/{P}/{S}/{max_talk_turns}`
  substituted (the latter from `Phase.game_cfg`). A single user message = memory diary + phase
  context (+ correction appended on JSON parse retry, max 2 retries, then `ActParseError` — the
  pairing is aborted, no substitution/fallback). The diary and phase context are glued into one
  user message (not sent as consecutive same-role messages).
- JSON extraction is lenient (raw / fenced / balanced-brace); validators per phase.
- DECIDE/PREDICT inputs are traced at DEBUG via the `src.core.agent` logger
  (`_render_trace`); keep the trace in sync if you change prompt assembly.
- Determinism: `seed` drives population build and a derived matchmaker rng stream;
  the LLM is the only nondeterministic part.
</important>

<important if="you are debugging what agents actually see or why they chose a number">
Run the demo with `LLM_TRACE=1` (env var or `.env`) — it prints the full system
prompt, memory diary, and decide/predict context per provider attempt. In tests,
use `caplog.set_level(logging.DEBUG, logger="src.core.agent")` (examples at the
bottom of `tests/core/test_agent.py`).
</important>

<important if="you need to understand the architecture, game rules, or data flow">
- `docs/architecture.md` — layers, the game, pairing flow, strategies, seams
- `docs/conventions.md` — language rules, code patterns, ownership, dependency rules

English overviews link to the authoritative **Russian design docs**, also under `docs/`:

- `docs/agent-games-plan.md` — master plan: research question, fixed contract
- `docs/agent-games-mvp-arch.md` — MVP code architecture, interfaces, round flow
- `docs/agent-games-mvp-explained.md` — the five layers in plain words
- `docs/agent-games-mvp-sequence.md` — one-round sequence diagram (Mermaid)
- `docs/agent-games-{provider,agent,game,matching,orchestrator}-plan.md` —
  per-layer design + its test slices

Specs/plans for individual features live in `docs/superpowers/{specs,plans}/`.
</important>

## Doc index

Read the relevant doc before starting work on a related area:

- `docs/architecture.md` — layered design, ReputationPD rules, pairing flow,
  strategies, agent phases, LLM input trace, intentional seams (Logger, selection)
- `docs/configuration.md` — episode YAML reference, provider anchors, population
  pools, how to add a config knob
- `docs/testing.md` — unit vs smoke tests, ScriptedProvider, determinism
- `docs/conventions.md` — Russian/English language rules, code patterns,
  output ownership, dependency direction
