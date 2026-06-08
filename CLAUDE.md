# CLAUDE.md

Research simulation engine studying whether **reputation** emerges on its own in
populations of LLM agents that repeatedly play a coordination game with cheap talk.

Python 3.12, asyncio, httpx (OpenAI-compatible LLM calls), PyYAML, python-dotenv.
Tooling: `uv`, `pytest` + `pytest-asyncio` (auto mode). No web framework — driven
from `examples/`.

## Project map

```
src/
├── providers/      OpenAI-compatible HTTP client + retries (Together.ai, Ollama)
├── core/           Agent, Memory, frozen-dataclass config, orchestrator
├── games/          ReputationPD game + English prompt builders
├── strategy/       PlayStrategy (direct | prediction) + prediction mappings
├── population/     Population (live roster, provider cache) + RosterGenerator
├── matchmaking/    Matchmaker (random) — who plays whom each round
config/             one YAML = one episode (example.yaml, example_prediction.yaml)
examples/           runnable demos; orchestrator_demo.py is the main entry point
tests/              mirrors src/; unit tests stub the LLM, smoke tests hit Ollama
docs/               English overviews + per-layer design docs (see relevant block)
```

Dependency flow is one-directional bottom→top: providers → core → games/strategy →
population/matchmaking → orchestrator. `games` and `strategy` reference each other
via lazy imports.

<important if="you need to run commands to install, test, or run an episode">

```bash
uv sync --extra dev                                   # install deps (incl. pytest)
uv run pytest                                          # all tests
uv run pytest tests/strategy/test_prediction.py       # single test file
PYTHONPATH=. .venv/bin/python examples/orchestrator_demo.py                          # run example.yaml episode
PYTHONPATH=. .venv/bin/python examples/orchestrator_demo.py config/example_prediction.yaml
```

Running an episode needs a reachable provider; the API key is read from `.env`
(`TOGETHER_API_KEY`). `.env` is gitignored — never commit it.
</important>

<important if="you are writing docstrings, prints, logging, or error messages">
- Docstrings (Google-style), `print`s, logging, and error messages are in **Russian**.
- Keep established English terms (payoff, cheap-talk, matchmaker) untranslated.
- LLM-facing prompt text stays in **English** (see `src/games/prompts.py`).
</important>

<important if="you are adding or removing a dependency">
- Manage dependencies only via `uv` (`uv add` / `uv remove` / `uv sync`) — never
  `pip` / `poetry`.
</important>

<important if="you are writing or modifying tests">
- Write the failing test first (AAA, behavioural name), then minimal code.
- Unit tests use the `ScriptedProvider` stub — no network.
- Smoke tests (`test_smoke_*_ollama.py`) hit a local Ollama and auto-skip if absent.
- Details and conventions: `docs/testing.md`.
</important>

<important if="you are adding a new provider, strategy, matchmaker, or population generator">
- Implement the relevant Protocol and register it through its `make_*` factory
  (`LLMProvider`, `PlayStrategy`, `Matchmaker`, `PopulationGenerator`).
</important>

<important if="you are modifying the orchestrator or the episode run loop">
- The orchestrator's **only output channel is the `observer` callback** — no printing
  or persistence inside `src/`.
- The **caller owns the `Population`**: it builds it and must `await pop.aclose()`.
- Deep dive: `docs/architecture.md`.
</important>

<important if="you are adding or changing configuration fields or episode YAML">
- Config objects are frozen dataclasses in `src/core/config.py`; wire new fields
  through the `_*_cfg` builders / `load_episode`.
- Validate input once at load (`_validate`), fail fast with a Russian message.
- Episode YAML, provider anchors, population pools: `docs/configuration.md`.
</important>

<important if="you need to understand the architecture, layers, game rules, or data flow">

- `docs/architecture.md` — layers, the game, pairing flow, strategies, seams
- `docs/conventions.md` — language rules, code patterns, ownership, dependency rules

English overviews link to the authoritative **Russian design docs**, also under `docs/`:

- `docs/agent-games-plan.md` — master plan: research question, fixed contract
- `docs/agent-games-mvp-arch.md` — MVP code architecture, interfaces, round flow
- `docs/agent-games-mvp-explained.md` — the five layers in plain words
- `docs/agent-games-mvp-sequence.md` — one-round sequence diagram (Mermaid)
- `docs/agent-games-{provider,agent,game,matching,orchestrator}-plan.md` —
  per-layer design + its test slices
</important>
