# Testing

`pytest` with `pytest-asyncio` in **auto mode** (`asyncio_mode = "auto"` in
`pyproject.toml`) — `async def test_*` functions need no `@pytest.mark.asyncio`.
`pythonpath = ["."]` is set so tests import `src.*` directly.

## Commands

```bash
uv run pytest                          # all tests
uv run pytest tests/strategy           # one package
uv run pytest tests/core/test_agent.py # one file
uv run pytest -k resolve               # by name
```

## Two kinds of tests

### Unit tests — no network, use `ScriptedProvider`

The default. Network is replaced by `ScriptedProvider` (defined per-package in
`tests/games/conftest.py` and others): a `LLMProvider` test double that returns a
queued list of reply strings in order and records every `(system, messages)` call
it received. Drive an agent/game/strategy by queuing the exact JSON replies the LLM
would have produced, then assert on outcomes **and** on what the agent was prompted
with.

### Smoke tests — live, against local Ollama, auto-skipped

Files named `test_smoke_*_ollama.py`. They hit a local Ollama at
`http://localhost:11434/v1` and are guarded by a `_ollama_up()` reachability check —
`@pytest.mark.skipif(not _ollama_up(), ...)` — so they silently skip when Ollama
isn't running. Example: `tests/core/test_smoke_orchestrator_ollama.py`. These prove
the real HTTP path end-to-end; they assert only on shape (valid outcome, number in
range), not exact values.

## Conventions (from the user's global TDD rules)

- **Write the failing test first**, then minimal implementation to pass.
- AAA structure (Arrange–Act–Assert); prefer one assertion per test.
- Behavioural test names: `test_<behaviour>` describing what should happen.
- Test files mirror the `src/` package layout under `tests/`.

## Determinism

`seed` makes runs reproducible: the population build and a derived matchmaker rng
stream (`Random(f"{seed}:matchmaker")`) are both seeded. The LLM itself is the only
nondeterministic part — hence unit tests stub it with `ScriptedProvider`.

## Further reading

Each per-layer design doc (the `agent-games-*-plan.md` files in this folder) ends with
its testing strategy ("срезы" / slices) and lists what each slice asserts:
[provider](./agent-games-provider-plan.md) ·
[agent](./agent-games-agent-plan.md) ·
[game](./agent-games-game-plan.md) ·
[matchmaking](./agent-games-matching-plan.md) ·
[orchestrator](./agent-games-orchestrator-plan.md).
