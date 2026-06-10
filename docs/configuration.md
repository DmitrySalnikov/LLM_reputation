# Configuration

One episode = one YAML file under `config/`. Loaded and validated once by
`load_episode` (`src/core/config.py:119`); invalid configs fail fast with a Russian
`ValueError`. All config objects are **frozen dataclasses** (`src/core/config.py`).

## Running an episode

```bash
PYTHONPATH=. .venv/bin/python examples/orchestrator_demo.py [config/episode.yaml]
```

Defaults to `config/example.yaml` if no path is given. The entry point
(`examples/orchestrator_demo.py`) calls `load_dotenv()` so provider API keys come
from `.env`.

Set `LLM_TRACE=1` (env var or `.env`) to print the exact LLM input of every
DECIDE/PREDICT call while the episode runs (see `docs/architecture.md`,
"LLM input trace").

## Reference configs

- `config/example.yaml` — direct strategy (agents pick numbers themselves).
- `config/example_prediction.yaml` — prediction strategy with `one_above` mapping.

## Top-level fields (`EpisodeCfg`, `src/core/config.py:49`)

| field | meaning |
|-------|---------|
| `seed` | drives population build + a derived matchmaker rng stream |
| `rounds` | number of rounds in the episode |
| `matchmaker` | only `random` is implemented |
| `context_window` | per-agent memory window; `null` = unbounded |
| `idle_payoff` | what an odd-one-out agent scores when it sits a round out |
| `max_concurrency` | semaphore size for concurrent pairings |
| `play_strategy` | `direct` (default) or `prediction` |
| `prediction_mapping` | only used when `play_strategy: prediction`; `match` or `one_above` |
| `game` | `GameCfg`: `payoffs {R,T,P,S}`, `max_talk_turns`, `rationale` (ask for reasoning before the number in DECIDE/PREDICT; default `true`), `reflection` (extra post-game REFLECT call per agent, stored in memory; default `false`) |
| `population` | `PopulationCfg` (see below) |

## Provider blocks & YAML anchors

A provider block is shared across agents via a YAML `&anchor` / `*alias`. pyyaml
resolves these itself, so the same dict reaches every agent (`load_episode`
docstring). Example:

```yaml
provider_default: &default
  base_url: https://api.together.xyz/v1
  api_key_env: TOGETHER_API_KEY      # key read from .env
  model: Qwen/Qwen2.5-7B-Instruct-Turbo
  temperature: 0.7
  max_tokens: 1000
```

`ProviderCfg` (`src/core/config.py:8`) is OpenAI-compatible — point `base_url` at
any `/chat/completions` endpoint (Together.ai in prod, Ollama for smoke tests).

## Population block

```yaml
population:
  kind: roster                       # only roster is implemented
  n_agents: 4
  first_name_pool: [...]             # >= n_agents unique names, validated
  last_name_pool:  [...]             # >= n_agents unique names, validated
  agents:                            # shorter than n_agents -> cycled at build time
    - {persona: "...", provider: *default}
    - {persona: "...", provider: *default}
```

Agent ids are sampled as unique `First Last` strings from the two pools. `_validate`
(`src/core/config.py:88`) enforces: both pools present, no duplicates within a pool,
each pool ≥ `n_agents`. Empty pools fall back to `A1`, `A2`, … ids.

## Adding a config knob

1. Add the field to the relevant frozen dataclass in `src/core/config.py`.
2. Wire it through the matching `_*_cfg` builder and/or `load_episode`.
3. If it constrains valid input, extend `_validate` (fail fast, Russian message).

## Further reading

- [agent-games-mvp-arch.md](./agent-games-mvp-arch.md) §5 — config schema
  as originally designed (note: SQLite/`db_path` there is the future Logger layer, not
  the current in-memory orchestrator).
- [agent-games-orchestrator-plan.md](./agent-games-orchestrator-plan.md) —
  why the provider cache keys on `(base_url, model)` and how the episode is assembled.
