# Configuration

One episode = one YAML file under `config/`. Loaded and validated once by
`load_episode` (`src/core/config.py:119`); invalid configs fail fast with a Russian
`ValueError`. All config objects are **frozen dataclasses** (`src/core/config.py`).

## Running an episode

```bash
uv run python examples/orchestrator_demo.py [config/episode.yaml]
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
| `game` | `GameCfg`: `payoffs {R,T,P,S}`, `max_talk_turns`, `rationale` (ask for reasoning before the number in DECIDE/PREDICT; default `true`), `reflection` (extra post-game REFLECT call per agent, stored in memory; default `false`), `memory_notes_every` (0 = off; every N rounds an agent has actually played — counted per-agent, idle rounds excluded — it rewrites its memory into private notes via a NOTE call that then replace the raw round history; default `0`), and the prompt templates `rules`, `talk_prompt`, `talk_open_prompt` (first turn of a round, empty feed — the agent opens the talk; no `{feed}`), `decide_prompt`, `predict_prompt`, `reflect_prompt`, `notes_prompt` (placeholders `{round} {score}`; used only when `memory_notes_every > 0`) — each defaults to a `DEFAULT_*` in `src/core/config.py`; delete a key to use the default. `decide_prompt`/`predict_prompt` support a `{answer}` placeholder filled per the `rationale` flag (reason+number vs bare number); `decide_prompt` also takes `{reason}` (how the chat closed — `reason_limit`/`reason_agreed`, the **same wording** the history close line uses). The whole LLM input is one **game transcript**: past rounds are replayed by `Memory.render` with the tags the rules declare (`<game>`/`<you>`/`<opponent name>`), and the current round's `{feed}` uses the same tags. The line templates are shared so a given line type reads identically in history and live: the live `talk_prompt` round-open carries `{opener}` (who started the round — `opener_self`/`opener_partner`, the same phrase history uses), `opener_self` is the exact text `talk_open_prompt` opens with, and `history_close_prompt` matches the live decide close line. Transcript line templates are config (defaulted): `history_round_prompt` (`{round} {partner} {opener}`), `opener_self`/`opener_partner` (`{partner} starts first:`), `msg_self`/`msg_partner` (one cheap-talk line, `{text}`/`{partner}`), `history_close_prompt` (`{reason}`), `reason_limit`/`reason_agreed`, `history_result_prompt` (`{round} {partner} {partner_number} {payoff} {partner_payoff} {total}`, where `{total}` is the score **after** the round; the agent's own number is shown just above as a `<you>` line). The running score therefore lives in the history result lines, not in the talk/decide headers. In cheap talk the agent-facing JSON key to close the chat is `finish` (stored internally as `ready`) |
| `population` | `PopulationCfg` (see below) |
| `judge` | `JudgeCfg` or absent/`null` — optional LLM judge (see below) |

## LLM judge block (`JudgeCfg`, `src/core/config.py:148`)

An optional top-level `judge:` block enables a separate LLM that reads the episode's
public cheap-talk transcript once after the episode ends and returns a verdict on
whether a reputation institute emerged.

```yaml
judge:
  provider:
    base_url: https://api.together.xyz/v1
    api_key_env: TOGETHER_API_KEY        # env-var name (not the value itself)
    model: Qwen/Qwen2.5-72B-Instruct-Turbo
  # prompt: optional override of the default English judge prompt ({transcript} placeholder)
```

| sub-field | meaning |
|-----------|---------|
| `provider` | required; same `ProviderCfg` shape as agent providers — YAML anchors (`*alias`) work here too |
| `prompt` | optional; overrides `DEFAULT_JUDGE_PROMPT` (`src/core/config.py`); must contain `{transcript}` — the placeholder is replaced literally with the full public cheap-talk feed |

Notes:
- Omitting `judge:` (or setting it to `null`) disables the judge entirely (`cfg.judge is None`).
- The judge config field is **excluded from the `run_id` hash** — adding or removing the
  judge does not create a new run entry; only the game/population config matters for de-duplication.
- Re-running an **identical** config: if the stored run is **finished**, the runner does nothing
  ("nothing to do — change seed or config"); if it is **unfinished** (crashed/aborted, no
  `finished_at`), the runner **deletes** that run (`Storage.delete_run` — a single
  `DELETE FROM runs`, child rows go via `ON DELETE CASCADE`) and re-runs from scratch. The
  decision lives in `runner._handle_existing_run` (a future extension point — could resume
  instead of delete).
- The verdict is printed by the runner / demo and stored in the `judge_verdicts` table in
  the SQLite DB; `replay.py` highlights cited messages in yellow and appends a JUDGE VERDICT section.

## Provider blocks & YAML anchors

The provider lives on the **population** (`population.provider`), one LLM shared by every
agent — model variability between agents isn't a research dimension, so like the rules and
the identity prompt it is a single fixed frame for the episode. A common pattern is to define
the provider as a top-level `&anchor` and reference it with `*alias`; pyyaml resolves these
itself. Example:

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
  provider: *default                 # one LLM provider, shared by all agents (required)
  identity_prompt: "You are AI agent {id}."   # system opener shared by all agents (optional)
  n_agents: 4
  first_name_pool: [...]             # >= n_agents unique names, validated
  last_name_pool:  [...]             # >= n_agents unique names, validated
  agents:                            # shorter than n_agents -> cycled at build time
    - {persona: "...", count: 2}
    - {}                             # persona optional -> omitted
```

`provider` — **required**, no default; the LLM used by every agent (see above). `identity_prompt`
— the system-prompt opener placed before the persona; `{id}` is replaced with the agent id.
Both live on the **population**, not the agent: like the game rules they are the same fixed
frame for every agent in the episode. `identity_prompt` is **optional** — when omitted it
defaults to `"You are AI agent {id}."` (`DEFAULT_IDENTITY_PROMPT` in `src/core/config.py`).

Per-agent keys: `persona` (optional; omit/`null` to drop it), `count`.

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
