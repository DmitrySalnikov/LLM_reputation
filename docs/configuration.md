# Configuration

One episode = one YAML file under `config/`. Loaded and validated once by `load_episode`
(`src/core/config.py`); an invalid config fails fast with a `ValueError`. All
config objects are **frozen dataclasses**.

Reference configs:

- `config/experiment.yaml` — the main single-episode config used by `experiment.py`.
- `config/research.yaml` — the design used by the `research.py` model sweep.
- `config/judge_qwen3_vllm.yaml` — a judge-only config (points the LLM judge at a vLLM/Ollama endpoint).

## Running an episode

```bash
uv run python experiment.py                       # config/experiment.yaml -> experiment.db
uv run python experiment.py config/other.yaml     # a different episode
uv run python experiment.py config/other.yaml "run name"   # + a human label (runs.name)
```

Each invocation appends a **new** run (fresh incremental `run_id`) to `experiment.db`.
`experiment.py` calls `load_dotenv()`, so provider API keys come from `.env`. Set
`LLM_TRACE=1` (env var or `.env`) to print the exact LLM input of every DECIDE/PREDICT/REFLECT
call while it runs.

## Resuming or extending a run

A stored run can be continued by its integer `run_id` — to **finish an unfinished run**
(crashed/aborted, no `finished_at`) or to **extend a finished one** to more rounds:

```bash
uv run python experiment.py --resume 87              # finish #87 to its configured rounds
uv run python experiment.py --resume 87 --rounds 20  # grow #87 to 20 rounds (extend)
```

`resume_run` reloads the run's stored config, rebuilds the population from the same seed
(identical agent ids), rehydrates each agent's score and memory from the DB, and plays
only the missing rounds (`last_round + 1` → target). Past rounds are read from the DB, so
any stored run is resumable. Extending updates
`runs.config`'s `rounds` but **not** `config_hash` (which excludes `rounds`), so the run
stays in its design family. The LLM judge is not re-run on resume/extend.

## Top-level fields (`EpisodeCfg`)

| field | meaning |
|-------|---------|
| `seed` | drives population build + a derived matchmaker rng stream. An int = fixed seed; `random` = draw a fresh seed at load time. The drawn int is persisted (`runs.seed`/`config`), so the run stays reproducible and resume reuses it. |
| `rounds` | number of rounds in the episode |
| `matchmaker` | only `random` is implemented |
| `context_window` | per-agent memory window; `null` = unbounded |
| `idle_payoff` | what an odd-one-out agent scores when it sits a round out |
| `max_concurrency` | semaphore size for concurrent pairings |
| `game` | `GameCfg` — payoffs, talk rules, prompts (see below) |
| `population` | `PopulationCfg` (see below) |
| `judge` | optional `JudgeCfg` (see below) |
| `schedule` | optional list of per-round change-points (see below) |

## The `game` block (`GameCfg`)

Core knobs:

| field | meaning |
|-------|---------|
| `payoffs` | `{R, T, P, S}` — reference `3, 5, 1, 0`; must satisfy `T > R > P > S` and `2R > T + S` |
| `max_talk_turns` | hard ceiling on cheap-talk messages per round |
| `talk_stop_rule` | how cheap talk ends; all variants stop only when **both** agents set `finish: true` — `both_ready_latch` (default), `both_ready_revocable`, `both_ready_committed` (see [architecture.md](./architecture.md)) |
| `rationale` | `true` (default) = agents reason before choosing (and it is stored); `false` = number only |
| `reflection` | `true` = one extra post-game REFLECT call per agent, stored in memory (default `false`) |
| `memory_notes_every` | `0` = off; every N rounds an agent has actually played it rewrites its memory into private notes (default `0`) |

**Prompt templates.** Every prompt is a **static template**; only named placeholders are
substituted — never assembled from text chunks. Each defaults to a `DEFAULT_*` in
`src/core/config.py`; delete a key to use the default.

- Instruction prompts: `talk_prompt`, `talk_open_prompt` (first turn of a round, empty
  feed), `decide_prompt` / `decide_prompt_bare`, `predict_prompt` / `predict_prompt_bare`,
  `reflect_prompt`, `notes_prompt`. The `rationale` flag picks the full vs `_bare` DECIDE/PREDICT
  template. `predict_prompt` mirrors `decide_prompt` byte-for-byte except the directive.
  Placeholders: `{round} {partner} {feed} {reason}` (`{reason}` = how the chat closed).
- Transcript line templates (how history is replayed): `history_round_prompt`,
  `msg_self` / `msg_partner`, `history_close_prompt` / `history_close_prompt_rationale`,
  `reason_limit` / `reason_agreed`, `history_result_prompt` (carries the running score as
  `{total}` — the score lives in result lines, not talk/decide headers), and three optional
  **private trace** lines, each behind its own flag: `history_rationale_prompt`
  (`show_rationale`), `history_predicted_prompt` (`show_predicted`; prediction agents only),
  `history_reflection_prompt` (`show_reflection`). All three flags default `true`.
- Notes rendering: `notes_block_prompt` (`<you>{notes}</you>` — the agent's private memo),
  `notes_header` / `buffer_header` (`<game>`-tagged section labels framing notes vs the raw
  buffer).
- Parse-retry corrections (appended when a reply is unparseable): `talk_correction`,
  `decide_correction` / `decide_correction_bare`, `predict_correction` /
  `predict_correction_bare`, `reflect_correction`, `note_correction`. DECIDE/PREDICT pick the
  `_bare` vs full correction by the **same** `rationale` flag as the prompt.

## The `population` block (`PopulationCfg`)

```yaml
# The whole system prompt is ONE per-agent string (anchor-shareable). Define once,
# reference with *alias. The engine substitutes only {id} and the payoffs
# {R}/{T}/{P}/{S}/{max_talk_turns}.
system_pragmatic: &system_pragmatic |-
  You are AI agent {id}, a pragmatic player. ...full rules with {R}/{T}/{P}/{S}...

population:
  kind: roster                       # only roster is implemented
  provider: *default                 # one LLM provider, shared by all agents (required)
  n_agents: 4
  first_name_pool: [...]             # >= n_agents unique names (optional)
  last_name_pool:  [...]             # >= n_agents unique names (optional)
  agents:                            # shorter than n_agents -> cycled at build time
    - {count: 2, play_strategy: direct, system_prompt: *system_pragmatic}
    - {count: 2, play_strategy: prediction, prediction_mapping: one_above, system_prompt: *system_pragmatic}
```

- `provider` — **required**, no default; the single LLM used by every agent. Model
  variability between agents is not a research dimension, so it is one fixed frame per
  episode (each agent's wording lives in its own `system_prompt`).
- `system_prompt` — the agent's **entire** system message, taken verbatim. Only `{id}` and
  the payoff placeholders are substituted; keep the canonical tag conventions
  (`<game>` / `<you>` / `<Name>`) so history rendering stays in sync. Optional — omitted, it
  defaults to `DEFAULT_SYSTEM_PROMPT`.
- Per-agent keys: `count`, `play_strategy` (`direct` default, or `prediction`),
  `prediction_mapping` (`match` default, or `one_above`; only for prediction agents),
  `system_prompt`. **Strategy is per-agent**, so a population can mix both.

Agent ids come from the name pools in one of three modes:

- **both pools** → unique `First Last` strings;
- **one pool only** → the id is the pool entry itself, no surname (e.g. a single
  `first_name_pool: [Player 348, Player 712, ...]`);
- **no pools** → fall back to `A1`, `A2`, ….

`_validate` enforces, per non-empty pool: no duplicates and size ≥ `n_agents`.

## The `judge` block (`JudgeCfg`)

An optional top-level `judge:` enables a separate LLM that reads the episode's public
cheap-talk once, after the episode ends, and returns a verdict on whether a reputation
institution emerged.

```yaml
judge:
  provider:
    base_url: https://api.together.xyz/v1
    api_key_env: TOGETHER_API_KEY        # env-var NAME, not the value
    model: Qwen/Qwen2.5-72B-Instruct-Turbo
  # prompt:     optional override of DEFAULT_JUDGE_PROMPT (must contain {transcript})
  # correction: optional override of the parse-retry reminder
```

- `provider` uses the same `ProviderCfg` shape as agents; YAML anchors work here too. The
  judge can therefore live on a different endpoint than the agents.
- Omitting `judge:` (or `null`) disables it entirely.
- The judge is **excluded from `config_hash`** — adding/removing it does not change a run's
  design.
- The verdict is stored in `judge_verdicts`; `replay.py` highlights cited messages in
  yellow and appends a JUDGE VERDICT section.

Judge stored runs after the fact with `judge_runs.py`; see [database.md](./database.md).

## The `schedule` block (per-round change-points)

The config can **vary per round**. A top-level `schedule:` is an ordered list of
change-points, each with a `from_round` and a `patch` — a sparse partial config deep-merged
onto the base from `from_round` onward (later change-points override earlier; lists are
replaced, not merged). `cfg_for_round(cfg, r)` materializes the full `EpisodeCfg` for round
`r`, and the orchestrator rebuilds the game from it each round.

```yaml
schedule:
  - from_round: 4
    patch: {game: {payoffs: {T: 6}}}     # from round 4 onward, betrayal pays 6
  - from_round: 6
    patch: {game: {max_talk_turns: 0}}   # from round 6 onward, decision-only (no cheap talk)
```

- Patches are **sticky** (active from `from_round` onward). `seed`, `max_concurrency` and the
  matchmaker *kind* are whole-run frame and are not patched.
- Every folded phase is **validated eagerly at load** — an invalid patch fails fast.
- The schedule is part of the **design** (included in `config_hash`); `rounds` stays excluded,
  so extending a run honours the schedule for the new rounds.

## Provider blocks & YAML anchors

Define the provider once as a top-level `&anchor` and reference it with `*alias`; PyYAML
resolves these itself.

```yaml
provider_default: &default
  base_url: https://api.together.xyz/v1
  api_key_env: TOGETHER_API_KEY      # key read from .env
  model: Qwen/Qwen2.5-7B-Instruct-Turbo
  temperature: 0.7
  max_tokens: 1000
  reasoning: true                    # reasoning models (e.g. DeepSeek-V4-Pro); false = Non-think
  reasoning_effort: ""               # "" | high | max (sent only when non-empty)
```

`ProviderCfg` is OpenAI-compatible — point `base_url` at any `/chat/completions` endpoint
(Together.ai in production, Ollama for smoke tests). `reasoning: false` sends
`{"reasoning": {"enabled": false}}` (Together's Non-think for DeepSeek-V4-Pro); the default
`true` sends nothing (non-reasoning models ignore the absent field). `reasoning_effort` is
forwarded only when non-empty. The research sweep sets `reasoning: false` so every model
runs Non-think for a like-for-like comparison.

## Adding a config knob

1. Add the field to the relevant frozen dataclass in `src/core/config.py`.
2. Wire it through the matching `_*_cfg` builder and/or `load_episode`.
3. If it constrains valid input, extend `_validate` (fail fast, clear message).
