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

## Resuming or extending a run

A stored run can be continued by its integer `run_id` — to **finish an unfinished run**
(crashed/aborted, no `finished_at`) or to **extend a finished one** to more rounds:

```bash
uv run python experiment.py --resume 87              # finish #87 to its configured rounds
uv run python experiment.py --resume 87 --rounds 20  # grow #87 to 20 rounds (extend)
```

`runner.resume_run` reloads the run's stored config (`runs.config` → `config.episode_from_dict`),
rebuilds the population from the same seed (identical agent ids), rehydrates each agent's score
and memory from the DB (`Storage.load_state` — diary from `messages`/`pairings`, notes from
`a_notes`/`b_notes`, score = Σ payoffs + idle), and plays from `last_round + 1` to the target.
Past rounds are read from the DB (the actual recorded pairings), only new rounds are played —
so runs recorded before the per-round-rng change are resumable too. With `--rounds` ≤ what's
already played, it's a no-op **only when the run is already finished**; a run whose final round
aborted has all its round rows written (the failed pairing is persisted *before* `EpisodeAborted`)
yet no `finished_at` — resume plays zero new rounds and just stamps `finished_at` to close it,
instead of getting stuck forever on "nothing to do". Extending updates `runs.config`'s `rounds` but **not** `config_hash`
(which excludes `rounds`), so the run stays in its design family. The LLM judge is not re-run on
resume/extend (it's a whole-episode analytics pass — score the run separately).

## Reference configs

- `config/example.yaml` — direct strategy (agents pick numbers themselves).
- `config/example_prediction.yaml` — prediction strategy with `one_above` mapping.

## Top-level fields (`EpisodeCfg`, `src/core/config.py:49`)

| field | meaning |
|-------|---------|
| `seed` | drives population build + a derived matchmaker rng stream. Int = fixed seed; `random` = draw a fresh seed at load time (each run differs). The drawn int is persisted into the run (`runs.seed`/`config`), so the run stays reproducible by that number, and resume/extend reuses the stored int. |
| `rounds` | number of rounds in the episode |
| `matchmaker` | only `random` is implemented |
| `context_window` | per-agent memory window; `null` = unbounded |
| `idle_payoff` | what an odd-one-out agent scores when it sits a round out |
| `max_concurrency` | semaphore size for concurrent pairings |
| `game` | `GameCfg`: `payoffs {R,T,P,S}`, `max_talk_turns`, `talk_stop_rule` (how cheap talk ends — all variants stop only when **both** agents set `finish: true`, varying along two axes (keep talking? revocable flag?): `both_ready_latch` (default) latches the first finisher silent until the other matures; `both_ready_revocable` lets it keep talking with a revocable `finish` flag (a later `finish: false` takes readiness back); `both_ready_committed` lets it keep talking but the `finish` flag is sticky — once set it cannot be revoked), `reflection` (extra post-game REFLECT call per agent, stored in memory; default `false`), `memory_notes_every` (0 = off; every N rounds an agent has actually played — counted per-agent, idle rounds excluded — it rewrites its memory into private notes via a NOTE call that then replace the raw round history; default `0`), and the prompt templates `talk_prompt`, `talk_open_prompt` (first turn of a round, empty feed — the agent opens the talk; no `{feed}`), `decide_prompt`, `predict_prompt`, `reflect_prompt`, `notes_prompt` (placeholders `{round} {score}`; used only when `memory_notes_every > 0`; `<game>`-wrapped like the other instructions), `notes_block_prompt` (`{notes}` — how saved notes are rendered back into the transcript: `<you>{notes}</you>`, since the notes are the agent's own private memo), `notes_header`/`buffer_header` (`<game>`-tagged section labels framing the notes vs the raw buffer of rounds since consolidation; the buffer header's `<game>` fuses with the first buffered round's `<game>` at the seam) — each defaults to a `DEFAULT_*` in `src/core/config.py`; delete a key to use the default. **All prompts are static templates — only the named placeholders are substituted, never assembled from text chunks.** DECIDE/PREDICT come as two complete static templates each: `decide_prompt`/`predict_prompt` (ask to reason first, then the number) and `decide_prompt_bare`/`predict_prompt_bare` (number only). The `rationale` flag (default `true`) picks **one whole template** — `rationale: false` selects the `_bare` variant — and also gates whether the returned reasoning is stored (blanked when off). There is no `{answer}` chunk-assembly. `predict_prompt` mirrors `decide_prompt` byte-for-byte except the directive (predict the opponent's number vs choose your own); all four take placeholders `{round} {partner} {feed} {reason}` (`{reason}` = how the chat closed — `reason_limit`/`reason_agreed`, the **same wording** the history close line uses). The whole LLM input is one **game transcript**: past rounds are replayed by `Memory.render` with the tags the rules declare (`<game>`/`<you>`/`<opponent name>`), and the current round's `{feed}` uses the same tags. The line templates are shared so a given line type reads identically in history and live; the history close line mirrors the live decide close line — `history_close_prompt` for `rationale: false`, `history_close_prompt_rationale` (asks to reason first) for `rationale: true`, picked by the same `rationale` flag. Each prompt is a complete template — who opened the round is not injected as a separate `{opener}` phrase (the transcript order already shows it). Transcript line templates are config (defaulted): `history_round_prompt` (`{round} {partner}`), `msg_self`/`msg_partner` (one cheap-talk line, `{text}`/`{partner}`), `history_close_prompt`/`history_close_prompt_rationale` (`{reason}`; bare vs reason-first close, picked by `rationale`), `reason_limit`/`reason_agreed`, `history_result_prompt` (`{round} {partner} {partner_number} {payoff} {partner_payoff} {total}`, where `{total}` is the score **after** the round; the agent's own number is shown just above — inside the rationale answer block when `show_rationale`, otherwise as a standalone `msg_self` `<you>` line). The running score therefore lives in the history result lines, not in the talk/decide headers. Three **private trace** lines (the agent's own `<you>` scratch notes) may appear, each behind its **own flag** and rendered only when its field is populated. `history_rationale_prompt` (`{my_rationale} {my_number}`, gated by `show_rationale`) is the agent's **answer block**: a single `<you>` line replaying the JSON response — rationale first, then the number — rendered in place of the bare number line, before the revealing result (mirroring the live decide order). After the result line follow `history_predicted_prompt` (`{partner} {my_predicted}`, gated by `show_predicted`; set only for prediction agents) and `history_reflection_prompt` (`{my_reflection}`, gated by `show_reflection`). All three flags default `true`. In cheap talk the agent-facing JSON key to close the chat is `finish` (stored internally as `ready`). **Parse-retry corrections** are config too: when a phase reply is unparseable the engine appends a fixed JSON-format reminder (`talk_correction`, `decide_correction`/`decide_correction_bare`, `predict_correction`/`predict_correction_bare`, `reflect_correction`, `note_correction`; the judge has its own `judge.correction`). DECIDE/PREDICT pick the `_bare` vs full correction by the **same** `rationale` flag as the prompt — so in `rationale: false` the retry asks only for `{"number"}`, no longer contradicting a bare prompt (the previous hardcoded suffix always demanded `rationale`). These lived in `src/core/agent.py`/`src/judge/judge.py` before; they now default to `DEFAULT_*_CORRECTION` in `src/core/config.py` and are overridable per key |
| `population` | `PopulationCfg` (see below) |
| `judge` | `JudgeCfg` or absent/`null` — optional LLM judge (see below) |
| `schedule` | optional list of **change-points** that vary the config per round (see below); absent = one config for the whole run |

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
  # correction: optional override of the parse-retry reminder (DEFAULT_JUDGE_CORRECTION)
```

| sub-field | meaning |
|-----------|---------|
| `provider` | required; same `ProviderCfg` shape as agent providers — YAML anchors (`*alias`) work here too |
| `prompt` | optional; overrides `DEFAULT_JUDGE_PROMPT` (`src/core/config.py`); must contain `{transcript}` — the placeholder is replaced literally with the full public cheap-talk feed |
| `correction` | optional; overrides `DEFAULT_JUDGE_CORRECTION` — the JSON-format reminder appended on the single parse-retry |

Notes:
- Omitting `judge:` (or setting it to `null`) disables the judge entirely (`cfg.judge is None`).
- The judge config field is **excluded from `config_hash`** (see below) — adding or removing
  the judge does not change a run's design hash.
- **Run identity is an incremental integer `run_id`** (1, 2, 3 …), allocated by SQLite
  (`INTEGER PRIMARY KEY AUTOINCREMENT`), **not** a config hash. Every `run` creates a **new**
  run: re-running the same config no longer de-dups, it just gets the next number — handy for
  repeated runs of one config under a noisy LLM. The **design** is recorded separately in
  `runs.config_hash` = SHA-256 of the config **minus `judge` and `rounds`** (`store._hash_config_dict`).
  Runs of the same design — repeats and longer continuations — share one `config_hash` (a
  "family"). `rounds` is excluded because with per-round matchmaker rng round *r* is identical
  regardless of total length, so a longer run is just a shorter one continued. `replay.py`
  accepts either the integer `run_id` or a `config_hash` (the latter resolves to the earliest
  run of that family).
- The verdict is printed by the runner / demo and stored in the `judge_verdicts` table in
  the SQLite DB; `replay.py` highlights cited messages in yellow and appends a JUDGE VERDICT section.

## Per-round config: the `schedule` block (change-points)

The config can **vary per round**. A top-level `schedule:` is an ordered list of
**change-points**, each with a `from_round` and a `patch` — a sparse partial config that is
**deep-merged** onto the base from `from_round` onward (later change-points override earlier
ones; lists are replaced, not merged). `cfg_for_round(cfg, r)` (`src/core/config.py`)
materializes the full `EpisodeCfg` for round *r*; the orchestrator rebuilds the game from it
each round, so payoffs, talk turns, prompts, `idle_payoff`, reflection/notes, etc. can phase
across a run.

```yaml
schedule:
  - from_round: 4
    patch: {game: {payoffs: {T: 6}}}     # from round 4 onward, betrayal pays 6
  - from_round: 6
    patch: {game: {max_talk_turns: 0}}   # from round 6 onward, decision-only (no cheap talk)
```

- `patch` change-points are **sticky** (active from `from_round` onward). `seed`,
  `max_concurrency` and the matchmaker *kind* are whole-run frame and are not patched per round.
- Every folded phase is **validated eagerly at load** (`_validate`) — an invalid patch fails
  fast, not mid-run.
- The schedule is part of the **design**: it is included in `config_hash` (so a different
  schedule is a different family), while `rounds` stays excluded — extending a run honours the
  schedule for the new rounds (`runner.resume_run`).

> Roster join/leave, manual pairing and injecting a number for an agent (the imperative
> change-point kinds and the interactive/scripted **controller** that drives them) land in
> later phases — see `docs/superpowers/plans/2026-06-25-interventions-axis-bc.md`.

## Provider blocks & YAML anchors

The provider lives on the **population** (`population.provider`), one LLM shared by every
agent — model variability between agents isn't a research dimension, so it is a single fixed
frame for the episode (each agent's wording lives in its own `system_prompt`). A common pattern is to define
the provider as a top-level `&anchor` and reference it with `*alias`; pyyaml resolves these
itself. Example:

```yaml
provider_default: &default
  base_url: https://api.together.xyz/v1
  api_key_env: TOGETHER_API_KEY      # key read from .env
  model: Qwen/Qwen2.5-7B-Instruct-Turbo
  temperature: 0.7
  max_tokens: 1000
  reasoning: true                    # reasoning-модели (DeepSeek-V4-Pro): false -> Non-think
  reasoning_effort: ""               # "" | high | max (задел; шлётся только если непусто)
```

`ProviderCfg` (`src/core/config.py:8`) is OpenAI-compatible — point `base_url` at
any `/chat/completions` endpoint (Together.ai in prod, Ollama for smoke tests).
**`reasoning`/`reasoning_effort`** control thinking on reasoning-capable models:
`reasoning: false` sends `{"reasoning": {"enabled": false}}` (Together's Non-think for
DeepSeek-V4-Pro); the default `true` sends nothing (provider decides — non-reasoning
models simply ignore the absent field). `reasoning_effort` (`high`/`max`) is forwarded
only when non-empty. The research sweep (`config/research.yaml`) sets `reasoning: false`
so every model — including DeepSeek — runs Non-think for a like-for-like comparison.

## Population block

```yaml
# The whole system prompt is ONE per-agent string (anchor-shareable). Define once, reference
# with *alias. The engine substitutes only {id} and the payoffs {R}/{T}/{P}/{S}/{max_talk_turns}.
system_pragmatic: &system_pragmatic |-
  You are AI agent {id}, a pragmatic player. ...full rules with {R}/{T}/{P}/{S}...

population:
  kind: roster                       # only roster is implemented
  provider: *default                 # one LLM provider, shared by all agents (required)
  n_agents: 4
  first_name_pool: [...]             # >= n_agents unique names, validated (optional)
  last_name_pool:  [...]             # >= n_agents unique names, validated (optional)
  agents:                            # shorter than n_agents -> cycled at build time
    - {count: 2, play_strategy: direct, system_prompt: *system_pragmatic}
    - {count: 2, play_strategy: prediction, prediction_mapping: one_above, system_prompt: *system_pragmatic}
```

`provider` — **required**, no default; the LLM used by every agent (see above).

**`system_prompt`** is the agent's **entire** system message, taken **verbatim** — there is no
more `identity_prompt + persona + rules` assembly (those fields are gone). Write the whole frame
(identity, persona, game rules) into one string and share it across agents via a YAML
`&anchor` / `*alias`. The engine substitutes only `{id}` (agent id) and the payoff placeholders
`{R}/{T}/{P}/{S}/{max_talk_turns}` (from `game.payoffs` / `game.max_talk_turns`) — everything
else is literal, so keep the canonical tag conventions (`<game>/<you>/<Name>`) and the payoff
placeholders in the text if you want history rendering and payoffs to stay in sync. `system_prompt`
is **optional**; omitted, it defaults to `DEFAULT_SYSTEM_PROMPT` (preamble + canonical rules).
Legacy `identity_prompt` / `persona` / `game.rules` keys in old configs are silently ignored.

Per-agent keys: `count`, `play_strategy` (`direct` (default) or `prediction`),
`prediction_mapping` (`match` (default) or `one_above`; only used when `play_strategy:
prediction`), `system_prompt` (above). **Strategy is per-agent**, so a population can mix
direct and prediction agents in one episode; `_validate` checks each spec's strategy/mapping,
and the game builds each agent's strategy from `agent.setup` (`ReputationPD._strategy_for`).

Agent ids are sampled from the name pools, in one of three modes:

- **both pools** → unique `First Last` strings;
- **one pool only** → the id is the pool entry itself, no surname (e.g. a single
  `first_name_pool: [Player 348, Player 712, ...]` yields ids `Player 348` …); numeric
  entries are coerced to strings;
- **no pools** → fall back to `A1`, `A2`, … ids.

`_validate` (`src/core/config.py`) enforces, **per non-empty pool**: no duplicates within
the pool and size ≥ `n_agents`. Pools are optional and validated independently, so a single
pool is valid on its own.

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
