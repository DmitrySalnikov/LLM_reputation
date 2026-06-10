# LLM Judge for Reputation Emergence — Design

**Date:** 2026-06-10
**Status:** Approved

## Problem

The research question of this project is whether a **reputation institute** emerges
in populations of LLM agents playing the repeated coordination game with cheap talk.
Today a human has to read transcripts to answer that. We add an **LLM judge**: a
separate model that reads the episode's public communication once, at the end of the
run, and renders a verdict — did reputation emerge, and which messages show it.

## Decisions (made with the user)

1. **Scope:** the judge runs in all paths — `examples/orchestrator_demo.py` (YAML)
   and `experiment.py` → `src/runner.py` (SQLite). The verdict is persisted so
   `replay.py` can show it.
2. **Judge input:** **public cheap-talk messages only.** No private rationales, no
   reflections, no chosen numbers, no payoffs. The question is whether reputation is
   visible in public communication.
3. **Rendering:** the live run prints only the verdict (present/absent +
   explanation). Colored highlighting of cited messages happens **only in
   `replay.py`**, inline in the round-by-round narration, plus a verdict section at
   the end of the replay.
4. **Architecture:** Approach A — a new `src/judge/` module invoked by the callers
   (runner, demo) after `run_episode` returns. The orchestrator is untouched; its
   only output channel remains the `observer` callback.
5. **run_id:** the `judge` config is **excluded** from the run_id hash. Judge is
   analysis, not gameplay; toggling it must not force a re-run of an expensive
   episode. Consequence: re-running an already-stored config to add a verdict is
   skipped exactly like today (a re-judge tool may come later — out of scope).

## Configuration

In `src/core/config.py`:

```python
@dataclass(frozen=True)
class JudgeCfg:
    provider: ProviderCfg                 # judge's own model, independent of agents
    prompt: str = DEFAULT_JUDGE_PROMPT    # editable English template
```

- `EpisodeCfg` gains `judge: JudgeCfg | None = None`. **Absence = feature off**;
  there is no separate `enabled` flag.
- YAML: optional top-level `judge:` block; its `provider:` may reuse the existing
  YAML anchors or define a different model:

```yaml
judge:
  provider:
    base_url: https://api.together.xyz/v1
    model: Qwen/Qwen2.5-72B-Instruct-Turbo
    api_key_env: TOGETHER_API_KEY
  # prompt: optional override of DEFAULT_JUDGE_PROMPT
```

- `load_episode` builds `JudgeCfg` when the block is present; `_validate` fails fast
  (Russian message) if `judge` is present without a `provider`.
- `experiment.py` users construct `JudgeCfg(...)` directly in `CONFIG`.

`DEFAULT_JUDGE_PROMPT` lives with the other prompt defaults in `src/core/config.py`,
uses literal `{...}` placeholder replacement like the game prompts (placeholder:
`{transcript}`), and is therefore persisted verbatim into the stored run config.

## Judge module — `src/judge/`

```python
@dataclass(frozen=True)
class MessageRef:
    round: int
    pair: int
    turn: int

@dataclass(frozen=True)
class JudgeVerdict:
    emerged: bool
    explanation: str
    evidence: list[MessageRef]   # validated refs only

async def judge_episode(cfg: JudgeCfg, records: list[PairingRecord]) -> JudgeVerdict
```

Behavior:

- **Transcript rendering:** every public message across the episode is enumerated
  with a stable id `r<round>.p<pair>.t<turn>` (matching the storage keys
  `round_idx`/`pair_idx`/`turn_idx`), grouped by round and pairing, with speaker
  names. Nothing private is rendered. `pair` is the index of the record within its
  round (the same `pair_idx` Storage assigns via `enumerate`).
- **One LLM call** through `make_provider(cfg.provider)`; the judge owns its
  provider instance and closes it (`finally`).
- **Prompt** (English, default in config): briefly states the game rules context,
  defines the reputation institute (agents conditioning behavior on partners' past
  conduct, gossip / third-party information, history-based trust expressed in
  messages), and demands JSON only:

```json
{"emerged": <true|false>, "explanation": "<short explanation>",
 "evidence": ["r0.p1.t2", "..."]}
```

- **Parsing:** lenient JSON extraction (same raw / fenced / balanced-brace approach
  as `Agent.act`; extraction helper reused or shared, not duplicated). One
  correction retry on parse failure, then `JudgeError` is raised.
- **Evidence validation:** ids that don't match any actual message are dropped with
  a DEBUG-level warning (Russian); they never crash the run.

## Wiring

- `src/runner.py` (`run_experiment`): after `st.finish(pop)`, if `cfg.judge` is set —
  `await judge_episode(...)` over the collected records (runner keeps its own
  per-round list, same as the demo does), print the verdict, persist it via
  `Storage.save_verdict`. The records list is accumulated in the existing observer.
- `examples/orchestrator_demo.py`: same call after the scoreboard; print-only.
- **Error handling:** the episode must never be lost to a judge failure. Callers
  wrap the judge call in `try/except`; on failure print a Russian warning and
  continue — the run stays valid with no verdict row.
- Console output (live run): verdict only — emerged / not emerged + explanation +
  count of evidence messages. No colored excerpts (decision 3).

## Storage & replay

New table:

```sql
CREATE TABLE IF NOT EXISTS judge_verdicts (
    run_id      TEXT PRIMARY KEY,
    emerged     INTEGER NOT NULL,
    explanation TEXT NOT NULL,
    evidence    TEXT NOT NULL,      -- JSON: [{"round":0,"pair":1,"turn":2}, ...]
    model       TEXT NOT NULL,      -- judge provider model, for the record
    created_at  TEXT NOT NULL,
    FOREIGN KEY (run_id) REFERENCES runs(run_id)
);
```

- `Storage.save_verdict(verdict, model)` inserts one row.
- `_run_id` drops the `judge` key from `asdict(cfg)` before hashing (decision 5).
  Existing run_ids of judge-less configs are unaffected.
- `replay.py`: loads the verdict row if present; builds a set of cited
  `(round_idx, pair_idx, turn_idx)`; while narrating, cited messages are wrapped in
  ANSI yellow (only when stdout is a TTY); a `JUDGE VERDICT` section after the
  scoreboard prints emerged/not + explanation + the cited messages list. Runs
  without a verdict replay exactly as today.

## Testing (TDD, ScriptedProvider, no network)

- `tests/judge/`: `judge_episode` with a scripted verdict reply — parses into
  `JudgeVerdict`; the recorded prompt contains the enumerated public messages and
  contains **no** rationale/reflection text; invalid evidence ids are dropped;
  parse failure path (correction retry, then `JudgeError`).
- `tests/core/test_config.py` additions: YAML with `judge:` block loads `JudgeCfg`;
  YAML without it yields `judge=None`; `judge` without `provider` fails validation.
- `tests/storage/`: verdict save/load round-trip; run_id unchanged by adding a
  `judge` block to the config.
- `replay.py`: extract a small pure helper (e.g. `cited(refs, r, p, t)` / ANSI
  wrapper) and unit-test it; the script body itself stays manually verified.

## Out of scope

- Re-judging an already-stored run (post-hoc `judge.py <run_id>`).
- Multiple judges / judge ensembles, confidence scores, per-round judging.
- Highlighting in the live narration (it has already scrolled by when the judge runs).
