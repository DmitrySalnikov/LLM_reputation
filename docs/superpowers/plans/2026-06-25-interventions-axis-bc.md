# Interventions (Axis B/C) — Implementation Plan

> **⏸ STATUS — PAUSED (2026-06-25).** Feature shelved after Phase 1. **Phase 1 (per-round
> config / change-point `schedule`) is DONE but UNCOMMITTED** on branch `main` (sits with the
> rest of the uncommitted resume/extend work). Full suite 257 green. **Resume here → Phase 2**
> (roster join/leave + controller seam). Nothing below Phase 1 is started. The config-change
> mechanism that landed: `ChangePoint`/`schedule`/`cfg_for_round`/`_deep_merge` in
> `src/core/config.py`, orchestrator per-round resolution, hash+resume support — see the
> Phase 1 "Shipped" note. Design decisions are frozen in "Locked design decisions" below;
> do not re-litigate them on resume.

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking. Build phase-by-phase; each phase ships on its own.

**Goal:** Let a run's configuration **vary per round** and let the experimenter **intervene** — change payoffs/talk/prompts/strategy mid-run, force specific pairings, add/remove agents, and inject a number for an agent — either **scripted** (a declarative schedule in the config) or **interactive** (prompted each round). Scripted (B) and interactive (C) are the same mechanism: the interactive session *produces* the schedule that a scripted re-run *replays*.

**Architecture:** One authored artifact — the **schedule**: a base `EpisodeCfg` plus an ordered list of **change-points** `(from_round, …)`. It is consumed by two paths the orchestrator already has room for:

- **Pure config** (payoffs, talk turns, prompts, strategy, matchmaker policy) → resolved by `cfg_for_round(cfg, r)` (deep-merge of `patch` change-points). The round loop uses the per-round cfg instead of one fixed cfg.
- **Imperative directives** (roster join/leave, manual pairing, inject-a-number) → fed in through a new **controller** seam, inbound and symmetric to the existing `observer`. The default controller derives directives from the schedule (scripted); an interactive controller prompts the user and *writes* them back into the schedule.

The schedule lives in `runs.config` (already persisted), so resume/replay reconstruct everything from it — including the live roster (replay the join/leave change-points), which is the one thing the `seed → roster` function cannot recover on its own. Builds on the completed resume/extend work (`docs/configuration.md` "Resuming or extending a run", `src/runner.py::resume_run`, per-round matchmaker rng).

**Tech Stack:** Python 3.12, asyncio, frozen dataclasses, PyYAML, sqlite3, pytest + pytest-asyncio (auto mode), `ScriptedProvider` test double. Manage deps via `uv` only.

**Project conventions that apply to every task here:**
- Docstrings / prints / error messages in **Russian**; LLM-facing prompt text in **English**; established terms (payoff, cheap-talk, matchmaker, roster) stay English.
- `from __future__ import annotations` atop every module; rng objects are passed in, never created in `src/` library code.
- TDD: failing test first (AAA, behavioural name), then minimal code.
- No printing/persistence inside `src/` except the runner/caller layer and DEBUG logging via `src.core.agent`. The orchestrator's only *outbound* channel stays `observer`; the controller is its only new *inbound* channel and is owned by the caller (like `observer`).
- Keep docs in sync (`docs/configuration.md`, `docs/architecture.md`, `CLAUDE.md`).

---

## Locked design decisions (settled with the researcher)

1. **Store sparse, consume full.** The schedule is authored/stored as `base + change-points` (a human writes "from round 6, payoffs change", not 50 full configs). The engine *materializes* the full `EpisodeCfg` per round via `cfg_for_round`. Optionally cache the resolved per-round config in the `rounds` table for audit — it is a cache, not the source of truth.
2. **One run-level hash over the whole schedule.** `config_hash` = SHA-256 of `base + schedule` minus `judge` and `rounds` (extends today's `_hash_config_dict`). A schedule that kicks an agent or changes payoffs is a *different design* → different hash, correctly. Extending with identical params adds no change-point → hash unchanged (today's "family" invariant survives). **Do not** put the round number inside the hash — identical rounds must hash identically.
3. **Two change-point lifetimes.** `patch` (scalar config) and `roster` (join/leave) are **sticky** — active from `from_round` onward. `pairing` (manual partition) and `inject` (forced number) are **one-shot** — apply only at exactly that round. This split is intrinsic: "payoffs change" persists; "pair them *this* round" does not.
4. **Roster is recoverable only from the schedule.** `seed → roster` yields the *initial* roster only; mid-run join/leave leave no trace in the seed. So roster edits are recorded as change-points and **replayed** on resume to reconstruct the live roster (same "rebuild skeleton from seed, rehydrate from history" philosophy already used for memory/score).
5. **join = clean newborn:** empty memory, score 0, born at its `from_round`. No seeded history. (Decided explicitly.)
6. **B = C.** Interactive controller appends directives to an in-memory schedule and the runner persists it into `runs.config` at finish (recompute `config_hash` then). An interactive run thus becomes a fully reproducible scripted run. A lightweight `interventions` audit table is optional (Phase 5) for replay annotation.

Phases are independently shippable: **1** (scalar per-round config) already delivers scripted payoff/prompt/strategy phasing and composes with extend. **2** adds roster. **3** manual pairing. **4** inject. **5** the interactive driver + persistence + replay/docs.

---

## Phase 1 — Per-round config (scalar change-points) — ✅ DONE (2026-06-25)

Deliver `cfg_for_round` and make the round loop use it. No controller, no roster yet. After this phase a YAML `schedule:` of `patch` change-points lets payoffs/talk/prompts/strategy phase across rounds, and resume/extend honour it.

**Shipped:** `ChangePoint` + `schedule` field + `_deep_merge`/`cfg_for_round` (`src/core/config.py`); YAML `schedule:` loading via `_change_point` + eager per-phase `_validate`; orchestrator resolves `cfg_for_round(cfg, r)` per round and rebuilds `ReputationPD(cfg_r.game)`; `config_hash` includes the schedule for free (asdict carries it), resume/extend round-trip it unchanged. Tests: `tests/core/test_schedule.py`, `+` schedule cases in `test_config_load.py`/`test_orchestrator.py`/`test_storage.py`/`test_runner.py`. Docs: `docs/configuration.md` "Per-round config". Full suite 257 green. **Not committed yet** (on `main`).

### Task 1.1: `ChangePoint` + `schedule` field + deep-merge resolver

**Files:** `src/core/config.py`; test `tests/core/test_schedule.py`

- [ ] **Step 1: Failing tests** — create `tests/core/test_schedule.py`:

```python
from __future__ import annotations

from dataclasses import replace

from src.core.config import ChangePoint, EpisodeCfg, GameCfg, PopulationCfg, ProviderCfg, cfg_for_round


def _base(**kw):
    spec = ... # AgentSpec(persona="p", count=2) — mirror tests/test_runner.py::_cfg
    return EpisodeCfg(seed=0, rounds=10, matchmaker="random",
                      population=PopulationCfg(kind="roster", agents=[spec],
                                               provider=ProviderCfg(base_url="http://x/v1", model="m")),
                      game=GameCfg(max_talk_turns=0), **kw)


def test_no_schedule_returns_base_unchanged():
    base = _base()
    assert cfg_for_round(base, 5) == base


def test_scalar_patch_applies_from_its_round_onward():
    base = _base(schedule=(ChangePoint(from_round=4, patch={"game": {"payoffs": {"T": 6}}}),))
    assert cfg_for_round(base, 3).game.payoffs.T == base.game.payoffs.T   # до точки — старое
    assert cfg_for_round(base, 4).game.payoffs.T == 6                      # с точки — новое
    assert cfg_for_round(base, 9).game.payoffs.T == 6                      # и дальше (sticky)


def test_later_change_point_overrides_earlier():
    base = _base(schedule=(ChangePoint(from_round=2, patch={"game": {"max_talk_turns": 4}}),
                           ChangePoint(from_round=6, patch={"game": {"max_talk_turns": 0}})))
    assert cfg_for_round(base, 5).game.max_talk_turns == 4
    assert cfg_for_round(base, 6).game.max_talk_turns == 0


def test_deep_merge_does_not_mutate_sibling_fields():
    # патч одного payoff не должен затирать остальные
    base = _base(schedule=(ChangePoint(from_round=1, patch={"game": {"payoffs": {"T": 9}}}),))
    r = cfg_for_round(base, 1).game.payoffs
    assert r.T == 9 and r.R == base.game.payoffs.R and r.S == base.game.payoffs.S
```

- [ ] **Step 2:** `uv run pytest tests/core/test_schedule.py` → FAIL (no `ChangePoint`/`cfg_for_round`).

- [ ] **Step 3: Implement in `src/core/config.py`.** Add the dataclass near the other config blocks:

```python
@dataclass(frozen=True)
class ChangePoint:
    """Одна точка изменения расписания эпизода (с раунда `from_round`).

    Виды (могут сочетаться в одной точке):
      patch   — частичный override скалярного конфига (game/payoffs/промпты/стратегия),
                **липкий**: действует с from_round и далее (сворачивается deep-merge).
      roster  — {"join": [...], "leave": [...]} — мутация состава, событие (Фаза 2).
      pairing — явная разбивка на пары для ЭТОГО раунда, **разовая** (Фаза 3).
      inject  — {agent_id: number} — навязать число агенту в ЭТОМ раунде, разовая (Фаза 4).

    Хранится разреженно; полный конфиг раунда собирает cfg_for_round (patch), а
    императивные директивы — directives_for_round (Фаза 2)."""

    from_round: int
    patch: dict | None = None
    roster: dict | None = None
    pairing: tuple | None = None
    inject: dict | None = None
```

Add `schedule: tuple[ChangePoint, ...] = ()` to `EpisodeCfg` (last field, defaulted — keeps existing constructors working). Add the resolver + deep-merge near `episode_from_dict`:

```python
def _deep_merge(base: dict, patch: dict) -> dict:
    """Рекурсивно наложить patch на base (новый словарь). dict → вглубь; всё прочее
    (скаляры, списки) — замена целиком. Списки НЕ сливаются — это сделано осознанно:
    листовые поля заменяются, а состав (списочное поле) меняют roster-директивы, не patch."""
    out = dict(base)
    for k, v in patch.items():
        out[k] = _deep_merge(out[k], v) if isinstance(v, dict) and isinstance(out.get(k), dict) else v
    return out


def cfg_for_round(cfg: EpisodeCfg, r: int) -> EpisodeCfg:
    """Материализовать полный EpisodeCfg для раунда r, свернув липкие patch-точки.

    Чистая функция: одинаковый (cfg, r) → одинаковый результат. Императивные директивы
    (roster/pairing/inject) здесь НЕ применяются — их обрабатывает контроллер."""
    if not cfg.schedule:
        return cfg
    d = asdict(cfg)
    d.pop("schedule", None)                                  # расписание не входит в раундовый конфиг
    for cp in sorted(cfg.schedule, key=lambda c: c.from_round):
        if cp.from_round <= r and cp.patch:
            d = _deep_merge(d, cp.patch)
    return episode_from_dict(d)
```

`episode_from_dict` must tolerate a `schedule` key (Task 1.2 wires YAML); here we drop it before rebuilding so the resolved per-round cfg carries an empty schedule (it is a single round's config, not a schedule).

- [ ] **Step 4:** `uv run pytest tests/core/test_schedule.py` → PASS.
- [ ] **Step 5: Commit** — `feat: ChangePoint + cfg_for_round — пораундовый конфиг из липких patch-точек`.

### Task 1.2: Load `schedule:` from YAML + eager per-phase validation

**Files:** `src/core/config.py`; test `tests/core/test_config_load.py`

- [ ] **Step 1: Failing tests** — append to `tests/core/test_config_load.py`: a YAML with a `schedule:` list of `{from_round, patch}` loads into `cfg.schedule` as `ChangePoint`s; an invalid patch (e.g. `patch: {rounds: -1}` or a payoff that breaks `_validate`) raises `ValueError` **at load** (fail fast), not at round time.

- [ ] **Step 2:** run → FAIL.

- [ ] **Step 3: Implement.** In `episode_from_dict`, build `schedule=tuple(_change_point(c) for c in d.get("schedule", []))` with a `_change_point(c)` builder. In `_validate`, after validating the base, **resolve and validate every fold boundary**: for each distinct `from_round` among patch change-points, build `cfg_for_round`-equivalent dict and run the existing field validations on it (a patch can produce an invalid config — catch it once at load). Russian error messages.

- [ ] **Step 4:** run the config-load suite → PASS.
- [ ] **Step 5: Commit** — `feat: загрузка schedule из YAML + ранняя валидация каждой фазы конфига`.

### Task 1.3: Round loop uses `cfg_for_round`

**Files:** `src/core/orchestrator.py`; test `tests/core/test_orchestrator.py`

- [ ] **Step 1: Failing test** — a 3-round episode with a `patch` change-point at round 2 (e.g. `max_talk_turns` 0→2) observably changes round 2's behaviour but not round 1's. Assert via the observer / a `ScriptedProvider` whose call count reflects talk turns. Also: `start_round` resume path still resolves the right per-round cfg.

- [ ] **Step 2:** run → FAIL.

- [ ] **Step 3: Implement.** In `run_episode`, inside the round loop, replace uses of `cfg.*` that are round-sensitive with `cfg_r = cfg_for_round(cfg, r)`:
  - build the game/talk/payoff context from `cfg_r.game`;
  - matchmaker policy from `cfg_r.matchmaker`;
  - keep `cfg.seed` (seed is whole-run, never patched) and the per-round rng `Random(f"{cfg.seed}:matchmaker:{r}")` unchanged.
  Note in a comment that `ReputationPD` is currently built once from `cfg.game`; to honour per-round game params, either rebuild the game per round from `cfg_r.game` or pass `cfg_r.game` into the play call. Prefer rebuilding `ReputationPD(cfg_r.game)` per round (cheap; the game is stateless between pairings).

- [ ] **Step 4:** run orchestrator suite → PASS.
- [ ] **Step 5: Commit** — `feat: оркестратор берёт конфиг раунда через cfg_for_round`.

### Task 1.4: Storage hash + resume honour the schedule

**Files:** `src/storage/store.py`, `src/runner.py`; tests `tests/storage/test_storage.py`, `tests/test_runner.py`

- [ ] **Step 1: Failing tests** —
  - `_hash_config_dict` already drops `judge`/`rounds`; add a test that two cfgs differing only in `schedule` get **different** `config_hash`, and that adding `rounds` still doesn't change it.
  - resume/extend test: a run whose schedule has a `patch` at round 3, extended past it, plays later rounds with the patched params (assert via final scores against a straight run of the same schedule).

- [ ] **Step 2:** run → FAIL (hash ignores schedule; or resume mismatch).

- [ ] **Step 3: Implement.** `_hash_config_dict` already hashes the whole dict minus `judge`/`rounds`; since `schedule` is now in `asdict(cfg)`, it is included automatically — verify and add the test. `resume_run` already does `episode_from_dict(json.loads(config_json))`, so the schedule round-trips; confirm `cfg_for_round` is what the (now schedule-aware) orchestrator uses for new rounds. No structural change expected beyond tests — if a gap appears, fix minimally.

- [ ] **Step 4:** `uv run pytest` → PASS.
- [ ] **Step 5: Commit** — `test: schedule участвует в config_hash; resume/extend honours per-round patches`.

---

## Phase 2 — Roster join/leave + the controller seam

Introduce the inbound **controller** seam (default no-op), the scripted controller that reads `roster` change-points, `Population` add/remove, and the resume-time roster replay.

### Task 2.1: `Population.add` / `Population.remove`

**Files:** `src/population/base.py` (or wherever `Population` lives — verify); test `tests/population/test_population.py`

- [ ] **Step 1: Failing tests** — `add(spec)` appends a live `Agent` (empty memory, score 0, id from spec) reusing the population's provider cache (no new provider for the same `(base_url, model)`); `remove(id)` drops it from `ids()`/iteration; removing a missing id raises a Russian `KeyError`/`ValueError`; `aclose()` still closes shared providers exactly once.

- [ ] **Step 2:** run → FAIL.

- [ ] **Step 3: Implement.** Add `add`/`remove` to `Population`. `add` builds an `Agent` the same way `RosterGenerator.build` does for one spec (persona, strategy via `agent.setup`, provider from the cache keyed `(base_url, model)`), with a fresh empty `Memory` and `score=0`. `remove` deletes from the roster list; do **not** aclose the provider (shared). Keep `from __future__ import annotations`, rng passed in if any sampling is needed (joins use explicit ids — no rng).

- [ ] **Step 4:** run population suite → PASS.
- [ ] **Step 5: Commit** — `feat: Population.add/remove — динамический состав (чистый новорождённый, общий кэш провайдеров)`.

### Task 2.2: Controller protocol + no-op + scripted-from-schedule

**Files:** new `src/core/controller.py`; test `tests/core/test_controller.py`

- [ ] **Step 1: Failing tests** — `NoOpController.round_start(...)` returns empty `Directives`; `ScheduleController(cfg.schedule).round_start(r, ...)` returns the roster ops whose `from_round == r` (one-shot at that round) as `joins`/`leaves`, and (Phases 3–4) `pairing`/`inject` for that round. `directives_for_round(schedule, r)` pure helper is covered directly.

- [ ] **Step 2:** run → FAIL.

- [ ] **Step 3: Implement.** 

```python
@dataclass(frozen=True)
class Directives:
    """Что контроллер хочет сделать в начале раунда. Пустой = поведение по умолчанию."""
    joins: tuple = ()        # specs новых агентов (id, persona, play_strategy, prediction_mapping)
    leaves: tuple = ()       # id выбывающих
    pairing: tuple | None = None     # явная разбивка на пары (Фаза 3); None = матчер сам
    inject: dict | None = None       # {agent_id: number} на этот раунд (Фаза 4)


class Controller(Protocol):
    async def round_start(self, round: int, pop, scores: dict) -> Directives: ...


class NoOpController:
    async def round_start(self, round, pop, scores) -> Directives:
        return Directives()


class ScheduleController:
    """Скриптовый контроллер: проигрывает директивы из расписания (ось B)."""
    def __init__(self, schedule): self._schedule = schedule
    async def round_start(self, round, pop, scores) -> Directives:
        return directives_for_round(self._schedule, round)
```

`directives_for_round(schedule, r)` collects `roster`/`pairing`/`inject` from change-points with `from_round == r`. (`patch` is NOT here — it goes through `cfg_for_round`.)

- [ ] **Step 4:** run → PASS.
- [ ] **Step 5: Commit** — `feat: controller seam (NoOp + ScheduleController) — входной канал директив раунда`.

### Task 2.3: Orchestrator applies roster directives

**Files:** `src/core/orchestrator.py`; test `tests/core/test_orchestrator.py`

- [ ] **Step 1: Failing tests** — `run_episode(..., controller=ScheduleController(schedule))` with a `leave` at round 2 and a `join` at round 3: round 2 plays without the left agent; round 3 includes the newborn (verify via observer pairings/idle and `pop.ids()`); the newborn's memory is empty (its first round shows no history). Default `controller=None` ⇒ `NoOpController` ⇒ behaviour identical to today (regression).

- [ ] **Step 2:** run → FAIL.

- [ ] **Step 3: Implement.** Signature: `run_episode(cfg, pop, *, observer=None, controller=None, start_round=1)`. At the top of each round, before pairing:

```python
        d = await ctrl.round_start(r, pop, {a.id: a.score for a in pop})
        for spec in d.joins:
            pop.add(spec)                       # новорождённый: пустая память, 0 очков
        for aid in d.leaves:
            pop.remove(aid)
        rng = random.Random(f"{cfg.seed}:matchmaker:{r}")
        plan = (await mm.plan_round(pop.ids(), r, rng)) if d.pairing is None else _plan_from(d.pairing, pop.ids())
```

(`_plan_from` lands in Phase 3; for Phase 2 keep `d.pairing is None` always.) Because `plan_round` already reads `pop.ids()` fresh, the mutated roster flows into pairing automatically. The caller still owns `pop.aclose()`.

- [ ] **Step 4:** run orchestrator suite → PASS (incl. the no-controller regression).
- [ ] **Step 5: Commit** — `feat: оркестратор применяет roster-директивы (join/leave) в начале раунда`.

### Task 2.4: Storage records joins/leaves; resume replays the roster

**Files:** `src/storage/store.py`, `src/runner.py`; tests `tests/storage/test_storage.py`, `tests/test_runner.py`

- [ ] **Step 1: Failing tests** —
  - After a run with a mid-run join, the `agents` table has a row for the newborn (with its join round recorded) and a `final_score`; a left agent keeps its frozen `final_score`.
  - `load_state(run_id, idle_payoff)` reconstructs the **live roster as of last_round** by replaying the schedule's roster change-points: a left agent is absent from the rebuilt live pop; a joined agent is present with memory rebuilt only from its post-birth rows.
  - End-to-end resume of a run that had a join/leave matches a straight run of the same schedule (final scores per surviving agent).

- [ ] **Step 2:** run → FAIL.

- [ ] **Step 3: Implement.**
  - Storage learns each round's live ids from `plan` (pairings + idle = everyone alive). On `observe`, upsert an `agents` row for any id not seen before, recording `joined_round` (new nullable column; NULL = initial roster). When the runner applies a `leave`, call a new `Storage.record_leave(agent_id, score, round)` to freeze `final_score`/`left_round` (add a `left_round` column). Keep schema migrations additive (the DB already migrated once — see `project_resume_extend`).
  - `resume_run`: after `episode_from_dict`, rebuild the **base** pop from seed, then replay `directives_for_round(cfg.schedule, r)` roster ops for `r in 1..last_round` (apply `pop.add`/`pop.remove`) to get the live roster, **then** `load_state` + `_apply_run_state` over that roster. `load_state` itself stays roster-agnostic (it rebuilds memory/score per agent_id seen in the DB); the roster shape comes from the schedule replay.

- [ ] **Step 4:** `uv run pytest` → PASS.
- [ ] **Step 5: Commit** — `feat: запись join/leave в agents + восстановление живого состава на resume (replay roster-точек)`.

---

## Phase 3 — Manual pairing

`pairing` change-points (one-shot) force an explicit partition for a round; the matchmaker is bypassed for that round.

### Task 3.1: `_plan_from(pairing, ids)` + orchestrator wiring

**Files:** `src/core/orchestrator.py` (+ maybe `src/matchmaking/`); tests `tests/core/test_orchestrator.py`

- [ ] **Step 1: Failing tests** — a `pairing` directive `[["A","B"],["C","D"]]` produces exactly those pairs and the correct `idle` (anyone alive but unpaired); validation rejects a pairing that names an unknown/absent id or pairs an id twice (Russian error). With no `pairing`, the matchmaker decides (regression).
- [ ] **Step 2:** run → FAIL.
- [ ] **Step 3: Implement.** `_plan_from(pairing, ids)` builds a `RoundPlan` (same shape `plan_round` returns) from the explicit partition; unpaired alive ids → `idle`. Validate membership/disjointness. Wire `d.pairing` in the round loop (replace the Phase-2 placeholder). Add it to `directives_for_round` (already collected in 2.2).
- [ ] **Step 4:** run → PASS.
- [ ] **Step 5: Commit** — `feat: ручная разбивка на пары через pairing-директиву (матчер в обход на этот раунд)`.

---

## Phase 4 — Inject a number for an agent

`inject` change-points (one-shot) substitute an agent's DECIDE for a fixed number this round, bypassing the LLM choice.

### Task 4.1: Game-level decision override

**Files:** `src/games/reputation_pd.py`; tests `tests/games/test_reputation_pd.py`

- [ ] **Step 1: Failing tests** — when `play_pairing` is given an inject map `{"A": 7}`, agent A's chosen number is 7 without calling `strategy.decide` (assert via `ScriptedProvider` call count: A makes no DECIDE call), B still decides normally; talk still happens (inject overrides only the number, not the cheap-talk); the `PairingRecord` stores the injected number and a marker so it is auditable (e.g. `a_rationale` left as-is/blanked + a recorded intervention — see Phase 5). Inject of an out-of-range number raises a Russian `ValueError`.
- [ ] **Step 2:** run → FAIL.
- [ ] **Step 3: Implement.** Thread an optional `inject: dict[str,int] | None` into `ReputationPD.play_pairing` (or the per-round play call). In the DECIDE step, if the agent's id is in `inject`, use that number instead of `await strategy.decide(...)`. Keep memory/resolve/payoff identical otherwise. The orchestrator passes `cfg_r`-independent `d.inject` from the controller into the game call for that round.
- [ ] **Step 4:** run games suite → PASS.
- [ ] **Step 5: Commit** — `feat: inject — навязать число агенту в раунде (минуя DECIDE), с записью в PairingRecord`.

---

## Phase 5 — Interactive driver, persistence, replay, docs

The interactive controller (caller layer, outside `src/`) prompts the experimenter each round and *writes* a schedule; the runner persists it so the run is reproducible; replay surfaces interventions; docs updated.

### Task 5.1: Interactive controller (caller layer)

**Files:** new `interactive.py` at repo root (caller layer, like `experiment.py`/`replay.py`); test `tests/test_interactive.py` (drive it with a scripted input function)

- [ ] **Step 1: Failing tests** — an `InteractiveController` whose input is a stub callback (no real stdin) turns answers into `Directives` (make pairs / add / remove / toggle predict via a `patch` it accumulates / inject) and **appends** the corresponding `ChangePoint`s to an in-memory schedule it exposes. Empty/“just continue” answer → empty `Directives` and no change-point. Lives outside `src/` (it prints/reads — that is the caller layer).
- [ ] **Step 2:** run → FAIL.
- [ ] **Step 3: Implement.** `InteractiveController` implements the `Controller` protocol; its `round_start` asks (via an injected `ask`/`print` pair, default real stdin) what to do, builds `Directives`, records equivalent `ChangePoint`s into `self.schedule`. Russian prompts. No engine API widening — it consumes only the public seam.
- [ ] **Step 4:** run → PASS.
- [ ] **Step 5: Commit** — `feat: интерактивный контроллер (caller-слой) — директивы из ответов + накопление расписания`.

### Task 5.2: Runner wires the controller; persists generated schedule

**Files:** `src/runner.py`, `experiment.py`; tests `tests/test_runner.py`

- [ ] **Step 1: Failing tests** — `run_experiment(cfg, db, controller=...)` passes the controller to `run_episode`; after an interactive run, `runs.config`'s `schedule` equals the controller's accumulated schedule and `config_hash` is recomputed to match it; re-running that stored config with `ScheduleController` reproduces the interactive run (final scores match). `experiment.py --interactive` selects the interactive controller; default stays no-op.
- [ ] **Step 2:** run → FAIL.
- [ ] **Step 3: Implement.** Add an optional `controller` param to `run_experiment`/`run`; pass it to `run_episode`. At `finish`, if the controller produced a schedule (interactive), write it into `runs.config` (`Storage.update_schedule(run_id, schedule)`) and recompute `config_hash`. `experiment.py`: `--interactive` flag → build `InteractiveController`; otherwise `ScheduleController(cfg.schedule)` (scripted, the default that also replays a YAML schedule). Resume already replays the schedule (Phase 2.4).
- [ ] **Step 4:** `uv run pytest` → PASS.
- [ ] **Step 5: Commit** — `feat: runner принимает controller; интерактивный прогон сохраняется как расписание (B=C)`.

### Task 5.3: Optional `interventions` audit table + replay annotation

**Files:** `src/storage/schema.py`, `src/storage/store.py`, `replay.py`; tests `tests/storage/test_storage.py`, `tests/test_replay.py`

- [ ] **Step 1: Failing tests** — `Storage.record_intervention(run_id, round, kind, payload)` round-trips; `replay.py` prints a per-round "interventions" note (manual pairing / join / leave / inject) when present, and is a no-op on old DBs without the table (mirror the judge `OperationalError` guard in `replay.py`).
- [ ] **Step 2:** run → FAIL.
- [ ] **Step 3: Implement.** Additive `interventions (run_id, round_idx, kind, payload TEXT)` table. The runner records each applied directive per round (kind ∈ join/leave/pairing/inject). `replay.py` reads them and annotates the round; guard `sqlite3.OperationalError` for pre-migration DBs.
- [ ] **Step 4:** run → PASS.
- [ ] **Step 5: Commit** — `feat: таблица interventions (аудит вмешательств) + аннотация в replay`.

### Task 5.4: Example schedule config + docs

**Files:** `config/experiment.yaml` (or a new `config/example_schedule.yaml`), `docs/configuration.md`, `docs/architecture.md`, `CLAUDE.md`, `README.md`

- [ ] **Step 1:** Add a commented `schedule:` example (a payoff phase, a roster join/leave, a manual pairing, an inject) to a config and a one-paragraph "Interventions / scheduled config" section to `docs/configuration.md` (sticky vs one-shot semantics; config_hash includes the schedule). `docs/architecture.md`: the controller seam (inbound, symmetric to observer; NoOp/Schedule/Interactive; orchestrator applies roster/pairing/inject, `cfg_for_round` applies patches). `CLAUDE.md`: project-map line for `src/core/controller.py` and the new "modifying the orchestrator" note (controller is the inbound channel). `README.md`: short `--interactive` / scheduled-experiment paragraph.
- [ ] **Step 2:** `uv run pytest` → PASS (example config still loads; schedule validates).
- [ ] **Step 3: Commit** — `docs: пример schedule + справка по вмешательствам и controller-шву`.

---

## Final verification

- [ ] `uv run pytest` — full suite green after every phase.
- [ ] Regression: a no-schedule, no-controller run is byte-identical to pre-feature behaviour (the `NoOpController` + empty `schedule` paths).
- [ ] End-to-end (needs a provider): run a scripted `schedule` (payoff phase + join/leave + manual pair + inject), then `replay.py <run_id>` shows the interventions; resume/extend it and confirm reconstruction.
- [ ] B=C check: an `--interactive` run, then a scripted re-run of its persisted `runs.config` schedule, produce identical final scores.
- [ ] Use superpowers:verification-before-completion before claiming done.
- [ ] Update `project_resume_extend` memory: Axis B/C status, and link this plan.
```
