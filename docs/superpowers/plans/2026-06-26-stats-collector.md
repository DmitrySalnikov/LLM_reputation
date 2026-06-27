# Stats Collector + Judge Backfill + Visualization — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add three tools — judge-backfill (evaluate stored runs with the LLM judge), a stats collector (emergence rate per design with Wilson 95% CI → JSON/CSV), and a visualization script (bar chart with error bars) — backed by small pure modules in `src/`.

**Architecture:** Pure logic lives in `src/stats/` and `src/storage/records.py` (unit-tested, no I/O side effects); three thin top-level scripts (`judge_runs.py`, `collect_stats.py`, `plot_stats.py`) own all printing/files, mirroring the existing `experiment.py` / `replay.py` style. The judge code path stays single: `render_transcript`/`valid_refs` become pair-aware so backfill can preserve true `pair_idx` while live runs keep enumerating.

**Tech Stack:** Python 3.12, sqlite3, asyncio, existing `judge_episode`, matplotlib (new dep), pytest + pytest-asyncio (auto mode).

---

## File Structure

**New (`src/`, pure logic):**
- `src/stats/__init__.py` — empty package marker.
- `src/stats/wilson.py` — `wilson_interval(k, n, z=1.96)`.
- `src/stats/selection.py` — `RunFilter`, `selected_run_ids(conn, flt)`, `filter_from_argv(argv)`.
- `src/stats/aggregate.py` — `RunRow`, `DesignStat`, `load_judged_runs`, `aggregate_by_design`.
- `src/storage/records.py` — `ReplayRecord`, `reconstruct_records(conn, run_id)`.

**Modified (`src/`):**
- `src/judge/transcript.py` — make `render_transcript`/`valid_refs` pair-aware (backward compatible).
- `src/storage/store.py` — add `conn` property, `has_verdict(run_id)`, `save_verdict(..., run_id=None)`.

**New (top-level scripts):**
- `judge_runs.py` — backfill.
- `collect_stats.py` — aggregate + write artifacts.
- `plot_stats.py` — render PNG.

**New (tests):**
- `tests/stats/conftest.py`, `tests/stats/test_wilson.py`, `test_selection.py`, `test_aggregate.py`, `test_collect.py`, `test_plot.py`.
- `tests/storage/conftest.py`, `tests/storage/test_store_verdict.py`, `tests/storage/test_records.py`.
- `tests/scripts/test_judge_runs.py`.

---

## Task 1: Wilson interval

**Files:**
- Create: `src/stats/__init__.py`
- Create: `src/stats/wilson.py`
- Test: `tests/stats/test_wilson.py`

- [ ] **Step 1: Create the package marker**

Create `src/stats/__init__.py` with a single line:

```python
"""Сбор статистики по оценённым судьёй прогонам (emergence rate + интервалы)."""
```

- [ ] **Step 2: Write the failing test**

Create `tests/stats/test_wilson.py`:

```python
from __future__ import annotations

import pytest

from src.stats.wilson import wilson_interval


def test_zero_successes_lower_bound_is_zero():
    lo, hi = wilson_interval(0, 10)
    assert lo == 0.0
    assert hi == pytest.approx(0.2775, abs=1e-3)


def test_all_successes_upper_bound_is_one():
    lo, hi = wilson_interval(10, 10)
    assert hi == 1.0
    assert lo == pytest.approx(0.7225, abs=1e-3)


def test_midpoint_is_symmetric():
    lo, hi = wilson_interval(5, 10)
    assert lo == pytest.approx(0.2366, abs=1e-3)
    assert hi == pytest.approx(0.7634, abs=1e-3)


def test_interval_narrows_as_n_grows():
    _, hi_small = wilson_interval(5, 10)
    _, hi_big = wilson_interval(50, 100)
    assert (hi_big - 0.5) < (hi_small - 0.5)


def test_rejects_bad_inputs():
    with pytest.raises(ValueError):
        wilson_interval(0, 0)
    with pytest.raises(ValueError):
        wilson_interval(11, 10)
```

- [ ] **Step 3: Run test to verify it fails**

Run: `uv run pytest tests/stats/test_wilson.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'src.stats.wilson'`.

- [ ] **Step 4: Write minimal implementation**

Create `src/stats/wilson.py`:

```python
from __future__ import annotations

import math


def wilson_interval(k: int, n: int, z: float = 1.96) -> tuple[float, float]:
    """Доверительный интервал Вилсона для доли k/n (по умолчанию 95%, z=1.96).

    Корректен у краёв 0/1 и при малом n, в отличие от нормального приближения (Wald).

    Args:
        k: Число «успехов» (прогонов, где институт репутации возник).
        n: Общее число испытаний (оценённых прогонов). Должно быть > 0.
        z: z-квантиль (1.96 ≈ 95%).

    Returns:
        Пара (lo, hi), обрезанная в [0, 1].

    Raises:
        ValueError: n <= 0 или k вне диапазона [0, n].
    """
    if n <= 0:
        raise ValueError("n должно быть > 0")
    if not 0 <= k <= n:
        raise ValueError("k должно быть в диапазоне [0, n]")
    p = k / n
    denom = 1 + z * z / n
    center = (p + z * z / (2 * n)) / denom
    half = (z / denom) * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n))
    return max(0.0, center - half), min(1.0, center + half)
```

- [ ] **Step 5: Run test to verify it passes**

Run: `uv run pytest tests/stats/test_wilson.py -v`
Expected: PASS (5 passed).

- [ ] **Step 6: Commit**

```bash
git add src/stats/__init__.py src/stats/wilson.py tests/stats/test_wilson.py
git commit -m "feat(stats): Wilson score interval for proportions"
```

---

## Task 2: Run selection (RunFilter + filter_from_argv)

**Files:**
- Create: `src/stats/selection.py`
- Create: `tests/stats/conftest.py`
- Test: `tests/stats/test_selection.py`

- [ ] **Step 1: Create the test conftest (shared seeding helpers)**

Create `tests/stats/conftest.py`:

```python
from __future__ import annotations

import sqlite3

import pytest

from src.storage.schema import init_schema


@pytest.fixture
def conn():
    """Чистая in-memory БД с реальной схемой L1."""
    c = sqlite3.connect(":memory:")
    init_schema(c)
    yield c
    c.close()


def add_run(conn, run_id, *, config_hash="d1", name=None, finished=True,
            seed=0, emerged=None):
    """Вставить прогон (+ опционально вердикт судьи) в тестовую БД.

    created_at делается уникальным по run_id, чтобы порядок выборки был детерминирован."""
    conn.execute(
        "INSERT INTO runs(run_id, name, config, config_hash, seed, created_at, finished_at) "
        "VALUES (?,?,?,?,?,?,?)",
        (run_id, name, "{}", config_hash, seed, f"2026-01-01T00:00:{run_id:02d}",
         "2026-01-01T01:00:00" if finished else None),
    )
    if emerged is not None:
        conn.execute(
            "INSERT INTO judge_verdicts(run_id, emerged, explanation, evidence, model, created_at) "
            "VALUES (?,?,?,?,?,?)",
            (run_id, int(emerged), "expl", "[]", "judge-m", "2026-01-01T01:00:00"),
        )
    conn.commit()
```

- [ ] **Step 2: Write the failing test**

Create `tests/stats/test_selection.py`:

```python
from __future__ import annotations

from conftest import add_run

from src.stats.selection import RunFilter, filter_from_argv, selected_run_ids


def test_no_filter_returns_all_finished(conn):
    add_run(conn, 1, finished=True)
    add_run(conn, 2, finished=False)
    assert selected_run_ids(conn, RunFilter()) == [1]


def test_finished_only_can_be_disabled(conn):
    add_run(conn, 1, finished=True)
    add_run(conn, 2, finished=False)
    assert selected_run_ids(conn, RunFilter(finished_only=False)) == [1, 2]


def test_include_designs_is_a_whitelist(conn):
    add_run(conn, 1, config_hash="da")
    add_run(conn, 2, config_hash="db")
    assert selected_run_ids(conn, RunFilter(include_designs=("db",))) == [2]


def test_exclude_designs_and_names(conn):
    add_run(conn, 1, config_hash="da", name="keep")
    add_run(conn, 2, config_hash="db", name="drop")
    flt = RunFilter(exclude_designs=("db",), exclude_names=("drop",))
    assert selected_run_ids(conn, flt) == [1]


def test_filter_from_argv_collects_repeated_flags():
    argv = ["--design", "d1", "--design", "d2", "--exclude-name", "bad", "--out", "x.json"]
    flt = filter_from_argv(argv)
    assert flt.include_designs == ("d1", "d2")
    assert flt.exclude_names == ("bad",)
    assert flt.include_names == ()
```

- [ ] **Step 3: Run test to verify it fails**

Run: `uv run pytest tests/stats/test_selection.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'src.stats.selection'`.

- [ ] **Step 4: Write minimal implementation**

Create `src/stats/selection.py`:

```python
from __future__ import annotations

import sqlite3
from dataclasses import dataclass


@dataclass(frozen=True)
class RunFilter:
    """Фильтр выбора прогонов из реплей-БД (общий для backfill и сборщика).

    include_* (если непусты) работают как белый список; exclude_* — чёрный список поверх;
    finished_only отбрасывает прогоны без finished_at (оборванные/в процессе)."""

    include_designs: tuple[str, ...] = ()
    exclude_designs: tuple[str, ...] = ()
    include_names: tuple[str, ...] = ()
    exclude_names: tuple[str, ...] = ()
    finished_only: bool = True


def selected_run_ids(conn: sqlite3.Connection, flt: RunFilter) -> list[int]:
    """Номера прогонов, прошедших фильтр, в порядке created_at."""
    rows = conn.execute(
        "SELECT run_id, config_hash, name, finished_at FROM runs ORDER BY created_at"
    ).fetchall()
    out: list[int] = []
    for run_id, config_hash, name, finished_at in rows:
        if flt.finished_only and not finished_at:
            continue
        if flt.include_designs and config_hash not in flt.include_designs:
            continue
        if config_hash in flt.exclude_designs:
            continue
        if flt.include_names and name not in flt.include_names:
            continue
        if name in flt.exclude_names:
            continue
        out.append(run_id)
    return out


def _collect_flag(argv: list[str], name: str) -> tuple[str, ...]:
    """Все значения повторяемого флага `--name V` в порядке появления."""
    return tuple(argv[i + 1] for i, a in enumerate(argv)
                 if a == name and i + 1 < len(argv))


def filter_from_argv(argv: list[str]) -> RunFilter:
    """Собрать RunFilter из argv: --design / --exclude-design / --name / --exclude-name."""
    return RunFilter(
        include_designs=_collect_flag(argv, "--design"),
        exclude_designs=_collect_flag(argv, "--exclude-design"),
        include_names=_collect_flag(argv, "--name"),
        exclude_names=_collect_flag(argv, "--exclude-name"),
    )
```

- [ ] **Step 5: Run test to verify it passes**

Run: `uv run pytest tests/stats/test_selection.py -v`
Expected: PASS (5 passed).

- [ ] **Step 6: Commit**

```bash
git add src/stats/selection.py tests/stats/conftest.py tests/stats/test_selection.py
git commit -m "feat(stats): RunFilter + run selection from DB and argv"
```

---

## Task 3: Aggregation by design

**Files:**
- Create: `src/stats/aggregate.py`
- Test: `tests/stats/test_aggregate.py`

- [ ] **Step 1: Write the failing test**

Create `tests/stats/test_aggregate.py`:

```python
from __future__ import annotations

import pytest
from conftest import add_run

from src.stats.aggregate import aggregate_by_design, load_judged_runs


def test_load_skips_runs_without_verdict(conn):
    add_run(conn, 1, config_hash="d1", emerged=True)
    add_run(conn, 2, config_hash="d1", emerged=None)   # нет вердикта
    rows = load_judged_runs(conn, [1, 2])
    assert [r.run_id for r in rows] == [1]
    assert rows[0].emerged is True


def test_aggregate_groups_and_computes_rate(conn):
    add_run(conn, 1, config_hash="d1", name="base", emerged=True)
    add_run(conn, 2, config_hash="d1", name="base", emerged=False)
    add_run(conn, 3, config_hash="d2", name="alt", emerged=True)
    stats = aggregate_by_design(load_judged_runs(conn, [1, 2, 3]))
    assert [s.config_hash for s in stats] == ["d1", "d2"]
    d1 = stats[0]
    assert d1.n == 2 and d1.n_emerged == 1 and d1.rate == pytest.approx(0.5)
    assert d1.name == "base"
    assert d1.run_ids == (1, 2)
    assert d1.ci_lo == pytest.approx(0.2366, abs=1e-3)
    assert d1.ci_hi == pytest.approx(0.7634, abs=1e-3)


def test_aggregate_empty_input():
    assert aggregate_by_design([]) == []
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/stats/test_aggregate.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'src.stats.aggregate'`.

- [ ] **Step 3: Write minimal implementation**

Create `src/stats/aggregate.py`:

```python
from __future__ import annotations

import sqlite3
from dataclasses import dataclass

from src.stats.wilson import wilson_interval


@dataclass(frozen=True)
class RunRow:
    """Один оценённый судьёй прогон: исход emerged + к какому дизайну относится."""

    run_id: int
    config_hash: str
    name: str | None
    emerged: bool


@dataclass(frozen=True)
class DesignStat:
    """Статистика по одному дизайну (config_hash): доля + интервал Вилсона."""

    config_hash: str
    name: str | None
    n: int
    n_emerged: int
    rate: float
    ci_lo: float
    ci_hi: float
    run_ids: tuple[int, ...]


def load_judged_runs(conn: sqlite3.Connection, run_ids: list[int]) -> list[RunRow]:
    """Прочитать (config_hash, name, emerged) для прогонов, у которых есть вердикт.

    Прогоны без вердикта молча выпадают (INNER JOIN). Порядок результата — как в run_ids."""
    if not run_ids:
        return []
    found = {
        rid: (ch, name, bool(em))
        for rid, ch, name, em in conn.execute(
            "SELECT r.run_id, r.config_hash, r.name, v.emerged "
            "FROM runs r JOIN judge_verdicts v ON v.run_id = r.run_id "
            f"WHERE r.run_id IN ({','.join('?' * len(run_ids))})",
            run_ids,
        )
    }
    return [RunRow(rid, *found[rid]) for rid in run_ids if rid in found]


def aggregate_by_design(rows: list[RunRow]) -> list[DesignStat]:
    """Сгруппировать прогоны по config_hash; доля + 95% интервал Вилсона на группу.

    Порядок групп — по первому появлению config_hash в rows. name группы — name первого
    прогона в ней (подпись для графика)."""
    order: list[str] = []
    groups: dict[str, list[RunRow]] = {}
    for r in rows:
        if r.config_hash not in groups:
            groups[r.config_hash] = []
            order.append(r.config_hash)
        groups[r.config_hash].append(r)
    out: list[DesignStat] = []
    for ch in order:
        grp = groups[ch]
        n = len(grp)
        k = sum(1 for r in grp if r.emerged)
        lo, hi = wilson_interval(k, n)
        out.append(DesignStat(
            config_hash=ch, name=grp[0].name, n=n, n_emerged=k,
            rate=k / n, ci_lo=lo, ci_hi=hi,
            run_ids=tuple(r.run_id for r in grp),
        ))
    return out
```

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest tests/stats/test_aggregate.py -v`
Expected: PASS (3 passed).

- [ ] **Step 5: Commit**

```bash
git add src/stats/aggregate.py tests/stats/test_aggregate.py
git commit -m "feat(stats): aggregate emergence rate + Wilson CI by design"
```

---

## Task 4: Pair-aware judge transcript

**Files:**
- Modify: `src/judge/transcript.py`
- Test: `tests/judge/test_transcript.py`

- [ ] **Step 1: Write the failing test (append to existing file)**

Append to `tests/judge/test_transcript.py`:

```python
from types import SimpleNamespace


def _rec_with_pair(round, pair):
    """Запись с явным pair_idx (как восстановленная из БД ReplayRecord)."""
    return SimpleNamespace(
        round=round, pair=pair, a_id="A1", b_id="A2",
        transcript=[{"speaker": "A1", "text": "hi there", "ready": False}],
    )


def test_explicit_pair_overrides_enumeration():
    out = render_transcript([_rec_with_pair(0, 5)])   # истинный pair_idx=5, не позиция 0
    assert "[r0.p5.t0] A1: hi there" in out
    assert "Pairing r0.p5:" in out


def test_valid_refs_uses_explicit_pair():
    refs = valid_refs([_rec_with_pair(1, 3)])
    assert refs == {(1, 3, 0)}
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/judge/test_transcript.py -k "explicit_pair or uses_explicit_pair" -v`
Expected: FAIL — current code enumerates, so it renders `r0.p0.t0` not `r0.p5.t0`.

- [ ] **Step 3: Edit the implementation**

In `src/judge/transcript.py`, add this helper after the imports (above `_by_round`):

```python
def _pair_index(rec, fallback: int) -> int:
    """pair_idx записи: явный rec.pair (если задан) — иначе позиция в раунде.

    Живой судья передаёт PairingRecord без поля pair → нумерация позицией (как раньше);
    backfill передаёт ReplayRecord с истинным pair_idx из БД → ссылки совпадают с messages."""
    p = getattr(rec, "pair", None)
    return fallback if p is None else p
```

Replace the body of `render_transcript`'s round loop. Change:

```python
    for rnd in sorted(grouped):
        lines.append(f"ROUND {rnd}")
        for p, rec in enumerate(grouped[rnd]):
            lines.append(f"  Pairing r{rnd}.p{p}: {rec.a_id} vs {rec.b_id}")
            if not rec.transcript:
                lines.append("    (no messages exchanged)")
            for t, msg in enumerate(rec.transcript):
                lines.append(f"    [r{rnd}.p{p}.t{t}] {msg['speaker']}: {msg['text']}")
```

to:

```python
    for rnd in sorted(grouped):
        lines.append(f"ROUND {rnd}")
        for i, rec in enumerate(grouped[rnd]):
            p = _pair_index(rec, i)
            lines.append(f"  Pairing r{rnd}.p{p}: {rec.a_id} vs {rec.b_id}")
            if not rec.transcript:
                lines.append("    (no messages exchanged)")
            for t, msg in enumerate(rec.transcript):
                lines.append(f"    [r{rnd}.p{p}.t{t}] {msg['speaker']}: {msg['text']}")
```

Replace the body of `valid_refs`'s loop. Change:

```python
    for rnd, recs in grouped.items():
        for p, rec in enumerate(recs):
            for t in range(len(rec.transcript)):
                refs.add((rnd, p, t))
```

to:

```python
    for rnd, recs in grouped.items():
        for i, rec in enumerate(recs):
            p = _pair_index(rec, i)
            for t in range(len(rec.transcript)):
                refs.add((rnd, p, t))
```

- [ ] **Step 4: Run the transcript tests to verify all pass**

Run: `uv run pytest tests/judge/test_transcript.py -v`
Expected: PASS (all — new pair-aware tests AND the existing enumeration tests, proving backward compatibility).

- [ ] **Step 5: Commit**

```bash
git add src/judge/transcript.py tests/judge/test_transcript.py
git commit -m "feat(judge): pair-aware transcript (explicit pair_idx overrides enumeration)"
```

---

## Task 5: Storage — conn property, has_verdict, save_verdict(run_id)

**Files:**
- Modify: `src/storage/store.py`
- Create: `tests/storage/conftest.py`
- Test: `tests/storage/test_store_verdict.py`

- [ ] **Step 1: Create the storage test conftest**

Create `tests/storage/conftest.py`:

```python
from __future__ import annotations

import sqlite3

import pytest

from src.storage.schema import init_schema


@pytest.fixture
def conn():
    """Чистая in-memory БД с реальной схемой L1 (FK выключены — удобно для точечных вставок)."""
    c = sqlite3.connect(":memory:")
    init_schema(c)
    yield c
    c.close()


def insert_run_row(conn, run_id, *, config_hash="d1", name=None):
    """Минимальная строка runs (для тестов, где нужен родитель под judge_verdicts)."""
    conn.execute(
        "INSERT INTO runs(run_id, name, config, config_hash, seed, created_at, finished_at) "
        "VALUES (?,?,?,?,?,?,?)",
        (run_id, name, "{}", config_hash, 0, "2026-01-01T00:00:00", "2026-01-01T01:00:00"),
    )
    conn.commit()
```

- [ ] **Step 2: Write the failing test**

Create `tests/storage/test_store_verdict.py`:

```python
from __future__ import annotations

from src.judge import JudgeVerdict, MessageRef
from src.storage import Storage


def _verdict():
    return JudgeVerdict(emerged=True, explanation="gossip",
                        evidence=[MessageRef(round=0, pair=0, turn=0)])


def test_has_verdict_false_then_true_after_save():
    st = Storage(":memory:")
    try:
        st.conn.execute(
            "INSERT INTO runs(run_id, name, config, config_hash, seed, created_at, finished_at) "
            "VALUES (1, NULL, '{}', 'd1', 0, '2026-01-01T00:00:00', '2026-01-01T01:00:00')"
        )
        st.conn.commit()
        assert st.has_verdict(1) is False
        st.save_verdict(_verdict(), model="judge-m", run_id=1)
        assert st.has_verdict(1) is True
        row = st.conn.execute(
            "SELECT emerged, model FROM judge_verdicts WHERE run_id=1"
        ).fetchone()
        assert row == (1, "judge-m")
    finally:
        st.close()
```

- [ ] **Step 3: Run test to verify it fails**

Run: `uv run pytest tests/storage/test_store_verdict.py -v`
Expected: FAIL with `AttributeError: 'Storage' object has no attribute 'conn'`.

- [ ] **Step 4: Edit the implementation**

In `src/storage/store.py`, add a `conn` property just after `__init__` (after the `self._run_id` line):

```python
    @property
    def conn(self) -> sqlite3.Connection:
        """Доступ к соединению на чтение (реконструкция записей, выборка прогонов)."""
        return self._conn

    def has_verdict(self, run_id: int) -> bool:
        """True, если у прогона уже есть вердикт судьи."""
        row = self._conn.execute(
            "SELECT 1 FROM judge_verdicts WHERE run_id=?", (run_id,)
        ).fetchone()
        return row is not None
```

Then change `save_verdict` to target an explicit run. Replace its signature and the run_id it inserts. Change:

```python
    def save_verdict(self, verdict: JudgeVerdict, *, model: str) -> None:
        """Шаг J: сохранить вердикт LLM-судьи (одна строка на run)."""
        with self._conn:
            self._conn.execute(
                """INSERT INTO judge_verdicts(run_id, emerged, explanation, evidence, model, created_at)
                   VALUES (?,?,?,?,?,?)""",
                (
                    self._run_id,
```

to:

```python
    def save_verdict(self, verdict: JudgeVerdict, *, model: str, run_id: int | None = None) -> None:
        """Шаг J: сохранить вердикт LLM-судьи (одна строка на run).

        run_id=None — текущий прогон (живой путь); явный run_id — для backfill, где одна
        Storage оценивает много сохранённых прогонов."""
        rid = run_id if run_id is not None else self._run_id
        with self._conn:
            self._conn.execute(
                """INSERT INTO judge_verdicts(run_id, emerged, explanation, evidence, model, created_at)
                   VALUES (?,?,?,?,?,?)""",
                (
                    rid,
```

(The remaining lines of the INSERT tuple — `int(verdict.emerged)`, … — stay unchanged.)

- [ ] **Step 5: Run test to verify it passes**

Run: `uv run pytest tests/storage/test_store_verdict.py -v`
Expected: PASS (1 passed).

- [ ] **Step 6: Run the existing storage/replay tests to confirm no regression**

Run: `uv run pytest tests/test_replay.py -v`
Expected: PASS (the existing `save_verdict(..., model=...)` callers still work — `run_id` is optional).

- [ ] **Step 7: Commit**

```bash
git add src/storage/store.py tests/storage/conftest.py tests/storage/test_store_verdict.py
git commit -m "feat(storage): conn property, has_verdict, save_verdict(run_id=...)"
```

---

## Task 6: Reconstruct records from the DB

**Files:**
- Create: `src/storage/records.py`
- Test: `tests/storage/test_records.py`

- [ ] **Step 1: Write the failing test**

Create `tests/storage/test_records.py`:

```python
from __future__ import annotations

from conftest import insert_run_row

from src.storage.records import reconstruct_records


def _add_pairing(conn, run_id, round_idx, pair_idx, *, finished=1, a="A1", b="A2"):
    conn.execute(
        "INSERT INTO pairings(run_id, round_idx, pair_idx, a_id, b_id, finished, "
        "a_number, b_number, a_outcome, a_payoff, b_payoff) "
        "VALUES (?,?,?,?,?,?,?,?,?,?,?)",
        (run_id, round_idx, pair_idx, a, b, finished,
         7 if finished else None, 7 if finished else None,
         "CC" if finished else None, 3.0 if finished else None, 3.0 if finished else None),
    )


def _add_msg(conn, run_id, round_idx, pair_idx, turn_idx, speaker, text, ready=0):
    conn.execute(
        "INSERT INTO messages(run_id, round_idx, pair_idx, turn_idx, speaker, text, ready) "
        "VALUES (?,?,?,?,?,?,?)",
        (run_id, round_idx, pair_idx, turn_idx, speaker, text, ready),
    )


def test_reconstructs_finished_pairings_with_true_pair_idx(conn):
    insert_run_row(conn, 1)
    _add_pairing(conn, 1, 0, 0)
    _add_pairing(conn, 1, 0, 1)
    _add_msg(conn, 1, 0, 1, 0, "A1", "hi")
    _add_msg(conn, 1, 0, 1, 1, "A2", "pick 7", ready=1)
    conn.commit()

    recs = reconstruct_records(conn, 1)
    assert [(r.round, r.pair) for r in recs] == [(0, 0), (0, 1)]
    second = recs[1]
    assert second.pair == 1                        # истинный pair_idx сохранён
    assert [m["text"] for m in second.transcript] == ["hi", "pick 7"]
    assert second.transcript[1]["ready"] is True


def test_skips_aborted_pairings(conn):
    insert_run_row(conn, 1)
    _add_pairing(conn, 1, 0, 0, finished=0)        # сорвана — без результата
    conn.commit()
    assert reconstruct_records(conn, 1) == []
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/storage/test_records.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'src.storage.records'`.

- [ ] **Step 3: Write minimal implementation**

Create `src/storage/records.py`:

```python
from __future__ import annotations

import sqlite3
from dataclasses import dataclass, field


@dataclass(frozen=True)
class ReplayRecord:
    """Минимальная запись завершённой пары, восстановленная из БД для LLM-судьи.

    Несёт истинный pair_idx (поле pair), чтобы evidence-ссылки судьи совпадали с поиском
    по таблице messages в replay.py. transcript — публичный cheap-talk пары."""

    round: int
    pair: int
    a_id: str
    b_id: str
    transcript: list[dict] = field(default_factory=list)
    finished: bool = True


def reconstruct_records(conn: sqlite3.Connection, run_id: int) -> list[ReplayRecord]:
    """Восстановить завершённые пары прогона (round, истинный pair_idx, сообщения).

    Только finished=1 (судья видит лишь доигранные пары). Порядок — round_idx, pair_idx;
    сообщения — по turn_idx. Сорванные пары пропускаются."""
    pairings = conn.execute(
        "SELECT round_idx, pair_idx, a_id, b_id FROM pairings "
        "WHERE run_id=? AND finished=1 ORDER BY round_idx, pair_idx",
        (run_id,),
    ).fetchall()
    out: list[ReplayRecord] = []
    for round_idx, pair_idx, a_id, b_id in pairings:
        transcript = [
            {"speaker": s, "text": t, "ready": bool(r)}
            for s, t, r in conn.execute(
                "SELECT speaker, text, ready FROM messages "
                "WHERE run_id=? AND round_idx=? AND pair_idx=? ORDER BY turn_idx",
                (run_id, round_idx, pair_idx),
            )
        ]
        out.append(ReplayRecord(round=round_idx, pair=pair_idx,
                                a_id=a_id, b_id=b_id, transcript=transcript))
    return out
```

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest tests/storage/test_records.py -v`
Expected: PASS (2 passed).

- [ ] **Step 5: Commit**

```bash
git add src/storage/records.py tests/storage/test_records.py
git commit -m "feat(storage): reconstruct finished-pairing records from the DB"
```

---

## Task 7: judge_runs.py (backfill script)

**Files:**
- Create: `judge_runs.py`
- Test: `tests/scripts/test_judge_runs.py`

Backfill uses a single uniform judge config (`JUDGE_DEFAULT`) for all runs — none of the stored runs carry a `judge` block, and one judge across a study keeps verdicts comparable. The judge model is recorded per verdict in `judge_verdicts.model`.

- [ ] **Step 1: Write the failing test**

Create `tests/scripts/test_judge_runs.py`:

```python
from __future__ import annotations

import asyncio
import json

import pytest

from src.judge import judge as judge_mod
from src.storage import Storage

import judge_runs


class ScriptedProvider:
    """Минимальный двойник провайдера: отдаёт заранее заданный текст вердикта."""

    def __init__(self, replies):
        self._queue = list(replies)

    async def complete(self, *, system, messages, temperature, max_tokens):
        from src.providers.base import Completion
        text = self._queue.pop(0)
        return Completion(text=text, prompt_tokens=1, completion_tokens=1,
                          raw={}, request={}, attempts=())

    async def aclose(self):
        pass


def _verdict_json():
    return json.dumps({"emerged": True, "explanation": "gossip",
                       "evidence": ["r0.p0.t0"]})


def _seed_run(st, run_id=1):
    c = st.conn
    c.execute("INSERT INTO runs(run_id,name,config,config_hash,seed,created_at,finished_at) "
              "VALUES (?,?,?,?,?,?,?)",
              (run_id, None, "{}", "d1", 0, "2026-01-01T00:00:00", "2026-01-01T01:00:00"))
    for aid in ("A1", "A2"):
        c.execute("INSERT INTO agents(run_id,agent_id,system_prompt,provider) VALUES (?,?,?,?)",
                  (run_id, aid, "sys", "{}"))
    c.execute("INSERT INTO rounds(run_id,round_idx) VALUES (?,?)", (run_id, 0))
    c.execute("INSERT INTO pairings(run_id,round_idx,pair_idx,a_id,b_id,finished,"
              "a_number,b_number,a_outcome,a_payoff,b_payoff) VALUES (?,?,?,?,?,?,?,?,?,?,?)",
              (run_id, 0, 0, "A1", "A2", 1, 7, 7, "CC", 3.0, 3.0))
    c.executemany("INSERT INTO messages(run_id,round_idx,pair_idx,turn_idx,speaker,text,ready) "
                  "VALUES (?,?,?,?,?,?,?)",
                  [(run_id, 0, 0, 0, "A1", "hi", 0), (run_id, 0, 0, 1, "A2", "pick 7", 1)])
    c.commit()


def test_judge_run_writes_verdict_then_skips(monkeypatch):
    monkeypatch.setattr(judge_mod, "make_provider", lambda cfg: ScriptedProvider([_verdict_json()]))
    st = Storage(":memory:")
    try:
        _seed_run(st, 1)
        status = asyncio.run(judge_runs.judge_run(st, 1, judge_runs.JUDGE_DEFAULT, force=False))
        assert status == "judged"
        assert st.has_verdict(1) is True
        # второй прогон без --force — пропуск (провайдер не дёргается)
        status2 = asyncio.run(judge_runs.judge_run(st, 1, judge_runs.JUDGE_DEFAULT, force=False))
        assert status2 == "skipped"
    finally:
        st.close()


def test_judge_run_force_rejudges(monkeypatch):
    monkeypatch.setattr(judge_mod, "make_provider",
                        lambda cfg: ScriptedProvider([_verdict_json(), _verdict_json()]))
    st = Storage(":memory:")
    try:
        _seed_run(st, 1)
        asyncio.run(judge_runs.judge_run(st, 1, judge_runs.JUDGE_DEFAULT, force=False))
        status = asyncio.run(judge_runs.judge_run(st, 1, judge_runs.JUDGE_DEFAULT, force=True))
        assert status == "judged"
    finally:
        st.close()
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/scripts/test_judge_runs.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'judge_runs'`.

- [ ] **Step 3: Write minimal implementation**

Create `judge_runs.py` (repo root):

```python
"""Backfill: оценить сохранённые прогоны LLM-судьёй и записать вердикты.

Судья (judge_episode) сейчас работает только вживую в конце эпизода. Этот скрипт оценивает
уже лежащие в БД прогоны: восстанавливает публичный cheap-talk, зовёт судью, сохраняет
вердикт. Уже оценённые прогоны пропускаются (если не задан --force).

Один общий судья на все прогоны (JUDGE_DEFAULT) — для сопоставимости вердиктов в рамках
исследования. Модель судьи пишется в judge_verdicts.model.

    uv run python judge_runs.py [--design HASH ...] [--exclude-design HASH ...] \
                                [--name LABEL ...] [--exclude-name LABEL ...] [--force]
"""

from __future__ import annotations

import asyncio
import sys

from dotenv import load_dotenv

from src.core.config import JudgeCfg, ProviderCfg
from src.judge import JudgeError, judge_episode
from src.providers.base import ProviderError
from src.stats.selection import filter_from_argv, selected_run_ids
from src.storage import Storage
from src.storage.records import reconstruct_records

load_dotenv()                       # подхватить TOGETHER_API_KEY из .env

DB = "experiment.db"

# Судья по умолчанию (как закомментированный JUDGE в experiment.py). Отредактируй под свою
# модель/эндпойнт перед запуском.
JUDGE_DEFAULT = JudgeCfg(provider=ProviderCfg(
    base_url="https://api.together.xyz/v1",
    api_key_env="TOGETHER_API_KEY",
    model="Qwen/Qwen2.5-72B-Instruct-Turbo",
))


async def judge_run(st: Storage, run_id: int, judge_cfg: JudgeCfg, *, force: bool) -> str:
    """Оценить один сохранённый прогон. Возвращает статус-строку.

    Статусы: skipped (уже есть вердикт), no-records (нет завершённых пар),
    failed (судья не справился), judged (вердикт записан)."""
    if not force and st.has_verdict(run_id):
        return "skipped"
    records = reconstruct_records(st.conn, run_id)
    if not records:
        return "no-records"
    try:
        verdict = await judge_episode(judge_cfg, records)
    except (JudgeError, ProviderError) as e:
        print(f"  прогон {run_id}: судья не справился: {e}")
        return "failed"
    st.save_verdict(verdict, model=judge_cfg.provider.model, run_id=run_id)
    return "judged"


async def backfill(db_path: str, argv: list[str], judge_cfg: JudgeCfg) -> dict[str, int]:
    """Выбрать прогоны по фильтру из argv и оценить каждый; вернуть счётчик статусов."""
    force = "--force" in argv
    flt = filter_from_argv(argv)
    st = Storage(db_path)
    counts: dict[str, int] = {}
    try:
        run_ids = selected_run_ids(st.conn, flt)
        print(f"Под фильтр попало прогонов: {len(run_ids)}")
        for rid in run_ids:
            status = await judge_run(st, rid, judge_cfg, force=force)
            counts[status] = counts.get(status, 0) + 1
            print(f"  прогон {rid}: {status}")
    finally:
        st.close()
    return counts


def main() -> None:
    counts = asyncio.run(backfill(DB, sys.argv[1:], JUDGE_DEFAULT))
    print(f"\nИтог: {counts}")


if __name__ == "__main__":
    main()
```

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest tests/scripts/test_judge_runs.py -v`
Expected: PASS (2 passed).

- [ ] **Step 5: Commit**

```bash
git add judge_runs.py tests/scripts/test_judge_runs.py
git commit -m "feat: judge_runs.py — backfill judge verdicts for stored runs"
```

---

## Task 8: collect_stats.py (aggregate + write artifacts)

**Files:**
- Create: `collect_stats.py`
- Test: `tests/stats/test_collect.py`

- [ ] **Step 1: Write the failing test**

Create `tests/stats/test_collect.py`:

```python
from __future__ import annotations

import json
import sqlite3

import pytest
from conftest import add_run

import collect_stats
from src.stats.aggregate import DesignStat
from src.stats.selection import RunFilter
from src.storage.schema import init_schema


def test_collect_groups_by_design_from_file_db(tmp_path):
    db = tmp_path / "t.db"
    c = sqlite3.connect(db)
    init_schema(c)
    add_run(c, 1, config_hash="d1", emerged=True)
    add_run(c, 2, config_hash="d1", emerged=False)
    add_run(c, 3, config_hash="d2", emerged=True)
    c.close()

    stats = collect_stats.collect(str(db), RunFilter())
    by_hash = {s.config_hash: s for s in stats}
    assert by_hash["d1"].n == 2 and by_hash["d1"].n_emerged == 1
    assert by_hash["d2"].rate == pytest.approx(1.0)


def test_stats_to_json_shape():
    stats = [DesignStat("d1", "base", 2, 1, 0.5, 0.24, 0.76, (1, 2))]
    obj = collect_stats.stats_to_json(stats, RunFilter(include_designs=("d1",)))
    assert obj["filters"]["include_designs"] == ["d1"]
    d = obj["designs"][0]
    assert d["config_hash"] == "d1" and d["run_ids"] == [1, 2]


def test_write_json_and_csv_roundtrip(tmp_path):
    stats = [DesignStat("d1", "base", 2, 1, 0.5, 0.24, 0.76, (1, 2))]
    jp = tmp_path / "s.json"
    cp = tmp_path / "s.csv"
    collect_stats.write_json(str(jp), collect_stats.stats_to_json(stats, RunFilter()))
    collect_stats.write_csv(str(cp), stats)
    assert json.loads(jp.read_text())["designs"][0]["n"] == 2
    csv_text = cp.read_text()
    assert "config_hash,name,n,n_emerged,rate,ci_lo,ci_hi" in csv_text
    assert "d1,base,2,1,0.5000,0.2400,0.7600" in csv_text
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/stats/test_collect.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'collect_stats'`.

- [ ] **Step 3: Write minimal implementation**

Create `collect_stats.py` (repo root):

```python
"""Собрать статистику emergence rate по дизайнам из оценённых судьёй прогонов.

Берёт прогоны с вердиктом судьи, группирует по config_hash, считает долю «институт
репутации возник» с 95% интервалом Вилсона, печатает таблицу и пишет stats.json (+ stats.csv).

    uv run python collect_stats.py [--design HASH ...] [--exclude-design HASH ...] \
                                   [--name LABEL ...] [--exclude-name LABEL ...] \
                                   [--out stats.json] [--csv stats.csv]
"""

from __future__ import annotations

import csv
import json
import sqlite3
import sys
from dataclasses import asdict

from src.stats.aggregate import DesignStat, aggregate_by_design, load_judged_runs
from src.stats.selection import RunFilter, filter_from_argv, selected_run_ids

DB = "experiment.db"


def collect(db_path: str, flt: RunFilter) -> list[DesignStat]:
    """Прочитать БД и посчитать статистику по дизайнам (чистое чтение)."""
    conn = sqlite3.connect(db_path)
    try:
        run_ids = selected_run_ids(conn, flt)
        rows = load_judged_runs(conn, run_ids)
        return aggregate_by_design(rows)
    finally:
        conn.close()


def stats_to_json(stats: list[DesignStat], flt: RunFilter) -> dict:
    """Сериализуемый объект артефакта: применённые фильтры + список дизайнов.

    Кортежи RunFilter превращаются в списки, чтобы JSON был привычной формы."""
    filters = {k: (list(v) if isinstance(v, tuple) else v)
               for k, v in asdict(flt).items()}
    return {
        "filters": filters,
        "designs": [{**asdict(s), "run_ids": list(s.run_ids)} for s in stats],
    }


def write_json(path: str, obj: dict) -> None:
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)


def write_csv(path: str, stats: list[DesignStat]) -> None:
    with open(path, "w", encoding="utf-8", newline="") as f:
        w = csv.writer(f)
        w.writerow(["config_hash", "name", "n", "n_emerged", "rate", "ci_lo", "ci_hi"])
        for s in stats:
            w.writerow([s.config_hash, s.name or "", s.n, s.n_emerged,
                        f"{s.rate:.4f}", f"{s.ci_lo:.4f}", f"{s.ci_hi:.4f}"])


def print_table(stats: list[DesignStat]) -> None:
    """Человекочитаемая таблица в консоль."""
    if not stats:
        print("Нет оценённых прогонов под фильтр — статистики нет.")
        return
    print(f"{'design':16} {'name':12} {'n':>4} {'emrg':>5} {'rate':>6}  95% CI")
    for s in stats:
        print(f"{s.config_hash[:16]:16} {(s.name or '—'):12} {s.n:>4} {s.n_emerged:>5} "
              f"{s.rate:>6.2f}  [{s.ci_lo:.2f}, {s.ci_hi:.2f}]")


def _flag(args: list[str], name: str, default: str) -> str:
    return args[args.index(name) + 1] if name in args else default


def main() -> None:
    args = sys.argv[1:]
    flt = filter_from_argv(args)
    out = _flag(args, "--out", "stats.json")
    csv_path = _flag(args, "--csv", "stats.csv")
    stats = collect(DB, flt)
    print_table(stats)
    if not stats:
        sys.exit(1)
    write_json(out, stats_to_json(stats, flt))
    write_csv(csv_path, stats)
    print(f"\nЗаписано: {out}, {csv_path}")


if __name__ == "__main__":
    main()
```

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest tests/stats/test_collect.py -v`
Expected: PASS (3 passed).

- [ ] **Step 5: Commit**

```bash
git add collect_stats.py tests/stats/test_collect.py
git commit -m "feat: collect_stats.py — aggregate verdicts to stats.json + stats.csv"
```

---

## Task 9: plot_stats.py (visualization)

**Files:**
- Modify: `pyproject.toml` (add matplotlib via uv)
- Create: `plot_stats.py`
- Test: `tests/stats/test_plot.py`

- [ ] **Step 1: Add the matplotlib dependency**

Run: `uv add matplotlib`
Expected: `pyproject.toml` gains `matplotlib`; `uv.lock` updates; numpy is pulled transitively.

- [ ] **Step 2: Write the failing test**

Create `tests/stats/test_plot.py`:

```python
from __future__ import annotations

import pytest

import plot_stats


def _designs():
    return [
        {"config_hash": "d1", "name": "base", "n": 10, "n_emerged": 7,
         "rate": 0.7, "ci_lo": 0.4, "ci_hi": 0.9},
        {"config_hash": "d2", "name": None, "n": 5, "n_emerged": 1,
         "rate": 0.2, "ci_lo": 0.04, "ci_hi": 0.62},
    ]


def test_render_writes_nonempty_png(tmp_path):
    out = tmp_path / "p.png"
    plot_stats.render(_designs(), str(out))
    assert out.exists() and out.stat().st_size > 0


def test_render_rejects_empty(tmp_path):
    with pytest.raises(ValueError):
        plot_stats.render([], str(tmp_path / "p.png"))


def test_load_designs_reads_artifact(tmp_path):
    import json
    p = tmp_path / "s.json"
    p.write_text(json.dumps({"filters": {}, "designs": _designs()}))
    assert len(plot_stats.load_designs(str(p))) == 2
```

- [ ] **Step 3: Run test to verify it fails**

Run: `uv run pytest tests/stats/test_plot.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'plot_stats'`.

- [ ] **Step 4: Write minimal implementation**

Create `plot_stats.py` (repo root):

```python
"""Визуализация emergence rate по дизайнам со статистическими ошибками (интервал Вилсона).

Читает stats.json (артефакт collect_stats.py) и строит столбчатую диаграмму: высота столбца —
доля прогонов с возникновением института репутации, асимметричные усы — 95% интервал Вилсона.
Сохраняет PNG.

    uv run python plot_stats.py [--in stats.json] [--out stats.png] [--title TEXT]
"""

from __future__ import annotations

import json
import sys

import matplotlib

matplotlib.use("Agg")            # рендер в файл без дисплея
import matplotlib.pyplot as plt  # noqa: E402 (после use("Agg"))


def load_designs(path: str) -> list[dict]:
    """Прочитать список дизайнов из stats.json."""
    with open(path, encoding="utf-8") as f:
        return json.load(f).get("designs", [])


def render(designs: list[dict], out_path: str,
           title: str = "Emergence of reputation by design") -> None:
    """Построить bar chart с асимметричными усами интервала Вилсона и сохранить PNG.

    Raises:
        ValueError: пустой список дизайнов — рисовать нечего.
    """
    if not designs:
        raise ValueError("в stats.json нет дизайнов — нечего рисовать")
    labels = [d["name"] or d["config_hash"][:8] for d in designs]
    rates = [d["rate"] for d in designs]
    lo_err = [d["rate"] - d["ci_lo"] for d in designs]
    hi_err = [d["ci_hi"] - d["rate"] for d in designs]
    x = range(len(designs))

    fig, ax = plt.subplots(figsize=(max(6.0, len(designs) * 1.2), 5.0))
    ax.bar(x, rates, yerr=[lo_err, hi_err], capsize=6, color="#4878a8")
    ax.set_xticks(list(x))
    ax.set_xticklabels(labels, rotation=30, ha="right")
    ax.set_ylim(0, 1)
    ax.set_ylabel("Доля прогонов с возникновением института")
    ax.set_title(title)
    for i, d in enumerate(designs):                 # n под каждым столбцом
        ax.text(i, 0.02, f"n={d['n']}", ha="center", va="bottom", fontsize=8)
    fig.tight_layout()
    fig.savefig(out_path, dpi=120)
    plt.close(fig)


def _flag(args: list[str], name: str, default: str) -> str:
    return args[args.index(name) + 1] if name in args else default


def main() -> None:
    args = sys.argv[1:]
    in_path = _flag(args, "--in", "stats.json")
    out_path = _flag(args, "--out", "stats.png")
    title = _flag(args, "--title", "Emergence of reputation by design")
    render(load_designs(in_path), out_path, title)
    print(f"Записано: {out_path}")


if __name__ == "__main__":
    main()
```

- [ ] **Step 5: Run test to verify it passes**

Run: `uv run pytest tests/stats/test_plot.py -v`
Expected: PASS (3 passed).

- [ ] **Step 6: Commit**

```bash
git add pyproject.toml uv.lock plot_stats.py tests/stats/test_plot.py
git commit -m "feat: plot_stats.py — bar chart with Wilson CI error bars"
```

---

## Task 10: Full suite + docs note

**Files:**
- Modify: `CLAUDE.md` (Commands section — add the three new scripts)

- [ ] **Step 1: Run the entire test suite**

Run: `uv run pytest`
Expected: PASS (all tests green, including the pre-existing suite — no regressions).

- [ ] **Step 2: Add the new commands to CLAUDE.md**

In `CLAUDE.md`, under the `## Commands` section, after the existing `uv run python examples/...` lines, add:

```bash
uv run python judge_runs.py                            # backfill judge verdicts for stored runs
uv run python collect_stats.py                         # aggregate verdicts -> stats.json + stats.csv
uv run python plot_stats.py                            # stats.json -> stats.png (Wilson CI bars)
```

- [ ] **Step 3: Commit**

```bash
git add CLAUDE.md
git commit -m "docs: document judge_runs / collect_stats / plot_stats commands"
```

---

## Self-Review Notes

- **Spec coverage:** judge-backfill (Tasks 5–7), stats collector with Wilson CI + JSON/CSV (Tasks 1–3, 8), visualization with error bars (Task 9), shared `RunFilter` filtering (Task 2), pair-aware evidence refs (Task 4). The spec's "prefer per-run judge block" was simplified to one uniform judge config — see the note in Task 7; the spec is updated to match.
- **Type consistency:** `DesignStat`/`RunRow`/`RunFilter`/`ReplayRecord` field names match across `aggregate.py`, `selection.py`, `records.py`, and both scripts. `wilson_interval(k, n)` signature is consistent everywhere. `save_verdict(..., run_id=None)` keyword matches both the live caller (unchanged) and backfill.
- **No placeholders:** every code/test step contains complete content.
```
