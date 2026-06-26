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
