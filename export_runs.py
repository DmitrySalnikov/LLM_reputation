"""Разбить БД с прогонами на отдельные файлы — по одному прогону на файл.

Каждый файл — самостоятельная SQLite-база с полной схемой исходной БД, но строками
ровно одного прогона. Имя файла берётся из числа в конце имени прогона ('qwen3-FP8 16'
-> 16.db); если числа нет — из run_id. Уже существующие файлы по умолчанию не трогаем,
поэтому функцию можно дёргать после каждого прогона (см. research.py) — она докладывает
только новое.

    uv run python export_runs.py [--db qwen3.db] [--out DIR] [--overwrite]

  --db         исходная БД (по умолчанию qwen3.db)
  --out        папка для файлов (по умолчанию имя БД без расширения, напр. qwen3)
  --overwrite  пересоздавать уже существующие файлы
"""

from __future__ import annotations

import os
import re
import sqlite3
import sys

DB = "qwen3.db"

_NUM_RE = re.compile(r"(\d+)\s*$")


def _out_dir_for(db_path: str) -> str:
    """Папка по умолчанию: имя БД без расширения ('qwen3.db' -> 'qwen3')."""
    return os.path.splitext(os.path.basename(db_path))[0]


def _run_filename(name: str | None, run_id: int) -> str:
    """Имя файла прогона: число в конце имени ('… 16' -> '16'), иначе run_id."""
    m = _NUM_RE.search(name or "")
    return f"{m.group(1) if m else run_id}.db"


def _schema_ddl(conn: sqlite3.Connection) -> list[str]:
    """CREATE-операторы исходной схемы: сначала таблицы, потом индексы.

    Пропускаем sqlite_sequence и авто-индексы (sql IS NULL)."""
    return [sql for (sql,) in conn.execute(
        "SELECT sql FROM sqlite_master WHERE sql IS NOT NULL AND name<>'sqlite_sequence' "
        "ORDER BY (type='index')"
    )]


def _keyed_tables(conn: sqlite3.Connection) -> tuple[list[str], set[str]]:
    """Все пользовательские таблицы и подмножество тех, где есть колонка run_id."""
    tables = [t for (t,) in conn.execute(
        "SELECT name FROM sqlite_master WHERE type='table' AND name<>'sqlite_sequence'"
    )]
    keyed = {t for t in tables
             if any(col[1] == "run_id" for col in conn.execute(f"PRAGMA table_info({t})"))}
    return tables, keyed


def export_run(conn: sqlite3.Connection, run_id: int, out_dir: str,
               *, overwrite: bool = False) -> str | None:
    """Выгрузить один прогон в отдельный файл out_dir/<имя>.db (чистое чтение исходника).

    Returns:
        Путь к созданному файлу, либо None, если файл уже был и overwrite=False.
    """
    row = conn.execute("SELECT name FROM runs WHERE run_id=?", (run_id,)).fetchone()
    if row is None:
        raise ValueError(f"прогон run_id={run_id} не найден в БД")
    os.makedirs(out_dir, exist_ok=True)
    path = os.path.join(out_dir, _run_filename(row[0], run_id))
    if os.path.exists(path):
        if not overwrite:
            return None
        os.remove(path)

    out = sqlite3.connect(path)
    try:
        for sql in _schema_ddl(conn):
            out.execute(sql)
        out.commit()
    finally:
        out.close()

    tables, keyed = _keyed_tables(conn)
    conn.execute("ATTACH DATABASE ? AS out", (path,))
    try:
        for t in tables:
            if t in keyed:
                conn.execute(f"INSERT INTO out.{t} SELECT * FROM main.{t} WHERE run_id=?", (run_id,))
            else:
                conn.execute(f"INSERT INTO out.{t} SELECT * FROM main.{t}")
        conn.commit()
    finally:
        conn.execute("DETACH DATABASE out")
    return path


def export_all(db_path: str, out_dir: str, *, overwrite: bool = False) -> list[str]:
    """Выгрузить каждый прогон БД в свой файл; вернуть пути реально созданных файлов."""
    conn = sqlite3.connect(db_path)
    try:
        run_ids = [r for (r,) in conn.execute("SELECT run_id FROM runs ORDER BY run_id")]
        made = [p for rid in run_ids
                if (p := export_run(conn, rid, out_dir, overwrite=overwrite)) is not None]
    finally:
        conn.close()
    return made


def _has_flag(args: list[str], name: str) -> bool:
    """Есть ли булев флаг в argv."""
    return name in args


def _flag(args: list[str], name: str, default: str) -> str:
    """Значение одиночного флага в argv."""
    return args[args.index(name) + 1] if name in args else default


def main() -> None:
    """Точка входа CLI: разбить БД на файлы по прогонам (идемпотентно)."""
    args = sys.argv[1:]
    db_path = _flag(args, "--db", DB)
    out_dir = _flag(args, "--out", _out_dir_for(db_path))
    overwrite = _has_flag(args, "--overwrite")
    made = export_all(db_path, out_dir, overwrite=overwrite)
    print(f"{db_path} -> {out_dir}/: создано файлов {len(made)}")
    for p in made:
        print(f"  {p}")


if __name__ == "__main__":
    main()
