"""Backfill: оценить сохранённые прогоны LLM-судьёй и записать вердикты.

Судья (judge_episode) сейчас работает только вживую в конце эпизода. Этот скрипт оценивает
уже лежащие в БД прогоны: восстанавливает публичный cheap-talk, зовёт судью, сохраняет
вердикт. Уже оценённые прогоны пропускаются (если не задан --force).

Один общий судья на все прогоны (JUDGE_DEFAULT) — для сопоставимости вердиктов в рамках
исследования. Модель судьи пишется в judge_verdicts.model.

    uv run python judge_runs.py [--design HASH ...] [--exclude-design HASH ...] \\
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
    if force and st.has_verdict(run_id):
        with st.conn:
            st.conn.execute("DELETE FROM judge_verdicts WHERE run_id=?", (run_id,))
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
