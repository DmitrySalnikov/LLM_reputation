"""Backfill: evaluate stored runs with the LLM judge and record verdicts.

The judge (judge_episode) currently only runs live at the end of an episode. This script
evaluates runs already sitting in the DB: it reconstructs the public cheap-talk, calls the
judge, and saves the verdict. Runs that are already judged are skipped (unless --force is
given).

One shared judge for all runs — for verdict comparability within the study. Its model is
taken from the judge block of the YAML config (default config/experiment.yaml), not from
the code: change the model there (judge.provider, or the *provider anchor = the agents'
model). The judge's model is written to judge_verdicts.model.

    uv run python judge_runs.py [--db experiment.db] [--config config/experiment.yaml] \\
                                [--design HASH ...] [--exclude-design HASH ...] \\
                                [--name LABEL ...] [--exclude-name LABEL ...] [--force]
"""

from __future__ import annotations

import asyncio
import sys

from dotenv import load_dotenv

from src.core.config import JudgeCfg, ProviderCfg, load_episode
from src.judge import JudgeError, judge_episode
from src.providers.base import ProviderError
from src.stats.selection import filter_from_argv, selected_run_ids
from src.storage import Storage
from src.storage.records import reconstruct_records

load_dotenv()                       # pick up TOGETHER_API_KEY from .env

DB = "experiment.db"                        # default DB; overridden by the --db flag
JUDGE_CONFIG = "config/experiment.yaml"     # where to take the judge from (this config's judge block)

# Fallback judge, used if the config has no judge block. The model matches the main model
# of the default config and is available serverless on Together; usually unused — the judge comes from the config.
JUDGE_DEFAULT = JudgeCfg(provider=ProviderCfg(
    base_url="https://api.together.xyz/v1",
    api_key_env="TOGETHER_API_KEY",
    model="Qwen/Qwen3-235B-A22B-Instruct-2507-FP8",
))


def db_path_from_argv(argv: list[str], default: str = DB) -> str:
    """DB path from the `--db PATH` flag; `default` if the flag is absent."""
    if "--db" in argv:
        i = argv.index("--db")
        if i + 1 < len(argv):
            return argv[i + 1]
    return default


def load_judge_cfg(path: str = JUDGE_CONFIG) -> JudgeCfg:
    """Judge for backfill — from the judge block of the YAML config (same model as the agents'
    if judge.provider is set via the *provider anchor). No judge block -> fallback JUDGE_DEFAULT."""
    cfg = load_episode(path)
    return cfg.judge if cfg.judge is not None else JUDGE_DEFAULT


async def judge_run(st: Storage, run_id: int, judge_cfg: JudgeCfg, *, force: bool) -> str:
    """Evaluate one stored run. Returns a status string.

    Statuses: skipped (a verdict already exists), no-records (no finished pairs),
    failed (the judge failed), judged (verdict recorded)."""
    if not force and st.has_verdict(run_id):
        return "skipped"
    records = reconstruct_records(st.conn, run_id)
    if not records:
        return "no-records"
    try:
        verdict = await judge_episode(judge_cfg, records)
    except (JudgeError, ProviderError) as e:
        print(f"  run {run_id}: judge failed: {e}")
        return "failed"
    if force and st.has_verdict(run_id):
        with st.conn:
            st.conn.execute("DELETE FROM judge_verdicts WHERE run_id=?", (run_id,))
    st.save_verdict(verdict, model=judge_cfg.provider.model, run_id=run_id)
    return "judged"


async def backfill(db_path: str, argv: list[str], judge_cfg: JudgeCfg) -> dict[str, int]:
    """Select runs by the argv filter and evaluate each; return a status counter."""
    force = "--force" in argv
    flt = filter_from_argv(argv)
    st = Storage(db_path)
    counts: dict[str, int] = {}
    try:
        run_ids = selected_run_ids(st.conn, flt)
        print(f"Runs matching the filter: {len(run_ids)}")
        for rid in run_ids:
            status = await judge_run(st, rid, judge_cfg, force=force)
            counts[status] = counts.get(status, 0) + 1
            print(f"  run {rid}: {status}")
    finally:
        st.close()
    return counts


def main() -> None:
    args = sys.argv[1:]
    config_path = args[args.index("--config") + 1] if "--config" in args else JUDGE_CONFIG
    db_path = db_path_from_argv(args)
    judge_cfg = load_judge_cfg(config_path)
    print(f"Judge: {judge_cfg.provider.model} ({config_path})")
    print(f"DB: {db_path}")
    counts = asyncio.run(backfill(db_path, args, judge_cfg))
    print(f"\nSummary: {counts}")


if __name__ == "__main__":
    main()
