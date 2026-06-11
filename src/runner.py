"""Experiment runner — drive one episode, persist it, narrate it live.

This is the reusable engine behind `experiment.py`: it owns no config of its own.
`experiment.py` defines the config and calls `run(cfg, db_path, name)`; tests or other
scripts can do the same. Persistence (Storage) and the live console narration are wired
together into one observer so the DB and the printed transcript never diverge.
"""

from __future__ import annotations

import os
import random
import sqlite3

from src.core.config import EpisodeCfg
from src.core.orchestrator import EpisodeAborted, run_episode
from src.judge import JudgeError, JudgeVerdict, judge_episode
from src.population import make_population
from src.providers.base import ProviderError
from src.storage import Storage


def narrate_round(r, plan, recs) -> None:
    """Print one round to console as soon as it's played (the live observer)."""
    print(f"\n{'─' * 60}\n  ROUND {r}")
    if plan.idle:
        print(f"  idle (sat out): {', '.join(plan.idle)}")
    for rec in recs:
        print(f"\n  {rec.a_id} vs {rec.b_id}  ({rec.a_id} opens):")
        if rec.transcript:
            for i, t in enumerate(rec.transcript, 1):
                print(f"    {i}. {t['speaker']}: {t['text']}   [ready={t['ready']}]")
        else:
            print("    (no messages exchanged)")
        if not rec.finished:                       # aborted pairing: no result
            print("    (pairing aborted by LLM failure — no result)")
            continue
        # reasoning is shown before the choices it led to (absent when game.rationale=false)
        if rec.a_rationale:
            print(f"    {rec.a_id} reason: {rec.a_rationale}")
        if rec.b_rationale:
            print(f"    {rec.b_id} reason: {rec.b_rationale}")
        if rec.a_predicted is not None or rec.b_predicted is not None:
            print(
                f"    predicted: {rec.a_id} guessed {rec.b_id}={rec.a_predicted}, "
                f"{rec.b_id} guessed {rec.a_id}={rec.b_predicted}"
            )
        print(
            f"    choices: {rec.a_id}={rec.a_number}, {rec.b_id}={rec.b_number}"
            f"  ->  {rec.outcome}   (payoffs {rec.a_id}={rec.a_payoff:g}, {rec.b_id}={rec.b_payoff:g})"
        )
        if rec.a_reflection is not None:
            print(f"      {rec.a_id} reflects: {rec.a_reflection}")
        if rec.b_reflection is not None:
            print(f"      {rec.b_id} reflects: {rec.b_reflection}")


def print_verdict(verdict: JudgeVerdict) -> None:
    """Напечатать вердикт LLM-судьи (без цитат — подсветка живёт в replay)."""
    bar = "=" * 60
    print(f"\n{bar}\n  JUDGE VERDICT\n{bar}")
    print(f"  reputation institute emerged: {'YES' if verdict.emerged else 'NO'}")
    print(f"  {verdict.explanation}")
    print(f"  evidence: {len(verdict.evidence)} message(s) — replay подсветит их цветом")


async def _judge_and_store(cfg, records, st) -> None:
    """Вызвать судью после эпизода; его ошибка не должна терять результаты run'а."""
    try:
        verdict = await judge_episode(cfg, records)
    except (JudgeError, ProviderError) as e:
        print(f"\nсудья не смог вынести вердикт: {e} — run сохранён без вердикта")
        return
    print_verdict(verdict)
    st.save_verdict(verdict, model=cfg.provider.model)


async def run_experiment(cfg: EpisodeCfg, db_path: str, name: str | None = None) -> str | None:
    """Build the population, run the episode, persist + narrate each round, score it.
    Returns the run_id, or None if this exact config is already stored (de-dup)."""
    pop = make_population(cfg.population, context_window=cfg.context_window).build(
        random.Random(cfg.seed)
    )
    st = Storage(db_path)
    try:
        try:
            run_id = st.begin(cfg, pop, name)    # INSERT runs+agents; fails if already stored
        except sqlite3.IntegrityError:
            print("identical config already in DB — nothing to do "
                  "(change seed or config to re-run)")
            return None

        records: list = []                       # копим записи для LLM-судьи

        def observer(r, plan, recs):             # persist AND narrate each round live
            st.observe(r, plan, recs)
            narrate_round(r, plan, recs)
            records.extend(rec for rec in recs if rec.finished)   # judge sees only finished pairings

        try:
            await run_episode(cfg, pop, observer=observer)
        except EpisodeAborted as e:
            # the aborted pairing is already persisted; run stays without finished_at (crash marker)
            print(f"\nepisode aborted: {e} — run saved without finished_at")
            return run_id
        st.finish(pop)

        bar = "=" * 60
        print(f"\n{bar}\n  FINAL SCOREBOARD\n{bar}")
        for a in sorted(pop, key=lambda a: a.score, reverse=True):
            print(f"  {a.id}: {a.score:g}")

        if cfg.judge is not None:                # опциональный LLM-судья (см. JudgeCfg)
            await _judge_and_store(cfg.judge, records, st)
    finally:
        st.close()
        await pop.aclose()
    return run_id


async def run(cfg: EpisodeCfg, db_path: str, name: str | None = None) -> str | None:
    """Top-level entry: print a header, run the experiment, print the replay hint."""
    os.makedirs(os.path.dirname(db_path) or ".", exist_ok=True)
    n_agents = sum(a.count for a in cfg.population.agents)
    print(f"Running experiment{f' {name!r}' if name else ''} into {db_path}: "
          f"{n_agents} agents, {cfg.rounds} rounds, seed={cfg.seed}")

    run_id = await run_experiment(cfg, db_path, name)
    if run_id is not None:
        print(f"\nrun_id={run_id}   "
              f"(replay: uv run python replay.py {run_id})")
    return run_id
