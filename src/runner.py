"""Experiment runner — drive one episode, persist it, narrate it live.

This is the reusable engine behind `experiment.py`: it owns no config of its own.
`experiment.py` defines the config and calls `run(cfg, db_path, name)`; tests or other
scripts can do the same. Persistence (Storage) and the live console narration are wired
together into one observer so the DB and the printed transcript never diverge.
"""

from __future__ import annotations

import json
import os
import random
from dataclasses import replace

from src.core.config import EpisodeCfg, episode_from_dict
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
                mark = "   [finish=true]" if t["ready"] else ""   # finish=false не печатаем
                print(f"    {i}. {t['speaker']}: {t['text']}{mark}")
        else:
            print("    (no messages exchanged)")
        if not rec.finished:                       # aborted pairing: no result
            print("    (pairing aborted by LLM failure — no result)")
            continue
        # reasoning is shown before the choices it led to (empty when the prompt asked only for a number)
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


async def run_experiment(cfg: EpisodeCfg, db_path: str, name: str | None = None,
                         quiet: bool = False) -> int:
    """Build the population, run the episode, persist + narrate each round, score it.

    Returns the run_id (целочисленный автоинкремент). Каждый запуск — новый прогон:
    дедупа по конфигу больше нет (повторный запуск того же конфига создаёт новый номер).
    Возобновление/доращивание оборванных или завершённых прогонов — отдельный явный путь
    по номеру прогона (см. resume, A5).

    `quiet=True` глушит пораундовую narration и финальный scoreboard (для свипов на сотни
    прогонов, см. research.py) — персист в БД при этом полный; сообщения об обрыве остаются."""
    pop = make_population(cfg.population, context_window=cfg.context_window).build(
        random.Random(cfg.seed)
    )
    st = Storage(db_path)
    try:
        run_id = st.begin(cfg, pop, name)        # INSERT runs+agents; новый номер

        records: list = []                       # копим записи для LLM-судьи

        def observer(r, plan, recs):             # persist AND (если не quiet) narrate each round live
            st.observe(r, plan, recs)
            if not quiet:
                narrate_round(r, plan, recs)
            records.extend(rec for rec in recs if rec.finished)   # judge sees only finished pairings

        try:
            await run_episode(cfg, pop, observer=observer)
        except EpisodeAborted as e:
            # the aborted pairing is already persisted; run stays without finished_at (crash marker)
            print(f"\nepisode aborted: {e} — run saved without finished_at")
            return run_id
        st.finish(pop)

        if not quiet:
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


def _apply_run_state(pop, state) -> None:
    """Наложить восстановленное состояние (счёт + память) на свежесобранную популяцию.

    id агентов совпадают: популяция пересобрана из того же конфига тем же сидом, поэтому
    имена сэмплируются идентично. Окно памяти живёт на агенте, не на объекте Memory, —
    замена memory его не теряет."""
    for agent in pop:
        agent.score = state.scores[agent.id]
        agent.memory = state.memories[agent.id]


async def resume_run(run_id: int, db_path: str, rounds: int | None = None,
                     quiet: bool = False) -> int:
    """Возобновить или дорастить существующий прогон по его номеру.

    Без `rounds` — доигрываем оборванный прогон до `rounds` из его сохранённого конфига.
    С `rounds` — растим до этого числа (extend); если уже сыграно столько же или больше —
    ничего не делаем. Прошлые раунды читаются из БД (фактические пары), новые играются по
    per-round rng, поэтому возобновимы и прогоны, записанные до перехода на per-round rng.

    `quiet=True` (для свипов, см. research.py) глушит narration/заголовки/scoreboard —
    сообщения об обрыве остаются; БД пишется полностью."""
    st = Storage(db_path)
    config_json = st.run_config(run_id)
    if config_json is None:
        print(f"run {run_id} not found in {db_path}")
        st.close()
        return run_id

    cfg = episode_from_dict(json.loads(config_json))
    if rounds is not None:
        cfg = replace(cfg, rounds=rounds)               # extend: растим целевое число раундов
    state = st.load_state(run_id, cfg.idle_payoff)
    if state.last_round >= cfg.rounds:
        if not quiet:
            print(f"run {run_id}: уже сыграно {state.last_round} раундов (>= {cfg.rounds}) — nothing to do")
        st.close()
        return run_id

    pop = make_population(cfg.population, context_window=cfg.context_window).build(
        random.Random(cfg.seed)
    )
    _apply_run_state(pop, state)
    start = state.last_round + 1
    if not quiet:
        print(f"Resuming run {run_id} into {db_path}: rounds {start}..{cfg.rounds}, "
              f"{len(pop)} agents, seed={cfg.seed}")
    try:
        st.resume(run_id, cfg)                          # _run_id, снять finished_at, обновить config
        def observer(r, plan, recs):                    # persist AND (если не quiet) narrate each new round
            st.observe(r, plan, recs)
            if not quiet:
                narrate_round(r, plan, recs)

        try:
            await run_episode(cfg, pop, observer=observer, start_round=start)
        except EpisodeAborted as e:
            print(f"\nepisode aborted: {e} — run saved without finished_at")
            return run_id
        st.finish(pop)

        if not quiet:
            bar = "=" * 60
            print(f"\n{bar}\n  FINAL SCOREBOARD\n{bar}")
            for a in sorted(pop, key=lambda a: a.score, reverse=True):
                print(f"  {a.id}: {a.score:g}")
            if cfg.judge is not None:                    # судья — отдельная аналитика по полному эпизоду
                print("\n(LLM-судья не запускается при resume/extend — оцените прогон отдельно)")
    finally:
        st.close()
        await pop.aclose()
    if not quiet:
        print(f"\nrun_id={run_id}   (replay: uv run python replay.py {run_id})")
    return run_id


async def run(cfg: EpisodeCfg, db_path: str, name: str | None = None,
              quiet: bool = False) -> int:
    """Top-level entry: print a header, run the experiment, print the replay hint.

    `quiet=True` (для свипов, см. research.py) подавляет заголовок, narration и подсказку —
    наружу не идёт ничего, кроме сообщений об обрыве; БД пишется полностью."""
    os.makedirs(os.path.dirname(db_path) or ".", exist_ok=True)
    if not quiet:
        n_agents = sum(a.count for a in cfg.population.agents)
        print(f"Running experiment{f' {name!r}' if name else ''} into {db_path}: "
              f"{n_agents} agents, {cfg.rounds} rounds, seed={cfg.seed}")

    run_id = await run_experiment(cfg, db_path, name, quiet=quiet)
    if not quiet:
        print(f"\nrun_id={run_id}   "
              f"(replay: uv run python replay.py {run_id})")
    return run_id
