from __future__ import annotations

from dataclasses import replace

from src.core.agent import ActParseError, Agent, Phase, PhaseKind
from src.core.config import GameCfg
from src.core.memory import MemoryEntry
from src.games.base import PairingRecord
from src.games.prompts import reflect_context, rules_text, talk_context
from src.providers.base import ProviderError
from src.strategy.base import PlayStrategy

# Outcome from A's perspective -> outcome from B's perspective.
_FLIP = {"CC": "CC", "DD": "DD", "DC": "CD", "CD": "DC"}


class ReputationPD:
    def __init__(self, cfg: GameCfg, rules: str | None = None,
                 strategy: PlayStrategy | None = None):
        self.cfg = cfg
        self._rules = rules if rules is not None else rules_text(cfg)
        if strategy is None:
            from src.strategy.direct import DirectStrategy  # ленивый импорт: разрывает цикл games<->strategy
            strategy = DirectStrategy(cfg)
        self._strategy = strategy

    def resolve(self, x: int, y: int) -> tuple[str, float, float]:
        p = self.cfg.payoffs
        if x == y:
            return ("CC", p.R, p.R)
        if x == (y + 1) % 10:
            return ("DC", p.T, p.S)  # x betrayed y
        if y == (x + 1) % 10:
            return ("CD", p.S, p.T)  # y betrayed x
        return ("DD", p.P, p.P)

    async def play_pairing(self, a: Agent, b: Agent, round: int) -> PairingRecord:
        # No rng: the matcher fixes who opens cheap-talk via argument order (a opens).
        # Сырьё каждого LLM-вызова копим инкрементально: при LLM-сбое (ProviderError) пара
        # не падает, а возвращается как сорванная (finished=False) со всем накопленным L2-логом.
        transcript: list[dict] = []
        post_calls: list = []        # вызовы decide/predict/reflect (talk-каллы — из transcript)
        try:
            await self._cheap_talk(a, b, round, transcript)
            feed_a = _render_feed(transcript, a.id)   # эгоцентрично: свой ход помечен "(you)"
            feed_b = _render_feed(transcript, b.id)
            da = await self._strategy.decide(a, b.id, round, feed_a, self._rules)
            post_calls += list(da.calls)
            db = await self._strategy.decide(b, a.id, round, feed_b, self._rules)
            post_calls += list(db.calls)
            x, y = da.number, db.number
            outcome, pa, pb = self.resolve(x, y)
            a.score += pa
            b.score += pb

            ra = rb = None
            usages = [t["usage"] for t in transcript] + [da.usage, db.usage]
            if self.cfg.reflection:
                # Рефлексия идёт до записи в память: факты раунда приходят через контекст,
                # а дневник агента ещё показывает только прошлые раунды.
                ra, ua, ca = await self._reflect(a, b.id, round, feed_a, x, y, pa)
                post_calls += list(ca)
                rb, ub, cb = await self._reflect(b, a.id, round, feed_b, y, x, pb)
                post_calls += list(cb)
                usages += [ua, ub]

            public = _public(transcript)
            self._remember(a, b.id, round, public, da, y, outcome, pa, pb, ra)
            self._remember(b, a.id, round, public, db, x, _FLIP[outcome], pb, pa, rb)
            return PairingRecord(
                round=round, a_id=a.id, b_id=b.id, transcript=public,
                a_number=x, b_number=y,
                a_rationale=da.rationale, b_rationale=db.rationale,
                outcome=outcome, a_payoff=pa, b_payoff=pb, usage=_sum_usage(usages),
                a_predicted=da.predicted, b_predicted=db.predicted,
                a_reflection=ra, b_reflection=rb,
                finished=True, llm_calls=_talk_calls(transcript) + post_calls,
            )
        except (ProviderError, ActParseError) as e:
            # Aborted pairing: no results (numbers/outcome/payoff = None), but the full raw
            # L2 log is kept — talk calls + completed decide/reflect + the failing ones (e.calls).
            calls = _talk_calls(transcript) + post_calls + list(e.calls)
            return PairingRecord(
                round=round, a_id=a.id, b_id=b.id, transcript=_public(transcript),
                usage=_usage_from_calls(calls), finished=False, llm_calls=calls,
            )

    async def _reflect(self, agent: Agent, partner_id: str, round: int, feed: str,
                       my_number: int, partner_number: int,
                       payoff: float) -> tuple[str, tuple[int, int], tuple]:
        """Запросить у агента рефлексию по вскрытому исходу раунда.

        Args:
            agent: Агент, осмысляющий исход.
            partner_id: Идентификатор партнёра в текущем раунде.
            round: Номер раунда.
            feed: Отрендеренная история переговоров.
            my_number: Число, выбранное самим агентом.
            partner_number: Число, выбранное партнёром.
            payoff: Выигрыш агента в этом раунде.

        Returns:
            Тройка (текст рефлексии, usage запроса, сырые LLMCall'ы фазы).
        """
        ctx = reflect_context(self.cfg, partner_id, round, feed, me_id=agent.id,
                              my_number=my_number, partner_number=partner_number, payoff=payoff)
        res = await agent.act(Phase(PhaseKind.REFLECT, ctx, rules=self._rules))
        return res.data["reflection"], res.usage, res.calls

    async def _cheap_talk(self, a: Agent, b: Agent, round: int, transcript: list[dict]) -> None:
        # Наполняет переданный transcript на месте (чтобы при LLM-сбое частичный
        # cheap-talk и его L2-лог не терялись — каждая реплика несёт свои "calls").
        ready = {a.id: False, b.id: False}
        order = [a, b]  # a opens; the matcher sets orientation via pairing order
        i = 0
        while len(transcript) < self.cfg.max_talk_turns:
            cur, oth = order[i % 2], order[(i + 1) % 2]
            i += 1
            if ready[cur.id]:
                if ready[oth.id]:
                    break
                continue  # latched: stays silent while the other matures
            ctx = talk_context(self.cfg, oth.id, round, _render_feed(transcript, cur.id))
            res = await cur.act(Phase(PhaseKind.TALK, ctx, rules=self._rules))
            transcript.append(
                {
                    "speaker": cur.id,
                    "text": res.public_text,
                    "ready": res.data["ready"],
                    "usage": res.usage,
                    "calls": res.calls,      # сырой L2-лог реплики (turn_idx доклеит игра)
                }
            )
            ready[cur.id] = res.data["ready"]
            # Each agent necessarily speaks at least once: ending needs BOTH ready,
            # and an agent is marked ready only after it has spoken.
            if ready[a.id] and ready[b.id]:
                break

    def _remember(self, agent, partner_id, round, public_transcript, mine, partner_number,
                  outcome, payoff, partner_payoff, reflection=None):
        agent.memory.add(
            MemoryEntry(
                round=round,
                my_id=agent.id,
                partner_id=partner_id,
                transcript=public_transcript,
                my_number=mine.number,
                my_rationale=mine.rationale,
                partner_number=partner_number,
                outcome=outcome,
                payoff=payoff,
                partner_payoff=partner_payoff,
                my_predicted=mine.predicted,
                my_reflection=reflection,
            )
        )


def _public(transcript: list[dict]) -> list[dict]:
    return [{"speaker": t["speaker"], "text": t["text"], "ready": t["ready"]} for t in transcript]


def _talk_calls(transcript: list[dict]) -> list:
    """Собрать LLMCall'ы реплик cheap-talk, проставив каждому turn_idx (индекс в transcript)."""
    out: list = []
    for ti, t in enumerate(transcript):
        out += [replace(c, turn_idx=ti) for c in t.get("calls", ())]
    return out


def _usage_from_calls(calls: list) -> dict:
    """Usage сорванной пары — суммарные токены и число HTTP-вызовов из её L2-лога."""
    return {
        "prompt_tokens": sum(c.prompt_tokens for c in calls),
        "completion_tokens": sum(c.completion_tokens for c in calls),
        "calls": len(calls),
    }


def _sum_usage(usages: list) -> dict:
    pt = ct = calls = 0
    for u in usages:
        if u is None:
            continue
        pt += u[0]
        ct += u[1]
        calls += 1
    return {"prompt_tokens": pt, "completion_tokens": ct, "calls": calls}


def _render_feed(transcript: list[dict], me_id: str) -> str:
    # Эгоцентрично к зрителю: его реплики — "<имя> (you)", чужие — по имени (как в дневнике).
    def _label(speaker: str) -> str:
        return f"{speaker} (you)" if speaker == me_id else speaker

    def _line(t: dict) -> str:
        mark = " (ready=true)" if t["ready"] else ""   # ready=false не выводим
        return f"  {_label(t['speaker'])}: {t['text']}{mark}"

    return "\n".join(_line(t) for t in transcript)
