from __future__ import annotations

from src.core.agent import Agent, Phase, PhaseKind
from src.core.config import GameCfg
from src.core.memory import MemoryEntry
from src.games.base import PairingRecord
from src.games.prompts import reflect_context, rules_text, talk_context
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
        transcript = await self._cheap_talk(a, b, round)
        feed = _render_feed(transcript)
        da = await self._strategy.decide(a, b.id, round, feed, self._rules)
        db = await self._strategy.decide(b, a.id, round, feed, self._rules)
        x, y = da.number, db.number
        outcome, pa, pb = self.resolve(x, y)
        a.score += pa
        b.score += pb

        ra = rb = None
        usages = [t["usage"] for t in transcript] + [da.usage, db.usage]
        if self.cfg.reflection:
            # Рефлексия идёт до записи в память: факты раунда приходят через контекст,
            # а дневник агента ещё показывает только прошлые раунды.
            ra, ua = await self._reflect(a, b.id, round, feed, x, y, pa)
            rb, ub = await self._reflect(b, a.id, round, feed, y, x, pb)
            usages += [ua, ub]

        public = _public(transcript)
        self._remember(a, b.id, round, public, da, y, outcome, pa, ra)
        self._remember(b, a.id, round, public, db, x, _FLIP[outcome], pb, rb)
        return PairingRecord(
            round=round, a_id=a.id, b_id=b.id, transcript=public,
            a_number=x, b_number=y,
            a_rationale=da.rationale, b_rationale=db.rationale,
            outcome=outcome, a_payoff=pa, b_payoff=pb, usage=_sum_usage(usages),
            a_predicted=da.predicted, b_predicted=db.predicted,
            a_reflection=ra, b_reflection=rb,
        )

    async def _reflect(self, agent: Agent, partner_id: str, round: int, feed: str,
                       my_number: int, partner_number: int,
                       payoff: float) -> tuple[str, tuple[int, int]]:
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
            Пара (текст рефлексии, usage запроса).
        """
        ctx = reflect_context(self.cfg, partner_id, round, feed, my_number=my_number,
                              partner_number=partner_number, payoff=payoff)
        res = await agent.act(Phase(PhaseKind.REFLECT, ctx, rules=self._rules))
        return res.data["reflection"], res.usage

    async def _cheap_talk(self, a: Agent, b: Agent, round: int) -> list[dict]:
        transcript: list[dict] = []
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
            ctx = talk_context(self.cfg, oth.id, round, _render_feed(transcript))
            res = await cur.act(Phase(PhaseKind.TALK, ctx, rules=self._rules))
            transcript.append(
                {
                    "speaker": cur.id,
                    "text": res.public_text,
                    "ready": res.data["ready"],
                    "usage": res.usage,
                }
            )
            ready[cur.id] = res.data["ready"]
            # Each agent necessarily speaks at least once: ending needs BOTH ready,
            # and an agent is marked ready only after it has spoken.
            if ready[a.id] and ready[b.id]:
                break
        return transcript

    def _remember(self, agent, partner_id, round, public_transcript, mine, partner_number,
                  outcome, payoff, reflection=None):
        agent.memory.add(
            MemoryEntry(
                round=round,
                partner_id=partner_id,
                transcript=public_transcript,
                my_number=mine.number,
                my_rationale=mine.rationale,
                partner_number=partner_number,
                outcome=outcome,
                payoff=payoff,
                my_predicted=mine.predicted,
                my_reflection=reflection,
            )
        )


def _public(transcript: list[dict]) -> list[dict]:
    return [{"speaker": t["speaker"], "text": t["text"], "ready": t["ready"]} for t in transcript]


def _sum_usage(usages: list) -> dict:
    pt = ct = calls = 0
    for u in usages:
        if u is None:
            continue
        pt += u[0]
        ct += u[1]
        calls += 1
    return {"prompt_tokens": pt, "completion_tokens": ct, "calls": calls}


def _render_feed(transcript: list[dict]) -> str:
    return "\n".join(
        f"{t['speaker']}: {t['text']} (ready={str(bool(t['ready'])).lower()})" for t in transcript
    )
