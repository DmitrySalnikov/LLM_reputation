from __future__ import annotations

from dataclasses import replace

from src.core.agent import ActParseError, Agent, Phase, PhaseKind
from src.core.config import GameCfg
from src.core.memory import MemoryEntry, both_agreed, pick_opener, render_turns
from src.games.base import PairingRecord
from src.games.prompts import notes_context, reflect_context, talk_context
from src.games.talk_rules import make_talk_rule
from src.providers.base import ProviderError
from src.strategy.base import PlayStrategy

# Outcome from A's perspective -> outcome from B's perspective.
_FLIP = {"CC": "CC", "DD": "DD", "DC": "CD", "CD": "DC"}


class ReputationPD:
    def __init__(self, cfg: GameCfg, strategy: PlayStrategy | None = None):
        self.cfg = cfg
        # Правила больше не собираются здесь: весь system (с правилами) задаётся одной строкой
        # AgentSpec.system_prompt, а payoff'ы подставляются в Agent.system_prompt из Phase.game_cfg.
        # Стратегия теперь живёт на агенте (agent.setup.play_strategy). Объект собирается
        # лениво и кэшируется по (play_strategy, prediction_mapping). Явно переданный
        # `strategy` — однородный override (для тестов и простого случая) поверх per-agent.
        self._override = strategy
        self._strategy_cache: dict[tuple[str, str], PlayStrategy] = {}

    def _strategy_for(self, agent: Agent) -> PlayStrategy:
        if self._override is not None:
            return self._override
        key = (agent.setup.play_strategy, agent.setup.prediction_mapping)
        st = self._strategy_cache.get(key)
        if st is None:
            from src.strategy.base import make_strategy  # ленивый импорт: цикл games<->strategy
            st = make_strategy(key[0], key[1], self.cfg)
            self._strategy_cache[key] = st
        return st

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
            feed_a = self._feed(transcript, a.id)   # эгоцентрично: свои реплики помечены <you>
            feed_b = self._feed(transcript, b.id)
            # Та же причина закрытия, что увидят агенты в истории этого раунда (см. memory).
            reason = (self.cfg.reason_agreed if both_agreed(transcript, a.id, b.id)
                      else self.cfg.reason_limit)
            da = await self._strategy_for(a).decide(a, b.id, round, feed_a, reason)
            post_calls += list(da.calls)
            db = await self._strategy_for(b).decide(b, a.id, round, feed_b, reason)
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

            # Memory notes: каждые N раундов агенты сворачивают память в заметки (после
            # _remember — чтобы текущий раунд уже был в памяти). Note-вызовы привязаны к паре.
            # Свёртка — по числу сыгранных АГЕНТОМ партий (len(memory.entries) после
            # _remember), а не по номеру раунда: idle-раунды не считаются, и оба агента
            # решают независимо. Каждый note-вызов копим сразу (как reflect): если note(b)
            # сорвётся, успевший L2-лог note(a) не теряется и доезжает в сорванную пару.
            na = nb = None
            if self._notes_due(a):
                na, una, cna = await self._make_notes(a, round)
                post_calls += list(cna); usages.append(una)
            if self._notes_due(b):
                nb, unb, cnb = await self._make_notes(b, round)
                post_calls += list(cnb); usages.append(unb)
            return PairingRecord(
                round=round, a_id=a.id, b_id=b.id, transcript=public,
                a_number=x, b_number=y,
                a_rationale=da.rationale, b_rationale=db.rationale,
                outcome=outcome, a_payoff=pa, b_payoff=pb, usage=_sum_usage(usages),
                a_predicted=da.predicted, b_predicted=db.predicted,
                a_reflection=ra, b_reflection=rb,
                a_notes=na, b_notes=nb,
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
                              my_number=my_number, partner_number=partner_number, payoff=payoff,
                              score=agent.score)
        res = await agent.act(Phase(PhaseKind.REFLECT, ctx, game_cfg=self.cfg))
        return res.data["reflection"], res.usage, res.calls

    def _notes_due(self, agent: Agent) -> bool:
        """Пора ли агенту свернуть память: каждые memory_notes_every СЫГРАННЫХ им партий.

        Счётчик партий = len(agent.memory.entries) (одна запись на сыгранный раунд; idle
        записей не даёт). Вызывается после _remember, поэтому текущий раунд уже учтён."""
        every = self.cfg.memory_notes_every
        return bool(every) and len(agent.memory.entries) % every == 0

    async def _make_notes(self, agent: Agent, round: int) -> tuple[str, tuple[int, int], tuple]:
        """Свернуть память агента в личные заметки (фаза NOTE) и запомнить их в agent.memory.

        Агент видит свою память целиком (старые заметки + буфер новых раундов) как history
        в Agent.act и переписывает её в новые заметки; с этого момента render() шлёт заметки
        вместо полной истории. Сбой (ProviderError/ActParseError) пробрасывается и обрывает
        пару, как и любой LLM-вызов.

        Args:
            agent: Агент, сворачивающий свою память.
            round: Номер раунда (для {round} в шаблоне).

        Returns:
            Тройка (текст заметок, usage запроса, сырые LLMCall'ы фазы NOTE).
        """
        ctx = notes_context(self.cfg, round, score=agent.score)
        res = await agent.act(Phase(PhaseKind.NOTE, ctx, game_cfg=self.cfg))
        agent.memory.set_notes(res.data["notes"])
        return res.data["notes"], res.usage, res.calls

    async def _cheap_talk(self, a: Agent, b: Agent, round: int, transcript: list[dict]) -> None:
        # Наполняет переданный transcript на месте (чтобы при LLM-сбое частичный
        # cheap-talk и его L2-лог не терялись — каждая реплика несёт свои "calls").
        # Стоп-правило — подключаемый модуль (src/games/talk_rules.py): skip_turn решает,
        # молчит ли уже-готовый говорящий (защёлка), is_over — пора ли завершить переговоры.
        rule = make_talk_rule(self.cfg.talk_stop_rule)
        ready = {a.id: False, b.id: False}
        order = [a, b]  # a opens; the matcher sets orientation via pairing order
        i = 0
        while len(transcript) < self.cfg.max_talk_turns:
            cur, oth = order[i % 2], order[(i + 1) % 2]
            i += 1
            if rule.skip_turn(cur.id, ready):
                if rule.is_over(ready):
                    break
                continue  # latched: stays silent while the other matures
            opener = pick_opener(transcript, cur.id, oth.id,
                                 self.cfg.opener_self, self.cfg.opener_partner)
            ctx = talk_context(self.cfg, oth.id, round, self._feed(transcript, cur.id),
                               cur.score, opener)
            res = await cur.act(Phase(PhaseKind.TALK, ctx, game_cfg=self.cfg))
            transcript.append(
                {
                    "speaker": cur.id,
                    "text": res.public_text,
                    "ready": res.data["ready"],
                    "usage": res.usage,
                    "calls": res.calls,      # сырой L2-лог реплики (turn_idx доклеит игра)
                }
            )
            # next_ready решает, перезаписать флаг сигналом (отзываемо) или защёлкнуть (липко).
            ready[cur.id] = rule.next_ready(ready[cur.id], res.data["ready"])
            # Each agent necessarily speaks at least once: ending needs BOTH ready,
            # and an agent is marked ready only after it has spoken.
            if rule.is_over(ready):
                break

    def _feed(self, transcript: list[dict], me_id: str) -> str:
        # Живой фид текущего раунда — теми же тегами <you>/<имя>, что и история (общий код).
        return render_turns(transcript, me_id, self.cfg.msg_self, self.cfg.msg_partner)

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
                score=agent.score - payoff,   # счёт ДО этого раунда — как в фазовом хедере
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
