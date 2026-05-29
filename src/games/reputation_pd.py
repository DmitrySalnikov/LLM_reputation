from __future__ import annotations

from src.core.agent import Agent, Phase, PhaseKind
from src.core.config import GameCfg
from src.core.memory import MemoryEntry
from src.games.base import PairingRecord

# Outcome from A's perspective -> outcome from B's perspective.
_FLIP = {"CC": "CC", "DD": "DD", "DC": "CD", "CD": "DC"}


class ReputationPD:
    def __init__(self, cfg: GameCfg, rules: str | None = None):
        self.cfg = cfg
        self._rules = rules if rules is not None else _rules_text(cfg)

    def resolve(self, x: int, y: int) -> tuple[str, float, float]:
        p = self.cfg.payoffs
        if x == y:
            return ("CC", p.R, p.R)
        if x == (y + 1) % 10:
            return ("DC", p.T, p.S)  # x betrayed y
        if y == (x + 1) % 10:
            return ("CD", p.S, p.T)  # y betrayed x
        return ("DD", p.P, p.P)

    async def play_pairing(self, a: Agent, b: Agent, round: int, rng) -> PairingRecord:
        transcript = await self._cheap_talk(a, b, round, rng)
        feed = _render_feed(transcript)
        ra = await a.act(Phase(PhaseKind.DECIDE, _decide_context(b.id, round, feed), rules=self._rules))
        rb = await b.act(Phase(PhaseKind.DECIDE, _decide_context(a.id, round, feed), rules=self._rules))
        x, y = ra.data["number"], rb.data["number"]
        outcome, pa, pb = self.resolve(x, y)
        a.score += pa
        b.score += pb

        public = _public(transcript)
        self._remember(a, b.id, round, public, ra, y, outcome, pa)
        self._remember(b, a.id, round, public, rb, x, _FLIP[outcome], pb)
        usage = _sum_usage([t["usage"] for t in transcript] + [ra.usage, rb.usage])
        return PairingRecord(
            round=round, a_id=a.id, b_id=b.id, transcript=public,
            a_number=x, b_number=y,
            a_rationale=ra.data["rationale"], b_rationale=rb.data["rationale"],
            outcome=outcome, a_payoff=pa, b_payoff=pb, usage=usage,
        )

    async def _cheap_talk(self, a: Agent, b: Agent, round: int, rng) -> list[dict]:
        transcript: list[dict] = []
        ready = {a.id: False, b.id: False}
        order = [a, b] if rng.random() < 0.5 else [b, a]  # first speaker by seed
        i = 0
        while len(transcript) < self.cfg.max_talk_turns:
            cur, oth = order[i % 2], order[(i + 1) % 2]
            i += 1
            if ready[cur.id]:
                if ready[oth.id]:
                    break
                continue  # latched: stays silent while the other matures
            ctx = _talk_context(oth.id, round, _render_feed(transcript))
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

    def _remember(self, agent, partner_id, round, public_transcript, mine, partner_number, outcome, payoff):
        agent.memory.add(
            MemoryEntry(
                round=round,
                partner_id=partner_id,
                transcript=public_transcript,
                my_number=mine.data["number"],
                my_rationale=mine.data["rationale"],
                partner_number=partner_number,
                outcome=outcome,
                payoff=payoff,
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


def _rules_text(cfg: GameCfg) -> str:
    p = cfg.payoffs
    return (
        "You are playing a repeated game. Each round you face one partner.\n"
        "Both of you secretly pick an integer from 0 to 9 at the same time.\n"
        f"- Equal numbers: you each score {p.R:g}.\n"
        f"- Your number exactly one higher than your partner's (0 counts as following 9): "
        f"you score {p.T:g}, your partner {p.S:g}.\n"
        f"- Your partner's number exactly one higher than yours: they score {p.T:g}, you {p.S:g}.\n"
        f"- Any other case: you each score {p.P:g}.\n"
        "Before choosing you may exchange short messages. Messages are not binding; "
        "the final choice is secret and simultaneous. Maximize your own total score."
    )


def _talk_context(partner: str, round: int, feed: str) -> str:
    feed_block = feed if feed else "(no messages yet)"
    return (
        f"Your partner this round is {partner}. Round {round}.\n"
        f"Negotiation so far:\n{feed_block}\n\n"
        'Send a short message to your partner. Set "ready": true when you have nothing more to say.\n'
        'Respond ONLY as JSON: {"message": "<your message>", "ready": <true|false>}'
    )


def _decide_context(partner: str, round: int, feed: str) -> str:
    feed_block = feed if feed else "(no messages were exchanged)"
    return (
        f"Your partner this round is {partner}. Round {round}.\n"
        f"Negotiation:\n{feed_block}\n\n"
        "Now secretly choose your number from 0 to 9.\n"
        'Respond ONLY as JSON: {"number": <0-9>, "rationale": "<short reason>"}'
    )
