"""Context (prompt) builders for the game — shared by the game and the strategies.

Imports neither the game nor the strategies, to avoid import cycles. The prompt TEXT lives
in GameCfg (config layer); these builders just fill placeholders by literal replacement
(NOT str.format — the templates contain real JSON braces):
    rules:                {R} {T} {P} {S} {max_talk_turns}
    talk:                 {partner} {round} {feed}
    decide/predict:       {partner} {round} {feed} {answer} (хвост ответа по флагу rationale)
    reflect:              {partner} {round} {feed} {me} {my_number} {partner_number} {payoff}
"""

from __future__ import annotations

from src.core.config import GameCfg


def rules_text(cfg: GameCfg) -> str:
    """Static game-rules text (goes into the system prompt after the persona)."""
    p = cfg.payoffs
    return (
        cfg.rules
        .replace("{R}", f"{p.R:g}").replace("{T}", f"{p.T:g}")
        .replace("{P}", f"{p.P:g}").replace("{S}", f"{p.S:g}")
        .replace("{max_talk_turns}", str(cfg.max_talk_turns))
    )


def talk_context(cfg: GameCfg, partner: str, round: int, feed: str) -> str:
    """Cheap-talk turn context. Пустой фид = первый ход -> шаблон-опенер (без блока Talk)."""
    if not feed:
        return _fill(cfg.talk_open_prompt, partner, round, "")
    return _fill(cfg.talk_prompt, partner, round, feed)


def decide_context(cfg: GameCfg, partner: str, round: int, feed: str) -> str:
    """Final number-choice context (direct strategy)."""
    feed_block = feed if feed else "(no messages were exchanged)"
    return _fill(cfg.decide_prompt, partner, round, feed_block).replace("{answer}", _answer(cfg.rationale))


def predict_context(cfg: GameCfg, partner: str, round: int, feed: str) -> str:
    """Partner-number prediction context (prediction strategy)."""
    feed_block = feed if feed else "(no messages were exchanged)"
    return _fill(cfg.predict_prompt, partner, round, feed_block).replace("{answer}", _answer(cfg.rationale))


# Хвост ответа DECIDE/PREDICT, управляемый флагом rationale — подставляется в плейсхолдер
# {answer} (если он есть в шаблоне). У дефолтных шаблонов {answer} нет -> замена холостая.
_ANSWER_RATIONALE = (
    "Reason first, then commit to a number.\n"
    'Respond ONLY as JSON: {"rationale": "<short reason>", "number": <0-9>}'
)
_ANSWER_BARE = 'Respond ONLY as JSON: {"number": <0-9>}'


def _answer(rationale: bool) -> str:
    return _ANSWER_RATIONALE if rationale else _ANSWER_BARE


def reflect_context(cfg: GameCfg, partner: str, round: int, feed: str, *,
                    me_id: str, my_number: int, partner_number: int, payoff: float) -> str:
    """Post-game reflection context: both numbers are revealed, the payoff is known.

    `{me}` -> "<me_id> (you)" — сам агент именуется так же, как в дневнике и фиде.
    """
    feed_block = feed if feed else "(no messages were exchanged)"
    return (
        _fill(cfg.reflect_prompt, partner, round, feed_block)
        .replace("{me}", f"{me_id} (you)")
        .replace("{my_number}", str(my_number))
        .replace("{partner_number}", str(partner_number))
        .replace("{payoff}", f"{payoff:g}")
    )


def _fill(template: str, partner: str, round: int, feed: str) -> str:
    return (
        template
        .replace("{partner}", partner)
        .replace("{round}", str(round))
        .replace("{feed}", feed)
    )
