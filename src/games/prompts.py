"""Context (prompt) builders for the game — shared by the game and the strategies.

Imports neither the game nor the strategies, to avoid import cycles. The prompt TEXT lives
in GameCfg (config layer); these builders just fill placeholders by literal replacement
(NOT str.format — the templates contain real JSON braces):
    rules:                {R} {T} {P} {S} {max_talk_turns}
    talk:                 {partner} {round} {feed} {opener} (кто открыл раунд)
    decide:               {partner} {round} {feed} {answer} {reason} (как закрылся чат)
    predict:              {partner} {round} {feed} {answer} (хвост ответа по флагу rationale)
    reflect:              {partner} {round} {feed} {score} {me} {my_number} {partner_number} {payoff}
    notes:                {round} {score}

{feed} — реплики текущего раунда тегами <you>/<имя> (src.core.memory.render_turns). Свой
накопленный счёт агент читает из строк результата прошлых раундов (история), поэтому в
заголовках talk/decide его больше нет; {score} остаётся только в reflect/notes.
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


def talk_context(cfg: GameCfg, partner: str, round: int, feed: str, score: float = 0.0,
                 opener: str = "") -> str:
    """Cheap-talk turn context. Пустой фид = первый ход -> шаблон-опенер (без блока Talk).

    `opener` — кто открыл раунд (та же фраза, что в истории); подставляется в {opener}
    строки открытия talk_prompt. У talk_open_prompt опенер уже вшит (агент сам открывает)."""
    if not feed:
        return _fill(cfg.talk_open_prompt, partner, round, "", score)
    return _fill(cfg.talk_prompt, partner, round, feed, score).replace("{opener}", opener)


def decide_context(cfg: GameCfg, partner: str, round: int, feed: str, score: float = 0.0,
                   reason: str = "") -> str:
    """Final number-choice context (direct strategy).

    `reason` — почему закрылся чат (лимит реплик / обоюдное согласие); подставляется в
    {reason} строки закрытия, чтобы она читалась дословно как в истории прошлых раундов."""
    feed_block = feed if feed else "(no messages were exchanged)"
    return (
        _fill(cfg.decide_prompt, partner, round, feed_block, score)
        .replace("{answer}", _answer(cfg))
        .replace("{reason}", reason)
    )


def predict_context(cfg: GameCfg, partner: str, round: int, feed: str, score: float = 0.0) -> str:
    """Partner-number prediction context (prediction strategy)."""
    feed_block = feed if feed else "(no messages were exchanged)"
    return _fill(cfg.predict_prompt, partner, round, feed_block, score).replace("{answer}", _answer(cfg))


# Хвост ответа DECIDE/PREDICT — подставляется в плейсхолдер {answer} (если он есть в шаблоне);
# текст берётся из конфига (answer_bare / answer_rationale), выбор по флагу rationale.
def _answer(cfg: GameCfg) -> str:
    return cfg.answer_rationale if cfg.rationale else cfg.answer_bare


def reflect_context(cfg: GameCfg, partner: str, round: int, feed: str, *,
                    me_id: str, my_number: int, partner_number: int, payoff: float,
                    score: float = 0.0) -> str:
    """Post-game reflection context: both numbers are revealed, the payoff is known.

    `{me}` -> "<me_id> (you)" — сам агент именуется так же, как в дневнике и фиде.
    """
    feed_block = feed if feed else "(no messages were exchanged)"
    return (
        _fill(cfg.reflect_prompt, partner, round, feed_block, score)
        .replace("{me}", f"{me_id} (you)")
        .replace("{my_number}", str(my_number))
        .replace("{partner_number}", str(partner_number))
        .replace("{payoff}", f"{payoff:g}")
    )


def notes_context(cfg: GameCfg, round: int, score: float = 0.0) -> str:
    """Memory-consolidation context: агент переписывает свою память в личные заметки.

    Память (заметки + буфер) доезжает как history в Agent.act; здесь — только инструкция
    с плейсхолдерами {round}/{score} (партнёра/фида у note-вызова нет)."""
    return _fill(cfg.notes_prompt, "", round, "", score)


def _fill(template: str, partner: str, round: int, feed: str, score: float = 0.0) -> str:
    return (
        template
        .replace("{partner}", partner)
        .replace("{round}", str(round))
        .replace("{feed}", feed)
        .replace("{score}", f"{score:g}")    # накопленный счёт агента до текущего раунда
    )
