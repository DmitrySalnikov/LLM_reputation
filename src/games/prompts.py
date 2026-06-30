"""Context (prompt) builders for the game — shared by the game and the strategies.

Imports neither the game nor the strategies, to avoid import cycles. The prompt TEXT lives
in GameCfg (config layer); these builders just fill placeholders by literal replacement
(NOT str.format — the templates contain real JSON braces). Правила/payoff'ы в system
теперь подставляет Agent.system_prompt (из AgentSpec.system_prompt), не эти билдеры:
    talk:                 {partner} {round} {feed}
    decide:               {partner} {round} {feed} {reason}; флаг rationale выбирает decide_prompt|_bare
    predict:              {partner} {round} {feed} {reason}; флаг rationale выбирает predict_prompt|_bare
    reflect:              {partner} {round} {feed} {score} {me} {my_number} {partner_number} {payoff}
    notes:                {round} {score}

{feed} — реплики текущего раунда тегами <you>/<имя> (src.core.memory.render_turns). Свой
накопленный счёт агент читает из строк результата прошлых раундов (история), поэтому в
заголовках talk/decide его больше нет; {score} остаётся только в reflect/notes.
"""

from __future__ import annotations

from src.core.config import GameCfg


def talk_context(cfg: GameCfg, partner: str, round: int, feed: str, score: float = 0.0) -> str:
    """Cheap-talk turn context. Пустой фид = первый ход -> шаблон-опенер (без блока Talk)."""
    if not feed:
        return _fill(cfg.talk_open_prompt, partner, round, "", score)
    return _fill(cfg.talk_prompt, partner, round, feed, score)


def decide_context(cfg: GameCfg, partner: str, round: int, feed: str, score: float = 0.0,
                   reason: str = "") -> str:
    """Final number-choice context (direct strategy).

    `reason` — почему закрылся чат (лимит реплик / обоюдное согласие); подставляется в
    {reason} строки закрытия, чтобы она читалась дословно как в истории прошлых раундов.
    Флаг rationale выбирает целый шаблон: decide_prompt (рассуждать) или decide_prompt_bare."""
    feed_block = feed if feed else "(no messages were exchanged)"
    tmpl = cfg.decide_prompt if cfg.rationale else cfg.decide_prompt_bare
    return _fill(tmpl, partner, round, feed_block, score).replace("{reason}", reason)


def predict_context(cfg: GameCfg, partner: str, round: int, feed: str, score: float = 0.0,
                    reason: str = "") -> str:
    """Partner-number prediction context (prediction strategy).

    Зеркалит decide: тот же статичный транскрипт + строка закрытия с {reason}, только
    директива другая (предсказать число оппонента). Флаг rationale так же выбирает целый
    шаблон: predict_prompt или predict_prompt_bare."""
    feed_block = feed if feed else "(no messages were exchanged)"
    tmpl = cfg.predict_prompt if cfg.rationale else cfg.predict_prompt_bare
    return _fill(tmpl, partner, round, feed_block, score).replace("{reason}", reason)


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
