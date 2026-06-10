"""Построители контекста (промптов) для игры — общие для игры и стратегий.

Не импортирует ни игру, ни стратегии, чтобы исключить циклические импорты."""

from __future__ import annotations

from src.core.config import GameCfg


def rules_text(cfg: GameCfg) -> str:
    """Статический текст правил игры (идёт в системный промпт после персоны).

    Args:
        cfg: Конфигурация игры с матрицей выигрышей.

    Returns:
        Текст правил на английском языке для системного промпта.
    """
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


def talk_context(partner: str, round: int, feed: str) -> str:
    """Контекст хода переговоров (cheap-talk).

    Args:
        partner: Идентификатор партнёра в текущем раунде.
        round: Номер раунда.
        feed: Отрендеренная история переговоров.

    Returns:
        Текст контекста на английском языке для хода переговоров.
    """
    feed_block = feed if feed else "(no messages yet)"
    return (
        f"Your partner this round is {partner}. Round {round}.\n"
        f"Negotiation so far:\n{feed_block}\n\n"
        'Send a short message to your partner. Set "ready": true when you have nothing more to say.\n'
        'Respond ONLY as JSON: {"message": "<your message>", "ready": <true|false>}'
    )


def decide_context(partner: str, round: int, feed: str, *, rationale: bool = True) -> str:
    """Контекст финального выбора числа (стратегия direct).

    Args:
        partner: Идентификатор партнёра в текущем раунде.
        round: Номер раунда.
        feed: Отрендеренная история переговоров.
        rationale: Просить ли обоснование перед числом.

    Returns:
        Текст контекста на английском языке для выбора числа.
    """
    feed_block = feed if feed else "(no messages were exchanged)"
    head = (
        f"Your partner this round is {partner}. Round {round}.\n"
        f"Negotiation:\n{feed_block}\n\n"
        "Now secretly choose your number from 0 to 9."
    )
    if not rationale:
        return head + '\nRespond ONLY as JSON: {"number": <0-9>}'
    return (
        head + " Reason first, then commit to a number.\n"
        'Respond ONLY as JSON: {"rationale": "<short reason>", "number": <0-9>}'
    )


def predict_context(partner: str, round: int, feed: str, *, rationale: bool = True) -> str:
    """Контекст предсказания числа партнёра (стратегия prediction).

    Args:
        partner: Идентификатор партнёра в текущем раунде.
        round: Номер раунда.
        feed: Отрендеренная история переговоров.
        rationale: Просить ли обоснование перед числом.

    Returns:
        Текст контекста на английском языке для предсказания числа партнёра.
    """
    feed_block = feed if feed else "(no messages were exchanged)"
    head = (
        f"Your partner this round is {partner}. Round {round}.\n"
        f"Negotiation:\n{feed_block}\n\n"
        "Predict the number your partner will secretly choose, from 0 to 9."
    )
    if not rationale:
        return head + '\nRespond ONLY as JSON: {"number": <0-9>}'
    return (
        head + " Reason first, then commit to a number.\n"
        'Respond ONLY as JSON: {"rationale": "<short reason>", "number": <0-9>}'
    )


def reflect_context(partner: str, round: int, feed: str, *,
                    my_number: int, partner_number: int, payoff: float) -> str:
    """Контекст пост-игровой рефлексии: осмысление вскрытого исхода раунда.

    Args:
        partner: Идентификатор партнёра в текущем раунде.
        round: Номер раунда.
        feed: Отрендеренная история переговоров.
        my_number: Число, выбранное самим агентом.
        partner_number: Число, выбранное партнёром (уже вскрыто).
        payoff: Выигрыш агента в этом раунде.

    Returns:
        Текст контекста на английском языке для рефлексии по исходу.
    """
    feed_block = feed if feed else "(no messages were exchanged)"
    return (
        f"Your partner this round is {partner}. Round {round}.\n"
        f"Negotiation:\n{feed_block}\n\n"
        f"The round is over. You picked {my_number}, {partner} picked {partner_number}. "
        f"You scored {payoff:g} points.\n"
        "Reflect briefly on this outcome: what does it tell you about this partner, "
        "and what should you do differently (or keep doing) in future rounds?\n"
        'Respond ONLY as JSON: {"reflection": "<short reflection>"}'
    )
