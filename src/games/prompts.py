"""Context (prompt) builders for the game — shared by the game and the strategies.

Imports neither the game nor the strategies, to avoid import cycles. The prompt TEXT lives
in GameCfg (config layer); these builders just fill placeholders by literal replacement
(NOT str.format — the templates contain real JSON braces):
    rules:                {R} {T} {P} {S}        <- payoff values
    talk/decide/predict:  {partner} {round} {feed}
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
    )


def talk_context(cfg: GameCfg, partner: str, round: int, feed: str) -> str:
    """Cheap-talk turn context."""
    feed_block = feed if feed else "(no messages yet)"
    return _fill(cfg.talk_prompt, partner, round, feed_block)


def decide_context(cfg: GameCfg, partner: str, round: int, feed: str) -> str:
    """Final number-choice context (direct strategy)."""
    feed_block = feed if feed else "(no messages were exchanged)"
    return _fill(cfg.decide_prompt, partner, round, feed_block)


def predict_context(cfg: GameCfg, partner: str, round: int, feed: str) -> str:
    """Partner-number prediction context (prediction strategy)."""
    feed_block = feed if feed else "(no messages were exchanged)"
    return _fill(cfg.predict_prompt, partner, round, feed_block)


def _fill(template: str, partner: str, round: int, feed: str) -> str:
    return (
        template
        .replace("{partner}", partner)
        .replace("{round}", str(round))
        .replace("{feed}", feed)
    )
