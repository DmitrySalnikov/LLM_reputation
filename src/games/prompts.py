"""Context (prompt) builders for the game — shared by the game and the strategies.

Imports neither the game nor the strategies, to avoid import cycles. The prompt TEXT lives
in GameCfg (config layer); these builders just fill placeholders by literal replacement
(NOT str.format — the templates contain real JSON braces). The rules/payoffs in system are
now substituted by Agent.system_prompt (from AgentSpec.system_prompt), not these builders:
    talk:                 {partner} {round} {feed}
    decide:               {partner} {round} {feed} {reason}; the rationale flag selects decide_prompt|_bare
    predict:              {partner} {round} {feed} {reason}; the rationale flag selects predict_prompt|_bare
    reflect:              {partner} {round} {feed} {score} {me} {my_number} {partner_number} {payoff}
    notes:                {round} {score}

{feed} — messages of the current round tagged <you>/<name> (src.core.memory.render_turns). The
agent reads its own accumulated score from the result lines of past rounds (history), so it's
no longer in the talk/decide headers; {score} remains only in reflect/notes.
"""

from __future__ import annotations

from src.core.config import GameCfg


def talk_context(cfg: GameCfg, partner: str, round: int, feed: str, score: float = 0.0) -> str:
    """Cheap-talk turn context. Empty feed = first turn -> opener template (no Talk block)."""
    if not feed:
        return _fill(cfg.talk_open_prompt, partner, round, "", score)
    return _fill(cfg.talk_prompt, partner, round, feed, score)


def decide_context(cfg: GameCfg, partner: str, round: int, feed: str, score: float = 0.0,
                   reason: str = "") -> str:
    """Final number-choice context (direct strategy).

    `reason` — why the chat closed (turn limit / mutual agreement); substituted into the
    {reason} closing line so it reads word-for-word as in the history of past rounds.
    The rationale flag selects the whole template: decide_prompt (reasoning) or decide_prompt_bare."""
    feed_block = feed if feed else "(no messages were exchanged)"
    tmpl = cfg.decide_prompt if cfg.rationale else cfg.decide_prompt_bare
    return _fill(tmpl, partner, round, feed_block, score).replace("{reason}", reason)


def predict_context(cfg: GameCfg, partner: str, round: int, feed: str, score: float = 0.0,
                    reason: str = "") -> str:
    """Partner-number prediction context (prediction strategy).

    Mirrors decide: the same static transcript + closing line with {reason}, only the
    directive is different (predict the opponent's number). The rationale flag likewise
    selects the whole template: predict_prompt or predict_prompt_bare."""
    feed_block = feed if feed else "(no messages were exchanged)"
    tmpl = cfg.predict_prompt if cfg.rationale else cfg.predict_prompt_bare
    return _fill(tmpl, partner, round, feed_block, score).replace("{reason}", reason)


def reflect_context(cfg: GameCfg, partner: str, round: int, feed: str, *,
                    me_id: str, my_number: int, partner_number: int, payoff: float,
                    score: float = 0.0) -> str:
    """Post-game reflection context: both numbers are revealed, the payoff is known.

    `{me}` -> "<me_id> (you)" — the agent itself is named the same way as in the diary and the feed.
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
    """Memory-consolidation context: the agent rewrites its memory into personal notes.

    Memory (notes + buffer) arrives as history in Agent.act; here — only the instruction
    with {round}/{score} placeholders (there's no partner/feed for a notes call)."""
    return _fill(cfg.notes_prompt, "", round, "", score)


def _fill(template: str, partner: str, round: int, feed: str, score: float = 0.0) -> str:
    return (
        template
        .replace("{partner}", partner)
        .replace("{round}", str(round))
        .replace("{feed}", feed)
        .replace("{score}", f"{score:g}")    # accumulated score of the agent before the current round
    )
