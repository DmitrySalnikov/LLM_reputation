from __future__ import annotations

from src.core.config import GameCfg
from src.games.prompts import decide_context, predict_context, reflect_context


def test_decide_template_puts_rationale_before_number():
    ctx = decide_context(GameCfg(), "A2", 1, "feed")
    assert '"rationale"' in ctx and '"number"' in ctx
    assert ctx.index('"rationale"') < ctx.index('"number"')


def test_predict_template_puts_rationale_before_number():
    ctx = predict_context(GameCfg(), "A2", 1, "feed")
    assert ctx.index('"rationale"') < ctx.index('"number"')


def test_decide_template_without_rationale_asks_bare_number():
    ctx = decide_context(GameCfg(rationale=False), "A2", 1, "feed")
    assert "rationale" not in ctx.lower()
    assert '{"number": <0-9>}' in ctx


def test_predict_template_without_rationale_asks_bare_number():
    ctx = predict_context(GameCfg(rationale=False), "A2", 1, "feed")
    assert "rationale" not in ctx.lower()
    assert '{"number": <0-9>}' in ctx


def test_explicit_template_overrides_rationale_flag():
    cfg = GameCfg(rationale=False, decide_prompt="Custom {partner} r{round}: {feed}")
    assert decide_context(cfg, "A2", 1, "feed") == "Custom A2 r1: feed"


def test_reflect_context_states_result_and_asks_json():
    ctx = reflect_context(GameCfg(), "A2", 3, "A2: take 4 (ready=true)",
                          me_id="A1", my_number=4, partner_number=5, payoff=0.0)
    assert "A2" in ctx and "Round 3" in ctx
    assert "take 4" in ctx                      # negotiation feed is restated
    assert "4" in ctx and "5" in ctx and "0" in ctx  # both numbers and the payoff
    assert "A1 (you) picked 4" in ctx           # сам агент — "<имя> (you)", как в дневнике/фиде
    assert '"reflection"' in ctx                # answer contract


def test_reflect_context_without_feed():
    ctx = reflect_context(GameCfg(), "A2", 1, "", me_id="A1",
                          my_number=2, partner_number=2, payoff=3.0)
    assert "(no messages were exchanged)" in ctx
