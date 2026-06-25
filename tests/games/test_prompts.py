from __future__ import annotations

from src.core.config import GameCfg
from src.games.prompts import (
    decide_context, notes_context, predict_context, reflect_context, talk_context,
)


def test_score_placeholder_filled_in_contexts():
    cfg = GameCfg(talk_prompt="t {score}", talk_open_prompt="o {score}",
                  decide_prompt="d {score}", predict_prompt="p {score}")
    assert talk_context(cfg, "A2", 1, "  A2: hi", 7.0) == "t 7"     # с фидом -> talk_prompt
    assert talk_context(cfg, "A2", 1, "", 5.0) == "o 5"             # пустой фид -> опенер
    assert decide_context(cfg, "A2", 1, "f", 3.0) == "d 3"
    assert predict_context(cfg, "A2", 1, "f", 9.0) == "p 9"


def test_talk_context_uses_open_template_on_empty_feed():
    cfg = GameCfg(talk_open_prompt="OPEN {partner} r{round}", talk_prompt="REPLY {feed}")
    assert talk_context(cfg, "A2", 1, "") == "OPEN A2 r1"          # первый ход -> опенер
    assert talk_context(cfg, "A2", 1, "  A2: hi") == "REPLY   A2: hi"  # есть фид -> обычный шаблон


def test_notes_context_fills_round_and_score():
    cfg = GameCfg(notes_prompt="Consolidate at round {round}, score {score}.")
    assert notes_context(cfg, 4, 12.0) == "Consolidate at round 4, score 12."


def test_decide_template_puts_rationale_before_number():
    ctx = decide_context(GameCfg(), "A2", 1, "feed")
    assert '"rationale"' in ctx and '"number"' in ctx
    assert ctx.index('"rationale"') < ctx.index('"number"')


def test_predict_template_puts_rationale_before_number():
    ctx = predict_context(GameCfg(), "A2", 1, "feed")
    assert ctx.index('"rationale"') < ctx.index('"number"')


def test_rationale_flag_selects_whole_template():
    # флаг выбирает ЦЕЛЫЙ статичный шаблон (не склейку): rationale|bare
    cfg = GameCfg(decide_prompt="THINK {feed}", decide_prompt_bare="BARE {feed}",
                  predict_prompt="PTHINK {feed}", predict_prompt_bare="PBARE {feed}")
    assert decide_context(cfg, "A2", 1, "f") == "THINK f"                       # rationale=True (default)
    assert predict_context(cfg, "A2", 1, "f") == "PTHINK f"
    from dataclasses import replace
    off = replace(cfg, rationale=False)
    assert decide_context(off, "A2", 1, "f") == "BARE f"
    assert predict_context(off, "A2", 1, "f") == "PBARE f"


def test_predict_mirrors_decide_and_threads_reason():
    # V1: predict зеркалит decide — тот же транскрипт + строка закрытия с {reason}
    ctx = predict_context(GameCfg(), "A2", 1, "feed", reason="both players agreed to stop")
    assert "The chat has been closed as both players agreed to stop." in ctx
    assert "Predict the number your opponent will secretly choose" in ctx


def test_bare_template_asks_only_number():
    ctx = decide_context(GameCfg(rationale=False), "A2", 1, "feed")
    assert "rationale" not in ctx.lower()
    assert '{"number": <0-9>}' in ctx


def test_explicit_decide_template_used_verbatim():
    cfg = GameCfg(decide_prompt="Custom {partner} r{round}: {feed}")   # rationale=True -> decide_prompt
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
