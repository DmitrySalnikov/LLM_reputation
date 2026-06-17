from __future__ import annotations

from src.core.memory import Memory, MemoryEntry


def _entry(round=1, partner="A2", my=4, partner_num=4, outcome="CC",
           payoff=3.0, partner_payoff=3.0, score=0.0):
    return MemoryEntry(
        round=round,
        my_id="A1",
        partner_id=partner,
        transcript=[
            {"speaker": "A1", "text": "let us both take 4", "ready": False},
            {"speaker": partner, "text": "ok, 4", "ready": True},
        ],
        my_number=my,
        my_rationale="agreed on 4",
        partner_number=partner_num,
        outcome=outcome,
        payoff=payoff,
        partner_payoff=partner_payoff,
        score=score,
    )


def test_empty_renders_nothing():
    m = Memory()
    assert m.render(None) == []
    assert m.render(5) == []


def test_single_entry_content():
    m = Memory()
    m.add(_entry(round=3, partner="A5"))
    msgs = m.render(None)
    assert len(msgs) == 1
    msg = msgs[0]
    assert msg.role == "user"
    text = msg.content
    assert "<game>Round 3 · opponent A5" in text
    assert "<you>let us both take 4</you>" in text   # own line tagged <you>
    assert "<A5>ok, 4</A5>" in text                   # opponent line tagged with the id
    assert "The choice has been accepted. A5 chose 4" in text  # revealing result line
    assert "Payoffs: you = 3, A5 = 3" in text         # обе выплаты в одной строке
    assert "Outcome" not in text and "CC" not in text  # сырой код исхода в транскрипт не утекает


def test_result_line_shows_running_total():
    m = Memory()
    m.add(_entry(round=3, partner="A5", score=12.0))   # 12 до раунда + payoff 3 = 15 после
    text = m.render(None)[0].content
    assert "Your total score after round 3 is 15 points" in text


def test_close_reason_reflects_who_ended_the_chat():
    m = Memory()
    m.add(_entry())                                    # A1 ready=False -> упёрлись в лимит
    assert "the messages number limit has been reached" in m.render(None)[0].content
    m2 = Memory()
    e = _entry()
    e.transcript[0]["ready"] = True                    # теперь оба выставили finish
    m2.add(e)
    assert "both players agreed to stop" in m2.render(None)[0].content


def test_render_shows_both_payoffs_distinctly():
    m = Memory()
    m.add(_entry(partner="A5", payoff=5.0, partner_payoff=0.0))   # ты перебил соперника
    text = m.render(None)[0].content
    assert "Payoffs: you = 5, A5 = 0" in text


def test_reflection_rendered_after_outcome():
    m = Memory()
    e = _entry(round=2, partner="A3")
    e.my_reflection = "A3 kept the agreement; cooperating with them pays off"
    m.add(e)
    text = m.render(None)[0].content
    assert "A3 kept the agreement" in text
    # reflection comes after the revealing result line
    assert text.index("Your total score") < text.index("A3 kept the agreement")


def test_entry_without_reflection_renders_no_reflection_line():
    m = Memory()
    m.add(_entry())
    text = m.render(None)[0].content
    assert "reflection" not in text.lower()


def test_window_limits_to_last_k():
    m = Memory()
    for r in range(1, 6):
        m.add(_entry(round=r))
    text = m.render(2)[0].content
    assert "Round 4" in text and "Round 5" in text
    assert "Round 1" not in text and "Round 3" not in text


def test_window_none_returns_all():
    m = Memory()
    for r in range(1, 4):
        m.add(_entry(round=r))
    text = m.render(None)[0].content
    assert all(f"Round {r}" in text for r in (1, 2, 3))


def test_window_zero_returns_nothing():
    m = Memory()
    m.add(_entry())
    assert m.render(0) == []


def test_set_notes_marks_buffer_boundary():
    m = Memory()
    m.add(_entry(round=1))
    m.add(_entry(round=2))
    m.set_notes("rounds 1-2 went fine")
    assert m.notes == "rounds 1-2 went fine"
    assert m.noted_upto == 2          # обе записи свёрнуты -> буфер пуст


def test_render_with_notes_replaces_history_but_keeps_recent_buffer():
    m = Memory()
    m.add(_entry(round=1))
    m.add(_entry(round=2))
    m.set_notes("R1-2: opponent A2 keeps agreements")
    m.add(_entry(round=3, partner="A7"))   # сыгран после консолидации -> буфер
    text = m.render(None)[0].content
    assert "Your notes from earlier rounds:" in text
    assert "R1-2: opponent A2 keeps agreements" in text   # сжатые заметки вместо истории
    assert "Round 1" not in text and "Round 2" not in text  # свёрнутые раунды не рендерятся сырыми
    assert "Round 3" in text and "A7" in text               # свежий буфер — сырым


def test_render_with_notes_only_when_buffer_empty():
    m = Memory()
    m.add(_entry(round=1))
    m.set_notes("note text")
    msgs = m.render(None)                  # буфер пуст -> только заметки (не [])
    assert len(msgs) == 1
    assert "note text" in msgs[0].content
    assert "Round 1" not in msgs[0].content


def test_render_includes_prediction_line_when_present():
    from src.core.memory import Memory, MemoryEntry

    m = Memory()
    m.add(MemoryEntry(round=1, my_id="A1", partner_id="A2", transcript=[], my_number=5,
                      my_rationale="r", partner_number=5, outcome="CC", payoff=3.0,
                      partner_payoff=3.0, my_predicted=4))
    rendered = m.render(None)[0].content
    assert "predict" in rendered.lower()
    assert "4" in rendered
