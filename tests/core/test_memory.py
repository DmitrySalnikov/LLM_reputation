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
    assert "Payoffs: you = 3, A5 = 3" in text         # both payoffs on one line
    assert "Outcome" not in text and "CC" not in text  # the raw outcome code doesn't leak into the transcript


def test_result_line_shows_running_total():
    m = Memory()
    m.add(_entry(round=3, partner="A5", score=12.0))   # 12 before the round + payoff 3 = 15 after
    text = m.render(None)[0].content
    assert "Your total score after round 3 is 15 points" in text


def test_close_reason_reflects_who_ended_the_chat():
    m = Memory()
    m.add(_entry())                                    # A1 ready=False -> hit the limit
    assert "the messages number limit has been reached" in m.render(None)[0].content
    m2 = Memory()
    e = _entry()
    e.transcript[0]["ready"] = True                    # now both set finish
    m2.add(e)
    assert "both players agreed to stop" in m2.render(None)[0].content


def test_render_shows_both_payoffs_distinctly():
    m = Memory()
    m.add(_entry(partner="A5", payoff=5.0, partner_payoff=0.0))   # you outbid the opponent
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
    assert m.noted_upto == 2          # both entries collapsed -> buffer empty


def test_render_with_notes_replaces_history_but_keeps_recent_buffer():
    m = Memory()
    m.add(_entry(round=1))
    m.add(_entry(round=2))
    m.set_notes("R1-2: opponent A2 keeps agreements")
    m.add(_entry(round=3, partner="A7"))   # played after consolidation -> buffer
    text = m.render(None)[0].content
    # section header in <game>, the notes themselves — in <you>
    assert "<game>Your notes from earlier rounds:</game>\n<you>R1-2: opponent A2 keeps agreements</you>" in text
    assert "<game>Your rounds since those notes:</game>" in text  # label above the fresh buffer, in <game>
    assert "Round 1" not in text and "Round 2" not in text  # collapsed rounds are not rendered raw
    assert "Round 3" in text and "A7" in text               # fresh buffer — rendered raw


def test_render_with_notes_only_when_buffer_empty():
    m = Memory()
    m.add(_entry(round=1))
    m.set_notes("note text")
    msgs = m.render(None)                  # buffer empty -> notes only (not [])
    assert len(msgs) == 1
    assert msgs[0].content == "<game>Your notes from earlier rounds:</game>\n<you>note text</you>"  # <game> label + <you> notes
    assert "Your rounds since those notes:" not in msgs[0].content  # no buffer -> no label either
    assert "Round 1" not in msgs[0].content


def _trace_entry():
    from src.core.memory import MemoryEntry
    e = MemoryEntry(round=1, my_id="A1", partner_id="A2", transcript=[], my_number=5,
                    my_rationale="risky", partner_number=5, outcome="CC", payoff=3.0,
                    partner_payoff=3.0, my_predicted=4)
    e.my_reflection = "trust holds"
    return e


def test_private_traces_rendered_when_flags_on():
    # all three flags default to True: traces are printed per their templates
    from src.core.config import GameCfg
    from src.core.memory import Memory

    m = Memory()
    m.add(_trace_entry())
    rendered = m.render(None, GameCfg())[0].content
    assert "I predicted A2 would pick 4" in rendered   # {partner}/{my_predicted} substituted
    # rationale and number — one <you> block (as in the JSON reply), rationale before the number
    assert "<you>rationale: risky\nnumber: 5</you>" in rendered
    assert "my takeaway: trust holds" in rendered
    # the answer block comes BEFORE the revealing result: reasoned -> chose -> learned the outcome
    assert rendered.index("rationale: risky") < rendered.index("The choice has been accepted")


def test_each_private_trace_has_its_own_flag():
    # three DIFFERENT flags: turn off one at a time — the rest remain
    from src.core.config import GameCfg
    from src.core.memory import Memory

    def render(**flags):
        m = Memory()
        m.add(_trace_entry())
        return m.render(None, GameCfg(**flags))[0].content

    no_pred = render(show_predicted=False)
    assert "predict" not in no_pred.lower() and "rationale: risky" in no_pred and "my takeaway: trust holds" in no_pred

    no_rat = render(show_rationale=False)
    # rationale disabled -> no answer block, the number is rendered as a separate msg_self line
    assert "rationale:" not in no_rat and "<you>5</you>" in no_rat
    assert "I predicted A2 would pick 4" in no_rat and "my takeaway: trust holds" in no_rat

    no_ref = render(show_reflection=False)
    assert "my takeaway" not in no_ref and "I predicted A2 would pick 4" in no_ref and "rationale: risky" in no_ref

    all_off = render(show_predicted=False, show_rationale=False, show_reflection=False)
    # the close line when rationale=True legitimately contains the word "rationale" — we check
    # specifically for the answer-block marker "rationale:" (with a colon), not the bare word.
    assert all(s not in all_off.lower() for s in ("predict", "rationale:", "takeaway"))
