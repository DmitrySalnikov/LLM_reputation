# Reasoning-before-choice + post-game reflection

Date: 2026-06-10
Status: approved (autonomous session — recommended defaults applied)

## Problem

1. In the DECIDE/PREDICT phases the agent answers `{"number": ..., "rationale": ...}`:
   the number is generated *before* the reasoning tokens, so the rationale is a
   post-hoc justification rather than chain-of-thought that informs the choice.
   The demo also prints choices before reasons.
2. After a pairing resolves, agents never explicitly reflect on the outcome.
   The research question (does reputation emerge?) benefits from a per-round
   reflection produced by the agent itself and carried in its memory into
   future rounds.

## Design

### Part 1 — reasoning first

- Swap the JSON field order in `decide_context` / `predict_context`
  (`src/games/prompts.py`) and in `_CORRECTION` (`src/core/agent.py`) to
  `{"rationale": "<short reason>", "number": <0-9>}`. Parsing is key-based and
  order-agnostic, so no parser changes are needed.
- Reorder `narrate_round` in `examples/orchestrator_demo.py`: print each
  agent's rationale (and prediction, when present) *before* the choices line.

### Part 2 — post-game reflection (extra LLM call)

- New phase `PhaseKind.REFLECT` in `src/core/agent.py`. Response contract:
  `{"reflection": "<short reflection>"}`; same 2-retry correction loop;
  fallback `{"reflection": ""}` on parse failure; included in the DEBUG
  LLM-input trace alongside DECIDE/PREDICT.
- New builder `reflect_context(partner, round, feed, my_number,
  partner_number, payoff)` in `src/games/prompts.py`: restates the negotiation
  and the revealed result ("you picked X, partner picked Y, you scored Z") and
  asks what this outcome means for future rounds with this and other partners.
- `GameCfg.reflection: bool = False` (`src/core/config.py`) — off by default so
  existing configs/tests keep their cost and behaviour; enabled in both example
  YAMLs.
- `ReputationPD.play_pairing` (`src/games/reputation_pd.py`): after `resolve`,
  when `cfg.reflection` is set, each agent makes one REFLECT call. The call
  happens *before* `_remember`, so the agent's memory shows only past rounds;
  the current round's facts arrive via the context. Reflection text is stored:
  - in `MemoryEntry.my_reflection` (new optional field) and rendered in the
    diary (`Memory._render_entry`) as a line after the choices/outcome line —
    this is what carries it into the next rounds' LLM input;
  - in `PairingRecord.a_reflection` / `b_reflection` (new optional fields) for
    the observer/demo;
  - reflection usage is summed into `PairingRecord.usage`.
- Privacy matches rationale: a reflection is visible only to its author and in
  the record for analysis, never to the partner.
- Demo prints `... reflects: <text>` lines after the outcome block.

## Out of scope

- No new strategy kind: reflection is strategy-independent and lives in the game.
- No persistence changes (Logger layer does not exist yet).

## Testing

- `tests/core/test_agent.py`: REFLECT parses, retries on bad JSON, falls back.
- `tests/core/test_memory.py`: rendered diary includes the reflection line;
  entries without reflection render unchanged.
- `tests/games/test_prompts.py`: rationale precedes number in decide/predict
  templates; reflect context contains both numbers and the payoff.
- `tests/games/test_play_pairing.py`: reflection off ⇒ no extra LLM calls and
  `my_reflection is None`; reflection on ⇒ stored in memory + record, usage
  counts the two extra calls, partner's reflection does not leak.
- `tests/core/test_config_load.py`: `game.reflection` loads; defaults to False.
