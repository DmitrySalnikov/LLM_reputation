"""LLM judge: a single model call over the public cheap-talk of a finished episode.

Called OUTSIDE the engine (runner / demo) after run_episode — the orchestrator is
not modified. The judge owns its provider (a separate model from JudgeCfg) and
closes it itself.
"""

from __future__ import annotations

import logging
import re

from src.core.config import JudgeCfg
from src.core.jsonextract import extract_json_obj
from src.games.base import PairingRecord
from src.judge.base import JudgeError, JudgeVerdict, MessageRef
from src.judge.transcript import render_transcript, valid_refs
from src.providers.base import Message
from src.providers.openai_compat import make_provider

_log = logging.getLogger(__name__)

_REF_RE = re.compile(r"^r(\d+)\.p(\d+)\.t(\d+)$")


async def judge_episode(cfg: JudgeCfg, records: list[PairingRecord]) -> JudgeVerdict:
    """Produce a verdict on whether an institution of reputation emerged in the episode.

    A single LLM call: prompt = cfg.prompt with the public transcript substituted in.
    On an unparsable response — one retry with a correction, then JudgeError.

    Args:
        cfg: Judge configuration (separate provider + prompt template).
        records: All pairing records of the episode in observation order.

    Returns:
        JudgeVerdict with validated references to evidence messages.

    Raises:
        JudgeError: The response did not parse into a valid verdict after a retry.
        ProviderError: Network/HTTP errors from the provider (propagated as-is).
    """
    prompt = cfg.prompt.replace("{transcript}", render_transcript(records))
    provider = make_provider(cfg.provider)
    pcfg = cfg.provider
    try:
        messages = [Message("user", prompt)]
        for _attempt in range(2):
            comp = await provider.complete(
                system="",
                messages=messages,
                temperature=pcfg.temperature,
                max_tokens=pcfg.max_tokens,
            )
            obj = extract_json_obj(comp.text)
            data = _validate_verdict(obj) if obj is not None else None
            if data is not None:
                return JudgeVerdict(
                    emerged=data["emerged"],
                    explanation=data["explanation"],
                    evidence=_validate_evidence(data["evidence"], records),
                )
            messages = [
                Message("user", prompt),
                Message("assistant", comp.text),
                Message("user", cfg.correction),
            ]
        raise JudgeError("the judge returned an unparsable response after a retry")
    finally:
        await provider.aclose()


def _validate_verdict(obj: dict) -> dict | None:
    """Validate the parsed verdict JSON; None -> retry with a correction."""
    emerged = obj.get("emerged")
    if not isinstance(emerged, bool):
        return None
    explanation = obj.get("explanation", "")
    if not isinstance(explanation, str):
        explanation = str(explanation)
    evidence = obj.get("evidence", [])
    if not isinstance(evidence, list):
        return None
    return {"emerged": emerged, "explanation": explanation,
            "evidence": [str(e) for e in evidence]}


def _validate_evidence(ids: list[str], records: list[PairingRecord]) -> list[MessageRef]:
    """Filter the judge's citations: keep only messages that actually exist."""
    existing = valid_refs(records)
    out: list[MessageRef] = []
    for raw in ids:
        m = _REF_RE.match(raw.strip())
        if m:
            ref = MessageRef(round=int(m.group(1)), pair=int(m.group(2)), turn=int(m.group(3)))
            if (ref.round, ref.pair, ref.turn) in existing:
                out.append(ref)
                continue
        _log.debug("judge referenced a non-existent message: %r — reference dropped", raw)
    return out
