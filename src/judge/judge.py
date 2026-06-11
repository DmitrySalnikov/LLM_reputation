"""LLM-судья: один вызов модели над публичным cheap-talk законченного эпизода.

Вызывается СНАРУЖИ движка (runner / демо) после run_episode — оркестратор не
изменяется. Судья владеет своим провайдером (отдельная модель из JudgeCfg) и
закрывает его сам.
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

_CORRECTION = (
    "Respond with ONLY valid JSON, nothing else: "
    '{"emerged": <true|false>, "explanation": "<short explanation>", '
    '"evidence": ["<message id>", ...]}'
)


async def judge_episode(cfg: JudgeCfg, records: list[PairingRecord]) -> JudgeVerdict:
    """Вынести вердикт о возникновении института репутации в эпизоде.

    Один LLM-вызов: промпт = cfg.prompt с подставленным публичным транскриптом.
    При неразборчивом ответе — одна повторная попытка с поправкой, затем JudgeError.

    Args:
        cfg: Конфигурация судьи (отдельный провайдер + шаблон промпта).
        records: Все записи пар эпизода в порядке наблюдения.

    Returns:
        JudgeVerdict с проверенными ссылками на сообщения-доказательства.

    Raises:
        JudgeError: Ответ не разобрался в валидный вердикт после повтора.
        ProviderError: Сетевые/HTTP-ошибки провайдера (пробрасываются как есть).
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
                Message("user", _CORRECTION),
            ]
        raise JudgeError("судья вернул неразборчивый ответ после повторной попытки")
    finally:
        await provider.aclose()


def _validate_verdict(obj: dict) -> dict | None:
    """Проверить разобранный JSON вердикта; None -> повтор с поправкой."""
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
    """Отфильтровать цитаты судьи: оставить только существующие сообщения."""
    existing = valid_refs(records)
    out: list[MessageRef] = []
    for raw in ids:
        m = _REF_RE.match(raw.strip())
        if m:
            ref = MessageRef(round=int(m.group(1)), pair=int(m.group(2)), turn=int(m.group(3)))
            if (ref.round, ref.pair, ref.turn) in existing:
                out.append(ref)
                continue
        _log.debug("судья сослался на несуществующее сообщение: %r — ссылка отброшена", raw)
    return out
