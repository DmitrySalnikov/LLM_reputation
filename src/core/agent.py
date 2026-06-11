from __future__ import annotations

import logging
import random
from dataclasses import dataclass
from enum import Enum

from src.core.config import ProviderCfg
from src.core.jsonextract import extract_json_obj
from src.core.memory import Memory
from src.providers.base import LLMProvider, Message

_MAX_PARSE_RETRIES = 2

_log = logging.getLogger(__name__)


class PhaseKind(Enum):
    TALK = "talk"
    DECIDE = "decide"
    PREDICT = "predict"
    REFLECT = "reflect"


@dataclass(frozen=True)
class Phase:
    kind: PhaseKind
    context: str          # rendered situation + output instruction (becomes a user message)
    rules: str = ""       # static game rules; the agent puts them in `system` after the persona


@dataclass(frozen=True)
class ActResult:
    public_text: str | None     # TALK -> message; DECIDE -> None
    data: dict                  # TALK -> {message, ready}; DECIDE -> {number, rationale}
    usage: tuple[int, int]      # (prompt_tokens, completion_tokens), summed over all attempts


@dataclass(frozen=True)
class AgentSetup:
    persona: str
    provider_cfg: ProviderCfg


_CORRECTION = {
    PhaseKind.DECIDE: (
        "Respond with ONLY valid JSON, nothing else: "
        '{"rationale": "<short reason>", "number": <integer 0-9>}'
    ),
    PhaseKind.TALK: (
        "Respond with ONLY valid JSON, nothing else: "
        '{"message": "<your message>", "ready": <true|false>}'
    ),
    PhaseKind.PREDICT: (
        "Respond with ONLY valid JSON, nothing else: "
        '{"rationale": "<short reason>", "number": <integer 0-9>}'
    ),
    PhaseKind.REFLECT: (
        "Respond with ONLY valid JSON, nothing else: "
        '{"reflection": "<short reflection>"}'
    ),
}


class Agent:
    def __init__(
        self,
        id: str,
        setup: AgentSetup,
        provider: LLMProvider,
        *,
        context_window: int | None = None,
    ):
        self.id = id
        self.setup = setup
        self.provider = provider
        self.memory = Memory()
        self.score = 0.0
        self.parse_failures = 0
        self._window = context_window

    def system_prompt(self, rules: str = "") -> str:
        system = f"You are agent {self.id}.\n\n{self.setup.persona}"
        if rules:
            system = f"{system}\n\n{rules}"
        return system

    async def act(self, phase: Phase) -> ActResult:
        system = self.system_prompt(phase.rules)
        base = self.memory.render(self._window) + [Message("user", phase.context)]
        cfg = self.setup.provider_cfg

        prompt_toks = 0
        comp_toks = 0
        correction: str | None = None
        for attempt in range(1, _MAX_PARSE_RETRIES + 2):
            messages = base if correction is None else base + [Message("user", correction)]
            if phase.kind is not PhaseKind.TALK and _log.isEnabledFor(logging.DEBUG):
                _log.debug(_render_trace(self.id, phase.kind, attempt, system, messages))
            comp = await self.provider.complete(
                system=system,
                messages=messages,
                temperature=cfg.temperature,
                max_tokens=cfg.max_tokens,
            )
            prompt_toks += comp.prompt_tokens
            comp_toks += comp.completion_tokens
            data = _parse(phase.kind, comp.text)
            if data is not None:
                return _result(phase.kind, data, (prompt_toks, comp_toks))
            correction = _CORRECTION[phase.kind]

        self.parse_failures += 1
        return _result(phase.kind, _fallback(phase.kind), (prompt_toks, comp_toks))


def _result(kind: PhaseKind, data: dict, usage: tuple[int, int]) -> ActResult:
    public = data["message"] if kind is PhaseKind.TALK else None
    return ActResult(public_text=public, data=data, usage=usage)


def _parse(kind: PhaseKind, text: str) -> dict | None:
    obj = extract_json_obj(text)
    if obj is None:
        return None
    if kind in (PhaseKind.DECIDE, PhaseKind.PREDICT):
        return _validate_decide(obj)
    if kind is PhaseKind.TALK:
        return _validate_talk(obj)
    if kind is PhaseKind.REFLECT:
        return _validate_reflect(obj)
    return None


def _validate_decide(obj: dict) -> dict | None:
    n = obj.get("number")
    if isinstance(n, bool):  # bool is a subclass of int — reject explicitly
        return None
    if not isinstance(n, int):
        try:
            n = int(str(n).strip())
        except (TypeError, ValueError):
            return None
    if not (0 <= n <= 9):
        return None
    rationale = obj.get("rationale", "")
    if not isinstance(rationale, str):
        rationale = str(rationale)
    return {"number": n, "rationale": rationale}


def _validate_talk(obj: dict) -> dict | None:
    message = obj.get("message")
    if message is None:  # message is required; trigger a retry
        return None
    if not isinstance(message, str):
        message = str(message)
    return {"message": message, "ready": _coerce_bool(obj.get("ready"))}


def _validate_reflect(obj: dict) -> dict | None:
    reflection = obj.get("reflection")
    if reflection is None:  # ключ обязателен; иначе повтор с поправкой
        return None
    if not isinstance(reflection, str):
        reflection = str(reflection)
    return {"reflection": reflection}


def _coerce_bool(value) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return bool(value)
    if isinstance(value, str):
        s = value.strip().lower()
        if s in ("true", "yes", "y", "1"):
            return True
    return False  # missing / unknown -> default False (keep talking)


def _fallback(kind: PhaseKind) -> dict:
    if kind in (PhaseKind.DECIDE, PhaseKind.PREDICT):
        return {"number": random.randint(0, 9), "rationale": "(unparsed)"}
    if kind is PhaseKind.REFLECT:
        return {"reflection": ""}
    return {"message": "", "ready": True}


def _render_trace(agent_id: str, kind: PhaseKind, attempt: int,
                  system: str, messages: list[Message]) -> str:
    """Отрендерить точный вход LLM (system + все сообщения) для записи в лог.

    Args:
        agent_id: Идентификатор агента, делающего запрос.
        kind: Фаза запроса (DECIDE, PREDICT или REFLECT).
        attempt: Номер попытки запроса (1..3, повторы из-за ошибок парсинга).
        system: Полный системный промпт (персона + правила).
        messages: Сообщения запроса (дневник памяти, контекст фазы, поправка).

    Returns:
        Многострочный текст записи лога.
    """
    parts = [
        f"LLM-вход: агент {agent_id}, фаза {kind.value}, попытка {attempt}",
        f"--- system ---\n{system}",
    ]
    parts += [f"--- {m.role} ---\n{m.content}" for m in messages]
    return "\n".join(parts)
