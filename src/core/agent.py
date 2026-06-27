from __future__ import annotations

import logging
import re
from dataclasses import dataclass
from enum import Enum

from src.core.config import GameCfg, ProviderCfg
from src.core.jsonextract import extract_json_obj
from src.core.memory import Memory
from src.providers.base import LLMProvider, Message, ProviderError

_MAX_PARSE_RETRIES = 2

_log = logging.getLogger(__name__)

# Стык двух соседних <game>-блоков: закрывающий </game>, за которым через один лишь
# пробельный промежуток идёт открывающий <game>. Склеиваем такие блоки в один (например,
# строку результата раунда и открытие следующего раунда), сохраняя сам разделитель.
_GAME_SEAM = re.compile(r"</game>(\s*)<game>")


def _merge_game_blocks(text: str) -> str:
    """Убрать стыки </game>…<game> между смежными блоками — получается один транскрипт."""
    return _GAME_SEAM.sub(r"\1", text)


class ActParseError(Exception):
    """The agent could not get a valid phase response after all parse-retries.

    No fallback / substitution: the pairing is aborted (finished=0) and the episode stops.
    Carries the failed call log (`calls`) so the pairing can persist it before aborting,
    exactly like a ProviderError.
    """

    calls: tuple = ()
    agent_id: str | None = None
    phase: str | None = None


class PhaseKind(Enum):
    TALK = "talk"
    DECIDE = "decide"
    PREDICT = "predict"
    REFLECT = "reflect"
    NOTE = "note"


@dataclass(frozen=True)
class Phase:
    kind: PhaseKind
    context: str          # rendered situation + output instruction (becomes a user message)
    game_cfg: GameCfg | None = None  # шаблоны транскрипта истории + payoff'ы для подстановки в system; едет от игры


@dataclass(frozen=True)
class LLMCall:
    """Сырой L2-лог одного `provider.complete()` (включая парс-ретраи).

    Самодостаточная запись: всё, кроме `round_idx`/`pair_idx`/`call_idx` (их добавляет
    `Storage.observe`). `turn_idx` проставляет игра для фазы TALK.

    Attributes:
        agent_id: Кто вызывал.
        phase: Фаза (talk/decide/predict/reflect).
        attempt: Парс-попытка Agent.act (1..3).
        http_attempt: Сетевой ретрай внутри одного complete() (1..5).
        status: ok | parse_error | bad_json | bad_shape | http_error | server_error | network.
        status_code: HTTP-код попытки (None при сетевой ошибке).
        request: Дословный отправленный payload (None, если провайдер его не отдал).
        response: Извлечённый текст ответа (только на финальной ok-попытке).
        response_raw: Дословное тело ответа строкой (resp.text, вкл. тело 5xx).
        error: Сообщение сбоя.
        prompt_tokens: Токены промпта (на финальной ok-попытке).
        completion_tokens: Токены ответа (на финальной ok-попытке).
        turn_idx: Индекс реплики для TALK; None для остальных фаз.
    """

    agent_id: str
    phase: str
    attempt: int
    http_attempt: int
    status: str
    status_code: int | None
    request: dict | None
    response: str | None
    response_raw: str | None
    error: str | None
    prompt_tokens: int
    completion_tokens: int
    turn_idx: int | None = None


@dataclass(frozen=True)
class ActResult:
    public_text: str | None     # TALK -> message; DECIDE -> None
    data: dict                  # TALK -> {message, ready}; DECIDE -> {number, rationale}
    usage: tuple[int, int]      # (prompt_tokens, completion_tokens), summed over all attempts
    calls: tuple[LLMCall, ...] = ()   # L2-лог всех попыток этого act()


@dataclass(frozen=True)
class AgentSetup:
    system_prompt: str        # полный system агента (ОДНА строка-шаблон); {id} и payoff'ы подставит Agent.system_prompt
    provider_cfg: ProviderCfg
    # Стратегия игры этого агента — простые строки (core не импортирует strategy; объект
    # стратегии собирается на уровне игры, см. ReputationPD._strategy_for).
    play_strategy: str = "direct"        # "direct" | "prediction"
    prediction_mapping: str = "match"    # отображение predict->выбор (только для prediction)


# Текст поправок на ретрае живёт в конфиге (GameCfg.*_correction), не зашит здесь. Для фаз,
# собранных без game_cfg (юнит-тесты), берём дефолтный GameCfg(). DECIDE/PREDICT выбирают
# bare/rationale-вариант по тому же флагу rationale, что и сам промпт фазы — поэтому в bare-режиме
# поправка больше не требует rationale (прежний _CORRECTION требовал, противореча промпту).
_DEFAULT_GAME_CFG = GameCfg()


def _correction(game_cfg: "GameCfg | None", kind: PhaseKind) -> str:
    cfg = game_cfg if game_cfg is not None else _DEFAULT_GAME_CFG
    if kind is PhaseKind.DECIDE:
        return cfg.decide_correction if cfg.rationale else cfg.decide_correction_bare
    if kind is PhaseKind.PREDICT:
        return cfg.predict_correction if cfg.rationale else cfg.predict_correction_bare
    if kind is PhaseKind.TALK:
        return cfg.talk_correction
    if kind is PhaseKind.REFLECT:
        return cfg.reflect_correction
    return cfg.note_correction  # PhaseKind.NOTE


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

    def system_prompt(self, game_cfg: "GameCfg | None" = None) -> str:
        """Полный system агента: шаблон self.setup.system_prompt с подставленными {id} и payoff'ами.

        Прежней склейки (identity + persona + rules) нет — system задаётся одной строкой.
        {id} подставляется всегда; payoff-параметры {R}/{T}/{P}/{S}/{max_talk_turns} — когда
        известен game_cfg (для всех боевых фаз он едет в Phase.game_cfg)."""
        system = self.setup.system_prompt.replace("{id}", self.id)
        if game_cfg is not None:
            p = game_cfg.payoffs
            system = (system
                      .replace("{R}", f"{p.R:g}").replace("{T}", f"{p.T:g}")
                      .replace("{P}", f"{p.P:g}").replace("{S}", f"{p.S:g}")
                      .replace("{max_talk_turns}", str(game_cfg.max_talk_turns)))
        return system

    async def act(self, phase: Phase) -> ActResult:
        system = self.system_prompt(phase.game_cfg)
        # NOTE сворачивает память в заметки — для этого видит её целиком (без окна),
        # чтобы ничего не потерять при консолидации; остальные фазы — с окном.
        window = None if phase.kind is PhaseKind.NOTE else self._window
        diary = self.memory.render(window, phase.game_cfg)  # [] или [user-сообщение с транскриптом истории]
        history = f"{diary[0].content}\n\n" if diary else ""
        cfg = self.setup.provider_cfg

        prompt_toks = 0
        comp_toks = 0
        calls: list[LLMCall] = []
        correction: str | None = None
        for attempt in range(1, _MAX_PARSE_RETRIES + 2):
            # одно user-сообщение: дневник памяти + контекст фазы (+ поправка на ретрае парсинга)
            content = history + phase.context
            if correction is not None:
                content = f"{content}\n\n{correction}"
            content = _merge_game_blocks(content)   # склеить стыки </game>…<game>
            messages = [Message("user", content)]
            if phase.kind is not PhaseKind.TALK and _log.isEnabledFor(logging.DEBUG):
                _log.debug(_render_trace(self.id, phase.kind, attempt, system, messages))
            try:
                comp = await self.provider.complete(
                    system=system,
                    messages=messages,
                    temperature=cfg.temperature,
                    max_tokens=cfg.max_tokens,
                )
            except ProviderError as e:
                # a failed call is also L2-log rows; attach context to the exception, re-raise
                calls.extend(_calls_from_attempts(self.id, phase.kind, attempt, e.attempts))
                e.agent_id, e.phase, e.attempt = self.id, phase.kind.value, attempt
                e.calls = tuple(calls)
                raise
            prompt_toks += comp.prompt_tokens
            comp_toks += comp.completion_tokens
            data = _parse(phase.kind, comp.text)
            calls.extend(_calls_from_attempts(
                self.id, phase.kind, attempt, comp.attempts, parsed=data is not None))
            if data is not None:
                return _result(phase.kind, data, (prompt_toks, comp_toks), tuple(calls))
            correction = _correction(phase.game_cfg, phase.kind)

        # all parse-retries exhausted: no fallback — abort the pairing (finished=0)
        self.parse_failures += 1
        err = ActParseError(
            f"{self.id} {phase.kind.value}: no valid JSON after {_MAX_PARSE_RETRIES + 1} attempts")
        err.agent_id, err.phase = self.id, phase.kind.value
        err.calls = tuple(calls)
        raise err


def _calls_from_attempts(agent_id, kind, attempt, attempts, *, parsed: bool | None = None):
    """Развернуть HTTP-попытки одного complete() в строки LLMCall (по одной на попытку).

    На финальной удачной по HTTP попытке статус `ok` меняем на `parse_error`, если фазовый
    валидатор отверг текст (`parsed=False`). Для сбойного complete() `parsed` не задаётся.
    """
    out = []
    n = len(attempts)
    for i, at in enumerate(attempts, start=1):
        status = at.status
        if parsed is not None and i == n and at.status == "ok" and not parsed:
            status = "parse_error"
        out.append(LLMCall(
            agent_id=agent_id, phase=kind.value, attempt=attempt, http_attempt=i,
            status=status, status_code=at.status_code, request=at.request,
            response=at.response, response_raw=at.response_raw, error=at.error,
            prompt_tokens=at.prompt_tokens, completion_tokens=at.completion_tokens,
        ))
    return out


def _result(kind: PhaseKind, data: dict, usage: tuple[int, int],
            calls: tuple[LLMCall, ...] = ()) -> ActResult:
    public = data["message"] if kind is PhaseKind.TALK else None
    return ActResult(public_text=public, data=data, usage=usage, calls=calls)


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
    if kind is PhaseKind.NOTE:
        return _validate_notes(obj)
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
    # Агент-facing ключ — "finish" (закрыть чат); внутри по-прежнему храним как "ready".
    return {"message": message, "ready": _coerce_bool(obj.get("finish"))}


def _validate_reflect(obj: dict) -> dict | None:
    reflection = obj.get("reflection")
    if reflection is None:  # ключ обязателен; иначе повтор с поправкой
        return None
    if not isinstance(reflection, str):
        reflection = str(reflection)
    return {"reflection": reflection}


def _validate_notes(obj: dict) -> dict | None:
    notes = obj.get("notes")
    if notes is None:  # ключ обязателен; иначе повтор с поправкой
        return None
    if not isinstance(notes, str):
        notes = str(notes)
    return {"notes": notes}


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


def _render_trace(agent_id: str, kind: PhaseKind, attempt: int,
                  system: str, messages: list[Message]) -> str:
    """Отрендерить точный вход LLM (system + все сообщения) для записи в лог.

    Args:
        agent_id: Идентификатор агента, делающего запрос.
        kind: Фаза запроса (DECIDE, PREDICT или REFLECT).
        attempt: Номер попытки запроса (1..3, повторы из-за ошибок парсинга).
        system: Полный системный промпт (персона + правила).
        messages: Сообщения запроса — одно user-сообщение (дневник памяти + контекст
            фазы + поправка), склеенное в Agent.act.

    Returns:
        Многострочный текст записи лога.
    """
    parts = [
        f"LLM-вход: агент {agent_id}, фаза {kind.value}, попытка {attempt}",
        f"--- system ---\n{system}",
    ]
    parts += [f"--- {m.role} ---\n{m.content}" for m in messages]
    return "\n".join(parts)
