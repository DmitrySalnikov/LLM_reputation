"""Толерантное извлечение JSON-объекта из текста ответа LLM.

Используется агентом (фазы TALK/DECIDE/PREDICT/REFLECT) и LLM-судьёй: модели часто
оборачивают JSON в прозу или код-ограждения, поэтому пробуем несколько кандидатов.
"""

from __future__ import annotations

import json
import re


def extract_json_obj(text: str) -> dict | None:
    """Извлечь первый JSON-объект из текста: сырой / в ```-ограждении / по скобкам.

    Args:
        text: Полный текст ответа модели.

    Returns:
        Словарь, если какой-то кандидат разобрался в JSON-объект, иначе None.
    """
    candidates = [text.strip()]
    fenced = re.search(r"```(?:json)?\s*(.*?)```", text, re.DOTALL | re.IGNORECASE)
    if fenced:
        candidates.append(fenced.group(1).strip())
    block = _first_brace_block(text)
    if block:
        candidates.append(block)
    for candidate in candidates:
        try:
            obj = json.loads(candidate)
        except ValueError:
            continue
        if isinstance(obj, dict):
            return obj
    return None


def _first_brace_block(text: str) -> str | None:
    # Naive balanced-brace scan: good enough for prose-wrapped JSON; does not account
    # for braces inside string values (rare in our outputs).
    start = text.find("{")
    if start == -1:
        return None
    depth = 0
    for i in range(start, len(text)):
        if text[i] == "{":
            depth += 1
        elif text[i] == "}":
            depth -= 1
            if depth == 0:
                return text[start : i + 1]
    return None
