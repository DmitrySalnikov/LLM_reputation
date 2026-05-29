from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class ProviderCfg:
    base_url: str
    model: str
    api_key_env: str = ""
    temperature: float = 0.7
    max_tokens: int = 512
    timeout_s: float = 120.0
