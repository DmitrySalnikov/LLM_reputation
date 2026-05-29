from __future__ import annotations

import asyncio
import json

import httpx
import pytest

from src.core.config import ProviderCfg
from src.providers import (
    Message,
    OpenAICompatibleProvider,
    ProviderHTTPError,
    ProviderParseError,
    ProviderUnavailable,
    make_provider,
)


async def _no_sleep(*_a, **_k):
    return None


def _ok_response(content="hi", usage=None):
    body = {"choices": [{"message": {"content": content}, "finish_reason": "stop"}]}
    if usage is not None:
        body["usage"] = usage
    return httpx.Response(200, json=body)


def _provider_with(handler, **kw):
    client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    return OpenAICompatibleProvider("http://x/v1", "k", "m", client=client, **kw)


async def _call(provider, **kw):
    defaults = dict(
        system="s", messages=[Message("user", "q")], temperature=0.0, max_tokens=10
    )
    defaults.update(kw)
    return await provider.complete(**defaults)


async def test_request_shape():
    captured = {}

    def handler(req):
        captured["req"] = req
        captured["body"] = json.loads(req.content)
        return _ok_response(usage={"prompt_tokens": 3, "completion_tokens": 5})

    p = _provider_with(handler)
    await _call(
        p,
        system="SYS",
        messages=[
            Message("user", "hello"),
            Message("assistant", "hi"),
            Message("user", "again"),
        ],
        temperature=0.3,
        max_tokens=64,
    )
    body = captured["body"]
    assert body["model"] == "m"
    assert body["messages"][0] == {"role": "system", "content": "SYS"}
    assert body["messages"][1] == {"role": "user", "content": "hello"}
    assert body["messages"][2] == {"role": "assistant", "content": "hi"}
    assert body["temperature"] == 0.3
    assert body["max_tokens"] == 64
    assert captured["req"].headers["Authorization"] == "Bearer k"
    assert captured["req"].url.path == "/v1/chat/completions"


async def test_parses_text_and_usage():
    p = _provider_with(
        lambda req: _ok_response(
            "answer", usage={"prompt_tokens": 7, "completion_tokens": 11}
        )
    )
    c = await _call(p)
    assert c.text == "answer"
    assert c.prompt_tokens == 7
    assert c.completion_tokens == 11
    assert c.raw["choices"][0]["message"]["content"] == "answer"


async def test_usage_missing_defaults_zero():
    p = _provider_with(lambda req: _ok_response("x"))
    c = await _call(p)
    assert (c.prompt_tokens, c.completion_tokens) == (0, 0)


async def test_null_content_becomes_empty():
    p = _provider_with(lambda req: _ok_response(content=None))
    c = await _call(p)
    assert c.text == ""


async def test_retries_5xx_then_succeeds(monkeypatch):
    monkeypatch.setattr(asyncio, "sleep", _no_sleep)
    calls = {"n": 0}

    def handler(req):
        calls["n"] += 1
        if calls["n"] < 3:
            return httpx.Response(503, text="busy")
        return _ok_response("ok", usage={"prompt_tokens": 1, "completion_tokens": 1})

    p = _provider_with(handler)
    c = await _call(p)
    assert c.text == "ok"
    assert calls["n"] == 3


async def test_retries_transport_error(monkeypatch):
    monkeypatch.setattr(asyncio, "sleep", _no_sleep)
    calls = {"n": 0}

    def handler(req):
        calls["n"] += 1
        if calls["n"] < 2:
            raise httpx.ConnectError("boom", request=req)
        return _ok_response("ok")

    p = _provider_with(handler)
    c = await _call(p)
    assert c.text == "ok"
    assert calls["n"] == 2


async def test_honors_retry_after(monkeypatch):
    slept = []

    async def fake_sleep(s):
        slept.append(s)

    monkeypatch.setattr(asyncio, "sleep", fake_sleep)
    calls = {"n": 0}

    def handler(req):
        calls["n"] += 1
        if calls["n"] == 1:
            return httpx.Response(429, headers={"Retry-After": "2"}, text="slow down")
        return _ok_response("ok")

    p = _provider_with(handler)
    await _call(p)
    assert slept == [2.0]


@pytest.mark.parametrize("code", [400, 401, 404])
async def test_4xx_fail_fast(code):
    calls = {"n": 0}

    def handler(req):
        calls["n"] += 1
        return httpx.Response(code, text="nope")

    p = _provider_with(handler)
    with pytest.raises(ProviderHTTPError) as ei:
        await _call(p)
    assert ei.value.status_code == code
    assert calls["n"] == 1


async def test_exhausts_retries(monkeypatch):
    monkeypatch.setattr(asyncio, "sleep", _no_sleep)
    calls = {"n": 0}

    def handler(req):
        calls["n"] += 1
        return httpx.Response(503, text="busy")

    p = _provider_with(handler)
    with pytest.raises(ProviderUnavailable):
        await _call(p)
    assert calls["n"] == 5


async def test_missing_choices_parse_error():
    p = _provider_with(lambda req: httpx.Response(200, json={"usage": {}}))
    with pytest.raises(ProviderParseError):
        await _call(p)


async def test_aclose_injected_not_closed():
    client = httpx.AsyncClient(transport=httpx.MockTransport(lambda req: _ok_response()))
    p = OpenAICompatibleProvider("http://x/v1", "k", "m", client=client)
    await p.aclose()
    assert not client.is_closed
    await client.aclose()


async def test_aclose_owned_closed():
    p = OpenAICompatibleProvider("http://x/v1", "k", "m")
    await p.aclose()
    assert p._client.is_closed


async def test_make_provider_uses_env_key(monkeypatch):
    monkeypatch.setenv("MY_KEY", "secret123")
    cfg = ProviderCfg(base_url="http://x/v1", model="m", api_key_env="MY_KEY")
    p = make_provider(cfg)
    try:
        assert p._headers["Authorization"] == "Bearer secret123"
    finally:
        await p.aclose()


async def test_make_provider_stub_key_when_unset(monkeypatch):
    monkeypatch.delenv("MISSING_KEY", raising=False)
    cfg = ProviderCfg(base_url="http://x/v1", model="m", api_key_env="MISSING_KEY")
    p = make_provider(cfg)
    try:
        assert p._headers["Authorization"] == "Bearer sk-noauth"
    finally:
        await p.aclose()
