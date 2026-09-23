"""Tests for the per-request downstream override (PCM_LLM_URL_ALLOWLIST).

The ``X-PCM-LLM-URL`` header lets a caller select an alternate downstream base
URL for a single request, but only when it is listed in
``PCM_LLM_URL_ALLOWLIST``.  An empty allowlist disables the feature entirely.
"""

from __future__ import annotations

import json

import httpx
import pytest
import respx

from app.config import Settings
from app.endpoints import normalize_llm_url, resolve_llm_url

DEFAULT_LLM = "http://llm.local/v1/chat/completions"
ALT_LLM = "http://alt.local/v1/chat/completions"


def completion(content: str = "ok") -> dict:
    return {
        "id": "chatcmpl-test",
        "object": "chat.completion",
        "created": 1,
        "model": "upstream",
        "choices": [
            {
                "index": 0,
                "message": {"role": "assistant", "content": content},
                "finish_reason": "stop",
            }
        ],
        "usage": {"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2},
    }


# ---------------------------------------------------------------------------
# Pure helpers
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "raw, expected",
    [
        ("http://h/v1", "http://h"),
        ("http://h/v1/", "http://h"),
        ("http://h/", "http://h"),
        ("  http://h/api/v1  ", "http://h/api"),
    ],
)
def test_normalize_llm_url(raw, expected):
    assert normalize_llm_url(raw) == expected


def test_resolve_llm_url_disabled_when_allowlist_empty():
    assert resolve_llm_url("http://default", frozenset(), "http://evil") == "http://default"
    assert resolve_llm_url("http://default", frozenset(), None) == "http://default"


def test_resolve_llm_url_allowed_and_normalized():
    allow = frozenset({"http://alt"})
    assert resolve_llm_url("http://default", allow, "http://alt/v1") == "http://alt"


def test_resolve_llm_url_rejects_unlisted():
    with pytest.raises(ValueError):
        resolve_llm_url("http://default", frozenset({"http://alt"}), "http://evil")


def test_config_parses_allowlist():
    settings = Settings(
        llm_url="http://default",
        llm_url_allowlist="http://a/v1, http://b/ ,, ",
    )
    assert settings.llm_url_allowlist == frozenset({"http://a", "http://b"})


# ---------------------------------------------------------------------------
# Endpoint behaviour
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
@respx.mock
async def test_allowed_override_routes_upstream(make_client):
    client = await make_client(llm_url_allowlist=frozenset({"http://alt.local"}))
    alt = respx.post(ALT_LLM).mock(return_value=httpx.Response(200, json=completion()))

    response = await client.post(
        "/v1/chat/completions",
        json={"model": "m", "messages": [{"role": "user", "content": "hi"}]},
        headers={"X-PCM-LLM-URL": "http://alt.local"},
    )

    assert response.status_code == 200
    assert alt.called
    # PCM's control header must not leak upstream.
    assert "x-pcm-llm-url" not in alt.calls.last.request.headers


@pytest.mark.asyncio
@respx.mock
async def test_override_normalizes_trailing_v1(make_client):
    client = await make_client(llm_url_allowlist=frozenset({"http://alt.local"}))
    alt = respx.post(ALT_LLM).mock(return_value=httpx.Response(200, json=completion()))

    response = await client.post(
        "/v1/chat/completions",
        json={"model": "m", "messages": [{"role": "user", "content": "hi"}]},
        headers={"X-PCM-LLM-URL": "http://alt.local/v1"},
    )

    assert response.status_code == 200
    assert alt.called


@pytest.mark.asyncio
async def test_unlisted_override_is_rejected(make_client):
    client = await make_client(llm_url_allowlist=frozenset({"http://alt.local"}))

    response = await client.post(
        "/v1/chat/completions",
        json={"model": "m", "messages": [{"role": "user", "content": "hi"}]},
        headers={"X-PCM-LLM-URL": "http://evil.local"},
    )

    assert response.status_code == 403


@pytest.mark.asyncio
@respx.mock
async def test_override_ignored_when_allowlist_empty(client):
    # Default settings: allowlist empty -> the header is ignored.
    route = respx.post(DEFAULT_LLM).mock(
        return_value=httpx.Response(200, json=completion())
    )

    response = await client.post(
        "/v1/chat/completions",
        json={"model": "m", "messages": [{"role": "user", "content": "hi"}]},
        headers={"X-PCM-LLM-URL": "http://evil.local"},
    )

    assert response.status_code == 200
    assert route.called


@pytest.mark.asyncio
@respx.mock
async def test_proxy_honours_override(make_client):
    client = await make_client(llm_url_allowlist=frozenset({"http://alt.local"}))
    alt = respx.get("http://alt.local/v1/models").mock(
        return_value=httpx.Response(200, json={"object": "list", "data": []})
    )

    response = await client.get(
        "/v1/models", headers={"X-PCM-LLM-URL": "http://alt.local"}
    )

    assert response.status_code == 200
    assert alt.called
    assert json.loads('{"object":"list","data":[]}') == response.json()
