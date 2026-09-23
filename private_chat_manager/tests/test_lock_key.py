"""The per-conversation lock must key on the answered-message hash, not hints.

The client session header is only a hint (it may be absent or rotated), so
concurrent representations of the same conversation must share one lock.
"""

from __future__ import annotations

import asyncio

import httpx
import pytest
import respx

import app.main as main
from app.session_lock import _locks

LLM_URL = "http://llm.local/v1/chat/completions"


def _completion(content: str = "ok") -> dict:
    return {
        "id": "chatcmpl-test",
        "object": "chat.completion",
        "created": 1,
        "model": "m",
        "choices": [
            {"index": 0, "message": {"role": "assistant", "content": content}, "finish_reason": "stop"}
        ],
        "usage": {"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2},
    }


@pytest.fixture(autouse=True)
def _clear_locks():
    _locks.clear()
    yield
    _locks.clear()


def _lock_keys():
    return set(_locks)


@pytest.mark.asyncio
@respx.mock
async def test_same_conversation_header_variants_share_lock(app, client, monkeypatch):
    """A headerless and a header-bearing request for one conversation share a lock.

    The second request carries a *different* client header; because the lock
    keys on the answered-message hash it must still serialize with the first.
    """
    respx.post(LLM_URL).mock(return_value=httpx.Response(200, json=_completion()))
    messages = [
        {"role": "user", "content": "u1"},
        {"role": "assistant", "content": "a1"},
        {"role": "user", "content": "u2"},
    ]

    await client.post(
        "/v1/chat/completions",
        headers={"X-Session-ID": "s1"},
        json={"model": "m", "messages": messages},
    )
    await client.post(
        "/v1/chat/completions",
        headers={"X-Session-ID": "rotated-header"},
        json={"model": "m", "messages": messages},
    )

    # Both requests must have used a single hash-keyed lock, not one per header.
    assert len(_lock_keys()) == 1
    assert next(iter(_lock_keys())).startswith("h:")


@pytest.mark.asyncio
@respx.mock
async def test_first_turn_falls_back_to_hash_when_headerless(app, client):
    """With no answered-user prefix and no header, the user hash keys the lock."""
    respx.post(LLM_URL).mock(return_value=httpx.Response(200, json=_completion()))

    await client.post(
        "/v1/chat/completions",
        json={"model": "m", "messages": [{"role": "user", "content": "u1"}]},
    )
    await client.post(
        "/v1/chat/completions",
        json={"model": "m", "messages": [{"role": "user", "content": "u2"}]},
    )

    keys = _lock_keys()
    assert len(keys) == 2
    assert all(key.startswith("h:") for key in keys)
