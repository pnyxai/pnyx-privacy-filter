"""Tests for conversation namespacing of the session key.

Distinct logical conversations that share one client session id (for example
an agent's title generator and its main chat) must not corrupt each other.
"""

from __future__ import annotations

import json

import httpx
import pytest
import respx

from app.privacy_manager import resolve_session_key

LLM_URL = "http://llm.local/v1/chat/completions"


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


TITLE_MESSAGES = [
    {"role": "system", "content": "You are a title generator."},
    {"role": "user", "content": "Generate a title for this conversation:\n"},
    {"role": "user", "content": "hi there"},
]
MAIN_MESSAGES = [
    {"role": "system", "content": "You are opencode."},
    {"role": "user", "content": "hi there"},
]


# ---------------------------------------------------------------------------
# resolve_session_key
# ---------------------------------------------------------------------------


def test_title_and_main_map_to_different_keys():
    title_key = resolve_session_key("ses-1", TITLE_MESSAGES)
    main_key = resolve_session_key("ses-1", MAIN_MESSAGES)
    assert title_key != main_key
    assert title_key.startswith("ses-1::")
    assert main_key.startswith("ses-1::")


def test_key_is_stable_across_turns_of_one_conversation():
    first_turn = [{"role": "user", "content": "first question"}]
    later_turn = [
        {"role": "user", "content": "first question"},
        {"role": "assistant", "content": "answer"},
        {"role": "user", "content": "second question"},
    ]
    assert resolve_session_key("ses-1", first_turn) == resolve_session_key(
        "ses-1", later_turn
    )


def test_already_namespaced_key_is_returned_unchanged():
    key = "ses-1::deadbeef0000"
    assert resolve_session_key(key, MAIN_MESSAGES) == key


def test_missing_base_uses_fingerprint_for_continuity():
    key = resolve_session_key(None, MAIN_MESSAGES)
    assert key.startswith("auto-")

    later_turn = [
        *MAIN_MESSAGES,
        {"role": "assistant", "content": "ok"},
        {"role": "user", "content": "second question"},
    ]
    assert resolve_session_key(None, later_turn) == key
    # A different conversation (different first user message) is separate.
    assert resolve_session_key(None, TITLE_MESSAGES) != key


# ---------------------------------------------------------------------------
# Endpoint behaviour
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
@respx.mock
async def test_title_and_main_do_not_collide(client):
    respx.post(LLM_URL).mock(return_value=httpx.Response(200, json=completion()))
    headers = {"X-Session-Id": "ses-xyz"}

    title = await client.post(
        "/v1/chat/completions",
        json={"model": "m", "messages": TITLE_MESSAGES},
        headers=headers,
    )
    main = await client.post(
        "/v1/chat/completions",
        json={"model": "m", "messages": MAIN_MESSAGES},
        headers=headers,
    )

    assert title.status_code == 200
    assert main.status_code == 200
    assert title.headers["x-session-id"] != main.headers["x-session-id"]

    # Inspecting the client-facing id resolves to the most recent (main)
    # conversation.
    inspect = await client.get("/v1/sessions/ses-xyz")
    assert inspect.status_code == 200
    assert inspect.json()["session_id"] == main.headers["x-session-id"]


@pytest.mark.asyncio
@respx.mock
async def test_main_conversation_continues_across_turns(client):
    route = respx.post(LLM_URL).mock(
        return_value=httpx.Response(200, json=completion())
    )
    headers = {"X-Session-Id": "ses-main"}

    first = await client.post(
        "/v1/chat/completions",
        json={"model": "m", "messages": [{"role": "user", "content": "q1"}]},
        headers=headers,
    )
    assert first.status_code == 200
    session_id = first.headers["x-session-id"]

    second = await client.post(
        "/v1/chat/completions",
        json={
            "model": "m",
            "messages": [
                {"role": "user", "content": "q1"},
                {"role": "assistant", "content": "ok"},
                {"role": "user", "content": "q2"},
            ],
        },
        headers=headers,
    )
    assert second.status_code == 200
    # Same conversation -> same namespaced key and accumulated history.
    assert second.headers["x-session-id"] == session_id
    upstream_messages = json.loads(route.calls.last.request.content)["messages"]
    assert len(upstream_messages) == 3


@pytest.mark.asyncio
@respx.mock
async def test_headerless_conversation_gets_continuity(client):
    route = respx.post(LLM_URL).mock(
        return_value=httpx.Response(200, json=completion())
    )

    first = await client.post(
        "/v1/chat/completions",
        json={"model": "m", "messages": [{"role": "user", "content": "q1"}]},
    )
    assert first.status_code == 200
    session_id = first.headers["x-session-id"]
    assert session_id.startswith("auto-")

    second = await client.post(
        "/v1/chat/completions",
        json={
            "model": "m",
            "messages": [
                {"role": "user", "content": "q1"},
                {"role": "assistant", "content": "ok"},
                {"role": "user", "content": "q2"},
            ],
        },
    )
    assert second.status_code == 200
    assert second.headers["x-session-id"] == session_id
    assert len(json.loads(route.calls.last.request.content)["messages"]) == 3
