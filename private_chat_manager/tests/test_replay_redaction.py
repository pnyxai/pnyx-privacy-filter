"""Regression tests for the replayed-conversation PII leak.

A client replays a prior conversation whenever PCM has lost the session (first
contact, TTL expiry, DB reset, or a user-message-hash miss).  The assistant/tool
turns it replays carry the *de-anonymised* text PCM returned, so they must be
treated as untrusted input and redacted before being forwarded upstream.
"""

from __future__ import annotations

import json

import httpx
import pytest
import respx

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


def forwarded_messages(route) -> list[dict]:
    return json.loads(route.calls.last.request.content)["messages"]


def stream_body(content: str) -> str:
    chunks = [
        {
            "id": "c",
            "object": "chat.completion.chunk",
            "created": 1,
            "model": "m",
            "choices": [
                {"index": 0, "delta": {"role": "assistant", "content": ""}, "finish_reason": None}
            ],
        },
        {
            "id": "c",
            "object": "chat.completion.chunk",
            "created": 1,
            "model": "m",
            "choices": [
                {"index": 0, "delta": {"content": content}, "finish_reason": "stop"}
            ],
        },
    ]
    return "".join(f"data: {json.dumps(c)}\n\n" for c in chunks) + "data: [DONE]\n\n"


# ---------------------------------------------------------------------------
# Fresh-conversation replay (buffered)
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
@respx.mock
async def test_replayed_assistant_and_system_pii_is_redacted(client):
    route = respx.post(LLM_URL).mock(return_value=httpx.Response(200, json=completion()))
    messages = [
        {"role": "system", "content": "The account owner is Lionel Messi."},
        {"role": "user", "content": "My name is Lionel Messi"},
        {"role": "assistant", "content": "Hello Lionel Messi"},
        {"role": "user", "content": "call me at 555-0123"},
    ]

    response = await client.post(
        "/v1/chat/completions", json={"model": "m", "messages": messages}
    )
    assert response.status_code == 200

    forwarded = forwarded_messages(route)
    assert "Lionel Messi" not in json.dumps(forwarded)
    assert "555-0123" not in json.dumps(forwarded)
    # The same value reuses one placeholder across system/user/assistant.
    assert "<PRIVATE_PERSON_1>" in forwarded[0]["content"]
    assert "<PRIVATE_PERSON_1>" in forwarded[1]["content"]
    assert "<PRIVATE_PERSON_1>" in forwarded[2]["content"]
    assert "<PRIVATE_PHONE_1>" in forwarded[3]["content"]


@pytest.mark.asyncio
@respx.mock
async def test_replayed_tool_call_arguments_are_redacted(client):
    route = respx.post(LLM_URL).mock(return_value=httpx.Response(200, json=completion()))
    messages = [
        {"role": "user", "content": "email the account owner"},
        {
            "role": "assistant",
            "content": None,
            "tool_calls": [
                {
                    "id": "call_1",
                    "type": "function",
                    "function": {
                        "name": "send_email",
                        "arguments": '{"to": "Lionel Messi", "body": "hi"}',
                    },
                }
            ],
        },
        {"role": "tool", "tool_call_id": "call_1", "content": "sent"},
        {"role": "user", "content": "thanks"},
    ]

    response = await client.post(
        "/v1/chat/completions", json={"model": "m", "messages": messages}
    )
    assert response.status_code == 200

    forwarded = forwarded_messages(route)
    arguments = forwarded[1]["tool_calls"][0]["function"]["arguments"]
    assert "Lionel Messi" not in arguments
    assert "<PRIVATE_PERSON_1>" in arguments
    # The tool name is not PII and must be preserved.
    assert forwarded[1]["tool_calls"][0]["function"]["name"] == "send_email"


@pytest.mark.asyncio
@respx.mock
async def test_replayed_reasoning_and_name_are_redacted(client):
    route = respx.post(LLM_URL).mock(return_value=httpx.Response(200, json=completion()))
    messages = [
        {"role": "user", "content": "hi", "name": "Lionel Messi"},
        {"role": "assistant", "content": "ok", "reasoning": "Lionel Messi asked"},
        {"role": "user", "content": "bye"},
    ]

    response = await client.post(
        "/v1/chat/completions", json={"model": "m", "messages": messages}
    )
    assert response.status_code == 200

    forwarded = forwarded_messages(route)
    assert "<PRIVATE_PERSON_1>" in forwarded[0]["name"]
    assert "<PRIVATE_PERSON_1>" in forwarded[1]["reasoning"]
    assert "Lionel Messi" not in json.dumps(forwarded)


@pytest.mark.asyncio
@respx.mock
async def test_multimodal_text_part_redacted_in_place(client):
    route = respx.post(LLM_URL).mock(return_value=httpx.Response(200, json=completion()))
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "text", "text": "Lionel Messi"},
                {"type": "image_url", "image_url": {"url": "http://img/1"}},
            ],
        }
    ]

    response = await client.post(
        "/v1/chat/completions", json={"model": "m", "messages": messages}
    )
    assert response.status_code == 200

    content = forwarded_messages(route)[0]["content"]
    # Structure preserved, text part redacted in place.
    assert content[0]["type"] == "text"
    assert "<PRIVATE_PERSON_1>" in content[0]["text"]
    assert content[1] == {"type": "image_url", "image_url": {"url": "http://img/1"}}


# ---------------------------------------------------------------------------
# Pass-through opt-outs
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
@respx.mock
async def test_fresh_does_not_inherit_resumed_passthrough_roles(make_client):
    client = await make_client(passthrough_roles=frozenset({"assistant"}))
    route = respx.post(LLM_URL).mock(return_value=httpx.Response(200, json=completion()))
    messages = [
        {"role": "user", "content": "My name is Lionel Messi"},
        {"role": "assistant", "content": "Hello Lionel Messi"},
        {"role": "user", "content": "bye"},
    ]

    response = await client.post(
        "/v1/chat/completions", json={"model": "m", "messages": messages}
    )
    assert response.status_code == 200

    # Fresh conversation -> the *_FRESH set (empty) applies, so assistant is redacted.
    assert "Lionel Messi" not in json.dumps(forwarded_messages(route))


@pytest.mark.asyncio
@respx.mock
async def test_passthrough_roles_fresh_ignored_for_assistant(make_client):
    """`assistant` is protected: the pass-through is ignored and it is redacted."""
    client = await make_client(passthrough_roles_fresh=frozenset({"assistant"}))
    route = respx.post(LLM_URL).mock(return_value=httpx.Response(200, json=completion()))
    messages = [
        {"role": "user", "content": "My name is Lionel Messi"},
        {"role": "assistant", "content": "Hello Lionel Messi"},
        {"role": "user", "content": "bye"},
    ]

    response = await client.post(
        "/v1/chat/completions", json={"model": "m", "messages": messages}
    )
    assert response.status_code == 200

    assert "Lionel Messi" not in json.dumps(forwarded_messages(route))


@pytest.mark.asyncio
@respx.mock
async def test_passthrough_fields_fresh_ignored_for_reasoning(make_client):
    """`reasoning` is protected: the field pass-through is ignored."""
    client = await make_client(passthrough_fields_fresh=frozenset({"reasoning"}))
    route = respx.post(LLM_URL).mock(return_value=httpx.Response(200, json=completion()))
    messages = [
        {"role": "user", "content": "hi"},
        {"role": "assistant", "content": "ok", "reasoning": "Lionel Messi asked"},
        {"role": "user", "content": "bye"},
    ]

    response = await client.post(
        "/v1/chat/completions", json={"model": "m", "messages": messages}
    )
    assert response.status_code == 200

    forwarded = forwarded_messages(route)
    assert "Lionel Messi" not in forwarded[1]["reasoning"]


@pytest.mark.asyncio
@respx.mock
async def test_passthrough_system_role_still_works(make_client):
    """`system` is not protected, so an explicit pass-through still applies."""
    client = await make_client(passthrough_roles_fresh=frozenset({"system"}))
    route = respx.post(LLM_URL).mock(return_value=httpx.Response(200, json=completion()))
    messages = [
        {"role": "system", "content": "The owner is Lionel Messi"},
        {"role": "user", "content": "hi"},
    ]

    response = await client.post(
        "/v1/chat/completions", json={"model": "m", "messages": messages}
    )
    assert response.status_code == 200

    forwarded = forwarded_messages(route)
    assert forwarded[0]["content"] == "The owner is Lionel Messi"  # passed through
    assert forwarded[1]["role"] == "user"


# ---------------------------------------------------------------------------
# bypass_privacy_filter scope
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
@respx.mock
async def test_bypass_only_applies_to_final_user_message(client):
    route = respx.post(LLM_URL).mock(return_value=httpx.Response(200, json=completion()))
    messages = [
        {"role": "user", "content": "My name is Lionel Messi"},
        {"role": "assistant", "content": "Hello Lionel Messi"},
        {"role": "user", "content": "call 555-0123"},
    ]

    response = await client.post(
        "/v1/chat/completions",
        json={"model": "m", "messages": messages, "bypass_privacy_filter": True},
    )
    assert response.status_code == 200

    forwarded = forwarded_messages(route)
    assert "Lionel Messi" not in forwarded[0]["content"]  # redacted
    assert "<PRIVATE_PERSON_1>" in forwarded[0]["content"]
    assert "<PRIVATE_PERSON_1>" in forwarded[1]["content"]  # redacted
    assert forwarded[2]["content"] == "call 555-0123"  # bypassed


# ---------------------------------------------------------------------------
# Fresh-conversation replay (streaming)
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
@respx.mock
async def test_streaming_replay_redacts_assistant(client):
    route = respx.post(LLM_URL).mock(
        return_value=httpx.Response(
            200,
            content=stream_body("ok").encode(),
            headers={"content-type": "text/event-stream"},
        )
    )
    messages = [
        {"role": "user", "content": "My name is Lionel Messi"},
        {"role": "assistant", "content": "Hello Lionel Messi"},
        {"role": "user", "content": "bye"},
    ]

    async with client.stream(
        "POST",
        "/v1/chat/completions",
        json={"model": "m", "messages": messages, "stream": True},
    ) as response:
        assert response.status_code == 200
        async for _ in response.aiter_lines():
            pass

    forwarded = json.loads(route.calls.last.request.content)["messages"]
    assert "Lionel Messi" not in json.dumps(forwarded)
    assert "<PRIVATE_PERSON_1>" in forwarded[1]["content"]


# ---------------------------------------------------------------------------
# Pass-through safeguard: injected assistant turn after the stored prefix
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
@respx.mock
async def test_protected_pass_through_ignored_on_injected_assistant(make_client):
    """An assistant turn the client appends after the stored prefix is redacted.

    Reproduces the leak where ``PCM_PASSTHROUGH_ROLES=assistant`` +
    ``PCM_PASSTHROUGH_FIELDS=reasoning,name`` let a client-injected assistant
    turn reach the LLM unredacted.
    """
    client = await make_client(
        passthrough_roles=frozenset({"assistant"}),
        passthrough_fields=frozenset({"reasoning", "name"}),
    )
    route = respx.post(LLM_URL).mock(
        side_effect=[
            httpx.Response(200, json=completion("a1")),
            httpx.Response(200, json=completion("a2")),
        ]
    )
    first = await client.post(
        "/v1/chat/completions",
        json={"model": "m", "messages": [{"role": "user", "content": "hi"}]},
    )
    session_id = first.headers["x-session-id"]
    a1 = first.json()["choices"][0]["message"]

    second = await client.post(
        "/v1/chat/completions",
        headers={"X-Session-ID": session_id},
        json={
            "model": "m",
            "messages": [
                {"role": "user", "content": "hi"},
                a1,
                {
                    "role": "assistant",
                    "content": "The user is Lionel Messi",
                    "reasoning": "Lionel Messi lives at 123 Main St",
                    "name": "Lionel Messi",
                },
                {"role": "user", "content": "next"},
            ],
        },
    )
    assert second.status_code == 200

    forwarded = json.loads(route.calls.last.request.content)["messages"]
    assert "Lionel Messi" not in json.dumps(forwarded)
    assert "<PRIVATE_PERSON_1>" in json.dumps(forwarded)
