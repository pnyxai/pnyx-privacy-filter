"""End-to-end tests for the streaming ``/v1/chat/completions`` path.

The downstream LLM is stubbed with ``respx``; the Triton privacy-filter is
stubbed by :class:`tests.conftest.FakeTritonClient`.
"""

from __future__ import annotations

import json

import httpx
import pytest
import respx

LLM_URL = "http://llm.local/v1/chat/completions"


def sse(payload: dict) -> str:
    return "data: " + json.dumps(payload) + "\n\n"


def chunk(delta: dict, finish_reason=None, **extra) -> dict:
    body = {
        "id": "chatcmpl-test",
        "object": "chat.completion.chunk",
        "created": 1,
        "model": "pocket_network",
        "choices": [
            {
                "index": 0,
                "delta": delta,
                "logprobs": None,
                "finish_reason": finish_reason,
            }
        ],
    }
    body.update(extra)
    return body


def stream_body(chunks: list[dict]) -> str:
    return "".join(sse(c) for c in chunks) + "data: [DONE]\n\n"


async def collect_events(client, payload: dict, headers: dict | None = None):
    events: list[dict] = []
    async with client.stream(
        "POST", "/v1/chat/completions", json=payload, headers=headers
    ) as response:
        session_id = response.headers.get("x-session-id")
        content_type = response.headers.get("content-type", "")
        raw_lines: list[str] = []
        async for line in response.aiter_lines():
            if not line.startswith("data:"):
                continue
            payload_str = line[len("data:"):].strip()
            raw_lines.append(payload_str)
            if payload_str == "[DONE]":
                continue
            events.append(json.loads(payload_str))
        return response.status_code, session_id, content_type, events, raw_lines


def joined_content(events: list[dict]) -> str:
    parts: list[str] = []
    for event in events:
        for choice in event.get("choices") or []:
            content = choice.get("delta", {}).get("content")
            if content:
                parts.append(content)
    return "".join(parts)


@pytest.mark.asyncio
@respx.mock
async def test_streaming_reassembles_placeholder_and_persists(client):
    body = stream_body(
        [
            chunk({"role": "assistant", "content": ""}),
            chunk({"content": "You said your name is "}),
            chunk({"content": "<PRIV"}),
            chunk({"content": "ATE_PERSON_"}),
            chunk({"content": "1>."}),
            chunk({}, finish_reason="stop"),
        ]
    )
    route = respx.post(LLM_URL).mock(
        return_value=httpx.Response(
            200, content=body.encode(), headers={"content-type": "text/event-stream"}
        )
    )

    payload = {
        "model": "pocket_network",
        "messages": [{"role": "user", "content": "My name is Lionel Messi."}],
        "stream": True,
    }
    status, session_id, content_type, events, raw_lines = await collect_events(
        client, payload
    )

    assert status == 200
    assert session_id
    assert content_type.startswith("text/event-stream")
    assert joined_content(events) == "You said your name is Lionel Messi."
    assert raw_lines[-1] == "[DONE]"
    # The placeholder was never leaked to the client.
    assert all("PRIVATE_PERSON" not in line for line in raw_lines)
    assert route.called

    # Hidden session history keeps the placeholder; raw history has the value.
    inspect = await client.get(f"/v1/sessions/{session_id}")
    session = inspect.json()
    assert session["hidden_messages"][0]["content"] == "My name is <PRIVATE_PERSON_1>."
    assert session["raw_messages"][0]["content"] == "My name is Lionel Messi."
    assert session["hidden_messages"][-1]["content"] == (
        "You said your name is <PRIVATE_PERSON_1>."
    )
    assert session["raw_messages"][-1]["content"] == (
        "You said your name is Lionel Messi."
    )


@pytest.mark.asyncio
@respx.mock
async def test_streaming_forwards_stream_options_and_temperature(client):
    body = stream_body(
        [
            chunk({"role": "assistant", "content": "hi"}),
            chunk({}, finish_reason="stop"),
        ]
    )
    route = respx.post(LLM_URL).mock(
        return_value=httpx.Response(200, content=body.encode())
    )

    payload = {
        "model": "pocket_network",
        "messages": [{"role": "user", "content": "hello"}],
        "stream": True,
        "stream_options": {"include_usage": True},
        "temperature": 0.3,
    }
    await collect_events(client, payload)

    forwarded = json.loads(route.calls.last.request.content)
    assert forwarded["stream"] is True
    assert forwarded["stream_options"] == {"include_usage": True}
    assert forwarded["temperature"] == 0.3
    # The redacted prompt is what actually reaches the LLM.
    assert forwarded["model"] == "pocket_network"


@pytest.mark.asyncio
@respx.mock
async def test_streaming_reasoning_is_deanonymised(client):
    body = stream_body(
        [
            chunk({"role": "assistant", "content": ""}),
            chunk({"reasoning": "The user is <PRIV"}),
            chunk({"reasoning": "ATE_PERSON_1>."}),
            chunk({"content": "Done."}),
            chunk({}, finish_reason="stop"),
        ]
    )
    respx.post(LLM_URL).mock(return_value=httpx.Response(200, content=body.encode()))

    payload = {
        "model": "pocket_network",
        "messages": [{"role": "user", "content": "My name is Lionel Messi."}],
        "stream": True,
    }
    _, _, _, events, raw_lines = await collect_events(client, payload)

    reasoning_parts = [
        choice.get("delta", {}).get("reasoning")
        for event in events
        for choice in event.get("choices") or []
        if choice.get("delta", {}).get("reasoning")
    ]
    assert "".join(reasoning_parts) == "The user is Lionel Messi."
    assert all("PRIVATE_PERSON" not in line for line in raw_lines)


@pytest.mark.asyncio
@respx.mock
async def test_streaming_usage_chunk_is_passed_through(client):
    body = stream_body(
        [
            chunk({"role": "assistant", "content": "hi"}),
            chunk({}, finish_reason="stop"),
            {
                "id": "chatcmpl-test",
                "object": "chat.completion.chunk",
                "created": 1,
                "model": "pocket_network",
                "choices": [],
                "usage": {"prompt_tokens": 5, "completion_tokens": 1, "total_tokens": 6},
            },
        ]
    )
    respx.post(LLM_URL).mock(return_value=httpx.Response(200, content=body.encode()))

    payload = {
        "model": "pocket_network",
        "messages": [{"role": "user", "content": "hello"}],
        "stream": True,
        "stream_options": {"include_usage": True},
    }
    _, _, _, events, _ = await collect_events(client, payload)
    usage_events = [e for e in events if e.get("usage")]
    assert len(usage_events) == 1
    assert usage_events[0]["usage"]["total_tokens"] == 6


@pytest.mark.asyncio
@respx.mock
async def test_streaming_upstream_error_emits_error_then_done(client):
    respx.post(LLM_URL).mock(return_value=httpx.Response(500, text="boom"))

    payload = {
        "model": "pocket_network",
        "messages": [{"role": "user", "content": "hello"}],
        "stream": True,
    }
    status, session_id, _, events, raw_lines = await collect_events(client, payload)

    assert status == 200
    assert session_id
    assert any("error" in event for event in events)
    assert raw_lines[-1] == "[DONE]"


@pytest.mark.asyncio
@respx.mock
async def test_streaming_unterminated_tag_is_flushed(client):
    body = stream_body(
        [
            chunk({"role": "assistant", "content": "trailing <PRIV"}),
            chunk({}, finish_reason="stop"),
        ]
    )
    respx.post(LLM_URL).mock(return_value=httpx.Response(200, content=body.encode()))

    payload = {
        "model": "pocket_network",
        "messages": [{"role": "user", "content": "My name is Lionel Messi."}],
        "stream": True,
    }
    _, _, _, events, raw_lines = await collect_events(client, payload)
    assert joined_content(events) == "trailing <PRIV"
    assert raw_lines[-1] == "[DONE]"


@pytest.mark.asyncio
@respx.mock
async def test_streaming_never_closed_placeholder_is_replayed_verbatim(client):
    # The LLM opens a placeholder but diverges before closing it.  Everything
    # held must be replayed to the client unchanged, split across deltas.
    deltas = ["<", "PRIVATE", "_PERSON_", "12", " is a good person"]
    body = stream_body(
        [chunk({"role": "assistant", "content": d}) for d in deltas]
        + [chunk({}, finish_reason="stop")]
    )
    respx.post(LLM_URL).mock(return_value=httpx.Response(200, content=body.encode()))

    payload = {
        "model": "pocket_network",
        "messages": [{"role": "user", "content": "My name is Lionel Messi."}],
        "stream": True,
    }
    _, session_id, _, events, raw_lines = await collect_events(client, payload)

    assert joined_content(events) == "".join(deltas)
    assert raw_lines[-1] == "[DONE]"

    # The session records the exact text in both views (nothing was replaced).
    session = (await client.get(f"/v1/sessions/{session_id}")).json()
    assert session["hidden_messages"][-1]["content"] == "".join(deltas)
    assert session["raw_messages"][-1]["content"] == "".join(deltas)



@pytest.mark.asyncio
@respx.mock
async def test_non_streaming_still_works_and_deanonymises_reasoning(client):
    body = {
        "id": "chatcmpl-test",
        "object": "chat.completion",
        "created": 1,
        "model": "pocket_network",
        "choices": [
            {
                "index": 0,
                "message": {
                    "role": "assistant",
                    "content": "Your name is <PRIVATE_PERSON_1>.",
                    "reasoning": "I saw <PRIVATE_PERSON_1>.",
                },
                "finish_reason": "stop",
            }
        ],
        "usage": {"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2},
    }
    route = respx.post(LLM_URL).mock(return_value=httpx.Response(200, json=body))

    payload = {
        "model": "pocket_network",
        "messages": [{"role": "user", "content": "My name is Lionel Messi."}],
        "stream": False,
    }
    response = await client.post("/v1/chat/completions", json=payload)
    assert response.status_code == 200
    data = response.json()
    message = data["choices"][0]["message"]
    assert message["content"] == "Your name is Lionel Messi."
    assert message["reasoning"] == "I saw Lionel Messi."

    forwarded = json.loads(route.calls.last.request.content)
    assert forwarded["stream"] is False


# ---------------------------------------------------------------------------
# Streaming tool calls (vLLM shape)
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
@respx.mock
async def test_streaming_tool_calls_arguments_are_deanonymised(client):
    body = stream_body(
        [
            chunk({"role": "assistant", "content": ""}),
            chunk(
                {
                    "tool_calls": [
                        {
                            "id": "call_1",
                            "type": "function",
                            "index": 0,
                            "function": {"name": "lookup"},
                        }
                    ]
                }
            ),
            chunk(
                {
                    "tool_calls": [
                        {"index": 0, "function": {"arguments": '{"location": "'}}
                    ]
                }
            ),
            chunk({"tool_calls": [{"index": 0, "function": {"arguments": "<PRIV"}}]}),
            chunk(
                {"tool_calls": [{"index": 0, "function": {"arguments": "ATE_PERSON_1>"}}]}
            ),
            chunk(
                {
                    "tool_calls": [
                        {"index": 0, "function": {"arguments": '", "unit": "C"}'}}
                    ]
                }
            ),
            chunk({}, finish_reason="tool_calls"),
        ]
    )
    respx.post(LLM_URL).mock(return_value=httpx.Response(200, content=body.encode()))

    payload = {
        "model": "pocket_network",
        "messages": [{"role": "user", "content": "My name is Lionel Messi."}],
        "stream": True,
    }
    _, session_id, _, events, raw_lines = await collect_events(client, payload)

    # The client-visible arguments are de-anonymised...
    args = ""
    for event in events:
        for choice in event.get("choices") or []:
            for tool_call in choice.get("delta", {}).get("tool_calls") or []:
                function = tool_call.get("function") or {}
                if function.get("arguments"):
                    args += function["arguments"]
    assert args == '{"location": "Lionel Messi", "unit": "C"}'
    assert raw_lines[-1] == "[DONE]"

    # ...while the hidden history keeps the placeholder and raw keeps the value.
    session = (await client.get(f"/v1/sessions/{session_id}")).json()
    hidden_tc = session["hidden_messages"][-1]["tool_calls"][0]
    raw_tc = session["raw_messages"][-1]["tool_calls"][0]
    assert hidden_tc["id"] == "call_1"
    assert hidden_tc["type"] == "function"
    assert hidden_tc["function"]["name"] == "lookup"
    assert hidden_tc["function"]["arguments"] == (
        '{"location": "<PRIVATE_PERSON_1>", "unit": "C"}'
    )
    assert raw_tc["function"]["arguments"] == '{"location": "Lionel Messi", "unit": "C"}'


# ---------------------------------------------------------------------------
# Final-message debug logging
# ---------------------------------------------------------------------------


def _log_payload() -> dict:
    return {
        "model": "pocket_network",
        "messages": [{"role": "user", "content": "My name is Lionel Messi."}],
        "stream": True,
    }


@pytest.mark.asyncio
@respx.mock
async def test_streaming_logs_assembled_response(debug_client, caplog):
    body = stream_body(
        [
            chunk({"role": "assistant", "content": ""}),
            chunk({"reasoning": "He is <PRIV"}),
            chunk({"reasoning": "ATE_PERSON_1>."}),
            chunk({"content": "You are <PRIV"}),
            chunk({"content": "ATE_PERSON_1>."}),
            chunk({}, finish_reason="stop"),
        ]
    )
    respx.post(LLM_URL).mock(return_value=httpx.Response(200, content=body.encode()))

    await collect_events(debug_client, _log_payload())
    text = caplog.text

    assert "stream response body" in text
    assert "stream raw response" in text
    # De-anonymised view has the real value; raw view keeps the placeholder.
    assert "Lionel Messi" in text
    assert "<PRIVATE_PERSON_1>" in text


@pytest.mark.asyncio
@respx.mock
async def test_streaming_response_not_logged_when_event_disabled(
    debug_silent_client, caplog
):
    body = stream_body(
        [
            chunk({"role": "assistant", "content": "hi"}),
            chunk({}, finish_reason="stop"),
        ]
    )
    respx.post(LLM_URL).mock(return_value=httpx.Response(200, content=body.encode()))

    await collect_events(debug_silent_client, _log_payload())

    assert "stream response body" not in caplog.text
    assert "stream raw response" not in caplog.text


# ---------------------------------------------------------------------------
# Buffered vs streaming session-history parity
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
@respx.mock
async def test_persisted_content_matches_between_stream_and_nonstream(client):
    non_stream_body = {
        "id": "chatcmpl-test",
        "object": "chat.completion",
        "created": 1,
        "model": "pocket_network",
        "choices": [
            {
                "index": 0,
                "message": {
                    "role": "assistant",
                    "content": "You said your name is <PRIVATE_PERSON_1>.",
                },
                "finish_reason": "stop",
            }
        ],
        "usage": {"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2},
    }
    stream = stream_body(
        [
            chunk({"role": "assistant", "content": ""}),
            chunk({"content": "You said your name is "}),
            chunk({"content": "<PRIV"}),
            chunk({"content": "ATE_PERSON_1>."}),
            chunk({}, finish_reason="stop"),
        ]
    )
    respx.post(LLM_URL).mock(
        side_effect=[
            httpx.Response(200, json=non_stream_body),
            httpx.Response(200, content=stream.encode()),
        ]
    )

    payload = {
        "model": "pocket_network",
        "messages": [{"role": "user", "content": "My name is Lionel Messi."}],
    }
    buffered = await client.post(
        "/v1/chat/completions",
        json={**payload, "stream": False},
        headers={"X-Session-Id": "parity-buffered"},
    )
    _, streamed_id, _, _, _ = await collect_events(
        client,
        {**payload, "stream": True},
        headers={"X-Session-Id": "parity-streamed"},
    )

    buffered_session = (
        await client.get(f"/v1/sessions/{buffered.headers['x-session-id']}")
    ).json()
    streamed_session = (await client.get(f"/v1/sessions/{streamed_id}")).json()

    assert buffered_session["hidden_messages"][-1] == streamed_session["hidden_messages"][-1]
    assert buffered_session["raw_messages"][-1] == streamed_session["raw_messages"][-1]


@pytest.mark.asyncio
@respx.mock
async def test_persisted_tool_calls_match_between_stream_and_nonstream(client):
    non_stream_body = {
        "id": "chatcmpl-test",
        "object": "chat.completion",
        "created": 1,
        "model": "pocket_network",
        "choices": [
            {
                "index": 0,
                "message": {
                    "role": "assistant",
                    "tool_calls": [
                        {
                            "id": "call_1",
                            "type": "function",
                            "function": {
                                "name": "lookup",
                                "arguments": '{"location": "<PRIVATE_PERSON_1>"}',
                            },
                        }
                    ],
                },
                "finish_reason": "tool_calls",
            }
        ],
        "usage": {"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2},
    }
    stream = stream_body(
        [
            chunk({"role": "assistant", "content": ""}),
            chunk(
                {
                    "tool_calls": [
                        {
                            "id": "call_1",
                            "type": "function",
                            "index": 0,
                            "function": {"name": "lookup"},
                        }
                    ]
                }
            ),
            chunk(
                {
                    "tool_calls": [
                        {
                            "index": 0,
                            "function": {
                                "arguments": '{"location": "<PRIVATE_PERSON_1>"}'
                            },
                        }
                    ]
                }
            ),
            chunk({}, finish_reason="tool_calls"),
        ]
    )
    respx.post(LLM_URL).mock(
        side_effect=[
            httpx.Response(200, json=non_stream_body),
            httpx.Response(200, content=stream.encode()),
        ]
    )

    payload = {
        "model": "pocket_network",
        "messages": [{"role": "user", "content": "My name is Lionel Messi."}],
    }
    buffered = await client.post(
        "/v1/chat/completions",
        json={**payload, "stream": False},
        headers={"X-Session-Id": "parity-buffered"},
    )
    _, streamed_id, _, _, _ = await collect_events(
        client,
        {**payload, "stream": True},
        headers={"X-Session-Id": "parity-streamed"},
    )

    buffered_session = (
        await client.get(f"/v1/sessions/{buffered.headers['x-session-id']}")
    ).json()
    streamed_session = (await client.get(f"/v1/sessions/{streamed_id}")).json()

    assert buffered_session["hidden_messages"][-1] == streamed_session["hidden_messages"][-1]
    assert buffered_session["raw_messages"][-1] == streamed_session["raw_messages"][-1]


@pytest.mark.asyncio
@respx.mock
async def test_stream_persists_session_before_done(settings, monkeypatch):
    """The assistant turn must be committed *before* ``[DONE]`` is emitted.

    A client is allowed to disconnect the instant it sees ``[DONE]``; the
    session must already be durable so the next request sees a coherent
    history.
    """
    from app import privacy_manager as pm
    from app.models import SessionData
    from app.session_store import ensure_schema

    await ensure_schema(settings.db_path)
    session = SessionData(
        session_id="sess-1",
        created_at=0.0,
        updated_at=0.0,
        hidden_messages=[{"role": "user", "content": "<PRIVATE_PERSON_1>"}],
        raw_messages=[{"role": "user", "content": "Lionel Messi"}],
    )
    prepared = pm.PreparedRequest(
        session_id="sess-1",
        session=session,
        placeholder_map={"<PRIVATE_PERSON_1>": "Lionel Messi"},
        llm_payload={
            "model": "pocket_network",
            "messages": session.hidden_messages,
            "stream": True,
        },
        headers={"Content-Type": "application/json"},
    )
    body = stream_body(
        [
            chunk({"role": "assistant", "content": ""}),
            chunk({"content": "You are <PRIVATE_PERSON_1>."}),
            chunk({}, finish_reason="stop"),
        ]
    )
    respx.post(LLM_URL).mock(return_value=httpx.Response(200, content=body.encode()))

    calls: list[str] = []
    original = pm._persist_stream_session

    async def spy(*args, **kwargs):
        calls.append("persist")
        return await original(*args, **kwargs)

    monkeypatch.setattr(pm, "_persist_stream_session", spy)

    generator = pm.handle_stream_request(prepared, settings)
    saw_done = False
    async for event in generator:
        if "[DONE]" in event:
            saw_done = True
            # The session was already written before we were handed [DONE].
            assert calls == ["persist"], "session must be persisted before [DONE]"
            break
    await generator.aclose()

    assert saw_done
    # Persisted exactly once (the finally fallback must not repeat it).
    assert calls == ["persist"]


@pytest.mark.asyncio
@respx.mock
async def test_residual_events_keep_reasoning_content_key(client):
    # No terminal finish chunk, so the drained partial tag is emitted through
    # _residual_events at end of stream and must keep the original key.
    body = stream_body(
        [
            chunk({"role": "assistant", "content": ""}),
            chunk({"reasoning_content": "think <SEC"}),
        ]
    )
    respx.post(LLM_URL).mock(return_value=httpx.Response(200, content=body.encode()))

    payload = {
        "model": "pocket_network",
        "messages": [{"role": "user", "content": "hi"}],
        "stream": True,
    }
    _, _, _, events, _ = await collect_events(client, payload)

    deltas = [c.get("delta", {}) for e in events for c in e.get("choices") or []]
    reasoning_content = "".join(d.get("reasoning_content", "") for d in deltas)
    assert reasoning_content == "think <SEC"
    assert not any(d.get("reasoning") for d in deltas)


@pytest.mark.asyncio
@respx.mock
async def test_streaming_forwards_model_and_injects_session_header(make_client):
    client = await make_client(llm_session_header="x-opencode-session")
    body = stream_body(
        [
            chunk({"role": "assistant", "content": "hi"}),
            chunk({}, finish_reason="stop"),
        ]
    )
    route = respx.post(LLM_URL).mock(
        return_value=httpx.Response(
            200, content=body.encode(), headers={"content-type": "text/event-stream"}
        )
    )
    payload = {
        "model": "minimax-m3",
        "messages": [{"role": "user", "content": "hi"}],
        "stream": True,
    }
    status, session_id, _, _, _ = await collect_events(
        client, payload, headers={"X-Session-Id": "sess-stream"}
    )

    assert status == 200
    assert session_id == "sess-stream"
    upstream = route.calls.last.request
    assert json.loads(upstream.content)["model"] == "minimax-m3"
    assert upstream.headers.get("x-opencode-session") == session_id


@pytest.mark.asyncio
@respx.mock
async def test_streaming_preserves_client_session_header_name(client):
    body = stream_body(
        [
            chunk({"role": "assistant", "content": "hi"}),
            chunk({}, finish_reason="stop"),
        ]
    )
    respx.post(LLM_URL).mock(
        return_value=httpx.Response(
            200, content=body.encode(), headers={"content-type": "text/event-stream"}
        )
    )
    payload = {
        "model": "minimax-m3",
        "messages": [{"role": "user", "content": "hi"}],
        "stream": True,
    }
    async with client.stream(
        "POST",
        "/v1/chat/completions",
        json=payload,
        headers={"x-session-affinity": "aff-s"},
    ) as response:
        assert response.headers["x-session-affinity"] == "aff-s"
        assert response.headers["x-session-id"] == "aff-s"
        async for _ in response.aiter_lines():
            pass


@pytest.mark.asyncio
@respx.mock
async def test_reasoning_only_stream_persists_user_turn(client):
    """A reasoning-only assistant turn must still persist the session.

    Some models (e.g. a reasoning model behind vLLM) return only a reasoning
    channel and no ``content``.  Reasoning is never persisted, but the user turn
    must be saved so the conversation can continue instead of vanishing.
    """
    body = stream_body(
        [
            chunk({"role": "assistant", "content": ""}),
            chunk({"reasoning": "Let me think about <PRIVATE_PERSON_1>."}),
            chunk({}, finish_reason="stop"),
        ]
    )
    respx.post(LLM_URL).mock(
        return_value=httpx.Response(
            200, content=body.encode(), headers={"content-type": "text/event-stream"}
        )
    )
    status, session_id, _, _, _ = await collect_events(
        client,
        {
            "model": "m",
            "messages": [{"role": "user", "content": "My name is Lionel Messi"}],
            "stream": True,
        },
    )
    assert status == 200

    inspect = await client.get(f"/v1/sessions/{session_id}")
    assert inspect.status_code == 200
    session = inspect.json()
    assert session["hidden_messages"][0]["role"] == "user"
    # Reasoning is not persisted as an assistant message.
    assert not any(m.get("role") == "assistant" for m in session["hidden_messages"])




