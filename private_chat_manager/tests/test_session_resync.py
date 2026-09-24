"""Session resynchronisation after a one-off harness content change.

A harness can change its system prompt (and/or rewrite the opening user message)
between turns — e.g. Hermes reports the provider as ``custom:pnyx_pf`` on the
turn where the model changes, then normalises to ``custom``.  On the *first*
turn the content-addressed identity therefore differs from every later replay,
so no stored session shares a prefix.

Two guarantees are required:

* a ``user_hash`` match with **no** shared conversation content (only the system
  prompt, or nothing) must not own the request — a new root is started instead
  of forking an unrelated side-channel (the title generator) at cursor 0;
* the session created at that divergence must be discoverable by the *answered
  prefix* the next request computes, so a tool continuation (whose prefix lags
  one user message behind the final hash) re-syncs instead of forking again.
"""

from __future__ import annotations

import time

import aiosqlite
import httpx
import pytest
import respx

from app.identity import compute_user_hash
from app.models import SessionData
from app.privacy_manager import resolve_session
from app.session_store import ensure_schema, save_session

LLM_URL = "http://llm.local/v1/chat/completions"


def completion(content: str = "ok") -> dict:
    return {
        "id": "chatcmpl-test",
        "object": "chat.completion",
        "created": 1,
        "model": "m",
        "choices": [
            {
                "index": 0,
                "message": {"role": "assistant", "content": content},
                "finish_reason": "stop",
            }
        ],
        "usage": {"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2},
    }


TOOL_CALLS = [
    {
        "id": "call_1",
        "type": "function",
        "function": {"name": "read_file", "arguments": '{"path":"main.tex"}'},
    }
]


def tool_call_completion() -> dict:
    return {
        "id": "chatcmpl-tool",
        "object": "chat.completion",
        "created": 1,
        "model": "m",
        "choices": [
            {
                "index": 0,
                "message": {
                    "role": "assistant",
                    "content": None,
                    "tool_calls": TOOL_CALLS,
                },
                "finish_reason": "tool_calls",
            }
        ],
        "usage": {"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2},
    }


# ---------------------------------------------------------------------------
# Unit: a candidate must share conversation content to own a request
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_header_match_with_only_system_shared_starts_new(settings):
    await ensure_schema(settings.db_path)
    system = {"role": "system", "content": "sys"}
    history = [system, {"role": "user", "content": "one"}, {"role": "assistant", "content": "a"}]
    async with aiosqlite.connect(settings.db_path) as conn:
        await conn.execute("PRAGMA journal_mode=WAL")
        await save_session(
            conn,
            SessionData(
                session_id="s1",
                created_at=time.time(),
                updated_at=time.time(),
                client_x_session_header="ses",
                raw_messages=list(history),
                hidden_messages=list(history),
                user_hash=compute_user_hash([{"role": "user", "content": "one"}]),
            ),
        )
        # Same header, same system prompt, but a completely different
        # conversation: the shared run is only the system message.
        picked = await resolve_session(
            conn,
            "ses",
            [
                system,
                {"role": "user", "content": "two"},
                {"role": "assistant", "content": "b"},
                {"role": "user", "content": "three"},
            ],
            settings,
        )

    assert picked.session_id != "s1"


@pytest.mark.asyncio
async def test_header_match_with_shared_conversation_is_selected(settings):
    await ensure_schema(settings.db_path)
    system = {"role": "system", "content": "sys"}
    history = [system, {"role": "user", "content": "one"}, {"role": "assistant", "content": "a"}]
    async with aiosqlite.connect(settings.db_path) as conn:
        await conn.execute("PRAGMA journal_mode=WAL")
        await save_session(
            conn,
            SessionData(
                session_id="s1",
                created_at=time.time(),
                updated_at=time.time(),
                client_x_session_header="ses",
                raw_messages=list(history),
                hidden_messages=list(history),
                user_hash=compute_user_hash([{"role": "user", "content": "one"}]),
            ),
        )
        picked = await resolve_session(
            conn,
            "ses",
            [
                system,
                {"role": "user", "content": "one"},
                {"role": "assistant", "content": "a"},
                {"role": "user", "content": "two"},
            ],
            settings,
        )

    assert picked.session_id == "s1"


# ---------------------------------------------------------------------------
# End-to-end: title collision + one-off content change + tool continuation
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
@respx.mock
async def test_tool_continuation_resyncs_after_one_off_content_change(app, client):
    respx.post(LLM_URL).mock(
        side_effect=[
            httpx.Response(200, json=completion("title")),
            httpx.Response(200, json=completion("a1")),
            httpx.Response(200, json=tool_call_completion()),
            httpx.Response(200, json=completion("summary")),
            httpx.Response(200, json=completion("welcome")),
        ]
    )

    opening = "Hello ... can you help me with some task? tell me when you are ready."
    note = (
        "[System: The active model for this chat has changed to pocket_network "
        "via provider custom:pnyx_pf. From this point forward, use this runtime "
        "metadata.]\n\n"
    )
    title_system = {"role": "system", "content": "You are a title generator."}
    sys_v1 = {"role": "system", "content": "You are Hermes. Provider: custom:pnyx_pf."}
    sys_v2 = {"role": "system", "content": "You are Hermes. Provider: custom."}
    tool_assistant = {"role": "assistant", "tool_calls": TOOL_CALLS}

    # Title generator (its user message collides with the opening).
    title = await client.post(
        "/v1/chat/completions",
        json={"model": "m", "messages": [title_system, {"role": "user", "content": opening}]},
    )
    assert title.status_code == 200
    title_sid = title.headers["x-session-id"]

    # Main turn 1: the opening carries an injected model-change note.
    first = await client.post(
        "/v1/chat/completions",
        json={"model": "m", "messages": [sys_v1, {"role": "user", "content": note + opening}]},
    )
    assert first.status_code == 200
    first_sid = first.headers["x-session-id"]
    assert first_sid != title_sid
    a1 = first.json()["choices"][0]["message"]

    # Turn 2: the system prompt changed once and the opening is replayed plain.
    # The hash-matched title session must NOT own it; a new root must be created.
    second = await client.post(
        "/v1/chat/completions",
        json={
            "model": "m",
            "messages": [
                sys_v2,
                {"role": "user", "content": opening},
                a1,
                {"role": "user", "content": "summarize the file"},
            ],
        },
    )
    assert second.status_code == 200
    second_sid = second.headers["x-session-id"]
    assert second_sid != title_sid
    assert second_sid != first_sid

    info = (await client.get(f"/v1/sessions/{second_sid}")).json()
    assert info["parent_session_id"] is None
    assert info["root_session_id"] == info["session_id"]

    # The changed system prompt was redacted exactly once (this new root).
    sys_v2_text = sys_v2["content"]
    calls_after_turn2 = list(app.state.triton_client.calls)
    assert sum(1 for c in calls_after_turn2 if c == sys_v2_text) == 1

    # Turn 3: tool continuation. Its lookup prefix is still the opening hash, so
    # without prefix-root registration it would fork again; it must continue the
    # session created at turn 2.
    tool_msg = {"role": "tool", "tool_call_id": "call_1", "content": "FILE CONTENT"}
    third = await client.post(
        "/v1/chat/completions",
        json={
            "model": "m",
            "messages": [
                sys_v2,
                {"role": "user", "content": opening},
                a1,
                {"role": "user", "content": "summarize the file"},
                tool_assistant,
                tool_msg,
            ],
        },
    )
    assert third.status_code == 200
    assert third.headers["x-session-id"] == second_sid

    calls_after_turn3 = app.state.triton_client.calls[len(calls_after_turn2):]
    # Only the new tool payload hit Triton — not the system prompt or prefix.
    assert sys_v2_text not in calls_after_turn3
    assert "FILE CONTENT" in calls_after_turn3

    # Turn 4: a normal new user turn continues the same session.
    summary = third.json()["choices"][0]["message"]
    fourth = await client.post(
        "/v1/chat/completions",
        json={
            "model": "m",
            "messages": [
                sys_v2,
                {"role": "user", "content": opening},
                a1,
                {"role": "user", "content": "summarize the file"},
                tool_assistant,
                tool_msg,
                summary,
                {"role": "user", "content": "ok, thanks"},
            ],
        },
    )
    assert fourth.status_code == 200
    assert fourth.headers["x-session-id"] == second_sid


@pytest.mark.asyncio
@respx.mock
async def test_reformatted_tool_arguments_do_not_fork(app, client):
    """A client re-serialising a tool call's arguments (whitespace/key order)
    must not be treated as a divergence: the tool continuation continues the
    same session instead of forking and re-redacting the prefix."""
    spaced = '{"path": "/mnt/WD_4TB/CVs/main.tex"}'
    compact = '{"path":"/mnt/WD_4TB/CVs/main.tex"}'

    def tool_completion(arguments: str) -> dict:
        return {
            "id": "chatcmpl-tool",
            "object": "chat.completion",
            "created": 1,
            "model": "m",
            "choices": [
                {
                    "index": 0,
                    "message": {
                        "role": "assistant",
                        "content": None,
                        "tool_calls": [
                            {
                                "id": "call_1",
                                "type": "function",
                                "function": {"name": "read_file", "arguments": arguments},
                            }
                        ],
                    },
                    "finish_reason": "tool_calls",
                }
            ],
            "usage": {"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2},
        }

    respx.post(LLM_URL).mock(
        side_effect=[
            httpx.Response(200, json=completion("ready")),        # turn 1
            httpx.Response(200, json=tool_completion(spaced)),    # turn 2 (stored spaced)
            httpx.Response(200, json=completion("summary")),      # turn 3
        ]
    )

    system = {"role": "system", "content": "You are Hermes."}
    first = await client.post(
        "/v1/chat/completions",
        json={"model": "m", "messages": [system, {"role": "user", "content": "hi"}]},
    )
    assert first.status_code == 200
    a1 = first.json()["choices"][0]["message"]

    second = await client.post(
        "/v1/chat/completions",
        json={
            "model": "m",
            "messages": [system, {"role": "user", "content": "hi"}, a1,
                         {"role": "user", "content": "read the file"}],
        },
    )
    assert second.status_code == 200
    sid = second.headers["x-session-id"]
    before = (await client.get(f"/v1/sessions/{sid}")).json()

    # The client replays the assistant tool call with compact arguments and an
    # empty (not null) content.
    replayed_tool_call = {
        "role": "assistant",
        "content": "",
        "tool_calls": [
            {
                "id": "call_1",
                "type": "function",
                "function": {"name": "read_file", "arguments": compact},
            }
        ],
    }
    tool_msg = {"role": "tool", "tool_call_id": "call_1", "content": "FILE DATA"}
    third = await client.post(
        "/v1/chat/completions",
        json={
            "model": "m",
            "messages": [system, {"role": "user", "content": "hi"}, a1,
                         {"role": "user", "content": "read the file"},
                         replayed_tool_call, tool_msg],
        },
    )
    assert third.status_code == 200
    assert third.headers["x-session-id"] == sid

    after = (await client.get(f"/v1/sessions/{sid}")).json()
    # Continued in place: same internal session, still a root, history grew.
    assert after["session_id"] == before["session_id"]
    assert after["parent_session_id"] is None
    assert len(after["raw_messages"]) > len(before["raw_messages"])
