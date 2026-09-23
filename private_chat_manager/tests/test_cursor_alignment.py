"""Tests for cursor alignment between the stored session and the client.

The cursor is computed by matching the client's replayed history against the
stored ``raw_messages`` (content-aligned), not assumed from the stored length.
A dropped/modified message — including an assistant turn with ``content: null``
and only reasoning — forks the session (append-only) instead of shifting every
later slice.
"""

from __future__ import annotations

import json

import httpx
import pytest
import respx

from app.privacy_manager import (
    _common_prefix_len,
    _matching_prefix_len,
    _same_message,
    _turn_start,
)

LLM_URL = "http://llm.local/v1/chat/completions"


def completion(content="ok", reasoning=None) -> dict:
    message: dict = {"role": "assistant", "content": content}
    if reasoning is not None:
        message["reasoning"] = reasoning
    return {
        "id": "chatcmpl-test",
        "object": "chat.completion",
        "created": 1,
        "model": "m",
        "choices": [{"index": 0, "message": message, "finish_reason": "stop"}],
        "usage": {"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2},
    }


def forwarded(route) -> list[dict]:
    return json.loads(route.calls.last.request.content)["messages"]


# ---------------------------------------------------------------------------
# Unit: matching helpers
# ---------------------------------------------------------------------------


def test_same_message_role_content_and_tool_ids():
    assert _same_message({"role": "user", "content": "a"}, {"role": "user", "content": "a"})
    assert not _same_message(
        {"role": "user", "content": "a"}, {"role": "assistant", "content": "a"}
    )
    assert not _same_message(
        {"role": "user", "content": "a"}, {"role": "user", "content": "b"}
    )
    assert _same_message(
        {"role": "assistant", "content": None, "tool_calls": [{"id": "c1"}]},
        {"role": "assistant", "content": None, "tool_calls": [{"id": "c1"}]},
    )
    assert not _same_message(
        {"role": "assistant", "content": None, "tool_calls": [{"id": "c1"}]},
        {"role": "assistant", "content": None, "tool_calls": [{"id": "c2"}]},
    )


def test_common_prefix_len():
    stored = [{"role": "user", "content": "u1"}, {"role": "assistant", "content": "a1"}]
    assert _common_prefix_len(stored, stored + [{"role": "user", "content": "u2"}]) == 2
    assert _common_prefix_len(stored, [{"role": "user", "content": "u1"}]) == 1
    assert _common_prefix_len(stored, [{"role": "user", "content": "other"}]) == 0


def test_turn_start_finds_last_reply_awaiting_message():
    assert _turn_start([{"role": "user", "content": "u1"}]) == 0
    # Trailing assistant turns are part of an answer, not a turn to answer.
    assert (
        _turn_start(
            [
                {"role": "user", "content": "u1"},
                {"role": "assistant", "content": "a1"},
                {"role": "user", "content": "u2"},
                {"role": "assistant", "content": "a2"},
            ]
        )
        == 2
    )
    # Tool/function results await a reply too.
    assert (
        _turn_start(
            [
                {"role": "user", "content": "u1"},
                {"role": "assistant", "content": None, "tool_calls": [{"id": "c1"}]},
                {"role": "tool", "content": "r1"},
            ]
        )
        == 2
    )
    assert _turn_start([{"role": "assistant", "content": "a1"}]) == -1


def test_matching_prefix_len_handles_continuation_and_rewind():
    stored = [
        {"role": "user", "content": "u1"},
        {"role": "assistant", "content": "a1"},
        {"role": "user", "content": "u2"},
        {"role": "assistant", "content": "a2"},
    ]
    # Continuation (new content after the prefix).
    assert (
        _matching_prefix_len(stored, stored + [{"role": "user", "content": "u3"}], allow_rewind=True)
        == 4
    )
    # Strict-prefix rewind (undo) is honoured only when rewind is allowed.
    assert _matching_prefix_len(stored, stored[:-1], allow_rewind=True) == 3
    assert _matching_prefix_len(stored, stored[:-1], allow_rewind=False) == 0
    # A strict prefix with no answered-user prefix (title-generator style) is not
    # a rewind, so the caller must pass allow_rewind=False for it.
    assert _matching_prefix_len(stored, stored[:1], allow_rewind=False) == 0
    # No shared prefix -> no match.
    assert _matching_prefix_len(stored, [{"role": "user", "content": "other"}], allow_rewind=True) == 0


# ---------------------------------------------------------------------------
# Buffered reasoning-only
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
@respx.mock
async def test_buffered_reasoning_only_persists_user_only(client):
    respx.post(LLM_URL).mock(
        return_value=httpx.Response(
            200, json=completion(content=None, reasoning="thinking about Lionel Messi")
        )
    )
    response = await client.post(
        "/v1/chat/completions",
        json={"model": "m", "messages": [{"role": "user", "content": "hi"}]},
    )
    assert response.status_code == 200

    session = (
        await client.get(f"/v1/sessions/{response.headers['x-session-id']}")
    ).json()
    # No empty assistant message is stored.
    assert [m["role"] for m in session["hidden_messages"]] == ["user"]


@pytest.mark.asyncio
@respx.mock
async def test_reasoning_only_then_client_drops_assistant_continues(client):
    respx.post(LLM_URL).mock(
        return_value=httpx.Response(
            200, json=completion(content=None, reasoning="thinking")
        )
    )
    first = await client.post(
        "/v1/chat/completions",
        json={"model": "m", "messages": [{"role": "user", "content": "q1"}]},
    )
    assert first.status_code == 200
    session_id = first.headers["x-session-id"]

    # The client does not replay the empty assistant message.
    second = await client.post(
        "/v1/chat/completions",
        headers={"X-Session-ID": session_id},
        json={
            "model": "m",
            "messages": [
                {"role": "user", "content": "q1"},
                {"role": "user", "content": "q2"},
            ],
        },
    )
    assert second.status_code == 200, second.text


# ---------------------------------------------------------------------------
# Fork on divergence
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
@respx.mock
async def test_fork_when_client_drops_stored_assistant(client):
    route = respx.post(LLM_URL).mock(
        return_value=httpx.Response(200, json=completion("answer-1"))
    )
    first = await client.post(
        "/v1/chat/completions",
        json={"model": "m", "messages": [{"role": "user", "content": "q1"}]},
    )
    session_id = first.headers["x-session-id"]

    # Client drops the stored assistant turn; previously this 422'd because the
    # positional cursor (2) exceeded the client's length.
    second = await client.post(
        "/v1/chat/completions",
        headers={"X-Session-ID": session_id},
        json={
            "model": "m",
            "messages": [
                {"role": "user", "content": "q1"},
                {"role": "user", "content": "q2"},
            ],
        },
    )
    assert second.status_code == 200, second.text

    messages = forwarded(route)
    # The stale assistant turn is gone; the branch is seeded from the shared prefix.
    assert [m["role"] for m in messages] == ["user", "user"]
    assert "answer-1" not in json.dumps(messages)


@pytest.mark.asyncio
@respx.mock
async def test_fork_when_client_modifies_a_message(client):
    route = respx.post(LLM_URL).mock(
        return_value=httpx.Response(200, json=completion("answer-1"))
    )
    first = await client.post(
        "/v1/chat/completions",
        json={"model": "m", "messages": [{"role": "user", "content": "q1"}]},
    )
    session_id = first.headers["x-session-id"]

    second = await client.post(
        "/v1/chat/completions",
        headers={"X-Session-ID": session_id},
        json={
            "model": "m",
            "messages": [
                {"role": "user", "content": "q1"},
                {"role": "assistant", "content": "edited answer"},
                {"role": "user", "content": "q2"},
            ],
        },
    )
    assert second.status_code == 200, second.text

    messages = forwarded(route)
    assert [m["role"] for m in messages] == ["user", "assistant", "user"]
    assert messages[1]["content"] == "edited answer"


@pytest.mark.asyncio
@respx.mock
async def test_normal_full_history_keeps_positional_cursor(client):
    route = respx.post(LLM_URL).mock(
        side_effect=[
            httpx.Response(200, json=completion("answer-1")),
            httpx.Response(200, json=completion("answer-2")),
        ]
    )
    first = await client.post(
        "/v1/chat/completions",
        json={"model": "m", "messages": [{"role": "user", "content": "q1"}]},
    )
    session_id = first.headers["x-session-id"]

    second = await client.post(
        "/v1/chat/completions",
        headers={"X-Session-ID": session_id},
        json={
            "model": "m",
            "messages": [
                {"role": "user", "content": "q1"},
                {"role": "assistant", "content": "answer-1"},
                {"role": "user", "content": "q2"},
            ],
        },
    )
    assert second.status_code == 200, second.text

    messages = forwarded(route)
    assert [m["role"] for m in messages] == ["user", "assistant", "user"]
    # No duplication of the first turn.
    assert messages[1]["content"] == "answer-1"


# ---------------------------------------------------------------------------
# Undo: rewrite the last user message(s) and continue in the same session
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
@respx.mock
async def test_undo_last_user_message_resumes_same_session(client):
    route = respx.post(LLM_URL).mock(
        side_effect=[
            httpx.Response(200, json=completion("a1")),
            httpx.Response(200, json=completion("a2")),
            httpx.Response(200, json=completion("a2-new")),
        ]
    )
    first = await client.post(
        "/v1/chat/completions",
        json={"model": "m", "messages": [{"role": "user", "content": "u1"}]},
    )
    session_id = first.headers["x-session-id"]
    await client.post(
        "/v1/chat/completions",
        headers={"X-Session-ID": session_id},
        json={
            "model": "m",
            "messages": [
                {"role": "user", "content": "u1"},
                {"role": "assistant", "content": "a1"},
                {"role": "user", "content": "u2"},
            ],
        },
    )

    # Undo u2 and write u2-new instead.
    third = await client.post(
        "/v1/chat/completions",
        headers={"X-Session-ID": session_id},
        json={
            "model": "m",
            "messages": [
                {"role": "user", "content": "u1"},
                {"role": "assistant", "content": "a1"},
                {"role": "user", "content": "u2-new"},
            ],
        },
    )
    assert third.status_code == 200, third.text
    # The same session is resumed, not orphaned into a new one.
    assert third.headers["x-session-id"] == session_id

    messages = forwarded(route)
    assert [m["role"] for m in messages] == ["user", "assistant", "user"]
    assert messages[2]["content"] == "u2-new"

    session = (await client.get(f"/v1/sessions/{session_id}")).json()
    assert [m["role"] for m in session["hidden_messages"]] == [
        "user",
        "assistant",
        "user",
        "assistant",
    ]
    # The undone turn (u2/a2) is gone.
    assert session["hidden_messages"][2]["content"] == "u2-new"


@pytest.mark.asyncio
@respx.mock
async def test_undo_multiple_user_messages(client):
    route = respx.post(LLM_URL).mock(
        side_effect=[
            httpx.Response(200, json=completion("a1")),
            httpx.Response(200, json=completion("a2")),
            httpx.Response(200, json=completion("a3")),
            httpx.Response(200, json=completion("a2b")),
        ]
    )
    first = await client.post(
        "/v1/chat/completions",
        json={"model": "m", "messages": [{"role": "user", "content": "u1"}]},
    )
    session_id = first.headers["x-session-id"]
    await client.post(
        "/v1/chat/completions",
        headers={"X-Session-ID": session_id},
        json={
            "model": "m",
            "messages": [
                {"role": "user", "content": "u1"},
                {"role": "assistant", "content": "a1"},
                {"role": "user", "content": "u2"},
            ],
        },
    )
    await client.post(
        "/v1/chat/completions",
        headers={"X-Session-ID": session_id},
        json={
            "model": "m",
            "messages": [
                {"role": "user", "content": "u1"},
                {"role": "assistant", "content": "a1"},
                {"role": "user", "content": "u2"},
                {"role": "assistant", "content": "a2"},
                {"role": "user", "content": "u3"},
            ],
        },
    )

    # Undo the last two user turns and write a new u2b.
    fourth = await client.post(
        "/v1/chat/completions",
        headers={"X-Session-ID": session_id},
        json={
            "model": "m",
            "messages": [
                {"role": "user", "content": "u1"},
                {"role": "assistant", "content": "a1"},
                {"role": "user", "content": "u2b"},
            ],
        },
    )
    assert fourth.status_code == 200, fourth.text
    assert fourth.headers["x-session-id"] == session_id

    messages = forwarded(route)
    assert [m["role"] for m in messages] == ["user", "assistant", "user"]
    assert messages[2]["content"] == "u2b"


@pytest.mark.asyncio
@respx.mock
async def test_strict_prefix_side_channel_does_not_hijack(client):
    """A history that is only a strict prefix must not resume the session.

    Guards the title-generator case: its message list can be a prefix of the
    main conversation's, so the header alone must not merge them.
    """
    respx.post(LLM_URL).mock(
        side_effect=[
            httpx.Response(200, json=completion("title")),
            httpx.Response(200, json=completion("main")),
        ]
    )
    title = await client.post(
        "/v1/chat/completions",
        headers={"X-Session-ID": "shared-id"},
        json={"model": "m", "messages": [{"role": "user", "content": "hi"}]},
    )
    assert title.status_code == 200
    title_pk = (await client.get("/v1/sessions/shared-id")).json()["session_id"]

    main = await client.post(
        "/v1/chat/completions",
        headers={"X-Session-ID": "shared-id"},
        json={"model": "m", "messages": [{"role": "user", "content": "hi"}]},
    )
    assert main.status_code == 200
    # A second, distinct internal session was created (not the title one).
    main_pk = (await client.get("/v1/sessions/shared-id")).json()["session_id"]
    assert main_pk != title_pk


# ---------------------------------------------------------------------------
# Headerless undo: matched back to its session via the historical hashes
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
@respx.mock
async def test_headerless_undo_reuses_session_via_history(app, client):
    """A headerless undo is matched to its session by a historical prefix root.

    The user-message prefix no longer matches the session's *current* hash, but
    it matches an earlier one, so PCM reuses the session and only re-redacts the
    changed tail instead of starting a new conversation and re-running Triton on
    the whole history.
    """
    route = respx.post(LLM_URL).mock(
        side_effect=[
            httpx.Response(200, json=completion("a1")),
            httpx.Response(200, json=completion("a2")),
            httpx.Response(200, json=completion("a2r")),
        ]
    )
    first = await client.post(
        "/v1/chat/completions",
        json={
            "model": "m",
            "messages": [{"role": "user", "content": "My name is Lionel Messi"}],
        },
    )
    session_id = first.headers["x-session-id"]
    a1 = first.json()["choices"][0]["message"]

    # Turn 2 (headerless) resumes by hash.
    second = await client.post(
        "/v1/chat/completions",
        json={
            "model": "m",
            "messages": [
                {"role": "user", "content": "My name is Lionel Messi"},
                a1,
                {"role": "user", "content": "What is my name? (v1)"},
            ],
        },
    )
    assert second.status_code == 200
    assert second.headers["x-session-id"] == session_id

    calls_before = len(app.state.triton_client.calls)

    # Headerless undo: rewrite the last user message.  No exact current-hash
    # match, but the prefix root "My name is Lionel Messi" is in the history.
    third = await client.post(
        "/v1/chat/completions",
        json={
            "model": "m",
            "messages": [
                {"role": "user", "content": "My name is Lionel Messi"},
                a1,
                {"role": "user", "content": "What is my name? (UNDONE)"},
            ],
        },
    )
    assert third.status_code == 200
    # Same conversation resumed (no new session).
    assert third.headers["x-session-id"] == session_id

    # Only the changed tail was sent to Triton (the shared prefix was reused).
    new_calls = app.state.triton_client.calls[calls_before:]
    assert new_calls == ["What is my name? (UNDONE)"]

    forwarded = json.loads(route.calls.last.request.content)["messages"]
    assert "Lionel Messi" not in json.dumps(forwarded)


@pytest.mark.asyncio
@respx.mock
async def test_headerless_new_conversation_is_not_matched(client):
    """A history with no shared prefix must start a new conversation."""
    respx.post(LLM_URL).mock(
        side_effect=[
            httpx.Response(200, json=completion("a1")),
            httpx.Response(200, json=completion("other")),
        ]
    )
    first = await client.post(
        "/v1/chat/completions",
        json={"model": "m", "messages": [{"role": "user", "content": "first topic"}]},
    )
    session_id = first.headers["x-session-id"]

    other = await client.post(
        "/v1/chat/completions",
        json={
            "model": "m",
            "messages": [
                {"role": "user", "content": "a totally different topic"},
                {"role": "user", "content": "and another one"},
            ],
        },
    )
    assert other.status_code == 200
    assert other.headers["x-session-id"] != session_id


# ---------------------------------------------------------------------------
# Rewind forks a new session and preserves the parent (recoverability)
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
@respx.mock
async def test_rewind_forks_new_session_and_preserves_parent(client):
    """Undo splits the conversation; the original stays recoverable.

    The branch gets a new internal ``session_id`` (same client-facing id so a
    fixed-header client keeps working), and the parent keeps its full history.
    A later redo replays the original history and resolves back to the parent.
    """
    respx.post(LLM_URL).mock(
        side_effect=[
            httpx.Response(200, json=completion("a1")),
            httpx.Response(200, json=completion("a2")),
            httpx.Response(200, json=completion("a2-new")),
            httpx.Response(200, json=completion("a3")),
        ]
    )
    headers = {"X-Session-ID": "s1"}

    first = await client.post(
        "/v1/chat/completions",
        headers=headers,
        json={"model": "m", "messages": [{"role": "user", "content": "u1"}]},
    )
    a1 = first.json()["choices"][0]["message"]
    parent_pk = (await client.get("/v1/sessions/s1")).json()["session_id"]

    second = await client.post(
        "/v1/chat/completions",
        headers=headers,
        json={
            "model": "m",
            "messages": [
                {"role": "user", "content": "u1"},
                a1,
                {"role": "user", "content": "u2"},
            ],
        },
    )
    a2 = second.json()["choices"][0]["message"]

    # Undo u2 -> split.
    third = await client.post(
        "/v1/chat/completions",
        headers=headers,
        json={
            "model": "m",
            "messages": [
                {"role": "user", "content": "u1"},
                a1,
                {"role": "user", "content": "u2-new"},
            ],
        },
    )
    assert third.status_code == 200

    branch = (await client.get("/v1/sessions/s1")).json()
    assert branch["session_id"] != parent_pk  # new branch session
    # Lineage: the branch points at the parent and shares its root.
    assert branch["parent_session_id"] == parent_pk
    assert branch["root_session_id"] == parent_pk
    assert branch["origin_message_count"] == 2

    # The parent is preserved untouched (its own root, no parent).
    preserved = (await client.get(f"/v1/sessions/{parent_pk}")).json()
    assert [m["role"] for m in preserved["hidden_messages"]] == [
        "user",
        "assistant",
        "user",
        "assistant",
    ]
    assert preserved["hidden_messages"][2]["content"] == "u2"

    # Redo: replay the original history + a new turn -> back to the parent.
    fourth = await client.post(
        "/v1/chat/completions",
        headers=headers,
        json={
            "model": "m",
            "messages": [
                {"role": "user", "content": "u1"},
                a1,
                {"role": "user", "content": "u2"},
                a2,
                {"role": "user", "content": "u3"},
            ],
        },
    )
    assert fourth.status_code == 200
    redone = (await client.get(f"/v1/sessions/{parent_pk}")).json()
    assert [m["role"] for m in redone["hidden_messages"]] == [
        "user",
        "assistant",
        "user",
        "assistant",
        "user",
        "assistant",
    ]
    assert redone["hidden_messages"][4]["content"] == "u3"


# ---------------------------------------------------------------------------
# Structural divergence also forks (sessions are append-only)
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
@respx.mock
async def test_structural_divergence_forks_and_preserves_parent(client):
    """Omitting a stored assistant turn forks too, leaving the parent intact.

    Sessions are append-only: any divergence (not just a user-message rewrite)
    creates a child instead of rewriting the parent.
    """
    respx.post(LLM_URL).mock(
        side_effect=[
            httpx.Response(200, json=completion("a1")),
            httpx.Response(200, json=completion("a2")),
        ]
    )
    headers = {"X-Session-ID": "s1"}

    first = await client.post(
        "/v1/chat/completions",
        headers=headers,
        json={"model": "m", "messages": [{"role": "user", "content": "u1"}]},
    )
    a1 = first.json()["choices"][0]["message"]
    parent_pk = (await client.get("/v1/sessions/s1")).json()["session_id"]

    # The client replays [u1, u2], omitting the stored assistant turn a1.
    second = await client.post(
        "/v1/chat/completions",
        headers=headers,
        json={
            "model": "m",
            "messages": [
                {"role": "user", "content": "u1"},
                {"role": "user", "content": "u2"},
            ],
        },
    )
    assert second.status_code == 200

    branch = (await client.get("/v1/sessions/s1")).json()
    assert branch["session_id"] != parent_pk
    assert branch["parent_session_id"] == parent_pk
    assert branch["origin_message_count"] == 1
    assert [m["role"] for m in branch["hidden_messages"]] == [
        "user",
        "user",
        "assistant",
    ]

    preserved = (await client.get(f"/v1/sessions/{parent_pk}")).json()
    assert [m["role"] for m in preserved["hidden_messages"]] == [
        "user",
        "assistant",
    ]
    assert a1["content"] == "a1"  # sanity


# ---------------------------------------------------------------------------
# Strict-prefix undo: resubmitting the turn unchanged reuses the prefix
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
@respx.mock
async def test_undo_strict_prefix_forks_and_reuses_prefix(app, client):
    """An undo that resubmits the turn unchanged (a strict prefix) forks.

    Hermes/opencode drop the trailing reply and resubmit the same user turn, so
    the body is a strict prefix of the stored history.  PCM must reuse the shared
    prefix (placeholder state + hidden prefix) and re-answer only the last turn,
    instead of starting a new conversation and re-running Triton on everything.
    """
    respx.post(LLM_URL).mock(
        side_effect=[
            httpx.Response(200, json=completion("a1")),
            httpx.Response(200, json=completion("a2")),
            httpx.Response(200, json=completion("a2r")),
        ]
    )
    headers = {"X-Session-ID": "s1"}

    first = await client.post(
        "/v1/chat/completions",
        headers=headers,
        json={"model": "m", "messages": [{"role": "user", "content": "My name is Lionel Messi"}]},
    )
    a1 = first.json()["choices"][0]["message"]
    parent_pk = (await client.get("/v1/sessions/s1")).json()["session_id"]

    second = await client.post(
        "/v1/chat/completions",
        headers=headers,
        json={
            "model": "m",
            "messages": [
                {"role": "user", "content": "My name is Lionel Messi"},
                a1,
                {"role": "user", "content": "What is my name?"},
            ],
        },
    )
    assert second.status_code == 200

    calls_before = len(app.state.triton_client.calls)

    # Undo: resubmit the same history without the trailing reply.
    third = await client.post(
        "/v1/chat/completions",
        headers=headers,
        json={
            "model": "m",
            "messages": [
                {"role": "user", "content": "My name is Lionel Messi"},
                a1,
                {"role": "user", "content": "What is my name?"},
            ],
        },
    )
    assert third.status_code == 200, third.text
    assert third.headers["x-session-id"] == "s1"

    branch = (await client.get("/v1/sessions/s1")).json()
    assert branch["session_id"] != parent_pk  # new branch session
    assert branch["parent_session_id"] == parent_pk
    assert branch["root_session_id"] == parent_pk
    assert branch["origin_message_count"] == 2  # [u1, a1] reused

    # Only the re-answered turn went back through Triton.
    assert app.state.triton_client.calls[calls_before:] == ["What is my name?"]

    # The branch reuses the redacted prefix verbatim.
    parent = (await client.get(f"/v1/sessions/{parent_pk}")).json()
    assert branch["hidden_messages"][:2] == parent["hidden_messages"][:2]

    # The parent is preserved untouched (append-only).
    assert [m["role"] for m in parent["hidden_messages"]] == [
        "user",
        "assistant",
        "user",
        "assistant",
    ]


@pytest.mark.asyncio
@respx.mock
async def test_undo_multiple_turns_strict_prefix_forks(app, client):
    """``/undo N`` (several turns) reuses the prefix and forks at the cut.

    Re-submitting N turns back drops N-1 complete turns and re-answers the N-th;
    the shared prefix is reused and only that one turn is re-redacted.
    """
    respx.post(LLM_URL).mock(
        side_effect=[
            httpx.Response(200, json=completion("a1")),
            httpx.Response(200, json=completion("a2")),
            httpx.Response(200, json=completion("a3")),
            httpx.Response(200, json=completion("a2r")),
        ]
    )
    headers = {"X-Session-ID": "s1"}

    first = await client.post(
        "/v1/chat/completions",
        headers=headers,
        json={"model": "m", "messages": [{"role": "user", "content": "u1"}]},
    )
    a1 = first.json()["choices"][0]["message"]
    second = await client.post(
        "/v1/chat/completions",
        headers=headers,
        json={
            "model": "m",
            "messages": [
                {"role": "user", "content": "u1"},
                a1,
                {"role": "user", "content": "u2"},
            ],
        },
    )
    a2 = second.json()["choices"][0]["message"]
    third = await client.post(
        "/v1/chat/completions",
        headers=headers,
        json={
            "model": "m",
            "messages": [
                {"role": "user", "content": "u1"},
                a1,
                {"role": "user", "content": "u2"},
                a2,
                {"role": "user", "content": "u3"},
            ],
        },
    )
    assert third.status_code == 200
    parent_pk = (await client.get("/v1/sessions/s1")).json()["session_id"]

    calls_before = len(app.state.triton_client.calls)

    # /undo 2: resubmit through u2 (u2's reply, u3 and its reply are undone).
    fourth = await client.post(
        "/v1/chat/completions",
        headers=headers,
        json={
            "model": "m",
            "messages": [
                {"role": "user", "content": "u1"},
                a1,
                {"role": "user", "content": "u2"},
            ],
        },
    )
    assert fourth.status_code == 200, fourth.text

    branch = (await client.get("/v1/sessions/s1")).json()
    assert branch["session_id"] != parent_pk
    assert branch["parent_session_id"] == parent_pk
    assert branch["root_session_id"] == parent_pk
    assert branch["origin_message_count"] == 2  # [u1, a1] reused
    assert app.state.triton_client.calls[calls_before:] == ["u2"]

    # The undone turns are gone from the branch, the parent keeps them.
    assert [m["role"] for m in branch["hidden_messages"]] == [
        "user",
        "assistant",
        "user",
        "assistant",
    ]
    parent = (await client.get(f"/v1/sessions/{parent_pk}")).json()
    assert [m["role"] for m in parent["hidden_messages"]] == [
        "user",
        "assistant",
        "user",
        "assistant",
        "user",
        "assistant",
    ]


@pytest.mark.asyncio
@respx.mock
async def test_headerless_undo_strict_prefix_reuses_history(app, client):
    """A headerless strict-prefix undo is matched by a historical root."""
    respx.post(LLM_URL).mock(
        side_effect=[
            httpx.Response(200, json=completion("a1")),
            httpx.Response(200, json=completion("a2")),
            httpx.Response(200, json=completion("a2r")),
        ]
    )

    first = await client.post(
        "/v1/chat/completions",
        json={"model": "m", "messages": [{"role": "user", "content": "My name is Lionel Messi"}]},
    )
    session_id = first.headers["x-session-id"]
    a1 = first.json()["choices"][0]["message"]

    second = await client.post(
        "/v1/chat/completions",
        json={
            "model": "m",
            "messages": [
                {"role": "user", "content": "My name is Lionel Messi"},
                a1,
                {"role": "user", "content": "What is my name?"},
            ],
        },
    )
    assert second.status_code == 200

    calls_before = len(app.state.triton_client.calls)

    # Headerless undo: resubmit the same history without the trailing reply.
    third = await client.post(
        "/v1/chat/completions",
        json={
            "model": "m",
            "messages": [
                {"role": "user", "content": "My name is Lionel Messi"},
                a1,
                {"role": "user", "content": "What is my name?"},
            ],
        },
    )
    assert third.status_code == 200, third.text
    # Same conversation family (forked), same client-facing id.
    assert third.headers["x-session-id"] == session_id

    branch = (await client.get(f"/v1/sessions/{session_id}")).json()
    assert branch["parent_session_id"] is not None
    assert app.state.triton_client.calls[calls_before:] == ["What is my name?"]


# ---------------------------------------------------------------------------
# A colliding side-channel session must not shadow the real conversation
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
@respx.mock
async def test_undo_not_shadowed_by_title_session_collision(app, client):
    """A hash-colliding title session must not shadow the real conversation.

    Hermes runs a title generator with the same client header; its single user
    message is the opening, so its ``user_hash`` equals the conversation's first
    answered-user prefix.  The hash alone cannot pick the session — the session
    that shares the actual messages must win, so the undo reuses the prefix and
    re-redacts only the re-asked turn (not the system prompt / earlier turns).
    """
    respx.post(LLM_URL).mock(
        side_effect=[
            httpx.Response(200, json=completion("a1")),
            httpx.Response(200, json=completion("title")),
            httpx.Response(200, json=completion("a2")),
            httpx.Response(200, json=completion("a2r")),
        ]
    )
    headers = {"X-Session-ID": "ses"}
    opening = "hi there"
    system = {"role": "system", "content": "You are opencode."}

    # Main conversation, turn 1.
    first = await client.post(
        "/v1/chat/completions",
        headers=headers,
        json={"model": "m", "messages": [system, {"role": "user", "content": opening}]},
    )
    a1 = first.json()["choices"][0]["message"]
    main_pk = (await client.get("/v1/sessions/ses")).json()["session_id"]

    # Title generator: same client header, single user message that collides.
    title = await client.post(
        "/v1/chat/completions",
        headers=headers,
        json={
            "model": "m",
            "messages": [
                {"role": "system", "content": "You are a title generator."},
                {"role": "user", "content": opening},
            ],
        },
    )
    assert title.status_code == 200
    title_pk = (await client.get("/v1/sessions/ses")).json()["session_id"]
    assert title_pk != main_pk

    # Main conversation, turn 2 — must continue the main session, not the title.
    second = await client.post(
        "/v1/chat/completions",
        headers=headers,
        json={
            "model": "m",
            "messages": [system, {"role": "user", "content": opening}, a1,
                         {"role": "user", "content": "more"}],
        },
    )
    assert second.status_code == 200
    assert (await client.get("/v1/sessions/ses")).json()["session_id"] == main_pk

    calls_before = len(app.state.triton_client.calls)

    # Undo turn 2: resubmit the turn unchanged (a strict prefix of main's history).
    third = await client.post(
        "/v1/chat/completions",
        headers=headers,
        json={
            "model": "m",
            "messages": [system, {"role": "user", "content": opening}, a1,
                         {"role": "user", "content": "more"}],
        },
    )
    assert third.status_code == 200, third.text

    branch = (await client.get("/v1/sessions/ses")).json()
    assert branch["parent_session_id"] == main_pk
    assert branch["root_session_id"] == main_pk
    assert branch["origin_message_count"] == 3
    # Only the re-asked turn hit Triton — not the system prompt, the opening, or
    # the assistant turn.
    assert app.state.triton_client.calls[calls_before:] == ["more"]


# ---------------------------------------------------------------------------
# session_hashes records the turn boundary (message_count)
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_session_hashes_records_message_count(settings):
    import aiosqlite

    from app.models import SessionData
    from app.session_store import ensure_schema, save_session

    await ensure_schema(settings.db_path)
    async with aiosqlite.connect(settings.db_path) as conn:
        await conn.execute("PRAGMA journal_mode=WAL")
        await save_session(
            conn,
            SessionData(
                session_id="s1",
                created_at=1.0,
                updated_at=1.0,
                raw_messages=[
                    {"role": "user", "content": "u1"},
                    {"role": "assistant", "content": "a1"},
                ],
                hidden_messages=[
                    {"role": "user", "content": "u1"},
                    {"role": "assistant", "content": "a1"},
                ],
                user_hash="roothash",
            ),
        )
        async with conn.execute(
            "SELECT message_count FROM session_hashes WHERE user_hash = ?",
            ("roothash",),
        ) as cursor:
            row = await cursor.fetchone()

    assert row is not None
    assert row[0] == 2
