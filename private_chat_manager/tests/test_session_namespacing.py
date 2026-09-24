"""Tests for conversation identity and session resolution.

A conversation is identified by the Merkle hash of its *answered* user
messages.  A client session header is only a hint: it can select a session
whose hash agrees with the incoming history, but it can never override the
hash.  Distinct logical conversations that share one client id (for example an
agent's title generator and its main chat) therefore stay isolated as long as
their user-message histories differ.
"""

from __future__ import annotations

import json
import time

import aiosqlite
import httpx
import pytest
import respx

from app.identity import compute_prefix_user_hash, compute_user_hash
from app.models import SessionData
from app.privacy_manager import resolve_session
from app.session_store import ensure_schema, save_session

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
# User-message hash helpers
# ---------------------------------------------------------------------------


def test_user_hash_is_order_sensitive_and_stable():
    a = [{"role": "user", "content": "one"}, {"role": "user", "content": "two"}]
    b = [{"role": "user", "content": "two"}, {"role": "user", "content": "one"}]
    assert compute_user_hash(a) == compute_user_hash(a)
    assert compute_user_hash(a) != compute_user_hash(b)


def test_user_hash_ignores_non_user_messages():
    only_user = [{"role": "user", "content": "one"}]
    mixed = [
        {"role": "system", "content": "sys"},
        {"role": "user", "content": "one"},
        {"role": "assistant", "content": "reply"},
        {"role": "tool", "content": "result"},
    ]
    assert compute_user_hash(only_user) == compute_user_hash(mixed)


def test_user_hash_empty_without_user_messages():
    assert compute_user_hash([{"role": "system", "content": "sys"}]) == ""


def test_prefix_hash_excludes_last_user_message():
    history = [{"role": "user", "content": "one"}]
    later = [
        *history,
        {"role": "assistant", "content": "answer"},
        {"role": "user", "content": "two"},
    ]
    assert compute_prefix_user_hash(later) == compute_user_hash(history)
    # The very first turn has no answered user message.
    assert compute_prefix_user_hash(history) == ""


# ---------------------------------------------------------------------------
# resolve_session (DB-backed)
# ---------------------------------------------------------------------------


async def _save(conn: aiosqlite.Connection, **overrides) -> SessionData:
    data = {
        "session_id": "s",
        "created_at": time.time(),
        "updated_at": time.time(),
        "client_x_session_header": None,
        "endpoint_x_session_header": None,
        "user_hash": None,
    }
    data.update(overrides)
    session = SessionData(**data)
    await save_session(conn, session)
    return session


@pytest.mark.asyncio
async def test_resolve_creates_new_session(settings):
    await ensure_schema(settings.db_path)
    async with aiosqlite.connect(settings.db_path) as conn:
        session = await resolve_session(
            conn, "ses-1", [{"role": "user", "content": "hi"}], settings
        )
    assert session.session_id
    assert session.client_x_session_header == "ses-1"
    # Not persisted yet, so no answered user messages.
    assert session.user_hash is None


@pytest.mark.asyncio
async def test_resolve_matches_by_hash_without_header(settings):
    await ensure_schema(settings.db_path)
    messages = [{"role": "user", "content": "hi"}]
    async with aiosqlite.connect(settings.db_path) as conn:
        first = await resolve_session(conn, None, messages, settings)
        first.user_hash = compute_user_hash(messages)
        first.raw_messages = [
            {"role": "user", "content": "hi"},
            {"role": "assistant", "content": "ok"},
        ]
        await save_session(conn, first)

        resumed = await resolve_session(
            conn,
            None,
            [
                *messages,
                {"role": "assistant", "content": "ok"},
                {"role": "user", "content": "again"},
            ],
            settings,
        )
    assert resumed.session_id == first.session_id


@pytest.mark.asyncio
async def test_header_is_only_a_hint_hash_is_authoritative(settings):
    await ensure_schema(settings.db_path)
    async with aiosqlite.connect(settings.db_path) as conn:
        await _save(
            conn,
            session_id="s1",
            client_x_session_header="ses-x",
            user_hash=compute_user_hash([{"role": "user", "content": "hello"}]),
        )
        # Same client header, but a different conversation history -> new.
        other = await resolve_session(
            conn,
            "ses-x",
            [{"role": "user", "content": "completely different"}],
            settings,
        )
    assert other.session_id != "s1"


@pytest.mark.asyncio
async def test_collision_picks_an_existing_session(settings):
    await ensure_schema(settings.db_path)
    messages = [
        {"role": "user", "content": "hi"},
        {"role": "assistant", "content": "a"},
        {"role": "user", "content": "next"},
    ]
    prefix = compute_prefix_user_hash(messages)
    history = [
        {"role": "user", "content": "hi"},
        {"role": "assistant", "content": "a"},
    ]
    async with aiosqlite.connect(settings.db_path) as conn:
        await _save(conn, session_id="a", user_hash=prefix, raw_messages=list(history))
        await _save(conn, session_id="b", user_hash=prefix, raw_messages=list(history))
        picked = await resolve_session(conn, None, messages, settings)
    assert picked.session_id in {"a", "b"}


@pytest.mark.asyncio
async def test_rotated_header_rebinds_to_hash_match(settings):
    await ensure_schema(settings.db_path)
    async with aiosqlite.connect(settings.db_path) as conn:
        await _save(
            conn,
            session_id="s1",
            client_x_session_header="old-id",
            user_hash=compute_user_hash([{"role": "user", "content": "hello"}]),
            raw_messages=[
                {"role": "user", "content": "hello"},
                {"role": "assistant", "content": "ok"},
            ],
        )
        # A sub-agent starts sending a brand-new id mid-conversation.
        resumed = await resolve_session(
            conn,
            "new-id",
            [
                {"role": "user", "content": "hello"},
                {"role": "assistant", "content": "ok"},
                {"role": "user", "content": "more"},
            ],
            settings,
        )
    assert resumed.session_id == "s1"
    assert resumed.client_x_session_header == "new-id"


# ---------------------------------------------------------------------------
# Endpoint behaviour
# ---------------------------------------------------------------------------


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
    assert session_id == "ses-main"

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


@pytest.mark.asyncio
@respx.mock
async def test_title_and_main_do_not_corrupt_each_other(client):
    route = respx.post(LLM_URL).mock(
        return_value=httpx.Response(200, json=completion())
    )
    headers = {"X-Session-Id": "ses-xyz"}

    title = await client.post(
        "/v1/chat/completions",
        json={"model": "m", "messages": TITLE_MESSAGES},
        headers=headers,
    )
    assert title.status_code == 200

    main = await client.post(
        "/v1/chat/completions",
        json={"model": "m", "messages": MAIN_MESSAGES},
        headers=headers,
    )
    assert main.status_code == 200

    # Continuing the main conversation must recover the *main* history, not the
    # title one (whose user-message history differs).
    follow_up = await client.post(
        "/v1/chat/completions",
        json={
            "model": "m",
            "messages": [*MAIN_MESSAGES, {"role": "assistant", "content": "ok"},
                         {"role": "user", "content": "more"}],
        },
        headers=headers,
    )
    assert follow_up.status_code == 200
    upstream = json.loads(route.calls.last.request.content)["messages"]
    assert upstream[1]["content"] == "hi there"


@pytest.mark.asyncio
@respx.mock
async def test_malformed_opencode_header_is_replaced(make_client):
    client = await make_client(llm_url="https://opencode.ai/zen/go")
    route = respx.post("https://opencode.ai/zen/go/v1/chat/completions").mock(
        return_value=httpx.Response(200, json=completion())
    )
    response = await client.post(
        "/v1/chat/completions",
        json={"model": "m", "messages": [{"role": "user", "content": "hi"}]},
        headers={"x-opencode-session": "not a valid id"},
    )

    assert response.status_code == 200
    # The client's malformed id is echoed back as the client-facing id ...
    assert response.headers["x-session-id"] == "not a valid id"
    # ... but a PCM-generated id is sent to the endpoint.
    sent = route.calls.last.request.headers["x-opencode-session"]
    assert sent != "not a valid id"
    assert sent


@pytest.mark.asyncio
@respx.mock
async def test_inspect_resolves_by_client_header(client):
    respx.post(LLM_URL).mock(return_value=httpx.Response(200, json=completion()))
    response = await client.post(
        "/v1/chat/completions",
        json={"model": "m", "messages": [{"role": "user", "content": "hi"}]},
        headers={"X-Session-ID": "inspect-me"},
    )
    assert response.status_code == 200

    inspect = await client.get("/v1/sessions/inspect-me")
    assert inspect.status_code == 200
    body = inspect.json()
    # The body carries the internal PK, not the client-facing id ...
    assert body["session_id"]
    assert body["session_id"] != "inspect-me"
    assert body["turn_count"] == 1
    # ... and that PK resolves on its own too.
    again = await client.get(f"/v1/sessions/{body['session_id']}")
    assert again.status_code == 200
