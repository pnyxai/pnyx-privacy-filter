"""Tests for ``PCM_SYSTEM_PROMPT_PII_INSTRUCTION`` injection."""

from __future__ import annotations

import json

import httpx
import pytest
import respx

from app.config import Settings
from app.privacy_manager import (
    _find_first_system_index,
    append_instruction_to_content,
)

LLM_URL = "http://llm.local/v1/chat/completions"
INSTRUCTION = (
    "# Privacy Tags:\n"
    "The user has a privacy filter active.\n"
    "- <PRIVATE_PERSON_i>"
)


# ---------------------------------------------------------------------------
# Unit: pure helpers
# ---------------------------------------------------------------------------


def test_append_instruction_none_content():
    assert append_instruction_to_content(None, INSTRUCTION) == INSTRUCTION


def test_append_instruction_empty_string_content():
    assert append_instruction_to_content("", INSTRUCTION) == INSTRUCTION


def test_append_instruction_string_content_separated_by_blank_line():
    result = append_instruction_to_content("You are helpful.", INSTRUCTION)
    assert result == f"You are helpful.\n\n{INSTRUCTION}"


def test_append_instruction_list_content_appends_text_part():
    content = [{"type": "text", "text": "You are helpful."}]
    result = append_instruction_to_content(content, INSTRUCTION)
    assert result == [
        {"type": "text", "text": "You are helpful."},
        {"type": "text", "text": INSTRUCTION},
    ]
    # Original list is not mutated.
    assert content == [{"type": "text", "text": "You are helpful."}]


def test_append_instruction_empty_instruction_is_noop():
    content = "untouched"
    assert append_instruction_to_content(content, "") == content
    assert append_instruction_to_content(None, "") is None


def test_find_first_system_index():
    messages = [
        {"role": "user", "content": "hi"},
        {"role": "system", "content": "a"},
        {"role": "system", "content": "b"},
    ]
    assert _find_first_system_index(messages) == 1
    assert _find_first_system_index([{"role": "user"}]) is None
    assert _find_first_system_index([]) is None


def test_config_unescapes_literal_newlines(tmp_path):
    settings = Settings(
        llm_url="http://llm.local",
        llm_model_name="m",
        db_path=str(tmp_path / "x.db"),
        system_prompt_pii_instruction="line1\\nline2\\tindented",
    )
    assert settings.system_prompt_pii_instruction == "line1\nline2\tindented"


def test_config_default_is_empty(tmp_path):
    settings = Settings(
        llm_url="http://llm.local",
        llm_model_name="m",
        db_path=str(tmp_path / "x.db"),
    )
    assert settings.system_prompt_pii_instruction == ""


# ---------------------------------------------------------------------------
# Endpoint: buffered + streaming
# ---------------------------------------------------------------------------


async def _collect_stream_events(client, payload):
    session_id = None
    async with client.stream("POST", "/v1/chat/completions", json=payload) as resp:
        session_id = resp.headers.get("x-session-id")
        assert resp.status_code == 200
        async for line in resp.aiter_lines():
            if line:
                assert line.startswith("data:")
    return session_id


def _json_response(content: str = "ok") -> dict:
    return {
        "id": "chatcmpl-test",
        "object": "chat.completion",
        "created": 1,
        "model": "pocket_network",
        "choices": [
            {
                "index": 0,
                "message": {"role": "assistant", "content": content},
                "finish_reason": "stop",
            }
        ],
        "usage": {"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2},
    }


def _stream_body(content: str = "ok") -> str:
    chunks = [
        {
            "id": "c",
            "object": "chat.completion.chunk",
            "created": 1,
            "model": "pocket_network",
            "choices": [
                {"index": 0, "delta": {"role": "assistant", "content": ""}, "finish_reason": None}
            ],
        },
        {
            "id": "c",
            "object": "chat.completion.chunk",
            "created": 1,
            "model": "pocket_network",
            "choices": [
                {"index": 0, "delta": {"content": content}, "finish_reason": "stop"}
            ],
        },
    ]
    return "".join(f"data: {json.dumps(c)}\n\n" for c in chunks) + "data: [DONE]\n\n"


@pytest.mark.asyncio
@respx.mock
async def test_instruction_injected_and_persisted_nonstream(make_client):
    client = await make_client(system_prompt_pii_instruction=INSTRUCTION)
    route = respx.post(LLM_URL).mock(return_value=httpx.Response(200, json=_json_response()))

    payload = {
        "model": "pocket_network",
        "messages": [
            {"role": "system", "content": "You are helpful."},
            {"role": "user", "content": "hi"},
        ],
        "stream": False,
    }
    response = await client.post("/v1/chat/completions", json=payload)
    assert response.status_code == 200
    session_id = response.headers["x-session-id"]

    forwarded = json.loads(route.calls.last.request.content)
    assert forwarded["messages"][0]["role"] == "system"
    assert forwarded["messages"][0]["content"] == f"You are helpful.\n\n{INSTRUCTION}"

    session = (await client.get(f"/v1/sessions/{session_id}")).json()
    # Persisted in the hidden history...
    assert session["hidden_messages"][0]["content"] == f"You are helpful.\n\n{INSTRUCTION}"
    # ...but not in the raw (client) history.
    assert session["raw_messages"][0]["content"] == "You are helpful."


@pytest.mark.asyncio
@respx.mock
async def test_instruction_injected_streaming(make_client):
    client = await make_client(system_prompt_pii_instruction=INSTRUCTION)
    route = respx.post(LLM_URL).mock(
        return_value=httpx.Response(200, content=_stream_body().encode())
    )

    payload = {
        "model": "pocket_network",
        "messages": [
            {"role": "system", "content": "You are helpful."},
            {"role": "user", "content": "hi"},
        ],
        "stream": True,
    }
    await _collect_stream_events(client, payload)

    forwarded = json.loads(route.calls.last.request.content)
    assert forwarded["messages"][0]["content"] == f"You are helpful.\n\n{INSTRUCTION}"


@pytest.mark.asyncio
@respx.mock
async def test_instruction_not_duplicated_on_second_turn(make_client):
    client = await make_client(system_prompt_pii_instruction=INSTRUCTION)
    route = respx.post(LLM_URL).mock(
        side_effect=[
            httpx.Response(200, json=_json_response("first")),
            httpx.Response(200, json=_json_response("second")),
        ]
    )

    base_system = {"role": "system", "content": "You are helpful."}
    first_user = {"role": "user", "content": "hi"}
    response = await client.post(
        "/v1/chat/completions",
        json={"model": "pocket_network", "messages": [base_system, first_user]},
    )
    session_id = response.headers["x-session-id"]
    assistant = response.json()["choices"][0]["message"]

    second = await client.post(
        "/v1/chat/completions",
        headers={"X-Session-ID": session_id},
        json={
            "model": "pocket_network",
            "messages": [
                base_system,  # client resends the *un-injected* system prompt
                first_user,
                assistant,
                {"role": "user", "content": "again"},
            ],
        },
    )
    assert second.status_code == 200, second.text

    forwarded = json.loads(route.calls.last.request.content)
    system_content = forwarded["messages"][0]["content"]
    assert system_content.count(INSTRUCTION) == 1, system_content
    assert system_content == f"You are helpful.\n\n{INSTRUCTION}"


@pytest.mark.asyncio
@respx.mock
async def test_synthetic_system_prompt_when_client_sends_none(make_client):
    client = await make_client(system_prompt_pii_instruction=INSTRUCTION)
    route = respx.post(LLM_URL).mock(
        side_effect=[
            httpx.Response(200, json=_json_response("first")),
            httpx.Response(200, json=_json_response("second")),
        ]
    )

    first_user = {"role": "user", "content": "hi"}
    response = await client.post(
        "/v1/chat/completions",
        json={"model": "pocket_network", "messages": [first_user]},
    )
    session_id = response.headers["x-session-id"]
    assistant = response.json()["choices"][0]["message"]

    forwarded = json.loads(route.calls.last.request.content)
    assert forwarded["messages"][0] == {"role": "system", "content": INSTRUCTION}
    assert forwarded["messages"][1]["role"] == "user"

    # The synthetic system prompt is payload-only: it must not be stored, so
    # the cursor stays aligned and the next turn is processed normally.
    session = (await client.get(f"/v1/sessions/{session_id}")).json()
    assert all(m.get("role") != "system" for m in session["hidden_messages"])

    second = await client.post(
        "/v1/chat/completions",
        headers={"X-Session-ID": session_id},
        json={
            "model": "pocket_network",
            "messages": [first_user, assistant, {"role": "user", "content": "again"}],
        },
    )
    assert second.status_code == 200, second.text
    forwarded = json.loads(route.calls.last.request.content)
    assert forwarded["messages"][0] == {"role": "system", "content": INSTRUCTION}
    assert forwarded["messages"][-1]["content"] == "again"


@pytest.mark.asyncio
@respx.mock
async def test_no_instruction_leaves_system_prompt_untouched(client):
    route = respx.post(LLM_URL).mock(return_value=httpx.Response(200, json=_json_response()))

    payload = {
        "model": "pocket_network",
        "messages": [
            {"role": "system", "content": "You are helpful."},
            {"role": "user", "content": "hi"},
        ],
    }
    response = await client.post("/v1/chat/completions", json=payload)
    assert response.status_code == 200
    forwarded = json.loads(route.calls.last.request.content)
    assert forwarded["messages"][0]["content"] == "You are helpful."
