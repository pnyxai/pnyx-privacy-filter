"""End-to-end tests for model pass-through and header forwarding.

The downstream LLM is stubbed with ``respx``; the Triton privacy-filter is
stubbed by :class:`tests.conftest.FakeTritonClient`.
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


def last_payload(route) -> dict:
    return json.loads(route.calls.last.request.content)


def last_headers(route) -> httpx.Headers:
    return route.calls.last.request.headers


@pytest.mark.asyncio
@respx.mock
async def test_client_model_wins_over_fallback(client):
    route = respx.post(LLM_URL).mock(
        return_value=httpx.Response(200, json=completion())
    )
    payload = {
        "model": "minimax-m3",
        "messages": [{"role": "user", "content": "hi"}],
    }
    response = await client.post("/v1/chat/completions", json=payload)

    assert response.status_code == 200
    assert last_payload(route)["model"] == "minimax-m3"


@pytest.mark.asyncio
@respx.mock
async def test_fallback_model_used_when_client_omits_it(client):
    route = respx.post(LLM_URL).mock(
        return_value=httpx.Response(200, json=completion())
    )
    payload = {"messages": [{"role": "user", "content": "hi"}]}
    response = await client.post("/v1/chat/completions", json=payload)

    assert response.status_code == 200
    # conftest settings fall back to "pocket_network".
    assert last_payload(route)["model"] == "pocket_network"


@pytest.mark.asyncio
async def test_missing_model_everywhere_is_rejected(make_client):
    client = await make_client(llm_model_name=None)
    payload = {"messages": [{"role": "user", "content": "hi"}]}
    response = await client.post("/v1/chat/completions", json=payload)

    assert response.status_code == 422
    assert "model" in response.json()["detail"].lower()


@pytest.mark.asyncio
@respx.mock
async def test_session_header_is_forwarded_and_used_as_session_id(make_client):
    client = await make_client(llm_session_header="x-opencode-session")
    route = respx.post(LLM_URL).mock(
        return_value=httpx.Response(200, json=completion())
    )
    payload = {
        "model": "minimax-m3",
        "messages": [{"role": "user", "content": "hi"}],
    }
    response = await client.post(
        "/v1/chat/completions",
        json=payload,
        headers={"X-Session-Id": "sess-abc"},
    )

    assert response.status_code == 200
    session_id = response.headers["x-session-id"]
    assert session_id.startswith("sess-abc::")
    assert last_headers(route).get("x-opencode-session") == session_id


@pytest.mark.asyncio
@respx.mock
async def test_specific_session_header_beats_x_session_id(make_client):
    client = await make_client(llm_session_header="x-opencode-session")
    route = respx.post(LLM_URL).mock(
        return_value=httpx.Response(200, json=completion())
    )
    payload = {
        "model": "minimax-m3",
        "messages": [{"role": "user", "content": "hi"}],
    }
    response = await client.post(
        "/v1/chat/completions",
        json=payload,
        headers={"X-Session-ID": "fallback", "x-session-affinity": "affinity"},
    )

    assert response.status_code == 200
    session_id = response.headers["x-session-id"]
    assert session_id.startswith("affinity::")
    assert last_headers(route).get("x-opencode-session") == session_id


@pytest.mark.asyncio
@respx.mock
async def test_custom_client_headers_are_forwarded(client):
    route = respx.post(LLM_URL).mock(
        return_value=httpx.Response(200, json=completion())
    )
    payload = {
        "model": "minimax-m3",
        "messages": [{"role": "user", "content": "hi"}],
    }
    await client.post(
        "/v1/chat/completions",
        json=payload,
        headers={"X-Custom-Trace": "trace-1"},
    )

    # (The strip-list itself is covered by the build_forward_headers unit
    # tests; httpx re-adds its own Connection/Accept-Encoding when sending.)
    assert last_headers(route).get("x-custom-trace") == "trace-1"


@pytest.mark.asyncio
@respx.mock
async def test_generic_proxy_forwards_and_strips_response_headers(make_client):
    client = await make_client(llm_api_key="server-key")
    route = respx.get("http://llm.local/v1/models").mock(
        return_value=httpx.Response(
            200,
            json={"object": "list", "data": []},
            headers={"Content-Length": "999", "X-Upstream-Marker": "1"},
        )
    )
    response = await client.get("/v1/models")

    assert response.status_code == 200
    assert response.json()["object"] == "list"
    assert route.called
    # The server key is injected for proxy requests too.
    assert route.calls.last.request.headers.get("authorization") == (
        "Bearer server-key"
    )
    # Normal response headers pass through, but the stale upstream
    # Content-Length must not (Starlette recomputes the correct one).
    assert response.headers.get("x-upstream-marker") == "1"
    assert response.headers.get("content-length") != "999"


@pytest.mark.asyncio
async def test_triton_failure_returns_502(app, client):
    async def boom(text):
        raise RuntimeError("triton down")

    app.state.triton_client.infer = boom
    payload = {
        "model": "minimax-m3",
        "messages": [{"role": "user", "content": "hi"}],
    }
    response = await client.post("/v1/chat/completions", json=payload)

    assert response.status_code == 502
    assert "privacy filter" in response.json()["detail"].lower()


@pytest.mark.asyncio
@respx.mock
async def test_upstream_error_status_is_propagated(client):
    respx.post(LLM_URL).mock(
        return_value=httpx.Response(
            400,
            json={
                "error": {
                    "message": "This response_format type is unavailable now",
                    "type": "invalid_request_error",
                }
            },
        )
    )
    payload = {
        "model": "minimax-m3",
        "messages": [{"role": "user", "content": "hi"}],
    }
    response = await client.post("/v1/chat/completions", json=payload)

    assert response.status_code == 400
    assert "response_format" in str(response.json()["detail"])


@pytest.mark.asyncio
@respx.mock
async def test_upstream_transport_error_returns_502(client):
    respx.post(LLM_URL).mock(side_effect=httpx.ConnectError("boom"))
    payload = {
        "model": "minimax-m3",
        "messages": [{"role": "user", "content": "hi"}],
    }
    response = await client.post("/v1/chat/completions", json=payload)

    assert response.status_code == 502


@pytest.mark.asyncio
@respx.mock
async def test_opencode_upstream_gets_session_header_without_config(make_client):
    client = await make_client(llm_url="https://opencode.ai/zen/go")
    route = respx.post("https://opencode.ai/zen/go/v1/chat/completions").mock(
        return_value=httpx.Response(200, json=completion())
    )
    payload = {
        "model": "minimax-m3",
        "messages": [{"role": "user", "content": "hi"}],
    }
    response = await client.post(
        "/v1/chat/completions", json=payload, headers={"X-Session-Id": "s1"}
    )

    assert response.status_code == 200
    # No PCM_LLM_SESSION_HEADER was configured; opencode.ai is auto-detected.
    assert route.calls.last.request.headers.get("x-opencode-session") == (
        response.headers["x-session-id"]
    )
