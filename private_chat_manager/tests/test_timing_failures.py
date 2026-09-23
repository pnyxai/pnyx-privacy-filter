"""Failure paths must still emit exactly one request-timing event."""

from __future__ import annotations

import httpx
import pytest
import respx

import app.privacy_manager as privacy_manager

LLM_URL = "http://llm.local/v1/chat/completions"


def _completion() -> dict:
    return {
        "id": "chatcmpl-test",
        "object": "chat.completion",
        "created": 1,
        "model": "m",
        "choices": [
            {"index": 0, "message": {"role": "assistant", "content": "ok"}, "finish_reason": "stop"}
        ],
        "usage": {"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2},
    }


def _timing_events(caplog) -> list[str]:
    return [r.message for r in caplog.records if "request timing" in r.message]


@pytest.mark.asyncio
async def test_triton_failure_emits_timing(client, app, caplog):
    async def _boom(_text):
        raise RuntimeError("triton down")

    app.state.triton_client.infer = _boom

    with caplog.at_level("INFO"):
        response = await client.post(
            "/v1/chat/completions",
            json={"model": "m", "messages": [{"role": "user", "content": "hi"}]},
        )

    assert response.status_code == 502
    assert len(_timing_events(caplog)) == 1
    assert "error=True" in caplog.text


@pytest.mark.asyncio
@respx.mock
async def test_upstream_failure_emits_timing(client, caplog):
    respx.post(LLM_URL).mock(return_value=httpx.Response(500, json={"error": "boom"}))

    with caplog.at_level("INFO"):
        response = await client.post(
            "/v1/chat/completions",
            json={"model": "m", "messages": [{"role": "user", "content": "hi"}]},
        )

    assert response.status_code == 500
    assert len(_timing_events(caplog)) == 1


@pytest.mark.asyncio
@respx.mock
async def test_success_emits_exactly_one_timing(client, caplog):
    respx.post(LLM_URL).mock(return_value=httpx.Response(200, json=_completion()))

    with caplog.at_level("INFO"):
        response = await client.post(
            "/v1/chat/completions",
            json={"model": "m", "messages": [{"role": "user", "content": "hi"}]},
        )

    assert response.status_code == 200
    assert len(_timing_events(caplog)) == 1
    assert "error=True" not in caplog.text


@pytest.mark.asyncio
async def test_rejected_override_emits_timing_on_chat(make_client, capsys):
    client = await make_client(llm_url_allowlist=frozenset({"http://alt.local"}))

    response = await client.post(
        "/v1/chat/completions",
        json={"model": "m", "messages": [{"role": "user", "content": "hi"}]},
        headers={"X-PCM-LLM-URL": "http://evil.local"},
    )

    assert response.status_code == 403
    text = capsys.readouterr().out
    assert text.count("request timing") == 1
    assert "error=True" in text


@pytest.mark.asyncio
async def test_rejected_override_emits_timing_on_proxy(make_client, capsys):
    client = await make_client(llm_url_allowlist=frozenset({"http://alt.local"}))

    response = await client.get(
        "/v1/models", headers={"X-PCM-LLM-URL": "http://evil.local"}
    )

    assert response.status_code == 403
    text = capsys.readouterr().out
    assert text.count("request timing") == 1
    assert "error=True" in text


@pytest.mark.asyncio
@respx.mock
async def test_streaming_prepare_failure_emits_timing(client, app, caplog):
    async def _boom(_text):
        raise RuntimeError("triton down")

    app.state.triton_client.infer = _boom

    with caplog.at_level("INFO"):
        response = await client.post(
            "/v1/chat/completions",
            json={"model": "m", "stream": True, "messages": [{"role": "user", "content": "hi"}]},
        )

    assert response.status_code == 502
    assert len(_timing_events(caplog)) == 1
