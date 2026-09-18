"""Unit tests for the shared HTTP header helpers."""

from __future__ import annotations

import pytest

from app.headers import (
    build_forward_headers,
    build_response_headers,
    default_session_header_for,
    is_opencode_endpoint,
    redact_headers,
    resolve_session_label,
)


def test_redact_headers_masks_sensitive_values():
    headers = {
        "Authorization": "Bearer secret",
        "Cookie": "session=abc",
        "X-Api-Key": "k",
        "x-opencode-session": "auto-123",
        "content-type": "application/json",
    }
    redacted = redact_headers(headers)

    assert redacted["authorization"] == "<redacted>"
    assert redacted["cookie"] == "<redacted>"
    assert redacted["x-api-key"] == "<redacted>"
    assert redacted["x-opencode-session"] == "auto-123"
    assert redacted["content-type"] == "application/json"


@pytest.mark.parametrize(
    "url, expected",
    [
        ("https://opencode.ai/zen/go/v1", True),
        ("https://opencode.ai", True),
        ("https://api.opencode.ai/v1", True),
        ("http://localhost:8080/v1", False),
        ("https://evilopencode.ai/v1", False),
        ("https://opencode.ai.evil.com/v1", False),
        ("", False),
        (None, False),
    ],
)
def test_is_opencode_endpoint(url, expected):
    assert is_opencode_endpoint(url) is expected


def test_default_session_header_for():
    assert (
        default_session_header_for("https://opencode.ai/zen/go")
        == "x-opencode-session"
    )
    assert default_session_header_for("http://localhost:8080") is None


@pytest.mark.parametrize(
    "headers, expected",
    [
        ({"X-Session-ID": "a"}, "a"),
        ({"x-session-id": "a"}, "a"),
        ({"x-session-affinity": "b"}, "b"),
        ({"x-opencode-session": "c"}, "c"),
        ({"X-Hermes-Session-Id": "d"}, "d"),
        ({"x-custom-session": "e"}, "e"),
        ({"authorization": "Bearer x"}, None),
        ({"x-request-id": "r"}, None),
        ({"session-id": "no-x-prefix"}, None),
        ({}, None),
    ],
)
def test_resolve_session_label(headers, expected):
    assert resolve_session_label(headers) == expected


def test_resolve_session_label_prefers_specific_over_x_session_id():
    headers = {"X-Session-ID": "fallback", "x-session-affinity": "specific"}
    assert resolve_session_label(headers) == "specific"


def test_resolve_session_label_specific_after_fallback_still_wins():
    headers = {"x-session-id": "fallback", "x-opencode-session": "specific"}
    assert resolve_session_label(headers) == "specific"


def test_build_forward_headers_forwards_client_headers():
    headers = {
        "User-Agent": "opencode/1.0",
        "x-session-affinity": "sess-1",
        "content-type": "application/json",
    }
    forwarded = build_forward_headers(headers)
    assert forwarded["user-agent"] == "opencode/1.0"
    assert forwarded["x-session-affinity"] == "sess-1"
    assert forwarded["content-type"] == "application/json"


def test_build_forward_headers_strips_hop_by_hop_and_technical():
    headers = {
        "Host": "localhost:8080",
        "Connection": "keep-alive",
        "Transfer-Encoding": "chunked",
        "Content-Length": "42",
        "Accept-Encoding": "br, gzip",
        "X-Keep": "yes",
    }
    forwarded = build_forward_headers(headers)
    assert forwarded == {"x-keep": "yes", "content-type": "application/json"}


def test_build_forward_headers_force_json_overrides_client_content_type():
    forwarded = build_forward_headers({"Content-Type": "text/plain"})
    assert forwarded["content-type"] == "application/json"


def test_build_forward_headers_proxy_keeps_content_type():
    forwarded = build_forward_headers(
        {"Content-Type": "multipart/form-data; boundary=x"},
        force_json=False,
    )
    assert forwarded["content-type"] == "multipart/form-data; boundary=x"


def test_build_forward_headers_api_key_overrides_client_authorization():
    forwarded = build_forward_headers(
        {"Authorization": "Bearer client"},
        api_key="server-key",
    )
    assert forwarded["authorization"] == "Bearer server-key"


def test_build_forward_headers_keeps_client_authorization_without_api_key():
    forwarded = build_forward_headers({"Authorization": "Bearer client"})
    assert forwarded["authorization"] == "Bearer client"


def test_build_forward_headers_injects_session_header():
    forwarded = build_forward_headers(
        {"X-Session-ID": "sess-1"},
        session_header="x-opencode-session",
        session_id="sess-1",
    )
    assert forwarded["x-opencode-session"] == "sess-1"


def test_build_forward_headers_does_not_override_existing_session_header():
    forwarded = build_forward_headers(
        {"x-opencode-session": "client-provided"},
        session_header="x-opencode-session",
        session_id="resolved",
    )
    assert forwarded["x-opencode-session"] == "client-provided"


def test_build_forward_headers_no_session_injection_without_config():
    forwarded = build_forward_headers({"X-Session-ID": "sess-1"})
    assert "x-opencode-session" not in forwarded


def test_build_response_headers_strips_decoded_headers():
    headers = {
        "Content-Encoding": "br",
        "Content-Length": "123",
        "Connection": "keep-alive",
        "Content-Type": "application/json",
    }
    stripped = build_response_headers(headers)
    assert stripped == {"content-type": "application/json"}
