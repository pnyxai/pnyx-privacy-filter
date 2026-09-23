"""HTTP header helpers shared by the chat endpoint and the pass-through proxy.

The privacy filter should add as little friction as possible between an agent
and its model backend.  Two concerns are handled here:

* **Session identity** — different agents name their session header
  differently (``X-Session-ID``, ``x-session-affinity``,
  ``x-opencode-session``, ``X-Hermes-Session-Id`` …).  Any ``x…session…``
  header is accepted and resolved to a single client-facing session id.
* **Header forwarding** — client headers are copied to the upstream LLM
  verbatim, except for hop-by-hop and protocol-managed headers that must be
  recomputed by the HTTP client.  The endpoint-specific session header is
  decided elsewhere (:mod:`app.endpoints`); this module only injects it.

Endpoint detection and the session-header policy derived from ``PCM_LLM_URL``
live in :mod:`app.endpoints`.
"""

from __future__ import annotations

import re
from collections.abc import Mapping

# Any header whose name starts with ``x`` and contains ``session`` is treated
# as a session identifier (case-insensitive), e.g. ``x-opencode-session``,
# ``x-session-affinity`` or ``X-Hermes-Session-Id``.
SESSION_HEADER_RE = re.compile(r"^x[-_].*session", re.IGNORECASE)

# Headers that must not be forwarded (hop-by-hop or host-specific).
HOP_BY_HOP = frozenset(
    {
        "host",
        "transfer-encoding",
        "te",
        "trailers",
        "connection",
        "keep-alive",
        "upgrade",
        "proxy-authorization",
    }
)

# Additionally dropped from *requests*: a stale ``Content-Length`` would not
# match a re-serialised body, a client-declared ``Accept-Encoding`` could
# request a compression scheme the proxy cannot decode, and ``X-PCM-LLM-URL``
# is a PCM-internal control header that must not leak upstream.
REQUEST_STRIPPED = HOP_BY_HOP | {
    "content-length",
    "accept-encoding",
    "x-pcm-llm-url",
}

# Additionally dropped from *responses*: ``httpx`` transparently decodes the
# body, so the original ``Content-Encoding``/``Content-Length`` no longer
# describe the bytes we return.
RESPONSE_STRIPPED = HOP_BY_HOP | {"content-encoding", "content-length"}


def is_forwardable_request_header(name: str) -> bool:
    """Return whether an inbound request header should be sent upstream."""
    return name.lower() not in REQUEST_STRIPPED


def resolve_session_header(
    headers: Mapping[str, str],
) -> tuple[str | None, str | None]:
    """Return the ``(name, value)`` of the client's session header, if any.

    Any header matching :data:`SESSION_HEADER_RE` is accepted, in arrival
    order.  ``X-Session-ID`` is only a fallback: a more specific session
    header (e.g. ``x-opencode-session``, ``x-session-affinity``,
    ``X-Hermes-Session-Id``) takes precedence even when it appears later.
    Returns ``(None, None)`` when no session header was sent.  The name is
    returned as received so callers can echo it back verbatim.
    """
    fallback: tuple[str, str] | None = None
    for name, value in headers.items():
        if not SESSION_HEADER_RE.match(name):
            continue
        if name.lower() == "x-session-id":
            fallback = (name, value)
            continue
        return name, value
    return fallback if fallback is not None else (None, None)


def resolve_session_label(headers: Mapping[str, str]) -> str | None:
    """Return the value of the client's session header, if present."""
    return resolve_session_header(headers)[1]


def build_forward_headers(
    headers: Mapping[str, str] | None,
    api_key: str = "",
    session_header: str | None = None,
    session_id: str | None = None,
    force_json: bool = True,
) -> dict[str, str]:
    """Build the header set to send to the upstream LLM.

    Args:
        headers:        Incoming request headers (any mapping; case-insensitive
                        is not required, names are compared lowercased).
        api_key:        When set, overrides any client ``Authorization`` with a
                        bearer token for the configured upstream.
        session_header: Optional upstream-specific session header name.
        session_id:     Value to emit under *session_header*.  PCM is
                        authoritative for the endpoint session header: when
                        both are provided it **overrides** any client-supplied
                        value so a malformed id cannot leak upstream.
        force_json:     When true (chat completions), ``Content-Type`` is forced
                        to ``application/json``.  The generic proxy passes
                        ``False`` to forward the original content type.
    """
    forwarded: dict[str, str] = {}
    for name, value in (headers or {}).items():
        if is_forwardable_request_header(name):
            forwarded[name.lower()] = value

    if force_json:
        forwarded["content-type"] = "application/json"
    if api_key:
        forwarded["authorization"] = f"Bearer {api_key}"
    if session_header and session_id:
        forwarded[session_header.lower()] = session_id

    return forwarded


def build_response_headers(headers: Mapping[str, str]) -> dict[str, str]:
    """Strip headers that no longer describe the (decoded) response body."""
    return {
        name.lower(): value
        for name, value in headers.items()
        if name.lower() not in RESPONSE_STRIPPED
    }


# Headers whose values must never be written to logs.
SENSITIVE_HEADERS = frozenset(
    {
        "authorization",
        "proxy-authorization",
        "cookie",
        "set-cookie",
        "api-key",
        "x-api-key",
    }
)


def redact_headers(headers: Mapping[str, str]) -> dict[str, str]:
    """Return a log-safe copy of *headers* with sensitive values masked."""
    return {
        name.lower(): ("<redacted>" if name.lower() in SENSITIVE_HEADERS else value)
        for name, value in headers.items()
    }
