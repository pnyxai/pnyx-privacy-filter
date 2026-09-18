#!/usr/bin/env python3
"""Live end-to-end test for PrivateChatManager (PCM) streaming.

Sends a user message containing several kinds of PII to the running PCM
(``POST /v1/chat/completions`` with ``stream: true``), prints the SSE deltas
that reach the client, then inspects the stored session to prove that:

* the hidden history forwarded to the LLM contains placeholders, not PII,
* the raw history contains the original text,
* the client received a valid SSE stream terminated by ``[DONE]``, and
* any placeholder produced by the LLM was de-anonymised before reaching the
  client (never leaked).

Run it against a live stack (see ``examples/docker-compose.yaml``)::

    python ./test/pcm/test.py --url http://localhost:8080

Requires ``httpx`` (already a PCM dependency).
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from typing import Any

import httpx

DEFAULT_URL = "http://localhost:8080"

# The message from the report. Kept verbatim (including typos) so the test
# exercises the exact input that prompted it.
DEFAULT_MESSAGE = (
    "Hello, this is a test for a privacy filter. There is a midlayer in "
    "between us, an i need that you tell me if can see my personal "
    "informacion in the next phrase.\n\n"
    "My name is Lionel Messi. I live at 123 Main St, New York. My email is "
    "alice.smith@example.com and my phone is 555-0123., my citized id is "
    "123456789\n\n"
    "What would you unser if my question is Who am i?"
)

# Strings that MUST NOT appear in the hidden (placeholder) history.
PII_STRINGS = (
    "Lionel Messi",
    "123 Main St, New York",
    "alice.smith@example.com",
    "555-0123",
)


def _parse_sse(lines) -> dict[str, Any]:
    """Consume an SSE line iterator and collect the streamed deltas."""
    content: dict[int, list[str]] = defaultdict(list)
    reasoning: dict[int, list[str]] = defaultdict(list)
    tool_args: dict[int, list[str]] = defaultdict(list)
    finish_reasons: dict[int, str] = {}
    usage: dict[str, Any] | None = None
    events: list[dict[str, Any]] = []
    saw_done = False

    for line in lines:
        if not line.startswith("data:"):
            continue
        data = line[len("data:"):].strip()
        if data == "[DONE]":
            saw_done = True
            break
        if not data:
            continue
        chunk = json.loads(data)
        events.append(chunk)
        if chunk.get("usage"):
            usage = chunk["usage"]
        for choice in chunk.get("choices") or []:
            index = choice.get("index", 0)
            delta = choice.get("delta") or {}
            if isinstance(delta.get("content"), str):
                content[index].append(delta["content"])
            for key in ("reasoning", "reasoning_content"):
                if isinstance(delta.get(key), str):
                    reasoning[index].append(delta[key])
            for tool_call in delta.get("tool_calls") or []:
                function = tool_call.get("function") or {}
                if isinstance(function.get("arguments"), str):
                    tool_args[index].append(function["arguments"])
            if choice.get("finish_reason") is not None:
                finish_reasons[index] = choice["finish_reason"]

    return {
        "content": {i: "".join(parts) for i, parts in content.items()},
        "reasoning": {i: "".join(parts) for i, parts in reasoning.items()},
        "tool_args": {i: "".join(parts) for i, parts in tool_args.items()},
        "finish_reasons": finish_reasons,
        "usage": usage,
        "events": events,
        "saw_done": saw_done,
    }


def stream_chat(url: str, message: str, session_id: str | None) -> dict[str, Any]:
    headers = {"Content-Type": "application/json"}
    if session_id:
        headers["X-Session-ID"] = session_id
    payload = {
        "messages": [{"role": "user", "content": message}],
        "stream": True,
    }

    with httpx.Client(timeout=180.0) as client:
        with client.stream(
            "POST", f"{url.rstrip('/')}/v1/chat/completions", json=payload, headers=headers
        ) as response:
            response.raise_for_status()
            parsed = _parse_sse(response.iter_lines())
            parsed["session_id"] = response.headers.get("x-session-id")
            parsed["content_type"] = response.headers.get("content-type", "")
            return parsed


def inspect_session(url: str, session_id: str) -> dict[str, Any]:
    response = httpx.get(f"{url.rstrip('/')}/v1/sessions/{session_id}", timeout=30.0)
    response.raise_for_status()
    return response.json()


def _check(name: str, condition: bool, detail: str = "") -> bool:
    status = "PASS" if condition else "FAIL"
    suffix = f"  ({detail})" if detail else ""
    print(f"  [{status}] {name}{suffix}")
    return condition


def main() -> int:
    parser = argparse.ArgumentParser(description="Live PCM streaming end-to-end test")
    parser.add_argument("--url", default=DEFAULT_URL, help="PCM base URL")
    parser.add_argument("--message", default=DEFAULT_MESSAGE, help="User message to send")
    parser.add_argument("--session-id", default=None, help="Reuse an existing session")
    args = parser.parse_args()

    print(f"PCM: {args.url}")
    with httpx.Client(timeout=10.0) as client:
        health = client.get(f"{args.url.rstrip('/')}/health/triton")
        print(f"health/triton: {health.status_code} {health.text}")
        health.raise_for_status()

    print("\n=== Sending stream=true request ===")
    result = stream_chat(args.url, args.message, args.session_id)
    session_id = result["session_id"]
    print(f"X-Session-ID   : {session_id}")
    print(f"content-type   : {result['content_type']}")
    print(f"chunks received: {len(result['events'])}")
    print(f"finish reasons : {result['finish_reasons']}")
    print(f"usage          : {result['usage']}")

    print("\n--- client-visible deltas ---")
    for index in sorted(result["content"]):
        print(f"[choice {index}] content  : {result['content'][index]!r}")
    for index in sorted(result["reasoning"]):
        preview = result["reasoning"][index]
        print(f"[choice {index}] reasoning: {preview[:400]!r}{'…' if len(preview) > 400 else ''}")
    for index in sorted(result["tool_args"]):
        print(f"[choice {index}] tool args: {result['tool_args'][index]!r}")

    print("\n=== Session state (GET /v1/sessions/{id}) ===")
    session = inspect_session(args.url, session_id)
    placeholder_map = session["privacy_state"]["placeholder_map"]
    print("placeholder map:")
    for placeholder, value in sorted(placeholder_map.items()):
        print(f"  {placeholder} -> {value!r}")

    hidden_user = (session["hidden_messages"][0].get("content") or "")
    raw_user = (session["raw_messages"][0].get("content") or "")
    print("\nhidden user:", hidden_user)
    print("\nraw user   :", raw_user)

    hidden_asst = session["hidden_messages"][-1]
    raw_asst = session["raw_messages"][-1]
    print("\nhidden assistant:", hidden_asst)
    print("raw assistant   :", raw_asst)

    # ------------------------------------------------------------------
    # Assertions
    # ------------------------------------------------------------------
    print("\n=== checks ===")
    checks: list[bool] = []

    checks.append(
        _check("HTTP stream is text/event-stream", "text/event-stream" in result["content_type"])
    )
    checks.append(_check("stream terminated with [DONE]", result["saw_done"]))
    checks.append(
        _check("at least one placeholder was created", bool(placeholder_map), str(list(placeholder_map)))
    )

    for pii in PII_STRINGS:
        checks.append(
            _check(
                f"hidden history does NOT contain {pii!r}",
                pii not in hidden_user,
            )
        )

    checks.append(
        _check(
            "hidden user contains a <PRIVATE_...> placeholder",
            "<PRIVATE_" in hidden_user,
        )
    )
    checks.append(_check("raw user preserves the original message", raw_user == args.message))

    for placeholder, value in placeholder_map.items():
        checks.append(
            _check(
                f"placeholder {placeholder} maps into the hidden text",
                placeholder in hidden_user or placeholder in json.dumps(hidden_asst),
            )
        )
        checks.append(
            _check(
                f"value {value!r} is restored in the raw history",
                value in raw_user or value in json.dumps(raw_asst),
            )
        )

    # No placeholder produced by the LLM may reach the client.
    client_text = "".join(result["content"].values())
    client_reasoning = "".join(result["reasoning"].values())
    leaked = [
        placeholder
        for placeholder in placeholder_map
        if placeholder in client_text or placeholder in client_reasoning
    ]
    checks.append(_check("no placeholder leaked to the client", not leaked, str(leaked)))

    # If the assistant produced content, the client view must match the raw
    # (de-anonymised) history for that choice.
    if isinstance(raw_asst.get("content"), str):
        checks.append(
            _check(
                "client content matches de-anonymised session history",
                result["content"].get(0, "") == raw_asst["content"],
                f"client={result['content'].get(0, '')!r} raw={raw_asst['content']!r}",
            )
        )

    if all(checks):
        print("\nPCM STREAM TEST OK")
        return 0

    print(f"\nPCM STREAM TEST FAILED ({checks.count(False)}/{len(checks)} checks)")
    return 1


if __name__ == "__main__":
    sys.exit(main())
