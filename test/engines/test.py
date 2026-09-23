#!/usr/bin/env python3
"""Live integration test for downstream engines and how they behave behind PCM.

The engines (llama.cpp, vLLM, …) and the PCM instance are **declared by the
operator** in ``test/engines/engines.yaml`` (copy ``engines.example.yaml``);
this script never hardcodes endpoints.  It validates:

1. every configured engine is reachable and serves the expected model, for both
   buffered and streaming completions;
2. the PCM instance is healthy and its Triton filter is ready;
3. PCM's privacy pipeline works against its configured downstream: the hidden
   history carries placeholders (no PII), the raw history keeps the PII, and no
   placeholder leaks back to the client.

By default PCM's downstream is whatever ``PCM_LLM_URL`` points at.  To exercise
a specific local engine end-to-end, point PCM at it and assert the match::

    # examples/.env: PCM_LLM_URL=http://localhost:9087
    docker compose up -d --force-recreate --no-deps pnyx-pcm
    private_chat_manager/.venv/bin/python test/engines/test.py --engine vllm

Requires ``httpx`` and ``pyyaml`` (both available in
``private_chat_manager/.venv``).
"""

from __future__ import annotations

import argparse
import json
import sys
import uuid
from pathlib import Path
from typing import Any

import httpx
import yaml

HERE = Path(__file__).resolve().parent
DEFAULT_CONFIG = HERE / "engines.yaml"

# A message with several PII kinds the Triton filter is expected to catch.
PII_MESSAGE = (
    "Hello, this is a privacy filter test. My name is Lionel Messi. "
    "I live at 123 Main St, New York. My email is alice.smith@example.com "
    "and my phone is 555-0123. What is my name?"
)
PII_STRINGS = (
    "Lionel Messi",
    "123 Main St, New York",
    "alice.smith@example.com",
    "555-0123",
)

REQUEST_TIMEOUT = 180.0


class Checks:
    """Collect pass/fail results and print them as they happen."""

    def __init__(self) -> None:
        self.results: list[bool] = []

    def check(self, name: str, condition: bool, detail: str = "") -> bool:
        suffix = f"  ({detail})" if detail else ""
        print(f"  [{'PASS' if condition else 'FAIL'}] {name}{suffix}")
        self.results.append(bool(condition))
        return bool(condition)

    @property
    def ok(self) -> bool:
        return all(self.results)

    @property
    def failed(self) -> int:
        return self.results.count(False)


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------


def load_config(path: Path) -> dict[str, Any]:
    if not path.exists():
        print(f"error: config not found: {path}", file=sys.stderr)
        print(
            "Create it first:\n"
            f"  cp {path.with_name('engines.example.yaml')} {path}\n"
            "then edit the endpoints/models.",
            file=sys.stderr,
        )
        raise SystemExit(2)
    data = yaml.safe_load(path.read_text()) or {}
    if not data.get("pcm", {}).get("url"):
        print("error: config is missing `pcm.url`", file=sys.stderr)
        raise SystemExit(2)
    if not data.get("engines"):
        print("error: config is missing `engines`", file=sys.stderr)
        raise SystemExit(2)
    return data


# ---------------------------------------------------------------------------
# HTTP helpers
# ---------------------------------------------------------------------------


def first_model_id(payload: dict[str, Any]) -> str | None:
    for key in ("data", "models"):
        entries = payload.get(key) or []
        if entries and isinstance(entries[0], dict):
            entry = entries[0]
            return entry.get("id") or entry.get("model") or entry.get("name")
    return None


def list_models(base_url: str) -> list[str]:
    response = httpx.get(f"{base_url.rstrip('/')}/v1/models", timeout=30.0)
    response.raise_for_status()
    payload = response.json()
    ids: list[str] = []
    for key in ("data", "models"):
        for entry in payload.get(key) or []:
            if isinstance(entry, dict):
                model_id = entry.get("id") or entry.get("model") or entry.get("name")
                if model_id:
                    ids.append(model_id)
    return ids


def buffered_chat(base_url: str, model: str, message: str) -> dict[str, Any]:
    payload = {
        "model": model,
        "messages": [{"role": "user", "content": message}],
        "max_tokens": 16,
        "temperature": 0,
    }
    response = httpx.post(
        f"{base_url.rstrip('/')}/v1/chat/completions",
        json=payload,
        timeout=REQUEST_TIMEOUT,
    )
    response.raise_for_status()
    return response.json()


def parse_sse(lines) -> dict[str, Any]:
    content: list[str] = []
    reasoning: list[str] = []
    saw_done = False
    events = 0
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
        events += 1
        for choice in chunk.get("choices") or []:
            delta = choice.get("delta") or {}
            if isinstance(delta.get("content"), str):
                content.append(delta["content"])
            for key in ("reasoning", "reasoning_content"):
                if isinstance(delta.get(key), str):
                    reasoning.append(delta[key])
    return {
        "content": "".join(content),
        "reasoning": "".join(reasoning),
        "saw_done": saw_done,
        "events": events,
    }


def stream_chat(
    base_url: str,
    model: str,
    message: str | None = None,
    headers: dict[str, str] | None = None,
    messages: list[dict[str, Any]] | None = None,
) -> tuple[dict[str, Any], str, str | None]:
    if messages is None:
        messages = [{"role": "user", "content": message}]
    payload = {
        "model": model,
        "messages": messages,
        "max_tokens": 512,
        "temperature": 0,
        "stream": True,
    }
    with httpx.Client(timeout=REQUEST_TIMEOUT) as client:
        with client.stream(
            "POST",
            f"{base_url.rstrip('/')}/v1/chat/completions",
            json=payload,
            headers=headers,
        ) as response:
            response.raise_for_status()
            parsed = parse_sse(response.iter_lines())
            return (
                parsed,
                response.headers.get("content-type", ""),
                response.headers.get("x-session-id"),
            )


def inspect_session(pcm_url: str, session_id: str) -> dict[str, Any]:
    response = httpx.get(
        f"{pcm_url.rstrip('/')}/v1/sessions/{session_id}", timeout=30.0
    )
    response.raise_for_status()
    return response.json()


# ---------------------------------------------------------------------------
# Test stages
# ---------------------------------------------------------------------------


def test_engine(name: str, engine: dict[str, Any], checks: Checks) -> str | None:
    """Validate one engine directly; return its resolved model id."""
    base_url = engine["base_url"]
    print(f"\n=== engine: {name} ({base_url}) ===")

    try:
        models = list_models(base_url)
    except Exception as exc:  # noqa: BLE001 - report and continue
        checks.check(f"{name}: reachable /v1/models", False, repr(exc))
        return None

    checks.check(f"{name}: reachable /v1/models", True, f"{len(models)} model(s)")
    model = engine.get("model") or (models[0] if models else None)
    if not model:
        checks.check(f"{name}: a model is available", False)
        return None
    checks.check(f"{name}: model '{model}' is advertised", model in models, str(models))

    try:
        result = buffered_chat(base_url, model, "Reply with the single word: OK")
        choices = result.get("choices") or []
        checks.check(
            f"{name}: buffered completion returns a choice",
            bool(choices) and bool(choices[0].get("message")),
        )
    except Exception as exc:  # noqa: BLE001
        checks.check(f"{name}: buffered completion", False, repr(exc))

    try:
        parsed, content_type, _ = stream_chat(base_url, model, "Count: one two")
        checks.check(
            f"{name}: streaming completion is text/event-stream",
            "text/event-stream" in content_type,
            content_type,
        )
        checks.check(
            f"{name}: streaming completion ends with [DONE]",
            parsed["saw_done"],
            f"{parsed['events']} chunk(s)",
        )
    except Exception as exc:  # noqa: BLE001
        checks.check(f"{name}: streaming completion", False, repr(exc))

    return model


def test_pcm_pipeline(
    pcm_url: str,
    model: str,
    checks: Checks,
    llm_url: str | None = None,
) -> None:
    """Run the PII redaction round-trip through PCM (buffered + streaming).

    When *llm_url* is given, ``X-PCM-LLM-URL`` selects that downstream for the
    request (requires the URL in ``PCM_LLM_URL_ALLOWLIST``).
    """
    label = "pcm" if llm_url is None else f"pcm[{model}]"
    headers = {"X-PCM-LLM-URL": llm_url} if llm_url else None
    print(f"\n=== PCM privacy pipeline (downstream: {llm_url or model}) ===")

    # --- buffered -------------------------------------------------------
    message = PII_MESSAGE + " (buffered)"
    try:
        response = httpx.post(
            f"{pcm_url.rstrip('/')}/v1/chat/completions",
            json={
                "model": model,
                "messages": [{"role": "user", "content": message}],
                "max_tokens": 512,
            },
            headers=headers,
            timeout=REQUEST_TIMEOUT,
        )
        if response.status_code == 403:
            checks.check(
                f"{label}[buffered]: override allowed",
                False,
                f"add {llm_url!r} to PCM_LLM_URL_ALLOWLIST and restart PCM once",
            )
            return
        response.raise_for_status()
        session_id = response.headers.get("x-session-id")
        checks.check(f"{label}[buffered]: returns X-Session-ID", bool(session_id), str(session_id))
        checks.check(f"{label}[buffered]: returns 200", True)
    except Exception as exc:  # noqa: BLE001
        checks.check(f"{label}[buffered]: request", False, repr(exc))
        return

    if session_id:
        session = inspect_session(pcm_url, session_id)
        hidden_user = session["hidden_messages"][0].get("content") or ""
        raw_user = session["raw_messages"][0].get("content") or ""
        placeholder_map = session["privacy_state"]["placeholder_map"]
        checks.check(f"{label}[buffered]: placeholders were created", bool(placeholder_map))
        checks.check(
            f"{label}[buffered]: hidden history contains a <PRIVATE_...> tag",
            "<PRIVATE_" in hidden_user,
        )
        for pii in PII_STRINGS:
            checks.check(
                f"{label}[buffered]: hidden history hides {pii!r}", pii not in hidden_user
            )
        checks.check(
            f"{label}[buffered]: raw history keeps the original text", raw_user == message
        )

    # --- streaming ------------------------------------------------------
    message = PII_MESSAGE + " (streaming)"
    try:
        parsed, content_type, session_id = stream_chat(pcm_url, model, message, headers)
        checks.check(
            f"{label}[stream]: response is text/event-stream",
            "text/event-stream" in content_type,
            content_type,
        )
        checks.check(f"{label}[stream]: ends with [DONE]", parsed["saw_done"])
    except Exception as exc:  # noqa: BLE001
        checks.check(f"{label}[stream]: request", False, repr(exc))
        return

    if session_id:
        session = inspect_session(pcm_url, session_id)
        hidden_user = session["hidden_messages"][0].get("content") or ""
        raw_user = session["raw_messages"][0].get("content") or ""
        placeholder_map = session["privacy_state"]["placeholder_map"]
        checks.check(f"{label}[stream]: placeholders were created", bool(placeholder_map))
        for pii in PII_STRINGS:
            checks.check(
                f"{label}[stream]: hidden history hides {pii!r}", pii not in hidden_user
            )
        checks.check(
            f"{label}[stream]: raw history keeps the original text", raw_user == message
        )

        client_text = parsed["content"] + parsed["reasoning"]
        leaked = [p for p in placeholder_map if p in client_text]
        checks.check(
            f"{label}[stream]: no placeholder leaked to the client", not leaked, str(leaked)
        )


# ---------------------------------------------------------------------------
# Conversation flow (multi-turn, headers, branching)
# ---------------------------------------------------------------------------

# Distinct PII kinds so each turn adds a new placeholder type.
CONV_PII = {
    "name": "Lionel Messi",
    "email": "alice.smith@example.com",
    "address": "123 Main St, New York",
    "phone": "555-0123",
}


def test_pcm_conversation(
    pcm_url: str,
    model: str,
    checks: Checks,
    llm_url: str | None = None,
) -> None:
    """Exercise PCM's conversation state against one downstream.

    Covers, in order: buffered multi-turn continuity, placeholder accumulation
    and no leak; a streaming turn; headerless hash-only recovery; a custom
    session header (stored + echoed); branching on a rewritten user message
    (fork lineage, parent left intact); redoing the original history back to the
    parent; a structural divergence (assistant turns omitted) which also forks;
    and a headerless rewind matched via a historical user-hash root.
    """
    label = "pcm" if llm_url is None else f"pcm[{model}]"
    base_headers = {"X-PCM-LLM-URL": llm_url} if llm_url else {}
    print(f"\n=== PCM conversation flow (downstream: {llm_url or model}) ===")

    # The session DB persists across runs, so make every prompt unique to this
    # run; otherwise headerless resolution matches identical conversations from
    # earlier runs and picks one at random.
    run = uuid.uuid4().hex[:8]

    def tag(text: str) -> str:
        return f"{text} (run {run})"

    def post(
        messages: list[dict[str, Any]],
        session_id: str | None = None,
        headers: dict[str, str] | None = None,
    ) -> httpx.Response:
        merged = dict(base_headers)
        if headers:
            merged.update(headers)
        if session_id:
            merged["X-Session-ID"] = session_id
        return httpx.post(
            f"{pcm_url.rstrip('/')}/v1/chat/completions",
            json={"model": model, "messages": messages, "max_tokens": 512},
            headers=merged,
            timeout=REQUEST_TIMEOUT,
        )

    def session(session_id: str) -> dict[str, Any]:
        return inspect_session(pcm_url, session_id)

    def history(session_id: str) -> list[dict[str, Any]]:
        return session(session_id)["raw_messages"]

    def hidden_text(session_id: str) -> str:
        return " ".join(
            m.get("content") or "" for m in session(session_id)["hidden_messages"]
        )

    def user_turns(session_id: str) -> int:
        # User messages are always persisted; a reasoning-only assistant turn is
        # intentionally not, so count user turns for continuity assertions.
        return sum(
            1 for m in session(session_id)["raw_messages"] if m.get("role") == "user"
        )

    def ok(response: httpx.Response, name: str) -> bool:
        if response.status_code != 200:
            checks.check(f"{label}[conv]: {name}", False, f"{response.status_code}: {response.text[:160]}")
            return False
        return True

    # --- 1. buffered multi-turn, carrying the session header ---------------
    r = post([{"role": "user", "content": tag(f"My name is {CONV_PII['name']}. What is my name?")}])
    if r.status_code != 200:
        checks.check(f"{label}[conv]: turn 1 request", False, r.text[:200])
        return
    sid = r.headers.get("x-session-id")
    checks.check(f"{label}[conv]: turn 1 returns X-Session-ID", bool(sid), str(sid))
    s1 = session(sid)
    pk = s1["session_id"]
    checks.check(
        f"{label}[conv]: turn 1 creates a session", user_turns(sid) == 1, f"user turns={user_turns(sid)}"
    )
    checks.check(
        f"{label}[conv]: turn 1 redacts the name", CONV_PII["name"] not in hidden_text(sid)
    )
    checks.check(
        f"{label}[conv]: turn 1 raw keeps the name",
        CONV_PII["name"] in (s1["raw_messages"][0].get("content") or ""),
    )

    r = post(
        history(sid) + [{"role": "user", "content": tag(f"My email is {CONV_PII['email']}. What is it?")}],
        session_id=sid,
    )
    if not ok(r, "turn 2 request"):
        return
    sid2 = r.headers.get("x-session-id")
    s2 = session(sid2)
    checks.check(
        f"{label}[conv]: turn 2 reuses the session",
        sid2 == sid and s2["session_id"] == pk,
        f"{sid2} vs {sid}",
    )
    checks.check(
        f"{label}[conv]: turn 2 accumulates history", user_turns(sid2) == 2, f"user turns={user_turns(sid2)}"
    )
    checks.check(
        f"{label}[conv]: turn 2 hides both PII values",
        CONV_PII["name"] not in hidden_text(sid2) and CONV_PII["email"] not in hidden_text(sid2),
    )
    checks.check(
        f"{label}[conv]: turn 2 placeholder map grew",
        len(s2["privacy_state"]["placeholder_map"]) >= 2,
        str(len(s2["privacy_state"]["placeholder_map"])),
    )

    # --- 2. streaming turn, carrying the header ---------------------------
    try:
        parsed, ctype, sid3 = stream_chat(
            pcm_url,
            model,
            headers={**base_headers, "X-Session-ID": sid},
            messages=history(sid2)
            + [{"role": "user", "content": tag(f"I live at {CONV_PII['address']}. Where do I live?")}],
        )
    except Exception as exc:  # noqa: BLE001
        checks.check(f"{label}[conv]: stream turn request", False, repr(exc))
        return
    checks.check(
        f"{label}[conv]: stream turn is text/event-stream", "text/event-stream" in ctype, ctype
    )
    checks.check(f"{label}[conv]: stream turn ends with [DONE]", parsed["saw_done"])
    s3 = session(sid3)
    checks.check(
        f"{label}[conv]: stream turn reuses the session",
        sid3 == sid and s3["session_id"] == pk,
        f"{sid3} vs {sid}",
    )
    checks.check(
        f"{label}[conv]: stream turn accumulates history", user_turns(sid3) == 3, f"user turns={user_turns(sid3)}"
    )
    leaked = [p for p in s3["privacy_state"]["placeholder_map"] if p in (parsed["content"] + parsed["reasoning"])]
    checks.check(f"{label}[conv]: stream turn leaks no placeholder", not leaked, str(leaked))

    # --- 3. headerless continuation (hash-only resolution) ----------------
    r = post(history(sid3) + [{"role": "user", "content": tag(f"My phone is {CONV_PII['phone']}. What is it?")}])
    if not ok(r, "headerless turn request"):
        return
    sid4 = r.headers.get("x-session-id")
    s4 = session(sid4)
    checks.check(
        f"{label}[conv]: headerless turn recovers the session by hash",
        s4["session_id"] == pk,
        f"{s4['session_id']} vs {pk}",
    )
    checks.check(
        f"{label}[conv]: headerless turn accumulates history", user_turns(sid4) == 4, f"user turns={user_turns(sid4)}"
    )
    checks.check(
        f"{label}[conv]: headerless turn redacts the phone", CONV_PII["phone"] not in hidden_text(sid4)
    )

    # --- 4. custom session header is honoured and echoed ------------------
    custom = f"hermes-{uuid.uuid4().hex[:10]}"
    r = post(
        [{"role": "user", "content": tag("Hi, I am Bob Smith.")}],
        headers={"X-Hermes-Session-Id": custom},
    )
    checks.check(f"{label}[conv]: custom header request ok", r.status_code == 200, r.text[:200])
    checks.check(
        f"{label}[conv]: custom header echoed back",
        r.headers.get("x-hermes-session-id") == custom,
        str(r.headers.get("x-hermes-session-id")),
    )
    checks.check(
        f"{label}[conv]: custom header stored as the client id",
        session(custom)["client_x_session_header"] == custom,
    )

    hist4 = history(sid4)
    root = s4["root_session_id"]

    # --- 5. branch: rewrite an answered user message -> fork --------------
    r = post(
        [hist4[0], {"role": "user", "content": tag("Actually my name is Alice Cooper. What is my name?")}],
        session_id=sid,
    )
    if not ok(r, "branch request"):
        return
    sid_b = r.headers.get("x-session-id")
    s_b = session(sid_b)
    checks.check(
        f"{label}[conv]: branch creates a new session",
        s_b["session_id"] != pk,
        f"{s_b['session_id']} vs {pk}",
    )
    checks.check(
        f"{label}[conv]: branch records its parent", s_b["parent_session_id"] == pk, str(s_b["parent_session_id"])
    )
    checks.check(
        f"{label}[conv]: branch keeps the root", s_b["root_session_id"] == root, str(s_b["root_session_id"])
    )
    checks.check(
        f"{label}[conv]: branch records the fork point",
        s_b["origin_message_count"] == 1,
        str(s_b["origin_message_count"]),
    )
    checks.check(f"{label}[conv]: branch keeps the client id", sid_b == sid, f"{sid_b} vs {sid}")
    s_parent = session(pk)
    checks.check(
        f"{label}[conv]: parent left intact (append-only)",
        user_turns(pk) == 4 and s_parent["parent_session_id"] is None,
        f"user turns={user_turns(pk)}",
    )

    # --- 6. redo the original history -> resolves back to the parent ------
    r = post(hist4 + [{"role": "user", "content": tag("Back on the original branch, thanks.")}], session_id=sid)
    if not ok(r, "redo request"):
        return
    sid_r = r.headers.get("x-session-id")
    s_r = session(sid_r)
    checks.check(
        f"{label}[conv]: redo resolves back to the parent",
        s_r["session_id"] == pk,
        f"{s_r['session_id']} vs {pk}",
    )
    checks.check(
        f"{label}[conv]: redo continues the parent", user_turns(sid_r) == 5, f"user turns={user_turns(sid_r)}"
    )

    # --- 7. structural divergence (assistant turns omitted) -> fork -------
    users4 = [m for m in hist4 if m.get("role") == "user"]
    r = post(users4 + [{"role": "user", "content": tag("One more thing, please.")}], session_id=sid)
    if not ok(r, "structural-fork request"):
        return
    sid_s = r.headers.get("x-session-id")
    s_s = session(sid_s)
    checks.check(
        f"{label}[conv]: omitted-assistant history forks",
        s_s["session_id"] != pk,
        f"{s_s['session_id']} vs {pk}",
    )
    checks.check(
        f"{label}[conv]: omitted-assistant fork has lineage",
        s_s["parent_session_id"] is not None and s_s["root_session_id"] == root,
        f"parent={s_s['parent_session_id']} root={s_s['root_session_id']}",
    )
    checks.check(
        f"{label}[conv]: parent intact after structural fork",
        user_turns(pk) == 5,
        f"user turns={user_turns(pk)}",
    )

    # --- 8. headerless rewind matched via a historical root ---------------
    # Editing an answered user message with no session header makes the current
    # hash miss; PCM falls back to a historical root (session_hashes) and forks
    # from that conversation instead of starting a brand-new one.
    rewound = [
        users4[0],
        users4[1],
        users4[2],
        {"role": "user", "content": tag("Let's take this in a new direction.")},
    ]
    r = post(rewound)
    if not ok(r, "headerless rewind request"):
        return
    sid_w = r.headers.get("x-session-id")
    s_w = session(sid_w)
    checks.check(
        f"{label}[conv]: headerless rewind stays in the same tree",
        s_w["root_session_id"] == root,
        f"root={s_w['root_session_id']} vs {root}",
    )
    checks.check(
        f"{label}[conv]: headerless rewind is a fork",
        s_w["parent_session_id"] is not None,
        str(s_w["parent_session_id"]),
    )

    # --- 9. strict-prefix undo (Hermes/opencode `/undo`) ------------------
    # `/undo` re-submits the history truncated at the last user turn, dropping
    # its reply.  When the engine persisted that reply this is a strict-prefix
    # rewind: PCM must fork and reuse the prefix (only the last turn re-redacted)
    # rather than start a new conversation.  Reasoning-only engines that never
    # persisted the reply are skipped (there is nothing to undo).
    last_user_idx = max(
        (i for i, m in enumerate(hist4) if m.get("role") == "user"), default=None
    )
    if last_user_idx is not None and last_user_idx < len(hist4) - 1:
        r = post(hist4[: last_user_idx + 1], session_id=sid)
        if ok(r, "undo request"):
            s_u = session(r.headers.get("x-session-id"))
            checks.check(
                f"{label}[conv]: strict-prefix undo forks",
                s_u["parent_session_id"] is not None,
                str(s_u["parent_session_id"]),
            )
            checks.check(
                f"{label}[conv]: strict-prefix undo reuses the prefix",
                s_u["origin_message_count"] == last_user_idx,
                f"origin={s_u['origin_message_count']} expected={last_user_idx}",
            )


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Live test for downstream engines behind PCM"
    )
    parser.add_argument(
        "--config", type=Path, default=DEFAULT_CONFIG, help="Path to engines.yaml"
    )
    parser.add_argument(
        "--engine",
        action="append",
        default=None,
        help="Only test this engine (repeatable). With --require-pcm-engine, "
        "also assert PCM is pointed at it.",
    )
    parser.add_argument(
        "--require-pcm-engine",
        metavar="NAME",
        default=None,
        help="Fail unless PCM's downstream model matches this engine.",
    )
    parser.add_argument(
        "--skip-pcm", action="store_true", help="Only test the engines directly"
    )
    parser.add_argument(
        "--skip-conversation",
        action="store_true",
        help="Skip the multi-turn/header/branching conversation-flow stage.",
    )
    parser.add_argument(
        "--via-override",
        action="store_true",
        help="Exercise each selected engine through PCM via X-PCM-LLM-URL "
        "(requires the engine URLs in PCM_LLM_URL_ALLOWLIST).",
    )
    args = parser.parse_args()

    config = load_config(args.config)
    pcm_url = config["pcm"]["url"]
    engines: dict[str, Any] = config["engines"]

    selected = args.engine or list(engines)
    unknown = [name for name in selected if name not in engines]
    if unknown:
        print(f"error: unknown engine(s) {unknown}; known: {list(engines)}", file=sys.stderr)
        return 2

    checks = Checks()
    print(f"PCM: {pcm_url}")
    print(f"config: {args.config}")

    # 1. Engine liveness + completions.
    resolved: dict[str, str | None] = {}
    for name in selected:
        resolved[name] = test_engine(name, engines[name], checks)

    # 2. PCM health.
    print("\n=== PCM health ===")
    try:
        health = httpx.get(f"{pcm_url.rstrip('/')}/health", timeout=10.0)
        checks.check("pcm: /health is ok", health.status_code == 200, health.text)
        triton = httpx.get(f"{pcm_url.rstrip('/')}/health/triton", timeout=10.0)
        checks.check(
            "pcm: /health/triton is ready", triton.status_code == 200, triton.text
        )
    except Exception as exc:  # noqa: BLE001
        checks.check("pcm: health", False, repr(exc))
        return 1

    if not args.skip_pcm and args.via_override:
        # 3a. Exercise every selected engine through PCM, selecting the
        #     downstream per request with X-PCM-LLM-URL (no PCM restart).
        for name in selected:
            model = resolved.get(name)
            if not model:
                continue
            engine = engines[name]
            test_pcm_pipeline(
                pcm_url,
                model,
                checks,
                llm_url=engine.get("pcm_url") or engine["base_url"],
            )
            if not args.skip_conversation:
                test_pcm_conversation(
                    pcm_url,
                    model,
                    checks,
                    llm_url=engine.get("pcm_url") or engine["base_url"],
                )
    elif not args.skip_pcm:
        # 3b. Detect PCM's configured downstream and optionally assert it.
        try:
            pcm_models = list_models(pcm_url)
        except Exception as exc:  # noqa: BLE001
            checks.check("pcm: downstream /v1/models reachable", False, repr(exc))
            return 1
        pcm_model = pcm_models[0] if pcm_models else None
        checks.check("pcm: downstream advertises a model", bool(pcm_model), str(pcm_model))

        active = next(
            (name for name, model in resolved.items() if model and model == pcm_model),
            None,
        )
        print(
            f"\nPCM downstream model: {pcm_model!r} "
            f"-> engine: {active or 'not one of the configured engines'}"
        )

        if args.require_pcm_engine:
            wanted = resolved.get(args.require_pcm_engine)
            checks.check(
                f"pcm: downstream is engine '{args.require_pcm_engine}'",
                wanted is not None and wanted == pcm_model,
                f"pcm_model={pcm_model!r} expected={wanted!r}",
            )

        # 4. Privacy pipeline round-trip.
        if pcm_model:
            test_pcm_pipeline(pcm_url, pcm_model, checks)
            if not args.skip_conversation:
                test_pcm_conversation(pcm_url, pcm_model, checks)

    print("\n=== summary ===")
    if checks.ok:
        print("ENGINE TEST OK")
        return 0
    print(f"ENGINE TEST FAILED ({checks.failed} check(s))")
    return 1


if __name__ == "__main__":
    sys.exit(main())
