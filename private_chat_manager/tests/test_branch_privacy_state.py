"""Tests that a forked branch does not inherit privacy state for discarded turns.

A fork keeps only the shared message prefix, so it must also prune the parent's
placeholder map / counters / audit log.  Otherwise a stale placeholder token
(whose PII came from a discarded turn) would de-anonymise that discarded value
if it were echoed or injected on the new branch.
"""

from __future__ import annotations

import httpx
import pytest
import respx

from app.models import PrivacyFilterState, SessionData
from app.privacy_manager import _branch_session, _prune_privacy_state

LLM_URL = "http://llm.local/v1/chat/completions"


def _session() -> SessionData:
    return SessionData(
        session_id="parent",
        created_at=1.0,
        updated_at=1.0,
        raw_messages=[
            {"role": "user", "content": "I am Alice"},
            {"role": "assistant", "content": "Hi <PRIVATE_PERSON_1>"},
            {"role": "user", "content": "call Bob"},
            {"role": "assistant", "content": "ok <PRIVATE_PERSON_2>"},
        ],
        hidden_messages=[
            {"role": "user", "content": "I am <PRIVATE_PERSON_1>"},
            {"role": "assistant", "content": "Hi <PRIVATE_PERSON_1>"},
            {"role": "user", "content": "call <PRIVATE_PERSON_2>"},
            {"role": "assistant", "content": "ok <PRIVATE_PERSON_2>"},
        ],
        privacy_state=PrivacyFilterState(
            placeholder_map={
                "<PRIVATE_PERSON_1>": "Alice",
                "<PRIVATE_PERSON_2>": "Bob",
            },
            type_counters={"PRIVATE_PERSON": 2},
            redaction_results=[
                {"detected_spans": [{"text": "Alice"}]},
                {"detected_spans": [{"text": "Bob"}]},
            ],
        ),
        root_session_id="parent",
    )


def test_branch_prunes_discarded_placeholder_map():
    parent = _session()
    branch = _branch_session(parent, cursor=2)

    assert branch.privacy_state.placeholder_map == {"<PRIVATE_PERSON_1>": "Alice"}
    # The discarded value must not be reachable from the branch.
    assert "Bob" not in branch.privacy_state.placeholder_map.values()
    # Counters are rebuilt from the retained token, so new spans continue at 2.
    assert branch.privacy_state.type_counters == {"PRIVATE_PERSON": 1}
    # Only the audit result for the retained turn survives.
    assert branch.privacy_state.redaction_results == [
        {"detected_spans": [{"text": "Alice"}]}
    ]


def test_branch_parent_untouched():
    parent = _session()
    _branch_session(parent, cursor=2)
    assert parent.privacy_state.placeholder_map == {
        "<PRIVATE_PERSON_1>": "Alice",
        "<PRIVATE_PERSON_2>": "Bob",
    }
    assert parent.privacy_state.type_counters == {"PRIVATE_PERSON": 2}


def test_prune_keeps_entries_used_anywhere_in_prefix():
    state = PrivacyFilterState(
        placeholder_map={"<A_1>": "x", "<B_2>": "y", "<C_3>": "z"},
        type_counters={"A": 1, "B": 2, "C": 3},
        redaction_results=[],
    )
    hidden = [{"role": "user", "content": "<A_1> and <C_3>"}]
    pruned = _prune_privacy_state(state, hidden, [])
    assert set(pruned.placeholder_map) == {"<A_1>", "<C_3>"}
    assert pruned.type_counters == {"A": 1, "C": 3}


@pytest.mark.asyncio
@respx.mock
async def test_discarded_pii_not_deanonymised_on_branch(client):
    """End-to-end: a token from a discarded turn is not mapped on the branch."""
    respx.post(LLM_URL).mock(
        side_effect=[
            httpx.Response(200, json=_completion("a1")),
            httpx.Response(200, json=_completion("a2")),
        ]
    )
    headers = {"X-Session-ID": "s1"}

    first = await client.post(
        "/v1/chat/completions",
        headers=headers,
        json={"model": "m", "messages": [{"role": "user", "content": "I am Lionel Messi"}]},
    )
    a1 = first.json()["choices"][0]["message"]

    # Second turn introduces a second PII value (555-0123) that will be discarded.
    await client.post(
        "/v1/chat/completions",
        headers=headers,
        json={
            "model": "m",
            "messages": [
                {"role": "user", "content": "I am Lionel Messi"},
                a1,
                {"role": "user", "content": "call 555-0123"},
            ],
        },
    )

    # Fork by editing the first turn; the discarded 555-0123 placeholder must not
    # leak into the branch's placeholder map.
    respx.post(LLM_URL).mock(return_value=httpx.Response(200, json=_completion("a1b")))
    third = await client.post(
        "/v1/chat/completions",
        headers=headers,
        json={"model": "m", "messages": [{"role": "user", "content": "I am Lionel Messi!"}]},
    )
    assert third.status_code == 200

    branch = (await client.get("/v1/sessions/s1")).json()
    values = branch["privacy_state"]["placeholder_map"].values()
    assert "555-0123" not in values


def _completion(content: str) -> dict:
    return {
        "id": "chatcmpl-test",
        "object": "chat.completion",
        "created": 1,
        "model": "m",
        "choices": [
            {"index": 0, "message": {"role": "assistant", "content": content}, "finish_reason": "stop"}
        ],
        "usage": {"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2},
    }
