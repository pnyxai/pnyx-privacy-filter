"""Tests for the pure conversation-identity helpers and prefix-root recording.

A conversation is identified by a Merkle hash over its user messages.  The same
normaliser must be used for an incoming request and for a stored session, and
``save_session`` must register **every** answered-user prefix root — not only the
final one — so a session created or forked mid-conversation is discoverable by
the exact prefix a future request computes.
"""

from __future__ import annotations

import aiosqlite
import pytest

from app.identity import (
    compute_prefix_user_hash,
    compute_user_hash,
    user_hash_prefixes,
)
from app.models import SessionData
from app.session_store import (
    ensure_schema,
    find_sessions_by_historical_user_hash,
    save_session,
)


def test_user_hash_prefixes_boundaries():
    messages = [
        {"role": "system", "content": "sys"},
        {"role": "user", "content": "u1"},
        {"role": "assistant", "content": "a1"},
        {"role": "user", "content": "u2"},
        {"role": "assistant", "content": "a2"},
    ]
    prefixes = user_hash_prefixes(messages)
    assert len(prefixes) == 2
    (root1, boundary1), (root2, boundary2) = prefixes
    assert root1 == compute_user_hash([{"role": "user", "content": "u1"}])
    assert root2 == compute_user_hash(
        [
            {"role": "user", "content": "u1"},
            {"role": "user", "content": "u2"},
        ]
    )
    # boundary = index of the next user message; len(messages) for the last root
    assert boundary1 == 3
    assert boundary2 == len(messages)


def test_user_hash_prefixes_empty_without_users():
    assert user_hash_prefixes([{"role": "system", "content": "s"}]) == []


def test_prefix_hash_uses_same_normalisation_for_client_and_stored():
    # A request and a stored session go through the same normaliser, so the same
    # history hashes identically; a list-vs-str content shape with the same text
    # is the same identity (documented, intentional).
    client = [
        {"role": "user", "content": "hi"},
        {"role": "assistant", "content": "a"},
        {"role": "user", "content": [{"type": "text", "text": "more"}]},
    ]
    stored = list(client)
    assert compute_user_hash(client) == compute_user_hash(stored)
    assert compute_prefix_user_hash(client) == compute_user_hash([{"role": "user", "content": "hi"}])
    assert compute_user_hash(
        [{"role": "user", "content": [{"type": "text", "text": "x"}]}]
    ) == compute_user_hash([{"role": "user", "content": "x"}])


@pytest.mark.asyncio
async def test_save_session_records_all_prefix_roots(settings):
    await ensure_schema(settings.db_path)
    raw = [
        {"role": "system", "content": "sys"},
        {"role": "user", "content": "u1"},
        {"role": "assistant", "content": "a1"},
        {"role": "user", "content": "u2"},
        {"role": "assistant", "content": "a2"},
    ]
    root_u1 = compute_user_hash([{"role": "user", "content": "u1"}])
    async with aiosqlite.connect(settings.db_path) as conn:
        await conn.execute("PRAGMA journal_mode=WAL")
        await save_session(
            conn,
            SessionData(
                session_id="s1",
                created_at=1.0,
                updated_at=1.0,
                raw_messages=raw,
                hidden_messages=raw,
                user_hash=compute_user_hash(raw),
            ),
        )
        async with conn.execute(
            "SELECT user_hash, message_count FROM session_hashes "
            "WHERE session_id = 's1'"
        ) as cursor:
            rows = await cursor.fetchall()
        found = await find_sessions_by_historical_user_hash(conn, root_u1)

    boundaries = dict(rows)
    assert boundaries[root_u1] == 3
    assert boundaries[compute_user_hash(raw)] == len(raw)
    # The first (inherited) prefix root makes the session discoverable by it.
    assert [s.session_id for s in found] == ["s1"]


@pytest.mark.asyncio
async def test_fork_inherited_prefix_roots_are_derived(settings):
    """A branch's raw carries the inherited prefix, so deriving reproduces the
    parent's boundary roots for the child without an explicit copy."""
    await ensure_schema(settings.db_path)
    inherited = [
        {"role": "system", "content": "sys"},
        {"role": "user", "content": "u1"},
        {"role": "assistant", "content": "a1"},
    ]
    child_raw = [
        *inherited,
        {"role": "user", "content": "u2"},
        {"role": "assistant", "content": "a2"},
    ]
    root_u1 = compute_user_hash([{"role": "user", "content": "u1"}])
    async with aiosqlite.connect(settings.db_path) as conn:
        await conn.execute("PRAGMA journal_mode=WAL")
        await save_session(
            conn,
            SessionData(
                session_id="child",
                created_at=1.0,
                updated_at=1.0,
                raw_messages=child_raw,
                hidden_messages=child_raw,
                user_hash=compute_user_hash(child_raw),
            ),
        )
        found = await find_sessions_by_historical_user_hash(conn, root_u1)

    assert [s.session_id for s in found] == ["child"]
