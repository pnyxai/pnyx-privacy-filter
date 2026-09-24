"""Tests for session TTL expiry: the purge helper and the lazy resolution check.

TTL is a sliding window over ``updated_at`` (last activity).  A non-positive TTL
disables expiry.
"""

from __future__ import annotations

import time

import aiosqlite
import pytest

from app.identity import compute_user_hash
from app.models import SessionData
from app.privacy_manager import resolve_session
from app.session_store import ensure_schema, purge_expired_sessions, save_session


async def _save(db_path: str, session_id: str, updated_at: float) -> None:
    async with aiosqlite.connect(db_path) as conn:
        await conn.execute("PRAGMA journal_mode=WAL")
        await save_session(
            conn,
            SessionData(
                session_id=session_id,
                created_at=updated_at,
                updated_at=updated_at,
            ),
        )


async def _remaining(db_path: str) -> set[str]:
    async with aiosqlite.connect(db_path) as conn:
        async with conn.execute("SELECT session_id FROM sessions") as cursor:
            return {row[0] for row in await cursor.fetchall()}


@pytest.mark.asyncio
async def test_purge_removes_only_expired(settings):
    await ensure_schema(settings.db_path)
    now = time.time()
    await _save(settings.db_path, "old", now - 7200)
    await _save(settings.db_path, "fresh", now - 10)

    deleted = await purge_expired_sessions(settings.db_path, ttl_seconds=3600)

    assert deleted == 1
    assert await _remaining(settings.db_path) == {"fresh"}


@pytest.mark.asyncio
async def test_purge_disabled_is_noop(settings):
    await ensure_schema(settings.db_path)
    await _save(settings.db_path, "old", time.time() - 999_999)

    assert await purge_expired_sessions(settings.db_path, ttl_seconds=0) == 0
    assert await _remaining(settings.db_path) == {"old"}


@pytest.mark.asyncio
async def test_expired_session_is_treated_as_new(settings):
    ttl_settings = settings.model_copy(update={"session_ttl": 60})
    await ensure_schema(settings.db_path)
    messages = [{"role": "user", "content": "hi"}]

    async with aiosqlite.connect(settings.db_path) as conn:
        await conn.execute("PRAGMA journal_mode=WAL")
        first = await resolve_session(conn, None, messages, ttl_settings)
        first.user_hash = compute_user_hash(messages)
        first.updated_at = time.time() - 120  # idle past the 60s TTL
        await save_session(conn, first)

        resumed = await resolve_session(
            conn,
            None,
            [
                *messages,
                {"role": "assistant", "content": "ok"},
                {"role": "user", "content": "again"},
            ],
            ttl_settings,
        )

    assert resumed.session_id != first.session_id
