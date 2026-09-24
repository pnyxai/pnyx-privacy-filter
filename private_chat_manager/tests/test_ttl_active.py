"""TTL must not delete a session that is actively being used.

A session is touched (``updated_at`` bumped) when it is resolved, and the
sweeper applies a grace margin before physical deletion, so an in-flight
redaction/upstream/stream cannot have its row and history swept mid-request.
"""

from __future__ import annotations

import time

import aiosqlite
import pytest

from app.identity import compute_user_hash
from app.models import SessionData
from app.privacy_manager import resolve_session
from app.session_store import (
    ensure_schema,
    purge_expired_sessions,
    save_session,
    touch_session,
)


async def _save(db_path: str, session_id: str, updated_at: float, user_hash: str = "") -> None:
    async with aiosqlite.connect(db_path) as conn:
        await conn.execute("PRAGMA journal_mode=WAL")
        await save_session(
            conn,
            SessionData(
                session_id=session_id,
                created_at=updated_at,
                updated_at=updated_at,
                user_hash=user_hash or None,
            ),
        )


async def _remaining(db_path: str) -> set[str]:
    async with aiosqlite.connect(db_path) as conn:
        async with conn.execute("SELECT session_id FROM sessions") as cursor:
            return {row[0] for row in await cursor.fetchall()}


@pytest.mark.asyncio
async def test_grace_margin_keeps_recently_expired(settings):
    """A session just past the TTL survives inside the grace margin."""
    await ensure_schema(settings.db_path)
    now = time.time()
    await _save(settings.db_path, "borderline", now - 3650)  # 50s past a 1h TTL

    deleted = await purge_expired_sessions(
        settings.db_path, ttl_seconds=3600, grace_seconds=120
    )

    assert deleted == 0
    assert await _remaining(settings.db_path) == {"borderline"}


@pytest.mark.asyncio
async def test_grace_margin_still_purges_truly_stale(settings):
    await ensure_schema(settings.db_path)
    now = time.time()
    await _save(settings.db_path, "stale", now - 7200)
    await _save(settings.db_path, "within-grace", now - 3650)

    deleted = await purge_expired_sessions(
        settings.db_path, ttl_seconds=3600, grace_seconds=120
    )

    assert deleted == 1
    assert await _remaining(settings.db_path) == {"within-grace"}


@pytest.mark.asyncio
async def test_touch_session_refreshes_ttl_and_preserves_hashes(settings):
    await ensure_schema(settings.db_path)
    messages = [{"role": "user", "content": "hi"}]
    async with aiosqlite.connect(settings.db_path) as conn:
        await conn.execute("PRAGMA journal_mode=WAL")
        session = await resolve_session(conn, None, messages, settings)
        session.user_hash = compute_user_hash(messages)
        session.updated_at = time.time() - 7200
        await save_session(conn, session)

        await touch_session(conn, session.session_id, time.time())

        async with conn.execute(
            "SELECT updated_at FROM sessions WHERE session_id = ?",
            (session.session_id,),
        ) as cursor:
            (refreshed,) = await cursor.fetchone()
        async with conn.execute(
            "SELECT COUNT(*) FROM session_hashes WHERE session_id = ?",
            (session.session_id,),
        ) as cursor:
            (hash_count,) = await cursor.fetchone()

    assert refreshed > time.time() - 60
    assert hash_count == 1

    # The refreshed session is not swept.
    deleted = await purge_expired_sessions(settings.db_path, ttl_seconds=3600)
    assert deleted == 0
    assert await _remaining(settings.db_path) == {session.session_id}


@pytest.mark.asyncio
async def test_resolve_touches_existing_session(settings):
    """Resolving a live session bumps its updated_at before long work."""
    ttl_settings = settings.model_copy(update={"session_ttl": 60})
    await ensure_schema(settings.db_path)
    messages = [{"role": "user", "content": "hi"}]

    async with aiosqlite.connect(settings.db_path) as conn:
        await conn.execute("PRAGMA journal_mode=WAL")
        first = await resolve_session(conn, None, messages, ttl_settings)
        first.user_hash = compute_user_hash(messages)
        first.raw_messages = [
            {"role": "user", "content": "hi"},
            {"role": "assistant", "content": "ok"},
        ]
        # Live, but near the expiry boundary.
        stale = time.time() - 50
        first.updated_at = stale
        await save_session(conn, first)

        resumed = await resolve_session(
            conn,
            None,
            [*messages, {"role": "assistant", "content": "ok"}, {"role": "user", "content": "again"}],
            ttl_settings,
        )
        async with conn.execute(
            "SELECT updated_at FROM sessions WHERE session_id = ?",
            (first.session_id,),
        ) as cursor:
            (stored,) = await cursor.fetchone()

    assert resumed.session_id == first.session_id
    # The resolve refreshed the stored updated_at (without waiting for the save).
    assert stored > stale
