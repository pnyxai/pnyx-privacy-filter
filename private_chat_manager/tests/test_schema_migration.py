"""Tests for upgrading a pre-existing sessions table to the current schema."""

from __future__ import annotations

import aiosqlite
import pytest

from app.models import SessionData
from app.session_store import ensure_schema, get_session, save_session

# The original 6-column schema (commit 3f5fc37) — no client/endpoint/user-hash
# or lineage columns.  ``ensure_schema`` must migrate it in place.
_OLD_SCHEMA = """
CREATE TABLE sessions (
    session_id    TEXT PRIMARY KEY,
    created_at    REAL NOT NULL,
    updated_at    REAL NOT NULL,
    raw_messages  TEXT NOT NULL,
    hidden_messages TEXT NOT NULL,
    privacy_state TEXT NOT NULL
)
"""


async def _columns(db_path: str) -> set[str]:
    async with aiosqlite.connect(db_path) as conn:
        async with conn.execute("PRAGMA table_info(sessions)") as cursor:
            return {row[1] for row in await cursor.fetchall()}


@pytest.mark.asyncio
async def test_ensure_schema_migrates_old_sessions_table(tmp_path):
    db_path = str(tmp_path / "sessions.db")
    async with aiosqlite.connect(db_path) as conn:
        await conn.execute(_OLD_SCHEMA)
        await conn.commit()

    # Must not raise: index creation on user_hash requires the migrated column.
    await ensure_schema(db_path)

    columns = await _columns(db_path)
    assert {
        "client_x_session_header",
        "endpoint_x_session_header",
        "user_hash",
        "parent_session_id",
        "root_session_id",
        "origin_message_count",
    } <= columns

    # The migrated table is usable end-to-end.
    async with aiosqlite.connect(db_path) as conn:
        session = SessionData(
            session_id="s1",
            created_at=1.0,
            updated_at=1.0,
            raw_messages=[],
            hidden_messages=[],
            user_hash="abc",
        )
        await save_session(conn, session)
        loaded = await get_session(conn, "s1")
    assert loaded is not None
    assert loaded.user_hash == "abc"


@pytest.mark.asyncio
async def test_ensure_schema_is_idempotent(tmp_path):
    db_path = str(tmp_path / "sessions.db")
    await ensure_schema(db_path)
    await ensure_schema(db_path)
    assert "user_hash" in await _columns(db_path)
