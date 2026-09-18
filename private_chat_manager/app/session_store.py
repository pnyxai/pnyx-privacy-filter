from __future__ import annotations

import json
import time

import aiosqlite

from .models import PrivacyFilterState, SessionData

_CREATE_TABLE_SQL = """
CREATE TABLE IF NOT EXISTS sessions (
    session_id    TEXT PRIMARY KEY,
    created_at    REAL NOT NULL,
    updated_at    REAL NOT NULL,
    raw_messages  TEXT NOT NULL,
    hidden_messages TEXT NOT NULL,
    privacy_state TEXT NOT NULL
)
"""


async def ensure_schema(db_path: str) -> None:
    """Create the sessions table if it does not yet exist.

    Called once at application startup; subsequent requests reuse the schema.
    """
    async with aiosqlite.connect(db_path) as conn:
        await conn.execute("PRAGMA journal_mode=WAL")
        await conn.execute(_CREATE_TABLE_SQL)
        await conn.commit()


async def get_session(
    conn: aiosqlite.Connection, session_id: str
) -> SessionData | None:
    """Return the stored SessionData for *session_id*, or None if not found."""
    async with conn.execute(
        "SELECT session_id, created_at, updated_at, "
        "raw_messages, hidden_messages, privacy_state "
        "FROM sessions WHERE session_id = ?",
        (session_id,),
    ) as cursor:
        row = await cursor.fetchone()

    if row is None:
        return None

    return SessionData(
        session_id=row[0],
        created_at=row[1],
        updated_at=row[2],
        raw_messages=json.loads(row[3]),
        hidden_messages=json.loads(row[4]),
        privacy_state=PrivacyFilterState(**json.loads(row[5])),
    )


async def find_sessions_by_base(
    conn: aiosqlite.Connection, base_id: str
) -> list[SessionData]:
    """Return all namespaced sessions whose key starts with ``base_id::``.

    Internal session keys are ``<client-id>::<conversation-fingerprint>``; this
    lets the inspect endpoint resolve a client-facing id to its conversations.
    """
    async with conn.execute(
        "SELECT session_id, created_at, updated_at, "
        "raw_messages, hidden_messages, privacy_state "
        "FROM sessions WHERE session_id LIKE ? ORDER BY updated_at DESC",
        (f"{base_id}::%",),
    ) as cursor:
        rows = await cursor.fetchall()

    return [
        SessionData(
            session_id=row[0],
            created_at=row[1],
            updated_at=row[2],
            raw_messages=json.loads(row[3]),
            hidden_messages=json.loads(row[4]),
            privacy_state=PrivacyFilterState(**json.loads(row[5])),
        )
        for row in rows
    ]


async def save_session(conn: aiosqlite.Connection, data: SessionData) -> None:
    """Upsert *data* into the sessions table."""
    await conn.execute(
        """
        INSERT INTO sessions
            (session_id, created_at, updated_at,
             raw_messages, hidden_messages, privacy_state)
        VALUES (?, ?, ?, ?, ?, ?)
        ON CONFLICT(session_id) DO UPDATE SET
            updated_at      = excluded.updated_at,
            raw_messages    = excluded.raw_messages,
            hidden_messages = excluded.hidden_messages,
            privacy_state   = excluded.privacy_state
        """,
        (
            data.session_id,
            data.created_at,
            data.updated_at,
            json.dumps(data.raw_messages),
            json.dumps(data.hidden_messages),
            json.dumps(data.privacy_state.model_dump()),
        ),
    )
    await conn.commit()
