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
    privacy_state TEXT NOT NULL,
    client_x_session_header TEXT,
    endpoint_x_session_header TEXT,
    user_hash     TEXT,
    parent_session_id TEXT,
    root_session_id   TEXT,
    origin_message_count INTEGER
)
"""

_CREATE_USER_HASH_INDEX_SQL = (
    "CREATE INDEX IF NOT EXISTS idx_sessions_user_hash ON sessions(user_hash)"
)

_CREATE_UPDATED_AT_INDEX_SQL = (
    "CREATE INDEX IF NOT EXISTS idx_sessions_updated_at ON sessions(updated_at)"
)

# Lineage lookups: list the branches of a conversation (by shared root) or the
# immediate children of a session (by parent).
_CREATE_PARENT_INDEX_SQL = (
    "CREATE INDEX IF NOT EXISTS idx_sessions_parent ON sessions(parent_session_id)"
)

_CREATE_ROOT_INDEX_SQL = (
    "CREATE INDEX IF NOT EXISTS idx_sessions_root ON sessions(root_session_id)"
)

# NOTE (future forking): ``parent_session_id`` / ``root_session_id`` /
# ``origin_message_count`` already record the git-like lineage, and
# ``session_hashes.message_count`` records each turn boundary, so future forking
# (list the branches of a conversation, fork at message N, reconstruct the tree)
# is a query/API exercise rather than a schema migration.

# History of the user-message Merkle root after every turn.  The current root is
# also on ``sessions.user_hash``; this table additionally remembers every earlier
# prefix root, so a headerless undo/rewind can be matched back to its session and
# only the changed tail needs re-redacting.  ``message_count`` is the history
# length at that root (the turn boundary), and the composite primary key doubles
# as the lookup index on ``user_hash``.
_CREATE_SESSION_HASHES_SQL = """
CREATE TABLE IF NOT EXISTS session_hashes (
    user_hash  TEXT NOT NULL,
    session_id TEXT NOT NULL,
    created_at REAL NOT NULL,
    message_count INTEGER,
    PRIMARY KEY (user_hash, session_id)
)
"""

_SELECT_COLUMNS = (
    "session_id, created_at, updated_at, raw_messages, hidden_messages, "
    "privacy_state, client_x_session_header, endpoint_x_session_header, user_hash, "
    "parent_session_id, root_session_id, origin_message_count"
)

# Same columns qualified for joins against the ``sessions`` alias ``s``.
_SELECT_COLUMNS_QUALIFIED = ", ".join(
    f"s.{column.strip()}" for column in _SELECT_COLUMNS.split(",")
)


def _row_to_session(row: tuple) -> SessionData:
    return SessionData(
        session_id=row[0],
        created_at=row[1],
        updated_at=row[2],
        raw_messages=json.loads(row[3]),
        hidden_messages=json.loads(row[4]),
        privacy_state=PrivacyFilterState(**json.loads(row[5])),
        client_x_session_header=row[6],
        endpoint_x_session_header=row[7],
        user_hash=row[8],
        parent_session_id=row[9],
        root_session_id=row[10],
        origin_message_count=row[11],
    )


async def ensure_schema(db_path: str) -> None:
    """Create the sessions table and indexes if they do not yet exist.

    Called once at application startup; subsequent requests reuse the schema.
    """
    async with aiosqlite.connect(db_path) as conn:
        await conn.execute("PRAGMA journal_mode=WAL")
        await conn.execute(_CREATE_TABLE_SQL)
        await conn.execute(_CREATE_SESSION_HASHES_SQL)
        await conn.execute(_CREATE_USER_HASH_INDEX_SQL)
        await conn.execute(_CREATE_UPDATED_AT_INDEX_SQL)
        await conn.execute(_CREATE_PARENT_INDEX_SQL)
        await conn.execute(_CREATE_ROOT_INDEX_SQL)
        await conn.commit()


async def purge_expired_sessions(db_path: str, ttl_seconds: int) -> int:
    """Delete sessions idle for longer than *ttl_seconds*.

    The TTL is a sliding window measured from ``updated_at``.  Returns the
    number of rows deleted; a non-positive *ttl_seconds* disables expiry.
    """
    if ttl_seconds <= 0:
        return 0
    cutoff = time.time() - ttl_seconds
    async with aiosqlite.connect(db_path) as conn:
        await conn.execute("PRAGMA journal_mode=WAL")
        cursor = await conn.execute(
            "DELETE FROM sessions WHERE updated_at < ?", (cutoff,)
        )
        deleted = cursor.rowcount
        # Drop history rows whose session is gone.
        await conn.execute(
            "DELETE FROM session_hashes WHERE session_id NOT IN "
            "(SELECT session_id FROM sessions)"
        )
        await conn.commit()
    return max(deleted, 0)


async def get_session(
    conn: aiosqlite.Connection, session_id: str
) -> SessionData | None:
    """Return the stored SessionData for *session_id*, or None if not found."""
    async with conn.execute(
        f"SELECT {_SELECT_COLUMNS} FROM sessions WHERE session_id = ?",
        (session_id,),
    ) as cursor:
        row = await cursor.fetchone()

    return _row_to_session(row) if row is not None else None


async def find_sessions_by_user_hash(
    conn: aiosqlite.Connection, user_hash: str
) -> list[SessionData]:
    """Return every session whose stored user-message hash matches."""
    async with conn.execute(
        f"SELECT {_SELECT_COLUMNS} FROM sessions "
        "WHERE user_hash = ? ORDER BY updated_at DESC",
        (user_hash,),
    ) as cursor:
        rows = await cursor.fetchall()

    return [_row_to_session(row) for row in rows]


async def find_sessions_by_client_header(
    conn: aiosqlite.Connection, client_x_session_header: str
) -> list[SessionData]:
    """Return every session that was recorded for a client session id."""
    async with conn.execute(
        f"SELECT {_SELECT_COLUMNS} FROM sessions "
        "WHERE client_x_session_header = ? ORDER BY updated_at DESC",
        (client_x_session_header,),
    ) as cursor:
        rows = await cursor.fetchall()

    return [_row_to_session(row) for row in rows]


async def find_sessions_by_historical_user_hash(
    conn: aiosqlite.Connection, user_hash: str
) -> list[SessionData]:
    """Return every session that had *user_hash* at any earlier turn.

    Unlike :func:`find_sessions_by_user_hash` (the *current* root only), this
    matches the full per-turn history, so a client that rewinds a user turn
    (undo/branch) can still be mapped back to its conversation.
    """
    async with conn.execute(
        f"SELECT {_SELECT_COLUMNS_QUALIFIED} FROM session_hashes h "
        "JOIN sessions s ON s.session_id = h.session_id "
        "WHERE h.user_hash = ? ORDER BY s.updated_at DESC",
        (user_hash,),
    ) as cursor:
        rows = await cursor.fetchall()

    return [_row_to_session(row) for row in rows]


async def save_session(conn: aiosqlite.Connection, data: SessionData) -> None:
    """Upsert *data* into the sessions table."""
    await conn.execute(
        """
        INSERT INTO sessions
            (session_id, created_at, updated_at,
             raw_messages, hidden_messages, privacy_state,
             client_x_session_header, endpoint_x_session_header, user_hash,
             parent_session_id, root_session_id, origin_message_count)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        ON CONFLICT(session_id) DO UPDATE SET
            updated_at      = excluded.updated_at,
            raw_messages    = excluded.raw_messages,
            hidden_messages = excluded.hidden_messages,
            privacy_state   = excluded.privacy_state,
            client_x_session_header = excluded.client_x_session_header,
            endpoint_x_session_header = excluded.endpoint_x_session_header,
            user_hash       = excluded.user_hash,
            parent_session_id = excluded.parent_session_id,
            root_session_id = excluded.root_session_id,
            origin_message_count = excluded.origin_message_count
        """,
        (
            data.session_id,
            data.created_at,
            data.updated_at,
            json.dumps(data.raw_messages),
            json.dumps(data.hidden_messages),
            json.dumps(data.privacy_state.model_dump()),
            data.client_x_session_header,
            data.endpoint_x_session_header,
            data.user_hash,
            data.parent_session_id,
            data.root_session_id,
            data.origin_message_count,
        ),
    )
    # Remember this turn's user-message root so a later headerless undo can be
    # matched back to this session (see find_sessions_by_historical_user_hash).
    # ``message_count`` records the turn boundary for that root.
    if data.user_hash:
        await conn.execute(
            "INSERT OR IGNORE INTO session_hashes "
            "(user_hash, session_id, created_at, message_count) VALUES (?, ?, ?, ?)",
            (
                data.user_hash,
                data.session_id,
                data.created_at,
                len(data.raw_messages),
            ),
        )
    await conn.commit()
