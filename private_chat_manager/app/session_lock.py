"""Per-conversation serialisation.

Requests are serialised per internal session key so that a read → process →
save cycle cannot interleave with another request for the same conversation
(e.g. two requests arriving back-to-back before the first save completes).

Locks are created lazily and kept for the process lifetime.  The number of
distinct conversations over a long-running process is bounded in practice;
entries are tiny (one ``asyncio.Lock`` each).
"""

from __future__ import annotations

import asyncio

_locks: dict[str, asyncio.Lock] = {}


def get_session_lock(key: str) -> asyncio.Lock:
    """Return the process-wide lock for *key*, creating it on first use."""
    lock = _locks.get(key)
    if lock is None:
        lock = asyncio.Lock()
        _locks[key] = lock
    return lock
