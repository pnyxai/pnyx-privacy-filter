"""Pure conversation-identity helpers.

A conversation is identified by a Merkle hash over its ordered ``role == "user"``
messages.  These helpers are pure (no I/O, no database) and are shared by the
request path (:mod:`app.privacy_manager`) and the persistence layer
(:mod:`app.session_store`) so that the hash computed for an incoming request is
byte-for-byte the same value derived from a stored session's ``raw_messages``.

Keeping the normalisation in one place is what makes session resolution sound:
the prefix a future request looks up is exactly a prefix root derived here.
"""
from __future__ import annotations

import hashlib
from collections.abc import Mapping
from typing import Any


def extract_text_content(content: str | list | None) -> str:
    """Return the plain-text body of a message content field.

    Handles both the simple ``str`` form and the multi-part ``list`` form used
    by vision/audio requests (only ``"text"`` parts are extracted).
    """
    if content is None:
        return ""
    if isinstance(content, str):
        return content
    # Multi-part content list
    parts: list[str] = []
    for part in content:
        if isinstance(part, dict) and part.get("type") == "text":
            parts.append(part.get("text", ""))
        elif isinstance(part, str):
            parts.append(part)
    return " ".join(parts)


def merkle_root(leaves: list[str]) -> str:
    """Return the Merkle root of ordered hex-digest *leaves*.

    Leaves are domain-separated (``\\x00``) from internal nodes (``\\x01``) to
    avoid second-preimage ambiguity.  An odd node is duplicated at each level.
    """
    if not leaves:
        return ""
    nodes = [
        hashlib.sha256(b"\x00" + bytes.fromhex(leaf)).digest() for leaf in leaves
    ]
    while len(nodes) > 1:
        if len(nodes) % 2:
            nodes.append(nodes[-1])
        nodes = [
            hashlib.sha256(b"\x01" + nodes[i] + nodes[i + 1]).digest()
            for i in range(0, len(nodes), 2)
        ]
    return nodes[0].hex()


def user_message_digest(message: Mapping[str, Any]) -> str:
    """SHA-256 digest of a user message's extracted text content."""
    content = extract_text_content(message.get("content"))
    return hashlib.sha256(content.encode("utf-8")).hexdigest()


def compute_user_hash(messages: list[dict[str, Any]]) -> str:
    """Merkle root over the ordered ``role == "user"`` messages.

    Returns the empty string when there are no user messages (e.g. before any
    user turn has been answered).
    """
    leaves = [
        user_message_digest(message)
        for message in messages
        if message.get("role") == "user"
    ]
    return merkle_root(leaves)


def compute_prefix_user_hash(messages: list[dict[str, Any]]) -> str:
    """Hash of the *answered* user messages (all but the last user message).

    The final user message is the new, not-yet-answered turn and must not take
    part in matching the session that should answer it.
    """
    last_user_index: int | None = None
    for index, message in enumerate(messages):
        if message.get("role") == "user":
            last_user_index = index
    if last_user_index is None:
        return ""
    return compute_user_hash(messages[:last_user_index])


def user_hash_prefixes(messages: list[dict[str, Any]]) -> list[tuple[str, int]]:
    """Return ``(root, boundary)`` for every answered-user prefix of *messages*.

    For the first ``k`` user messages the root is their Merkle hash and the
    boundary is the index of the ``(k+1)``-th user message, or ``len(messages)``
    for the final (all-users) root.  The boundary is the history length at that
    root — the cursor a future request answering the next user turn would use —
    and is recorded in ``session_hashes.message_count`` (groundwork for
    fork-at-N).

    Registering every prefix root (not just the final one) is what keeps a
    session discoverable by the exact value a future request looks up, including
    the roots a branch inherits from its parent.
    """
    user_indices = [
        index for index, message in enumerate(messages) if message.get("role") == "user"
    ]
    if not user_indices:
        return []

    prefixes: list[tuple[str, int]] = []
    leaves: list[str] = []
    for position, index in enumerate(user_indices):
        leaves.append(user_message_digest(messages[index]))
        root = merkle_root(leaves)
        if position + 1 < len(user_indices):
            boundary = user_indices[position + 1]
        else:
            boundary = len(messages)
        prefixes.append((root, boundary))
    return prefixes
