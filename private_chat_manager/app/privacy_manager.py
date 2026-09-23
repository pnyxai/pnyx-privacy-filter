from __future__ import annotations

import asyncio
import copy
import hashlib
import json
import random
import time
import uuid
from collections.abc import AsyncIterator, Mapping
from dataclasses import dataclass, field
from typing import Any

import aiosqlite
import httpx
from fastapi import HTTPException

from ._logging import get_logger
from .config import Settings
from .endpoints import endpoint_session_policy
from .headers import build_forward_headers, redact_headers
from .message_fields import message_fields
from .models import PrivacyFilterState, PrivateChatRequest, SessionData
from .session_store import (
    find_sessions_by_client_header,
    find_sessions_by_historical_user_hash,
    find_sessions_by_user_hash,
    save_session,
)
from .streaming import StreamResponseFilter
from .timing import RequestTimings
from .triton_client import TritonPrivacyFilterClient

logger = get_logger(__name__)


# ---------------------------------------------------------------------------
# Pure helper functions (no I/O, fully unit-testable)
# ---------------------------------------------------------------------------


def apply_placeholder_indexing(
    redaction_result: dict[str, Any],
    type_counters: dict[str, int],
    placeholder_map: dict[str, str],
) -> tuple[str, dict[str, int], dict[str, str]]:
    """Assign unique indexed placeholders to every detected span.

    Triton returns base placeholders like ``<PRIVATE_PERSON>`` that repeat
    whenever the same label appears more than once.  This function turns them
    into globally-unique, session-scoped tokens such as ``<PRIVATE_PERSON_1>``,
    ``<PRIVATE_PERSON_2>`` … by appending the running per-type counter.

    The counters are *global per session* (never reset between turns) so that
    a span redacted in turn 1 as ``<PRIVATE_PERSON_1>`` still de-anonymises
    correctly in the LLM's reply during turn 3.

    **Deduplication:** if a span's original text was already seen in a prior
    turn (or earlier in the same message), the existing indexed placeholder is
    reused — no new counter is incremented and no new map entry is created.
    This guarantees that the same real-world value always maps to the same
    token across the whole session, even if the user re-sends PII.

    Args:
        redaction_result: A RedactionResult dict as returned by Triton.
        type_counters:    Current per-label counters for this session (copied,
                          not mutated).
        placeholder_map:  Current placeholder→true-text mapping for this session
                          (copied, not mutated).

    Returns:
        A 3-tuple of
        - *indexed_redacted_text*: ``redacted_text`` with every base placeholder
          replaced by its indexed counterpart.
        - *new_counters*: Updated type counter dict.
        - *new_map*: Updated placeholder→true-text mapping.
    """
    new_counters = dict(type_counters)
    new_map = dict(placeholder_map)
    redacted_text: str = redaction_result.get("redacted_text", "")

    # Build a reverse lookup (original text → existing placeholder) once so
    # that repeated spans within the same message are also deduplicated in O(1).
    text_to_placeholder: dict[str, str] = {v: k for k, v in new_map.items()}

    for span in redaction_result.get("detected_spans", []):
        label: str = span["label"]
        base_placeholder: str = span["placeholder"]  # e.g. "<PRIVATE_PERSON>"
        span_text: str = span["text"]

        if span_text in text_to_placeholder:
            # Already known — reuse the existing indexed placeholder.
            # No counter bump; no new map entry needed.
            indexed_placeholder = text_to_placeholder[span_text]
        else:
            # New text — assign a fresh indexed placeholder.
            new_counters[label] = new_counters.get(label, 0) + 1
            count = new_counters[label]
            indexed_placeholder = f"{base_placeholder[:-1]}_{count}>"
            new_map[indexed_placeholder] = span_text
            text_to_placeholder[span_text] = indexed_placeholder

        # Replace the FIRST remaining occurrence of the base placeholder so
        # that repeated labels advance the index correctly.
        redacted_text = redacted_text.replace(base_placeholder, indexed_placeholder, 1)

    return redacted_text, new_counters, new_map


def deanonymize_text(text: str, placeholder_map: dict[str, str]) -> str:
    """Replace all indexed placeholders in *text* with their original values.

    Keys are applied in descending length order to avoid prefix-collision bugs
    (e.g. ``<PRIVATE_PERSON_10>`` must be substituted before
    ``<PRIVATE_PERSON_1>``).
    """
    if not placeholder_map:
        return text
    for placeholder in sorted(placeholder_map, key=len, reverse=True):
        text = text.replace(placeholder, placeholder_map[placeholder])
    return text


def _extract_text_content(content: str | list | None) -> str:
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


def _strip_none(d: dict[str, Any]) -> dict[str, Any]:
    """Return a shallow copy of *d* with all None-valued keys removed."""
    return {k: v for k, v in d.items() if v is not None}


def _assistant_has_output(msg: Mapping[str, Any]) -> bool:
    """True when an assistant message carries content or tool calls.

    Used by both the buffered and streaming paths so they persist exactly the
    same set of assistant turns (reasoning/refusal side channels are not
    persisted).  Keeping the two paths identical matters because the session
    cursor is positional (``len(hidden_messages)``): if one path stored an
    extra empty assistant message the client did not, every later slice would
    be off by one.
    """
    return bool(msg.get("content")) or bool(msg.get("tool_calls"))


def _same_message(stored: Mapping[str, Any], client: Mapping[str, Any]) -> bool:
    """Whether a stored raw message matches the client's replayed message.

    ``raw_messages`` holds exactly what the client sent (user/system/tool) or
    the de-anonymised assistant reply it received, so a matching client message
    is byte-identical.  Compared on role, content and every semantic tool-call
    field — not the reasoning/refusal side channels, which the client may or may
    not echo.

    Every tool-call field (arguments, function name, ids and the legacy
    ``function_call``/``function_calls`` shapes) is compared so that an edit to
    a call — not just its id — is detected as divergence and re-redacted on a
    new branch rather than silently skipped in favour of the cached hidden copy.

    NOTE (future forking): this is a whole-message comparison.  Giving each
    stored message a stable identity (a message id or content hash) would allow
    message-level diff/merge between branches; deferred until forking is built.
    """
    if stored.get("role") != client.get("role"):
        return False
    if stored.get("content") != client.get("content"):
        return False
    semantic_fields = (
        "name",
        "tool_call_id",
        "tool_calls",
        "function_call",
        "function_calls",
    )
    return all(stored.get(key) == client.get(key) for key in semantic_fields)


def _common_prefix_len(stored: list[dict[str, Any]], client: list[dict[str, Any]]) -> int:
    """Length of the shared leading run of *stored* and *client* messages."""
    limit = min(len(stored), len(client))
    index = 0
    while index < limit and _same_message(stored[index], client[index]):
        index += 1
    return index


def _turn_start(client: list[dict[str, Any]]) -> int:
    """Index of the last message that awaits an assistant reply.

    The trailing ``assistant`` message(s) are part of an answer, not a turn to
    answer, so the new region never starts after the last ``user``/``tool``/
    ``function`` message.  Returns ``-1`` when the history has none (an
    all-assistant list, which the request contract rejects).
    """
    for index in range(len(client) - 1, -1, -1):
        if client[index].get("role") != "assistant":
            return index
    return -1


def _matching_prefix_len(
    stored: list[dict[str, Any]],
    client: list[dict[str, Any]],
    *,
    allow_rewind: bool,
) -> int:
    """Prefix length used to pick the session to continue, fork or rewind.

    Returns the longest common prefix when the client added content beyond it
    (continuation/fork), or — when *allow_rewind* is set — the length of a
    strict-prefix rewind (undo), where the client re-submitted the turn being
    re-answered unchanged.  Returns ``0`` when there is nothing to match,
    including a strict prefix that is not a rewind (e.g. a title generator's
    shortened history), which must not hijack the session.

    ``allow_rewind`` is only true when the incoming history has a non-empty
    answered-user prefix, which is what separates an undo (a later turn is being
    re-answered) from a title generator's one-message prefix.
    """
    shared = _common_prefix_len(stored, client)
    if 0 < shared < len(client):
        return shared
    if allow_rewind and 0 < shared == len(client) < len(stored):
        return shared
    return 0


# Client-supplied messages are untrusted, so a configured pass-through is
# ignored for roles/fields that can carry de-anonymised PCM output (assistant /
# tool turns) or PII a harness replayed.  Only the `system` role and the
# `refusal` field may still be passed through.
_PROTECTED_ROLES = frozenset({"user", "assistant", "tool", "function", "developer"})
_PROTECTED_FIELDS = frozenset(
    {
        "content",
        "reasoning",
        "reasoning_content",
        "name",
        "tool_calls.arguments",
        "function_calls.arguments",
        "function_call.arguments",
    }
)


def append_instruction_to_content(content: Any, instruction: str) -> Any:
    """Append the configured PII instruction to a message ``content`` value.

    Handles the three OpenAI content shapes without mutating the input:

    * ``None``            → the instruction alone;
    * ``str``             → ``content`` + blank line + instruction (or just the
      instruction when the existing content is empty);
    * ``list`` (multi-part) → a new ``{"type": "text", ...}`` part appended.
    """
    if not instruction:
        return content
    if content is None:
        return instruction
    if isinstance(content, str):
        return f"{content}\n\n{instruction}" if content else instruction
    if isinstance(content, list):
        return [*content, {"type": "text", "text": instruction}]
    return content


def _find_first_system_index(messages: list[dict[str, Any]]) -> int | None:
    """Return the index of the first ``role == "system"`` message, if any."""
    for index, message in enumerate(messages):
        if isinstance(message, dict) and message.get("role") == "system":
            return index
    return None


def _llm_chat_url(llm_url: str) -> str:
    return f"{llm_url.rstrip('/')}/v1/chat/completions"


def _upstream_error_detail(response: httpx.Response) -> Any:
    """Best-effort extraction of the upstream error body for the client."""
    try:
        return response.json()
    except ValueError:
        text = response.text
        return text[:2000] if text else "Upstream LLM error"


def _merkle_root(leaves: list[str]) -> str:
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


def _user_message_digest(message: Mapping[str, Any]) -> str:
    content = _extract_text_content(message.get("content"))
    return hashlib.sha256(content.encode("utf-8")).hexdigest()


def compute_user_hash(messages: list[dict[str, Any]]) -> str:
    """Merkle root over the ordered ``role == "user"`` messages.

    Returns the empty string when there are no user messages (e.g. before any
    user turn has been answered).
    """
    leaves = [
        _user_message_digest(message)
        for message in messages
        if message.get("role") == "user"
    ]
    return _merkle_root(leaves)


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


def _collect_text(messages: list[dict[str, Any]]) -> str:
    """Concatenate every text-bearing field of *messages*."""
    parts: list[str] = []
    for message in messages:
        if not isinstance(message, dict):
            continue
        for field in message_fields():
            for slot in field.iter_slots(message):
                text = slot.get()
                if text:
                    parts.append(text)
    return "\n".join(parts)


def _counters_from_placeholder_map(
    placeholder_map: Mapping[str, str],
) -> dict[str, int]:
    """Rebuild per-label counters from the highest index in *placeholder_map*.

    Placeholder keys are ``<LABEL_N>``; the counter for ``LABEL`` is the largest
    ``N`` seen.  Used when pruning a forked session so new spans continue the
    numbering without colliding with (or skipping past) retained tokens.
    """
    counters: dict[str, int] = {}
    for placeholder in placeholder_map:
        label, _, index = placeholder.rstrip(">").lstrip("<").rpartition("_")
        if label and index.isdigit():
            counters[label] = max(counters.get(label, 0), int(index))
    return counters


def _prune_privacy_state(
    state: PrivacyFilterState,
    retained_hidden: list[dict[str, Any]],
    retained_raw: list[dict[str, Any]],
) -> PrivacyFilterState:
    """Restrict *state* to the privacy facts of a session's retained prefix.

    A fork keeps only the shared message prefix, so inheriting the parent's full
    privacy state would carry over placeholder→PII mappings and audit results
    for turns that no longer exist on the branch.  A stale token that is later
    echoed (by the LLM or injected by a client) would then de-anonymise to PII
    from a discarded turn.  This keeps only:

    * placeholder-map entries whose token literally appears in the retained
      hidden messages (exactly the ones needed to de-anonymise the prefix);
    * counters rebuilt from those tokens;
    * audit results whose detected span text occurs in the retained raw
      messages (a discarded-only value is never exposed).
    """
    retained_hidden_text = _collect_text(retained_hidden)
    placeholder_map = {
        placeholder: value
        for placeholder, value in state.placeholder_map.items()
        if placeholder in retained_hidden_text
    }
    retained_raw_text = _collect_text(retained_raw)
    redaction_results = [
        result
        for result in state.redaction_results
        if all(
            span.get("text") and span["text"] in retained_raw_text
            for span in result.get("detected_spans", [])
        )
    ]
    return PrivacyFilterState(
        placeholder_map=placeholder_map,
        type_counters=_counters_from_placeholder_map(placeholder_map),
        redaction_results=redaction_results,
    )


def _new_session(
    client_value: str | None,
    prefix_user_hash: str,
    settings: Settings,
    llm_url: str,
) -> SessionData:
    """Build a fresh session with its client- and endpoint-facing ids."""
    now = time.time()
    session_id = str(uuid.uuid4())
    policy = endpoint_session_policy(llm_url, settings.llm_session_header)

    if client_value:
        client_x = client_value
    elif prefix_user_hash:
        client_x = f"auto-{prefix_user_hash[:12]}"
    else:
        client_x = f"auto-{uuid.uuid4().hex[:12]}"

    if policy.header_name is None:
        endpoint_x = None
    elif client_value and policy.validate(client_value):
        endpoint_x = client_value
    else:
        endpoint_x = policy.generate()

    return SessionData(
        session_id=session_id,
        created_at=now,
        updated_at=now,
        client_x_session_header=client_x,
        endpoint_x_session_header=endpoint_x,
        # A brand-new conversation is the root of its own lineage.
        root_session_id=session_id,
    )


def _branch_session(parent: SessionData, cursor: int) -> SessionData:
    """Fork *parent* into a new session seeded with its first *cursor* messages.

    Used when a client rewrites already-stored history (undo, edit, reorder, a
    divergent sub-agent, …).  The parent session is left **intact** so that the
    original conversation can still be recovered (e.g. a later redo replays the
    original history and resolves back to the parent).  The child inherits the
    parent's placeholder state so the shared prefix de-anonymises consistently,
    and keeps the parent's client-facing id so a fixed-header client continues to
    resolve the same family.
    """
    now = time.time()
    retained_raw = list(parent.raw_messages[:cursor])
    retained_hidden = list(parent.hidden_messages[:cursor])
    return SessionData(
        session_id=str(uuid.uuid4()),
        created_at=now,
        updated_at=now,
        raw_messages=retained_raw,
        hidden_messages=retained_hidden,
        # Prune the parent's privacy state to the retained prefix: a fork must
        # not inherit placeholder mappings or audit results for discarded turns.
        privacy_state=_prune_privacy_state(
            parent.privacy_state, retained_hidden, retained_raw
        ),
        client_x_session_header=parent.client_x_session_header,
        endpoint_x_session_header=parent.endpoint_x_session_header,
        # Git-like lineage: record where this branch came from.
        parent_session_id=parent.session_id,
        root_session_id=parent.root_session_id or parent.session_id,
        origin_message_count=cursor,
    )


def _log_user_hash_lookup(
    *,
    matched: bool,
    source: str,
    user_hash: str,
    session: SessionData,
) -> None:
    """Debug-log whether the incoming user-message hash matched a session."""
    logger.debug(
        "user_hash lookup",
        matched=matched,
        source=source,
        user_hash=user_hash or None,
        session_id=session.session_id,
        client_x_session_header=session.client_x_session_header,
        endpoint_x_session_header=session.endpoint_x_session_header,
    )


async def resolve_session(
    conn: aiosqlite.Connection,
    client_value: str | None,
    messages: list[dict[str, Any]],
    settings: Settings,
    llm_url: str | None = None,
) -> SessionData:
    """Recover (or create) the session that owns *messages*.

    Message content is the source of truth: a conversation is identified by the
    Merkle hash of its answered user messages (see :func:`compute_prefix_user_hash`).
    A client-supplied session id is only a hint — it selects a session when its
    hash agrees with the incoming history, but can never override the hash.

    Several sessions may share one hash when their conversation prefixes are
    identical (for example an agent's title generator and its main chat before
    they diverge); one is then chosen at random.  When nothing matches, a new
    session is created with freshly derived client/endpoint session ids.
    """
    prefix = compute_prefix_user_hash(messages)

    # Lazy expiry: a session idle longer than the TTL is treated as absent (and
    # the background sweeper will delete it), so a late request cannot revive it.
    cutoff = (
        time.time() - settings.session_ttl if settings.session_ttl > 0 else None
    )

    def _expired(session: SessionData) -> bool:
        return cutoff is not None and session.updated_at < cutoff

    header_sessions: list[SessionData] = []
    if client_value:
        header_sessions = [
            session
            for session in await find_sessions_by_client_header(conn, client_value)
            if not _expired(session)
        ]
    by_hash = (
        [
            session
            for session in await find_sessions_by_user_hash(conn, prefix)
            if not _expired(session)
        ]
        if prefix
        else []
    )
    by_history = (
        [
            session
            for session in await find_sessions_by_historical_user_hash(conn, prefix)
            if not _expired(session)
        ]
        if prefix
        else []
    )
    header_ids = {session.session_id for session in header_sessions}

    # Gather every candidate — the header hint, a current user-hash match, and a
    # historical prefix root — and pick the one that actually shares the longest
    # run of raw messages with the request; an exact user-hash match only breaks
    # ties.  Scoring by the shared prefix keeps a side-channel session (e.g. a
    # title generator whose single user message collides with the conversation's
    # opening) from shadowing the real conversation, which shares the messages.
    # Byte-identical prefixes (a genuine hash collision) are broken at random.
    candidates: dict[str, SessionData] = {}
    for sessions in (header_sessions, by_hash, by_history):
        for session in sessions:
            candidates.setdefault(session.session_id, session)

    best_score = (0, False)  # (shared prefix length, exact user-hash match)
    best: list[SessionData] = []
    for session in candidates.values():
        shared = _matching_prefix_len(
            session.raw_messages, messages, allow_rewind=bool(prefix)
        )
        exact = bool(prefix) and session.user_hash == prefix
        score = (shared, exact)
        if score > best_score:
            best_score, best = score, [session]
        elif score == best_score and score != (0, False):
            best.append(session)

    if best:
        session = random.choice(best)
        if best_score[1]:
            source = "client_header" if session.session_id in header_ids else "user_hash"
        else:
            source = (
                "client_header_resync"
                if session.session_id in header_ids
                else "history"
            )
        _log_user_hash_lookup(
            matched=True, source=source, user_hash=prefix, session=session
        )
        if best_score[0] > 0:
            logger.info(
                "session resolved by shared prefix",
                session_id=session.session_id,
                source=source,
                shared_prefix=best_score[0],
                client_count=len(messages),
            )
        if len(best) > 1:
            logger.warning(
                "user-message hash collision across sessions; picking one",
                user_hash=prefix,
                session_count=len(best),
            )
        if client_value and session.session_id not in header_ids:
            logger.warning(
                "client session header does not match its conversation history; "
                "using the message hash instead",
                client_x_session_header=client_value,
            )
        if (
            source == "user_hash"
            and client_value
            and session.client_x_session_header != client_value
        ):
            # Client rotated its id (e.g. a sub-agent); keep the binding.
            session.client_x_session_header = client_value
        return session

    session = _new_session(
        client_value, prefix, settings, llm_url or settings.llm_url
    )
    _log_user_hash_lookup(
        matched=False, source="new", user_hash=prefix, session=session
    )
    logger.info(
        "session created",
        session_id=session.session_id,
        client_x_session_header=session.client_x_session_header,
        endpoint_x_session_header=session.endpoint_x_session_header,
        user_hash=prefix or None,
    )
    return session


def _resolve_model(request: PrivateChatRequest, settings: Settings) -> str:
    """Return the model to forward upstream (client value wins).

    The client-supplied model is authoritative so the agent can pick any model
    the upstream exposes.  ``PCM_LLM_MODEL_NAME`` is only an optional fallback
    for clients that omit it.
    """
    model = (request.model or "").strip() or (settings.llm_model_name or "").strip()
    if not model:
        raise HTTPException(
            status_code=422,
            detail=(
                "No model specified: send a 'model' field or set "
                "PCM_LLM_MODEL_NAME as a fallback."
            ),
        )
    return model


# ---------------------------------------------------------------------------
# Prepared request (shared by streaming and non-streaming paths)
# ---------------------------------------------------------------------------


@dataclass
class PreparedRequest:
    """Everything needed to (a) call the LLM and (b) finish the session.

    Produced by :func:`prepare_request` and consumed either by
    :func:`handle_request` (buffered response) or by
    :func:`handle_stream_request` (server-sent events).
    """

    session_id: str
    session: SessionData
    placeholder_map: dict[str, str]
    llm_payload: dict[str, Any]
    headers: dict[str, str] = field(default_factory=dict)
    client_x_session_header: str | None = None
    llm_url: str = ""
    timings: RequestTimings | None = None


@dataclass
class _RedactionState:
    """Mutable accumulator shared across every redacted field and message.

    Carrying one set of counters/map/audit-results across the whole batch means
    the same real-world value always maps to the same placeholder token.
    """

    counters: dict[str, int]
    placeholder_map: dict[str, str]
    results: list[dict[str, Any]]


async def _redact_text(
    text: str,
    *,
    triton_client: TritonPrivacyFilterClient,
    state: _RedactionState,
    settings: Settings,
    session_id: str,
    role: str,
    field_name: str,
) -> str:
    """Run *text* through Triton and return the indexed, redacted text."""
    _t0 = time.monotonic()
    try:
        redaction_result: dict[str, Any] = await triton_client.infer(text)
    except Exception as exc:
        logger.exception(
            "privacy filter inference failed",
            session_id=session_id,
            role=role,
            field=field_name,
            char_count=len(text),
        )
        raise HTTPException(
            status_code=502,
            detail=(
                "Privacy filter (Triton) failed to process the message; "
                "the request was not forwarded to the LLM."
            ),
        ) from exc

    redacted, state.counters, state.placeholder_map = apply_placeholder_indexing(
        redaction_result, state.counters, state.placeholder_map
    )
    state.results.append(redaction_result)
    logger.info(
        "message redacted",
        session_id=session_id,
        role=role,
        field=field_name,
        span_count=len(redaction_result.get("detected_spans", [])),
        elapsed_ms=round((time.monotonic() - _t0) * 1000),
    )
    # NOTE: log below may contain PII (original span text).
    if "redaction_result" in settings.verbose_log_events:
        logger.debug(
            "redaction result",
            session_id=session_id,
            redaction_result=redaction_result,
        )
    return redacted


# ---------------------------------------------------------------------------
# Session / payload preparation
# ---------------------------------------------------------------------------


async def prepare_request(
    request: PrivateChatRequest,
    client_session_value: str | None,
    settings: Settings,
    triton_client: TritonPrivacyFilterClient,
    conn: aiosqlite.Connection,
    client_headers: Mapping[str, str] | None = None,
    llm_url: str | None = None,
    timings: RequestTimings | None = None,
) -> PreparedRequest:
    """Load the session, redact new messages and build the LLM payload.

    This is the shared front half of both the buffered and streaming handlers.
    It performs no LLM I/O and no session write; the caller is responsible for
    persisting the completed assistant turn.

    Workflow
    --------
    1.  Resolve (or create) the session from message history.
    2.  Validate that the last message is not an assistant turn.
    3.  For every new message, redact every known text field through Triton
        (default-deny) unless the role/field is explicitly passed through;
        append the fully-redacted copy to ``hidden_messages``.
    4.  Build the downstream LLM payload from the hidden history.
    """
    # ------------------------------------------------------------------
    # 1. Resolve the session from message history (and, as a hint, the
    #    client's session header).  See :func:`resolve_session`.
    # ------------------------------------------------------------------
    timings = timings or RequestTimings()
    if not request.messages:
        raise HTTPException(status_code=422, detail="messages list is empty")

    effective_llm_url = llm_url or settings.llm_url
    session = await resolve_session(
        conn, client_session_value, request.messages, settings, effective_llm_url
    )
    session_id = session.session_id
    timings.mark("resolve")

    if "request_body" in settings.verbose_log_events:
        logger.debug(
            "request body",
            session_id=session_id,
            client_x_session_header=client_session_value,
            request=request.model_dump(),
        )

    if session.hidden_messages:
        logger.info(
            "session resumed",
            session_id=session_id,
            stored_msg_count=len(session.hidden_messages),
        )

    # ------------------------------------------------------------------
    # 2. Identify new messages
    #
    # The PCM owns the canonical history in session.hidden_messages.
    # request.messages is the full client-side history (OpenAI convention).
    # Everything beyond the cursor (= current hidden history length) is new.
    #
    # On the very first turn the cursor is 0, so all messages are new.
    # ------------------------------------------------------------------
    cursor = len(session.hidden_messages)

    # The cursor is the length of the prefix the client replayed from its own
    # history.  It is computed by matching against ``raw_messages`` (which holds
    # exactly what the client sent / received), not assumed from the stored
    # length: a dropped, modified or empty assistant turn would otherwise shift
    # every later slice by one.
    #
    # Sessions are append-only: when the client's history diverges from the
    # stored one (undo, edit, reorder, or a different assistant rendering), PCM
    # does not rewrite the existing session — it forks a child seeded with the
    # shared prefix.  The parent is left intact, so the original stays
    # recoverable, and the conversation forms a git-like lineage.
    stored_raw = session.raw_messages
    stored_len = min(len(stored_raw), len(session.hidden_messages))
    cursor = _common_prefix_len(stored_raw[:stored_len], request.messages)
    # The new region never starts after the last turn awaiting a reply.  This is
    # what makes a strict-prefix undo work: when the client re-submits the turn
    # unchanged (dropping its stored reply), the cursor drops back to that turn,
    # so it is re-answered on a branch (only that turn is re-redacted) instead of
    # failing with "no new messages".
    turn_start = _turn_start(request.messages)
    if 0 <= turn_start < cursor:
        logger.info(
            "cursor aligned to the last turn awaiting a reply",
            session_id=session_id,
            cursor=turn_start,
            role=request.messages[turn_start].get("role"),
        )
        cursor = turn_start
    resynced = cursor < stored_len

    if resynced:
        parent = session
        session = _branch_session(parent, cursor)
        session_id = session.session_id
        logger.info(
            "branched conversation (client history diverged)",
            parent_session_id=parent.session_id,
            session_id=session_id,
            root_session_id=session.root_session_id,
            shared_prefix=cursor,
            client_count=len(request.messages),
        )

    new_messages: list[dict[str, Any]] = [dict(m) for m in request.messages[cursor:]]
    logger.info(
        "processing new messages",
        session_id=session_id,
        cursor=cursor,
        new_msg_count=len(new_messages),
        resynced=resynced,
        bypass=request.bypass_privacy_filter,
    )

    if not new_messages:
        raise HTTPException(
            status_code=422,
            detail=(
                "No new messages detected. The submitted history matches what "
                "is already cached for this session."
            ),
        )

    last_role = new_messages[-1].get("role")
    if last_role == "assistant":
        raise HTTPException(
            status_code=422,
            detail=(
                "The last new message has role 'assistant'. "
                "Requests must end with a 'user', 'tool', or 'function' message."
            ),
        )

    # ------------------------------------------------------------------
    # Fresh conversation replaying prior turns.
    #
    # When cursor == 0 the client is submitting a full history for a session PCM
    # does not have (first contact, TTL expiry, DB reset, or a hash miss).  Any
    # assistant/tool turns in it may carry de-anonymised PII produced by PCM
    # itself, so they are untrusted input and are redacted like everything else.
    # (The old assumption that those turns were already exposed upstream was
    # wrong for conversations PCM had processed.)
    # ------------------------------------------------------------------
    if cursor == 0:
        replayed = [
            m
            for m in new_messages
            if m.get("role") in ("assistant", "tool", "function")
        ]
        if replayed:
            logger.warning(
                "fresh conversation replays prior turns; redacting them as untrusted",
                session_id=session_id,
                replayed_count=len(replayed),
            )

    # ------------------------------------------------------------------
    # 4 / 5. Process each new message.
    #
    # Default-deny: every role and every known text field is redacted unless
    # explicitly listed in the applicable pass-through setting.  Fresh
    # conversations use the *_FRESH sets, which do NOT inherit the resumed sets.
    #
    # type_counters and placeholder_map accumulate across ALL fields and
    # messages in this batch, so the same value always maps to the same token.
    # ------------------------------------------------------------------
    # A first turn and a resynchronised (replayed) tail are both untrusted:
    # use the strict *_FRESH pass-through sets for them.
    fresh = cursor == 0 or resynced
    passthrough_roles = (
        settings.passthrough_roles_fresh if fresh else settings.passthrough_roles
    )
    passthrough_fields = (
        settings.passthrough_fields_fresh if fresh else settings.passthrough_fields
    )

    # Client-supplied messages are untrusted: report (and ignore) any configured
    # pass-through for a protected role/field so operators know it had no effect.
    ignored_passthrough = sorted(
        (passthrough_roles & _PROTECTED_ROLES)
        | (passthrough_fields & _PROTECTED_FIELDS)
    )
    if ignored_passthrough:
        logger.warning(
            "ignoring configured pass-through for protected roles/fields on client messages",
            session_id=session_id,
            ignored=ignored_passthrough,
        )

    state = _RedactionState(
        counters=dict(session.privacy_state.type_counters),
        placeholder_map=dict(session.privacy_state.placeholder_map),
        results=[],
    )

    # The configured PII instruction is appended to the *first* client system
    # message in this batch (persisted once in hidden history).  If the client
    # sent no system message at all, a synthetic one is added to the forwarded
    # payload only — inserting it into hidden history would desynchronise the
    # cursor used to detect new messages on the next turn.
    instruction = settings.system_prompt_pii_instruction
    target_system_index = (
        _find_first_system_index(new_messages) if instruction else None
    )

    last_index = len(new_messages) - 1
    for index, msg in enumerate(new_messages):
        role = msg.get("role", "")
        session.raw_messages.append(msg)

        # A fully independent copy so the redacted history never aliases (or
        # mutates) the raw client history.
        hidden_msg: dict[str, Any] = copy.deepcopy(msg)

        # ``bypass_privacy_filter`` only skips redaction of the final user turn.
        bypass_this = (
            request.bypass_privacy_filter and index == last_index and role == "user"
        )

        # A configured pass-through never applies to a protected role/field on a
        # client-supplied message (see _PROTECTED_ROLES / _PROTECTED_FIELDS).
        role_passthrough = (
            role in passthrough_roles and role not in _PROTECTED_ROLES
        )

        if not role_passthrough and not bypass_this:
            for field in message_fields():
                if (
                    field.name in passthrough_fields
                    and field.name not in _PROTECTED_FIELDS
                ):
                    continue
                for slot in field.iter_slots(hidden_msg):
                    text = slot.get()
                    if not text or not text.strip():
                        continue
                    slot.set(
                        await _redact_text(
                            text,
                            triton_client=triton_client,
                            state=state,
                            settings=settings,
                            session_id=session_id,
                            role=role,
                            field_name=field.name,
                        )
                    )

        if index == target_system_index:
            # Appended *after* redaction so the instruction never reaches Triton.
            hidden_msg["content"] = append_instruction_to_content(
                hidden_msg.get("content"), instruction
            )

        session.hidden_messages.append(hidden_msg)

    timings.mark("redact")

    # Commit accumulated privacy state updates
    session.privacy_state = PrivacyFilterState(
        placeholder_map=state.placeholder_map,
        type_counters=state.counters,
        redaction_results=[
            *session.privacy_state.redaction_results,
            *state.results,
        ],
    )

    # ------------------------------------------------------------------
    # 6. Build the forwarded payload: preserve all original sampling params,
    #    use the client's model (falling back to PCM_LLM_MODEL_NAME), and
    #    replace messages with the hidden (redacted) history.
    # ------------------------------------------------------------------
    model = _resolve_model(request, settings)
    llm_payload = request.model_dump(
        exclude={"session_id", "bypass_privacy_filter", "messages", "model"},
        exclude_none=True,
    )
    llm_payload["model"] = model
    llm_payload["messages"] = session.hidden_messages
    llm_payload["stream"] = bool(request.stream)

    # No client system message anywhere in the session: inject a synthetic one
    # into the payload only (never persisted, to keep the cursor aligned).
    if instruction and _find_first_system_index(session.hidden_messages) is None:
        llm_payload["messages"] = [
            {"role": "system", "content": instruction},
            *session.hidden_messages,
        ]

    # Emit the PCM-controlled endpoint session id under the header the
    # endpoint expects (if any), overriding any client-supplied value.  The
    # endpoint id is derived per request, not only at creation: an existing
    # session resolved by hash/header may predate a policy change (e.g. an
    # ``X-PCM-LLM-URL`` override that selects an endpoint requiring a session
    # header), so it is validated/generated against the *effective* policy.
    policy = endpoint_session_policy(effective_llm_url, settings.llm_session_header)
    if policy.header_name is not None and not policy.validate(
        session.endpoint_x_session_header
    ):
        session.endpoint_x_session_header = (
            client_session_value
            if client_session_value and policy.validate(client_session_value)
            else policy.generate()
        )
    forward_headers = build_forward_headers(
        client_headers,
        api_key=settings.llm_api_key,
        session_header=policy.header_name,
        session_id=session.endpoint_x_session_header,
    )

    if "llm_payload" in settings.verbose_log_events:
        logger.debug(
            "LLM payload",  # hidden messages are already redacted — no PII
            session_id=session_id,
            payload=llm_payload,
            headers=redact_headers(forward_headers),
        )

    return PreparedRequest(
        session_id=session_id,
        session=session,
        placeholder_map=state.placeholder_map,
        llm_payload=llm_payload,
        headers=forward_headers,
        client_x_session_header=session.client_x_session_header,
        llm_url=effective_llm_url,
        timings=timings,
    )


# ---------------------------------------------------------------------------
# Buffered (non-streaming) handler
# ---------------------------------------------------------------------------


async def handle_request(
    request: PrivateChatRequest,
    client_session_value: str | None,
    settings: Settings,
    triton_client: TritonPrivacyFilterClient,
    conn: aiosqlite.Connection,
    client_headers: Mapping[str, str] | None = None,
    llm_url: str | None = None,
    timings: RequestTimings | None = None,
) -> tuple[dict[str, Any], str | None]:
    """Process one buffered chat-completion request through the privacy layer.

    Returns ``(response_dict, client_x_session_header)`` where *response_dict*
    is the de-anonymised OpenAI-compatible JSON payload and the second element
    is the client-facing session id to return in ``X-Session-ID``.
    """

    prepared = await prepare_request(
        request=request,
        client_session_value=client_session_value,
        settings=settings,
        triton_client=triton_client,
        conn=conn,
        client_headers=client_headers,
        llm_url=llm_url,
        timings=timings,
    )
    timings = prepared.timings or RequestTimings()
    session = prepared.session
    session_id = prepared.session_id
    placeholder_map = prepared.placeholder_map

    # ------------------------------------------------------------------
    # Forward hidden messages to the downstream LLM
    # ------------------------------------------------------------------
    logger.info(
        "forwarding to LLM",
        session_id=session_id,
        model=prepared.llm_payload.get("model"),
        hidden_msg_count=len(session.hidden_messages),
    )

    _t0 = time.monotonic()
    async with httpx.AsyncClient(timeout=120.0) as http_client:
        try:
            llm_resp = await http_client.post(
                _llm_chat_url(prepared.llm_url or settings.llm_url),
                json=prepared.llm_payload,
                headers=prepared.headers,
            )
            llm_resp.raise_for_status()
        except httpx.HTTPStatusError as exc:
            logger.error(
                "upstream LLM rejected the request",
                session_id=session_id,
                status_code=exc.response.status_code,
            )
            raise HTTPException(
                status_code=exc.response.status_code,
                detail=_upstream_error_detail(exc.response),
            ) from exc
        except httpx.HTTPError as exc:
            logger.error(
                "upstream LLM transport error",
                session_id=session_id,
                error=str(exc),
            )
            raise HTTPException(
                status_code=502,
                detail="Upstream LLM transport error",
            ) from exc
    _llm_elapsed_ms = round((time.monotonic() - _t0) * 1000)
    timings.mark("upstream")

    llm_data: dict[str, Any] = llm_resp.json()
    _finish_reason = (
        llm_data["choices"][0].get("finish_reason") if llm_data.get("choices") else None
    )
    _usage = llm_data.get("usage", {})
    logger.info(
        "LLM response received",
        session_id=session_id,
        status_code=llm_resp.status_code,
        elapsed_ms=_llm_elapsed_ms,
        finish_reason=_finish_reason,
        prompt_tokens=_usage.get("prompt_tokens"),
        completion_tokens=_usage.get("completion_tokens"),
    )

    if "llm_raw_response" in settings.verbose_log_events:
        logger.debug("LLM raw response body", session_id=session_id, response=llm_data)

    # ------------------------------------------------------------------
    # De-anonymise the LLM response for the caller.
    #    We work on a deep copy so that llm_data retains the original
    #    (placeholder) version for the session's hidden history.
    # ------------------------------------------------------------------
    response_data: dict[str, Any] = copy.deepcopy(llm_data)
    for choice in response_data.get("choices", []):
        msg = choice.get("message", {})
        _deanonymize_message(msg, placeholder_map)

    # ------------------------------------------------------------------
    # Persist the assistant turn in both session histories.
    #    hidden  → the raw LLM output (still contains placeholders)
    #    raw     → the de-anonymised version shown to the caller
    # ------------------------------------------------------------------
    if llm_data.get("choices"):
        orig_msg = llm_data["choices"][0].get("message", {})
        deano_msg = response_data["choices"][0].get("message", {})

        hidden_assistant = _strip_none(
            {
                "role": orig_msg.get("role", "assistant"),
                "content": orig_msg.get("content"),
                "tool_calls": orig_msg.get("tool_calls") or None,
            }
        )
        raw_assistant = _strip_none(
            {
                "role": deano_msg.get("role", "assistant"),
                "content": deano_msg.get("content"),
                "tool_calls": deano_msg.get("tool_calls") or None,
            }
        )

        # Persist the assistant turn only when it carries content or tool calls;
        # a reasoning/refusal-only reply (content null) is not stored.  This
        # keeps the cursor aligned with the client for clients that do not
        # replay an empty assistant message, and matches the streaming path.
        if _assistant_has_output(hidden_assistant) or _assistant_has_output(
            raw_assistant
        ):
            session.hidden_messages.append(hidden_assistant)
            session.raw_messages.append(raw_assistant)
        else:
            logger.warning(
                "assistant turn had no content/tool_calls; persisting user turn only",
                session_id=session_id,
            )

    # ------------------------------------------------------------------
    # Save session and return
    # ------------------------------------------------------------------
    session.user_hash = compute_user_hash(session.raw_messages)
    session.updated_at = time.time()
    await save_session(conn, session)
    logger.info(
        "session saved",
        session_id=session_id,
        raw_msg_count=len(session.raw_messages),
        hidden_msg_count=len(session.hidden_messages),
    )
    if "session_state" in settings.verbose_log_events:
        logger.debug(
            "session state",
            session_id=session_id,
            raw_messages=session.raw_messages,
            hidden_messages=session.hidden_messages,
            privacy_state=session.privacy_state.model_dump(),
        )

    timings.mark("finalize")
    timings.log_timing(
        session_id=session_id,
        model=prepared.llm_payload.get("model"),
        stream=False,
    )

    return response_data, prepared.client_x_session_header


def _deanonymize_message(msg: dict[str, Any], placeholder_map: dict[str, str]) -> None:
    """Substitute placeholders back into every text field of *msg* in place.

    Mirrors the field registry (:mod:`app.message_fields`): ``content``, the
    reasoning side channel (``reasoning`` / ``reasoning_content``), ``refusal``,
    ``name``, tool calls and the legacy ``function_calls`` / ``function_call``
    arguments.  The legacy fields are **not** persisted to the session history
    (see ``handle_request``); they are only de-anonymised here so an exotic
    backend cannot leak raw placeholders to the client.
    """
    if msg.get("content"):
        msg["content"] = deanonymize_text(msg["content"], placeholder_map)
    for key in ("reasoning", "reasoning_content", "refusal", "name"):
        if msg.get(key):
            msg[key] = deanonymize_text(msg[key], placeholder_map)
    # Tool calls
    for tc in msg.get("tool_calls") or []:
        fn = tc.get("function", {})
        if fn.get("arguments"):
            fn["arguments"] = deanonymize_text(fn["arguments"], placeholder_map)
    # Function calls (legacy role)
    for tc in msg.get("function_calls") or []:
        if tc.get("arguments"):
            tc["arguments"] = deanonymize_text(tc["arguments"], placeholder_map)
    # Function call (legacy single object)
    fc = msg.get("function_call")
    if isinstance(fc, dict) and fc.get("arguments"):
        fc["arguments"] = deanonymize_text(fc["arguments"], placeholder_map)


# ---------------------------------------------------------------------------
# Streaming (Server-Sent Events) handler
# ---------------------------------------------------------------------------


def _sse(payload: dict[str, Any]) -> str:
    """Serialize a chunk object as one SSE ``data:`` event."""
    return "data: " + json.dumps(payload, ensure_ascii=False, separators=(",", ":")) + "\n\n"


def _sse_error(message: str) -> str:
    """Emit an OpenAI-style streaming error event, then terminators.

    The raw upstream body is intentionally not forwarded — it can contain
    internal details.  Only a coarse, safe message is sent to the client.
    """
    error_chunk = {
        "error": {
            "message": message,
            "type": "upstream_error",
            "code": "stream_error",
        }
    }
    return _sse(error_chunk) + "data: [DONE]\n\n"


def _residual_events(
    meta: dict[str, Any],
    residuals: dict[int, dict[str, str]],
) -> list[str]:
    """Build chunks for text drained from partial tags at end of stream."""
    events: list[str] = []
    for index, fields in residuals.items():
        delta: dict[str, str] = {}
        for key in ("content", "reasoning", "reasoning_content"):
            if fields.get(key):
                delta[key] = fields[key]
        if not delta:
            continue
        events.append(
            _sse(
                {
                    "id": meta.get("id", ""),
                    "object": "chat.completion.chunk",
                    "created": meta.get("created", int(time.time())),
                    "model": meta.get("model", ""),
                    "choices": [
                        {
                            "index": index,
                            "delta": delta,
                            "logprobs": None,
                            "finish_reason": None,
                        }
                    ],
                }
            )
        )
    return events


async def _persist_stream_session(
    session: SessionData,
    stream_filter: StreamResponseFilter,
    session_id: str,
    settings: Settings,
) -> None:
    """Append the streamed assistant turn to both histories and save."""
    hidden_msg, raw_msg = stream_filter.assistant_history(0)
    hidden_msg = _strip_none(hidden_msg)
    raw_msg = _strip_none(raw_msg)

    has_output = _assistant_has_output(hidden_msg) or _assistant_has_output(raw_msg)
    # A choice with only reasoning (no content/tool_calls) is a legitimate turn:
    # reasoning is intentionally never persisted, but the user turn must still be
    # saved so the conversation can continue.  A stream that produced no choice
    # at all (upstream error before any chunk) is not persisted, so a retry of
    # the same request starts cleanly instead of hitting "no new messages".
    saw_choice = bool(stream_filter.choice_indices())
    if not has_output and not saw_choice:
        logger.warning("stream produced no assistant output; not persisting turn",
                       session_id=session_id)
        return

    if has_output:
        session.hidden_messages.append(hidden_msg)
        session.raw_messages.append(raw_msg)
    else:
        logger.warning(
            "stream produced reasoning only; persisting user turn without assistant reply",
            session_id=session_id,
        )

    session.user_hash = compute_user_hash(session.raw_messages)
    session.updated_at = time.time()

    async with aiosqlite.connect(settings.db_path) as conn:
        await conn.execute("PRAGMA journal_mode=WAL")
        await save_session(conn, session)

    logger.info(
        "stream session saved",
        session_id=session_id,
        raw_msg_count=len(session.raw_messages),
        hidden_msg_count=len(session.hidden_messages),
    )
    if "session_state" in settings.verbose_log_events:
        logger.debug(
            "stream session state",
            session_id=session_id,
            raw_messages=session.raw_messages,
            hidden_messages=session.hidden_messages,
            privacy_state=session.privacy_state.model_dump(),
        )


async def handle_stream_request(
    prepared: PreparedRequest,
    settings: Settings,
) -> AsyncIterator[str]:
    """Forward to the LLM with ``stream=true`` and de-anonymise deltas live.

    Yields raw SSE frames (``data: {json}\\n\\n``) suitable for a FastAPI
    :class:`~fastapi.responses.StreamingResponse`.  The assistant turn is
    persisted to the session when the stream terminates (normally, on error,
    or on client disconnect) so the next turn sees a coherent history.

    De-anonymisation is incremental: see :mod:`app.streaming` for the exact
    detection/replacement algorithm.
    """
    session = prepared.session
    session_id = prepared.session_id
    timings = prepared.timings or RequestTimings()
    stream_filter = StreamResponseFilter(
        prepared.placeholder_map,
        settings.placeholder_labels,
    )

    logger.info(
        "forwarding streaming request to LLM",
        session_id=session_id,
        model=prepared.llm_payload.get("model"),
        hidden_msg_count=len(session.hidden_messages),
    )

    # Metadata copied from the last received chunk so residual events can be
    # shaped like normal assistant chunks.
    meta: dict[str, Any] = {}
    finalized = False

    def _finalize() -> dict[int, dict[str, str]]:
        nonlocal finalized
        if finalized:
            return {}
        finalized = True
        residuals: dict[int, dict[str, str]] = {}
        for index in stream_filter.choice_indices():
            drained = stream_filter.flush_choice(index)
            if drained:
                residuals[index] = drained
        return residuals

    _t0 = time.monotonic()
    seen_usage = False
    finished = False

    async def _finish() -> None:
        """Flush, log and persist the streamed turn exactly once.

        Persisting *before* the terminal ``[DONE]`` frame guarantees the
        session is available to the very next request, and shields the write
        from a client that disconnects the instant it sees ``[DONE]``.
        """
        nonlocal finished
        if finished:
            return
        finished = True

        elapsed_ms = round((time.monotonic() - _t0) * 1000)
        _finalize()

        # Mirror the buffered path's ``llm_raw_response`` / ``response_body``
        # debug events using the fully assembled (post-flush) stream.
        if settings.verbose_log_events & {"llm_raw_response", "response_body"}:
            for choice_index in stream_filter.choice_indices():
                client_msg = stream_filter.client_message(choice_index)
                has_output = (
                    client_msg.get("content") is not None
                    or client_msg.get("reasoning")
                    or client_msg.get("tool_calls")
                )
                if not has_output:
                    continue
                if "llm_raw_response" in settings.verbose_log_events:
                    logger.debug(
                        "stream raw response",  # placeholders only — no PII
                        session_id=session_id,
                        choice_index=choice_index,
                        message=stream_filter.raw_message(choice_index),
                    )
                if "response_body" in settings.verbose_log_events:
                    logger.debug(
                        "stream response body",  # ⚠ de-anonymised — contains PII
                        session_id=session_id,
                        choice_index=choice_index,
                        message=client_msg,
                    )

        try:
            await _persist_stream_session(session, stream_filter, session_id, settings)
        except Exception:
            logger.exception("failed to persist streamed session", session_id=session_id)
        logger.info(
            "stream finished",
            session_id=session_id,
            elapsed_ms=elapsed_ms,
            usage_seen=seen_usage,
        )
        timings.mark("finalize")
        timings.log_timing(
            session_id=session_id,
            model=prepared.llm_payload.get("model"),
            stream=True,
        )

    try:
        # ``read=None`` disables the read timeout for long-lived streams.
        timeout = httpx.Timeout(connect=10.0, read=None, write=30.0, pool=10.0)
        async with httpx.AsyncClient(timeout=timeout) as http_client:
            async with http_client.stream(
                "POST",
                _llm_chat_url(prepared.llm_url or settings.llm_url),
                json=prepared.llm_payload,
                headers=prepared.headers,
            ) as llm_resp:
                llm_resp.raise_for_status()

                async for line in llm_resp.aiter_lines():
                    if line.startswith("data:"):
                        payload = line[len("data:"):].strip()
                        if payload == "[DONE]":
                            break
                        if not payload:
                            continue
                        try:
                            chunk = json.loads(payload)
                        except json.JSONDecodeError:
                            logger.warning(
                                "malformed streaming chunk skipped",
                                session_id=session_id,
                                raw=payload[:200],
                            )
                            continue

                        if chunk.get("id") or chunk.get("model"):
                            meta = {
                                "id": chunk.get("id", meta.get("id", "")),
                                "model": chunk.get("model", meta.get("model", "")),
                                "created": chunk.get("created", meta.get("created", int(time.time()))),
                            }
                        if chunk.get("usage"):
                            seen_usage = True

                        stream_filter.process_chunk(chunk)
                        yield _sse(chunk)
                    elif not line.strip():
                        # SSE event separator: nothing to forward.
                        continue
                    else:
                        # Preserve any non-data SSE field (comments, event lines).
                        yield line + "\n\n"

        timings.mark("upstream")
        for event in _residual_events(meta, _finalize()):
            yield event
        # Commit the session before signalling completion so a follow-up
        # request (or a client that disconnects on [DONE]) sees a coherent
        # history.  ``shield`` keeps the write alive under cancellation.
        await asyncio.shield(_finish())
        yield "data: [DONE]\n\n"

    except httpx.HTTPStatusError as exc:
        logger.error(
            "upstream LLM rejected streaming request",
            session_id=session_id,
            status_code=exc.response.status_code,
        )
        yield _sse_error(f"Upstream LLM returned HTTP {exc.response.status_code}")
    except httpx.HTTPError as exc:
        logger.error(
            "upstream LLM streaming transport error",
            session_id=session_id,
            error=str(exc),
        )
        yield _sse_error("Upstream LLM streaming transport error")
    except Exception:
        logger.exception("unexpected error during streaming", session_id=session_id)
        yield _sse_error("Internal error during streaming")
    finally:
        # Best-effort fallback for the error / client-disconnect paths.  The
        # shield ensures a disconnect cannot cancel the DB write mid-flight.
        try:
            await asyncio.shield(_finish())
        except BaseException:
            pass
