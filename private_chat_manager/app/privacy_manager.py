from __future__ import annotations

import asyncio
import copy
import hashlib
import json
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
from .headers import (
    build_forward_headers,
    default_session_header_for,
    redact_headers,
)
from .models import PrivacyFilterState, PrivateChatRequest, SessionData
from .session_store import get_session, save_session
from .streaming import StreamResponseFilter
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


def _llm_chat_url(settings: Settings) -> str:
    return f"{settings.llm_url.rstrip('/')}/v1/chat/completions"


def _upstream_error_detail(response: httpx.Response) -> Any:
    """Best-effort extraction of the upstream error body for the client."""
    try:
        return response.json()
    except ValueError:
        text = response.text
        return text[:2000] if text else "Upstream LLM error"


def _conversation_fingerprint(messages: list[dict[str, Any]]) -> str:
    """Return a short, stable hash identifying a logical conversation.

    Derived from the first user message (falling back to the first non-empty
    message).  An agent's title/side-channel request and its main chat have
    different first user messages, so they map to separate sessions instead of
    sharing — and corrupting — one history cursor.
    """
    for role in ("user",):
        for msg in messages:
            if msg.get("role") == role:
                content = _extract_text_content(msg.get("content"))
                if content.strip():
                    return hashlib.sha1(content.encode("utf-8")).hexdigest()[:12]
    for msg in messages:
        content = _extract_text_content(msg.get("content"))
        if content.strip():
            return hashlib.sha1(content.encode("utf-8")).hexdigest()[:12]
    return "0"


def resolve_session_key(
    base_id: str | None, messages: list[dict[str, Any]]
) -> str:
    """Resolve the internal session key for a request.

    When the client supplies a session id (from any ``x…session…`` header) it
    is namespaced by a conversation fingerprint.  When it supplies none — e.g.
    the Hermes agent with a custom endpoint — the fingerprint itself becomes
    the key, so turns of one conversation still accumulate history instead of
    each getting a random id.

    The message list sent to the LLM is unaffected.  A value already containing
    the ``::`` separator is assumed to be a previously-resolved key and
    returned unchanged, so clients that echo the ``X-Session-ID`` response
    header remain stable.
    """
    fingerprint = _conversation_fingerprint(messages)
    if not base_id:
        return f"auto-{fingerprint}"
    if "::" in base_id:
        return base_id
    return f"{base_id}::{fingerprint}"


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


# ---------------------------------------------------------------------------
# Session / payload preparation
# ---------------------------------------------------------------------------


async def prepare_request(
    request: PrivateChatRequest,
    session_id_header: str | None,
    settings: Settings,
    triton_client: TritonPrivacyFilterClient,
    conn: aiosqlite.Connection,
    client_headers: Mapping[str, str] | None = None,
) -> PreparedRequest:
    """Load the session, redact new messages and build the LLM payload.

    This is the shared front half of both the buffered and streaming handlers.
    It performs no LLM I/O and no session write; the caller is responsible for
    persisting the completed assistant turn.

    Workflow
    --------
    1.  Resolve (or generate) a session ID.
    2.  Load or initialise session state from SQLite.
    3.  Validate that the last message is not an assistant turn.
    4.  Pass-through any non-filterable messages (system prompts, etc.).
    5.  Send each filterable message to Triton, apply indexed placeholder
        substitution, append the redacted copy to ``hidden_messages``.
    6.  Build the downstream LLM payload from the hidden history.
    """
    # ------------------------------------------------------------------
    # 1. Resolve session ID: X-Session-ID header > new UUID
    # ------------------------------------------------------------------
    session_id: str = session_id_header or str(uuid.uuid4())
    logger.debug("session ID resolved", session_id=session_id)

    if "request_body" in settings.verbose_log_events:
        logger.debug(
            "request body",
            session_id=session_id,
            x_session_id_header=session_id_header,
            request=request.model_dump(),
        )

    # ------------------------------------------------------------------
    # 2. Load or initialise SessionData
    # ------------------------------------------------------------------
    session = await get_session(conn, session_id)
    if session is None:
        now = time.time()
        session = SessionData(
            session_id=session_id,
            created_at=now,
            updated_at=now,
        )
        logger.info("session created", session_id=session_id)
    else:
        logger.info(
            "session resumed",
            session_id=session_id,
            stored_msg_count=len(session.hidden_messages),
        )

    # ------------------------------------------------------------------
    # 3. Identify new messages
    #
    # The PCM owns the canonical history in session.hidden_messages.
    # request.messages is the full client-side history (OpenAI convention).
    # Everything beyond the cursor (= current hidden history length) is new.
    #
    # On the very first turn the cursor is 0, so all messages are new.
    # ------------------------------------------------------------------
    if not request.messages:
        raise HTTPException(status_code=422, detail="messages list is empty")

    cursor = len(session.hidden_messages)
    new_messages: list[dict[str, Any]] = [dict(m) for m in request.messages[cursor:]]
    logger.info(
        "processing new messages",
        session_id=session_id,
        cursor=cursor,
        new_msg_count=len(new_messages),
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
    # Pre-existing conversation warning.
    #
    # When cursor == 0 and new_messages contains assistant turns in the
    # middle, the client is replaying a full prior conversation that was
    # NOT originally processed by PCM.  Those assistant turns were generated
    # by an LLM that already received un-redacted user input, so the PII
    # was exposed upstream — filtering them now provides no privacy benefit.
    # We log a warning so operators are aware of the session origin.
    # ------------------------------------------------------------------
    if cursor == 0 and not request.bypass_privacy_filter:
        replayed_assistant_count = sum(
            1 for m in new_messages if m.get("role") == "assistant"
        )
        if replayed_assistant_count:
            logger.warning(
                "replaying pre-existing conversation not processed by PCM",
                session_id=session_id,
                replayed_assistant_count=replayed_assistant_count,
                note="PII in those turns was already exposed to the upstream LLM",
            )

    # ------------------------------------------------------------------
    # 4 / 5. Process each new message.
    #
    # Which roles are filtered is controlled by settings.filterable_roles
    # (env var PCM_FILTERABLE_ROLES, default: user,tool,function).
    # Roles absent from that set pass through unchanged.
    #
    # type_counters and placeholder_map accumulate across ALL new messages
    # in this batch so that two tool replies in the same request that
    # contain the same label class get sequential indices, not both "_1".
    # ------------------------------------------------------------------
    new_counters = dict(session.privacy_state.type_counters)
    new_map = dict(session.privacy_state.placeholder_map)
    new_redaction_results: list[dict[str, Any]] = []

    # The configured PII instruction is appended to the *first* client system
    # message in this batch (persisted once in hidden history).  If the client
    # sent no system message at all, a synthetic one is added to the forwarded
    # payload only — inserting it into hidden history would desynchronise the
    # cursor used to detect new messages on the next turn.
    instruction = settings.system_prompt_pii_instruction
    target_system_index = (
        _find_first_system_index(new_messages) if instruction else None
    )

    for index, msg in enumerate(new_messages):
        role = msg.get("role", "")
        session.raw_messages.append(msg)

        if role not in settings.filterable_roles or request.bypass_privacy_filter:
            # Pass through unchanged (role not in filterable_roles, or bypass active)
            hidden_content: Any = msg.get("content")
        else:
            # --- Filter this message through Triton ---
            content = _extract_text_content(msg.get("content"))
            _t0 = time.monotonic()
            try:
                redaction_result: dict[str, Any] = await triton_client.infer(content)
            except Exception as exc:
                logger.exception(
                    "privacy filter inference failed",
                    session_id=session_id,
                    role=role,
                    char_count=len(content),
                )
                raise HTTPException(
                    status_code=502,
                    detail=(
                        "Privacy filter (Triton) failed to process the message; "
                        "the request was not forwarded to the LLM."
                    ),
                ) from exc
            _elapsed_ms = round((time.monotonic() - _t0) * 1000)

            span_count = len(redaction_result.get("detected_spans", []))
            logger.info(
                "message redacted",
                session_id=session_id,
                role=role,
                span_count=span_count,
                elapsed_ms=_elapsed_ms,
            )
            # NOTE: log below may contain PII (original span text).
            if "redaction_result" in settings.verbose_log_events:
                logger.debug(
                    "redaction result",
                    session_id=session_id,
                    redaction_result=redaction_result,
                )

            hidden_content, new_counters, new_map = apply_placeholder_indexing(
                redaction_result,
                new_counters,
                new_map,
            )
            new_redaction_results.append(redaction_result)

        if index == target_system_index:
            # Appended *after* redaction so the instruction never reaches Triton.
            hidden_content = append_instruction_to_content(hidden_content, instruction)

        # Store a copy so hidden history never aliases (and cannot mutate) the
        # raw history for pass-through roles.
        session.hidden_messages.append({**msg, "content": hidden_content})

    # Commit accumulated privacy state updates
    session.privacy_state = PrivacyFilterState(
        placeholder_map=new_map,
        type_counters=new_counters,
        redaction_results=[
            *session.privacy_state.redaction_results,
            *new_redaction_results,
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

    # Default the upstream session header to the one an OpenCode relay
    # requires when the operator has not set one explicitly.
    session_header = settings.llm_session_header or default_session_header_for(
        settings.llm_url
    )
    forward_headers = build_forward_headers(
        client_headers,
        api_key=settings.llm_api_key,
        session_header=session_header,
        session_id=session_id,
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
        placeholder_map=new_map,
        llm_payload=llm_payload,
        headers=forward_headers,
    )


# ---------------------------------------------------------------------------
# Buffered (non-streaming) handler
# ---------------------------------------------------------------------------


async def handle_request(
    request: PrivateChatRequest,
    session_id_header: str | None,
    settings: Settings,
    triton_client: TritonPrivacyFilterClient,
    conn: aiosqlite.Connection,
    client_headers: Mapping[str, str] | None = None,
) -> tuple[dict[str, Any], str]:
    """Process one buffered chat-completion request through the privacy layer.

    Returns ``(response_dict, session_id)`` where *response_dict* is the
    de-anonymised OpenAI-compatible JSON payload.
    """

    prepared = await prepare_request(
        request=request,
        session_id_header=session_id_header,
        settings=settings,
        triton_client=triton_client,
        conn=conn,
        client_headers=client_headers,
    )
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
                _llm_chat_url(settings),
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

        session.hidden_messages.append(
            _strip_none(
                {
                    "role": orig_msg.get("role", "assistant"),
                    "content": orig_msg.get("content"),
                    "tool_calls": orig_msg.get("tool_calls") or None,
                }
            )
        )
        session.raw_messages.append(
            _strip_none(
                {
                    "role": deano_msg.get("role", "assistant"),
                    "content": deano_msg.get("content"),
                    "tool_calls": deano_msg.get("tool_calls") or None,
                }
            )
        )

    # ------------------------------------------------------------------
    # Save session and return
    # ------------------------------------------------------------------
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

    return response_data, session_id


def _deanonymize_message(msg: dict[str, Any], placeholder_map: dict[str, str]) -> None:
    """Substitute placeholders back into every text field of *msg* in place.

    Covers ``content``, the reasoning side channel (``reasoning`` /
    ``reasoning_content``), tool calls and — defensively — the legacy
    ``function_calls`` list.  The legacy list is **not** persisted to the
    session history (see ``handle_request``); it is only de-anonymised here so
    an exotic backend cannot leak raw placeholders to the client.
    """
    if msg.get("content"):
        msg["content"] = deanonymize_text(msg["content"], placeholder_map)
    for key in ("reasoning", "reasoning_content"):
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

    has_output = (
        hidden_msg.get("content") is not None
        or hidden_msg.get("tool_calls")
        or raw_msg.get("content") is not None
    )
    if not has_output:
        logger.warning("stream produced no assistant output; not persisting turn",
                       session_id=session_id)
        return

    session.hidden_messages.append(hidden_msg)
    session.raw_messages.append(raw_msg)
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

    try:
        # ``read=None`` disables the read timeout for long-lived streams.
        timeout = httpx.Timeout(connect=10.0, read=None, write=30.0, pool=10.0)
        async with httpx.AsyncClient(timeout=timeout) as http_client:
            async with http_client.stream(
                "POST",
                _llm_chat_url(settings),
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
