from __future__ import annotations

import copy
import time
import uuid
from typing import Any

import aiosqlite
import httpx
from fastapi import HTTPException

from ._logging import get_logger
from .config import Settings
from .models import PrivacyFilterState, PrivateChatRequest, SessionData
from .session_store import get_session, save_session
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


# ---------------------------------------------------------------------------
# Main request handler
# ---------------------------------------------------------------------------


async def handle_request(
    request: PrivateChatRequest,
    session_id_header: str | None,
    settings: Settings,
    triton_client: TritonPrivacyFilterClient,
    conn: aiosqlite.Connection,
) -> tuple[dict[str, Any], str]:
    """Process one chat-completion request through the privacy mid-layer.

    Workflow
    --------
    1.  Resolve (or generate) a session ID.
    2.  Load or initialise session state from SQLite.
    3.  Validate that the last message has ``role == "user"``.
    4.  Pass-through any non-final messages (system prompts, etc.) unchanged.
    5a. *Normal path*: send the last user message to Triton, apply indexed
        placeholder substitution, append to hidden_messages.
    5b. *Bypass path*: skip Triton entirely; append the raw message as-is.
    6.  Forward *hidden_messages* to the downstream LLM.
    7.  Deep-copy and de-anonymise the LLM response for the caller.
    8.  Persist both the placeholder (hidden) and de-anonymised (raw) assistant
        turn in the session.
    9.  Save updated session to SQLite and return the response + session ID.

    Args:
        request:            Validated ``PrivateChatRequest`` from the endpoint.
        session_id_header:  Value of the ``X-Session-ID`` HTTP request header
                            (may be None).
        settings:           Application settings (URLs, keys, …).
        triton_client:      Initialised ``TritonPrivacyFilterClient``.
        conn:               Open ``aiosqlite`` connection for this request.

    Returns:
        ``(response_dict, session_id)`` where *response_dict* is the
        de-anonymised OpenAI-compatible JSON payload.
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
    # Roles absent from that set pass through unchanged:
    #   - "assistant": always unfiltered — either it was produced by PCM
    #     (placeholders already in place) or it predates PCM (PII already
    #     exposed; filtering retroactively does nothing).
    #   - "system" by default: infrastructure prompts, not user data.
    #     Add it to PCM_FILTERABLE_ROLES if prompts contain personal data.
    #
    # type_counters and placeholder_map accumulate across ALL new messages
    # in this batch so that two tool replies in the same request that
    # contain the same label class get sequential indices, not both "_1".
    # ------------------------------------------------------------------
    new_counters = dict(session.privacy_state.type_counters)
    new_map = dict(session.privacy_state.placeholder_map)
    new_redaction_results: list[dict[str, Any]] = []

    for msg in new_messages:
        role = msg.get("role", "")
        session.raw_messages.append(msg)

        if role not in settings.filterable_roles or request.bypass_privacy_filter:
            # Pass through unchanged (role not in filterable_roles, or bypass active)
            session.hidden_messages.append(msg)
            continue

        # --- Filter this message through Triton ---
        content = _extract_text_content(msg.get("content"))
        _t0 = time.monotonic()
        redaction_result: dict[str, Any] = await triton_client.infer(content)
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

        indexed_text, new_counters, new_map = apply_placeholder_indexing(
            redaction_result,
            new_counters,
            new_map,
        )

        session.hidden_messages.append({**msg, "content": indexed_text})
        new_redaction_results.append(redaction_result)

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
    # 6. Forward hidden messages to the downstream LLM
    # ------------------------------------------------------------------
    # Build the forwarded payload: preserve all original sampling params,
    # override model/messages/stream.
    llm_payload = request.model_dump(
        exclude={"session_id", "bypass_privacy_filter", "messages", "model"},
        exclude_none=True,
    )
    llm_payload["model"] = settings.llm_model_name
    llm_payload["messages"] = session.hidden_messages
    llm_payload["stream"] = False  # streaming not supported by this mid-layer

    headers: dict[str, str] = {"Content-Type": "application/json"}
    if settings.llm_api_key:
        headers["Authorization"] = f"Bearer {settings.llm_api_key}"

    logger.info(
        "forwarding to LLM",
        session_id=session_id,
        model=settings.llm_model_name,
        hidden_msg_count=len(session.hidden_messages),
    )
    if "llm_payload" in settings.verbose_log_events:
        logger.debug(
            "LLM payload",  # hidden messages are already redacted — no PII
            session_id=session_id,
            payload=llm_payload,
        )

    _t0 = time.monotonic()
    async with httpx.AsyncClient(timeout=120.0) as http_client:
        llm_resp = await http_client.post(
            f"{settings.llm_url.rstrip('/')}/v1/chat/completions",
            json=llm_payload,
            headers=headers,
        )
        llm_resp.raise_for_status()
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
    # 7. De-anonymise the LLM response for the caller.
    #    We work on a deep copy so that llm_data retains the original
    #    (placeholder) version for the session's hidden history.
    # ------------------------------------------------------------------
    placeholder_map = session.privacy_state.placeholder_map
    response_data: dict[str, Any] = copy.deepcopy(llm_data)

    for choice in response_data.get("choices", []):
        msg = choice.get("message", {})
        if msg.get("content"):
            msg["content"] = deanonymize_text(msg["content"], placeholder_map)
        # Tool calls
        for tc in msg.get("tool_calls") or []:
            fn = tc.get("function", {})
            if fn.get("arguments"):
                fn["arguments"] = deanonymize_text(fn["arguments"], placeholder_map)
        # Function calls (legacy role)
        for tc in msg.get("function_calls") or []:
            if tc.get("arguments"):
                tc["arguments"] = deanonymize_text(tc["arguments"], placeholder_map)

    # ------------------------------------------------------------------
    # 8. Persist the assistant turn in both session histories.
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
                    "function_calls": orig_msg.get("function_calls") or None,
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
    # 9. Save session and return
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
