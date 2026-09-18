from __future__ import annotations

import time
from typing import Any, Literal

from openai.types.chat.chat_completion_audio import (
    ChatCompletionAudio as OpenAIChatCompletionAudio,
)
from openai.types.chat.chat_completion_message import Annotation as OpenAIAnnotation
from pydantic import BaseModel, ConfigDict, Field


# ---------------------------------------------------------------------------
# OpenAI-compatible message types
# ---------------------------------------------------------------------------


class FunctionCall(BaseModel):
    name: str
    arguments: str


class ToolCall(BaseModel):
    id: str
    type: Literal["function"] = "function"
    function: FunctionCall


class ChatMessage(BaseModel):
    """Structured message as returned by the LLM (assistant / tool / user).

    Mirrors vLLM's ``ChatMessage`` / OpenAI's ``ChatCompletionMessage``.
    ``extra="allow"`` ensures vLLM-specific fields (e.g. ``reasoning``) pass
    through transparently without being stripped.
    """

    model_config = ConfigDict(extra="allow")

    role: str
    content: str | None = None
    name: str | None = None
    refusal: str | None = None
    # Typed via the official openai SDK so we stay byte-compatible with vLLM
    annotations: list[OpenAIAnnotation] | None = None
    audio: OpenAIChatCompletionAudio | None = None
    function_call: FunctionCall | None = None  # legacy, pre-tool_calls
    tool_calls: list[ToolCall] | None = None
    tool_call_id: str | None = None


# ---------------------------------------------------------------------------
# Incoming request (OpenAI /v1/chat/completions + PCM extensions)
# ---------------------------------------------------------------------------


class PrivateChatRequest(BaseModel):
    # Core OpenAI fields
    messages: list[dict[str, Any]]
    model: str | None = None
    temperature: float | None = None
    top_p: float | None = None
    n: int | None = None
    stop: str | list[str] | None = None
    max_tokens: int | None = None
    max_completion_tokens: int | None = None
    presence_penalty: float | None = None
    frequency_penalty: float | None = None
    logit_bias: dict[str, float] | None = None
    logprobs: bool | None = None
    top_logprobs: int | None = None
    stream: bool | None = False
    stream_options: dict[str, Any] | None = None
    tools: list[dict[str, Any]] | None = None
    tool_choice: str | dict[str, Any] | None = None
    response_format: dict[str, Any] | None = None
    seed: int | None = None
    user: str | None = None

    # PCM-specific extensions
    bypass_privacy_filter: bool = False


# ---------------------------------------------------------------------------
# Response types (mirrors OpenAI ChatCompletion)
# ---------------------------------------------------------------------------


class UsageInfo(BaseModel):
    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0


class ChatCompletionResponseChoice(BaseModel):
    """Mirrors vLLM's ChatCompletionResponseChoice.

    ``extra="allow"`` lets vLLM-specific extras (e.g. ``stop_reason``,
    ``token_ids``) flow back to the caller without being dropped.
    """

    model_config = ConfigDict(extra="allow")

    index: int
    message: ChatMessage
    finish_reason: str | None = "stop"
    # vLLM extras (passed through transparently)
    logprobs: dict[str, Any] | None = None
    stop_reason: int | str | None = None
    token_ids: list[int] | None = None


class PrivateChatResponse(BaseModel):
    """Full ChatCompletionResponse returned to the caller.

    Mirrors ``ChatCompletionResponse`` from vLLM / OpenAI exactly.
    The resolved session ID is returned in the ``X-Session-ID`` response header,
    not in the body, so the response stays OpenAI-compatible.

    ``extra="allow"`` ensures vLLM-specific top-level fields
    (e.g. ``prompt_logprobs``, ``kv_transfer_params``) are preserved.
    """

    model_config = ConfigDict(extra="allow")

    id: str
    object: Literal["chat.completion"] = "chat.completion"
    created: int = Field(default_factory=lambda: int(time.time()))
    model: str
    choices: list[ChatCompletionResponseChoice]
    usage: UsageInfo
    service_tier: str | None = None
    system_fingerprint: str | None = None


# ---------------------------------------------------------------------------
# Session-state types (persisted in SQLite)
# ---------------------------------------------------------------------------


class PrivacyFilterState(BaseModel):
    """Accumulated privacy state for a single session."""

    # Maps indexed placeholder → original text, e.g. "<PRIVATE_PERSON_1>" → "Alice Smith"
    placeholder_map: dict[str, str] = Field(default_factory=dict)
    # Global per-session type counters (never reset across turns)
    type_counters: dict[str, int] = Field(default_factory=dict)
    # One RedactionResult dict per filtered user turn (audit log)
    redaction_results: list[dict[str, Any]] = Field(default_factory=list)


class SessionData(BaseModel):
    """Full session record stored in SQLite."""

    session_id: str
    created_at: float
    updated_at: float
    # Raw (original) messages including true PII — never sent to the LLM
    raw_messages: list[dict[str, Any]] = Field(default_factory=list)
    # Hidden (redacted) messages — what the LLM actually receives
    hidden_messages: list[dict[str, Any]] = Field(default_factory=list)
    privacy_state: PrivacyFilterState = Field(default_factory=PrivacyFilterState)


class SessionInspectResponse(BaseModel):
    """Payload returned by ``GET /v1/sessions/{session_id}``."""

    session_id: str
    created_at: float
    updated_at: float
    raw_messages: list[dict[str, Any]]
    hidden_messages: list[dict[str, Any]]
    privacy_state: PrivacyFilterState
    # Convenience counters
    turn_count: int
    """Number of complete request/response cycles stored for this session."""
    filtered_turn_count: int
    """Number of turns that were actually sent through the privacy filter."""
