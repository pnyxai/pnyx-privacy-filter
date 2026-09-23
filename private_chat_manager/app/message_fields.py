"""Registry of the text-bearing fields inside a chat message.

The privacy filter redacts *every* known text field of every message it forwards
upstream by default (see :mod:`app.privacy_manager`).  This module is the single
owner of "where does text live in a message", so new OpenAI/vLLM fields can be
covered with one :func:`register_message_field` call and operators can opt a
field out via ``PCM_PASSTHROUGH_FIELDS``.

Each :class:`MessageField` knows how to enumerate the :class:`TextSlot`\\ s of a
message dict.  A slot is a ``get``/``set`` pair over one text location, which
lets multimodal ``content`` parts be redacted in place (structure preserved)
rather than flattened.

The default-deny posture mirrors the endpoint registry in :mod:`app.endpoints`:
text is redacted unless explicitly passed through.
"""

from __future__ import annotations

from collections.abc import Callable, Iterator
from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class TextSlot:
    """One redactable text location inside a message."""

    get: Callable[[], str]
    set: Callable[[str], None]


@dataclass(frozen=True)
class MessageField:
    """A named text field and how to reach its slot(s) within a message."""

    name: str
    iter_slots: Callable[[dict[str, Any]], Iterator[TextSlot]]


# ---------------------------------------------------------------------------
# Built-in field locators
# ---------------------------------------------------------------------------


def _content_slots(msg: dict[str, Any]) -> Iterator[TextSlot]:
    """``content`` — a plain string, or each ``type=="text"`` multimodal part."""
    content = msg.get("content")
    if isinstance(content, str):
        yield TextSlot(
            get=lambda: msg["content"],
            set=lambda value: msg.__setitem__("content", value),
        )
    elif isinstance(content, list):
        for part in content:
            if (
                isinstance(part, dict)
                and part.get("type") == "text"
                and isinstance(part.get("text"), str)
            ):
                yield TextSlot(
                    get=lambda p=part: p["text"],
                    set=lambda value, p=part: p.__setitem__("text", value),
                )


def _str_field_slots(key: str) -> Callable[[dict[str, Any]], Iterator[TextSlot]]:
    """Locator for a top-level string field such as ``reasoning`` or ``name``."""

    def _iter(msg: dict[str, Any]) -> Iterator[TextSlot]:
        if isinstance(msg.get(key), str):
            yield TextSlot(
                get=lambda: msg[key],
                set=lambda value: msg.__setitem__(key, value),
            )

    return _iter


def _tool_call_argument_slots(msg: dict[str, Any]) -> Iterator[TextSlot]:
    """Each ``tool_calls[*].function.arguments`` JSON string."""
    for tool_call in msg.get("tool_calls") or []:
        if not isinstance(tool_call, dict):
            continue
        function = tool_call.get("function")
        if isinstance(function, dict) and isinstance(function.get("arguments"), str):
            yield TextSlot(
                get=lambda f=function: f["arguments"],
                set=lambda value, f=function: f.__setitem__("arguments", value),
            )


def _function_calls_argument_slots(msg: dict[str, Any]) -> Iterator[TextSlot]:
    """Legacy ``function_calls[*].arguments`` list."""
    for call in msg.get("function_calls") or []:
        if isinstance(call, dict) and isinstance(call.get("arguments"), str):
            yield TextSlot(
                get=lambda c=call: c["arguments"],
                set=lambda value, c=call: c.__setitem__("arguments", value),
            )


def _function_call_argument_slots(msg: dict[str, Any]) -> Iterator[TextSlot]:
    """Legacy single ``function_call.arguments`` object."""
    call = msg.get("function_call")
    if isinstance(call, dict) and isinstance(call.get("arguments"), str):
        yield TextSlot(
            get=lambda: call["arguments"],
            set=lambda value: call.__setitem__("arguments", value),
        )


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------

_REGISTERED_FIELDS: list[MessageField] = []


def register_message_field(field: MessageField) -> None:
    """Append *field* to the registry (checked in insertion order)."""
    _REGISTERED_FIELDS.append(field)


def message_fields() -> tuple[MessageField, ...]:
    """Return the registered message fields."""
    return tuple(_REGISTERED_FIELDS)


def known_field_names() -> frozenset[str]:
    """Return the canonical names of every registered field."""
    return frozenset(field.name for field in _REGISTERED_FIELDS)


# Built-in fields.  Add new ones here as the OpenAI/vLLM schema grows.
register_message_field(MessageField("content", _content_slots))
register_message_field(MessageField("reasoning", _str_field_slots("reasoning")))
register_message_field(
    MessageField("reasoning_content", _str_field_slots("reasoning_content"))
)
register_message_field(MessageField("refusal", _str_field_slots("refusal")))
register_message_field(MessageField("name", _str_field_slots("name")))
register_message_field(MessageField("tool_calls.arguments", _tool_call_argument_slots))
register_message_field(
    MessageField("function_calls.arguments", _function_calls_argument_slots)
)
register_message_field(
    MessageField("function_call.arguments", _function_call_argument_slots)
)
