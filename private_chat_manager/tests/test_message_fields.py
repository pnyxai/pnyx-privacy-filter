"""Unit tests for the message text-field registry."""

from __future__ import annotations

from app.message_fields import known_field_names, message_fields


def _field(name: str):
    return next(field for field in message_fields() if field.name == name)


def test_expected_builtin_fields_are_registered():
    assert {
        "content",
        "reasoning",
        "reasoning_content",
        "refusal",
        "name",
        "tool_calls.arguments",
        "function_calls.arguments",
        "function_call.arguments",
    } <= known_field_names()


def test_content_string_slot_roundtrip():
    msg = {"role": "user", "content": "hello"}
    slots = list(_field("content").iter_slots(msg))

    assert len(slots) == 1
    assert slots[0].get() == "hello"
    slots[0].set("bye")
    assert msg["content"] == "bye"


def test_content_multimodal_slots_redact_in_place():
    msg = {
        "role": "user",
        "content": [
            {"type": "text", "text": "a"},
            {"type": "image_url", "image_url": {"url": "u"}},
            {"type": "text", "text": "b"},
        ],
    }
    slots = list(_field("content").iter_slots(msg))

    assert len(slots) == 2  # only the text parts
    for slot, replacement in zip(slots, ("X", "Y")):
        slot.set(replacement)
    assert msg["content"][0]["text"] == "X"
    assert msg["content"][1] == {"type": "image_url", "image_url": {"url": "u"}}
    assert msg["content"][2]["text"] == "Y"


def test_tool_calls_arguments_slot():
    msg = {
        "role": "assistant",
        "tool_calls": [
            {"function": {"name": "f", "arguments": "a"}},
            {"function": {"name": "g", "arguments": "b"}},
        ],
    }
    slots = list(_field("tool_calls.arguments").iter_slots(msg))

    assert [s.get() for s in slots] == ["a", "b"]
    slots[1].set("B")
    assert msg["tool_calls"][1]["function"]["arguments"] == "B"


def test_legacy_function_call_slots():
    msg = {"role": "assistant", "function_call": {"name": "f", "arguments": "a"}}
    slots = list(_field("function_call.arguments").iter_slots(msg))

    assert len(slots) == 1
    slots[0].set("X")
    assert msg["function_call"]["arguments"] == "X"


def test_legacy_function_calls_list_slots():
    msg = {"role": "assistant", "function_calls": [{"arguments": "a"}]}
    slots = list(_field("function_calls.arguments").iter_slots(msg))

    assert [s.get() for s in slots] == ["a"]


def test_scalar_fields_present_only_when_string():
    msg = {"role": "assistant", "reasoning": "r", "refusal": None, "name": "n"}
    assert [s.get() for s in _field("reasoning").iter_slots(msg)] == ["r"]
    assert list(_field("refusal").iter_slots(msg)) == []
    assert [s.get() for s in _field("name").iter_slots(msg)] == ["n"]
