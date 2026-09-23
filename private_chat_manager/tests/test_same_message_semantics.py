"""Tests that _same_message detects semantic tool-call edits (not just ids)."""

from __future__ import annotations

from app.privacy_manager import _same_message


def _call(call_id="c1", name="send_email", arguments='{"to":"a"}'):
    return {
        "role": "assistant",
        "content": None,
        "tool_calls": [{"id": call_id, "function": {"name": name, "arguments": arguments}}],
    }


def test_identical_tool_calls_match():
    assert _same_message(_call(), _call())


def test_argument_edit_is_detected():
    assert not _same_message(_call(), _call(arguments='{"to":"b"}'))


def test_function_name_edit_is_detected():
    assert not _same_message(_call(), _call(name="delete_email"))


def test_tool_call_id_edit_is_detected():
    assert not _same_message(_call(), _call(call_id="c2"))


def test_tool_message_id_edit_is_detected():
    stored = {"role": "tool", "tool_call_id": "c1", "content": "ok"}
    client = {"role": "tool", "tool_call_id": "c2", "content": "ok"}
    assert not _same_message(stored, client)


def test_legacy_function_call_edit_is_detected():
    stored = {"role": "assistant", "content": None, "function_call": {"name": "f", "arguments": "a"}}
    edited = {"role": "assistant", "content": None, "function_call": {"name": "f", "arguments": "b"}}
    assert not _same_message(stored, edited)


def test_legacy_function_calls_edit_is_detected():
    stored = {"role": "assistant", "content": None, "function_calls": [{"arguments": "a"}]}
    edited = {"role": "assistant", "content": None, "function_calls": [{"arguments": "b"}]}
    assert not _same_message(stored, edited)


def test_reasoning_and_refusal_are_ignored():
    stored = {"role": "assistant", "content": "hi"}
    client = {"role": "assistant", "content": "hi", "reasoning": "thinking", "refusal": None}
    assert _same_message(stored, client)
