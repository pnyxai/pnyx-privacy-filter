"""Tests that _same_message detects semantic tool-call edits (not just ids)."""

from __future__ import annotations

from app.privacy_manager import _canonicalize_arguments, _same_message


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


def test_argument_whitespace_is_not_a_divergence():
    # Clients re-serialise parsed arguments before replaying (Hermes: compact +
    # sorted).
    assert _same_message(_call(arguments='{"path": "/x"}'), _call(arguments='{"path":"/x"}'))


def test_argument_key_order_is_not_a_divergence():
    assert _same_message(_call(arguments='{"a":1,"b":2}'), _call(arguments='{"b":2,"a":1}'))


def test_argument_numeric_equivalence_matches():
    assert _same_message(_call(arguments='{"n": 1e3}'), _call(arguments='{"n":1000.0}'))


def test_non_json_arguments_fall_back_to_exact():
    assert _same_message(_call(arguments="a"), _call(arguments="a"))
    assert not _same_message(_call(arguments="a"), _call(arguments="b"))


def test_deeply_nested_arguments_do_not_raise():
    # json.loads raises RecursionError on pathologically nested input; the
    # comparison must fall back to the raw string, not fail the request.
    deep = "[" * 20000 + "]" * 20000
    assert _canonicalize_arguments(deep) == deep


def test_legacy_function_call_whitespace_is_not_a_divergence():
    stored = {"role": "assistant", "content": None, "function_call": {"name": "f", "arguments": '{"a": 1}'}}
    client = {"role": "assistant", "content": None, "function_call": {"name": "f", "arguments": '{"a":1}'}}
    assert _same_message(stored, client)


def test_legacy_function_calls_whitespace_is_not_a_divergence():
    stored = {"role": "assistant", "content": None, "function_calls": [{"arguments": '{"a": 1}'}]}
    client = {"role": "assistant", "content": None, "function_calls": [{"arguments": '{"a":1}'}]}
    assert _same_message(stored, client)


def test_omitted_and_empty_content_are_equivalent():
    # PCM drops a null ``content`` (``_strip_none``); some clients replay ``""``.
    stored = _call()  # no ``content`` key
    client = {**_call(), "content": ""}
    assert _same_message(stored, client)
    assert _same_message({"role": "assistant", "content": None}, {"role": "assistant", "content": ""})


def test_empty_content_differs_from_text():
    assert not _same_message({"role": "assistant", "content": ""}, {"role": "assistant", "content": "x"})
    assert not _same_message({"role": "assistant", "content": None}, {"role": "assistant", "content": "x"})
    assert not _same_message({"role": "user", "content": ""}, {"role": "user", "content": "hi"})


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
