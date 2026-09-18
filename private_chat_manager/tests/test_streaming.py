"""Unit tests for the real-time streaming de-anonymiser.

The most important test in this file is
:meth:`test_property_every_partition_is_equivalent_to_buffered`, which asserts
the equivalence guarantee documented in ``app/streaming.py``: for *any* way of
splitting a text into chunks, incremental de-anonymisation yields exactly the
same concatenated output as the buffered ``deanonymize_text`` helper.
"""

from __future__ import annotations

import random

import pytest

from app.privacy_manager import deanonymize_text
from app.streaming import (
    DEFAULT_PLACEHOLDER_LABELS,
    StreamResponseFilter,
    StreamingDeanonymizer,
)

PLACEHOLDER_MAP = {
    "<PRIVATE_PERSON_1>": "Lionel Messi",
    "<PRIVATE_PERSON_10>": "Emanuel Ginobili",
    "<PRIVATE_ADDRESS_1>": "123 Main St, New York",
    "<PRIVATE_EMAIL_1>": "alice.smith@example.com",
    "<SECRET_1>": "hunter2",
}


def collect(filters: StreamingDeanonymizer, chunks: list[str]) -> str:
    return "".join(filters.feed(chunk) for chunk in chunks) + filters.flush()


def all_partitions(text: str) -> list[list[str]]:
    """Every possible partition of *text* into consecutive chunks.

    ``2**(len(text)-1)`` partitions; keep the caller's input short.
    """
    if not text:
        return [[]]
    n = len(text)
    partitions: list[list[str]] = []
    for mask in range(1 << (n - 1)):
        chunks: list[str] = []
        start = 0
        for i in range(n - 1):
            if mask & (1 << i):
                chunks.append(text[start : i + 1])
                start = i + 1
        chunks.append(text[start:])
        partitions.append(chunks)
    return partitions


def random_partitions(text: str, count: int, rng: random.Random) -> list[list[str]]:
    partitions: list[list[str]] = []
    for _ in range(count):
        cuts = sorted(rng.sample(range(1, len(text)), k=min(rng.randint(0, 4), max(len(text) - 1, 0)))) if len(text) > 1 else []
        chunks: list[str] = []
        prev = 0
        for cut in cuts:
            chunks.append(text[prev:cut])
            prev = cut
        chunks.append(text[prev:])
        partitions.append(chunks)
    return partitions


# ---------------------------------------------------------------------------
# Basic behaviour
# ---------------------------------------------------------------------------


def test_passthrough_when_no_placeholders():
    filt = StreamingDeanonymizer({})
    assert collect(filt, ["Hello ", "world <PRIV", "ATE_PERSON_1>"]) == (
        "Hello world <PRIVATE_PERSON_1>"
    )


def test_plain_text_is_returned_verbatim():
    filt = StreamingDeanonymizer(PLACEHOLDER_MAP)
    text = "Nothing to see here, just ordinary prose."
    assert collect(filt, [text]) == text
    assert filt.pending == ""


def test_complete_tag_in_single_chunk():
    filt = StreamingDeanonymizer(PLACEHOLDER_MAP)
    out = collect(filt, ["Your name is <PRIVATE_PERSON_1>!"])
    assert out == "Your name is Lionel Messi!"


def test_split_tag_across_three_chunks():
    filt = StreamingDeanonymizer(PLACEHOLDER_MAP)
    out = collect(filt, ["Your name is <PRIV", "ATE_PERSON_", "1>!"])
    assert out == "Your name is Lionel Messi!"


def test_tag_split_at_every_boundary_is_reassembled():
    filt = StreamingDeanonymizer(PLACEHOLDER_MAP)
    text = "Greetings <PRIVATE_ADDRESS_1>, welcome!"
    expected = deanonymize_text(text, PLACEHOLDER_MAP)
    for split in range(len(text) + 1):
        filt = StreamingDeanonymizer(PLACEHOLDER_MAP)
        assert collect(filt, [text[:split], text[split:]]) == expected


def test_longest_index_wins_over_prefix_collision():
    # <PRIVATE_PERSON_10> must not be mistaken for <PRIVATE_PERSON_1> + "0".
    filt = StreamingDeanonymizer(PLACEHOLDER_MAP)
    assert collect(filt, ["<PRIVATE_PERSON_10>"]) == "Emanuel Ginobili"


def test_index_split_so_ambiguity_is_resolved_only_when_complete():
    filt = StreamingDeanonymizer(PLACEHOLDER_MAP)
    # First chunk ends on the ambiguous "1"; nothing is emitted yet because
    # it could still grow into "_10".
    assert filt.feed("Pele and <PRIVATE_PERSON_1") == "Pele and "
    assert filt.pending == "<PRIVATE_PERSON_1"
    assert filt.feed("0>") == "Emanuel Ginobili"


def test_multiple_and_adjacent_tags():
    filt = StreamingDeanonymizer(PLACEHOLDER_MAP)
    text = "<PRIVATE_PERSON_1><PRIVATE_EMAIL_1>"
    assert collect(filt, [text[:7], text[7:15], text[15:]]) == (
        "Lionel Messialice.smith@example.com"
    )


def test_literal_angle_bracket_is_not_a_tag():
    filt = StreamingDeanonymizer(PLACEHOLDER_MAP)
    text = "5 < 10 and 3 <b> tags"
    assert collect(filt, [text]) == text


def test_literal_bracket_then_real_tag():
    filt = StreamingDeanonymizer(PLACEHOLDER_MAP)
    out = collect(filt, ["a < b <PRIVATE_PERSON_1>"])
    assert out == "a < b Lionel Messi"


def test_dangling_prefix_adjacent_to_complete_tag():
    # A partial opener immediately followed by another '<' can never become a
    # placeholder: it is emitted literally, and only the well-formed tag is
    # replaced (same result as the buffered `deanonymize_text`).
    filt = StreamingDeanonymizer(PLACEHOLDER_MAP)
    assert collect(filt, ["<PRIVATE_PERSO", "<PRIVATE_PERSON_1>"]) == (
        "<PRIVATE_PERSO" + "Lionel Messi"
    )
    assert collect(
        StreamingDeanonymizer(PLACEHOLDER_MAP),
        ["<PRIVATE_PERSO<PRIVATE_PERSON_1>"],
    ) == "<PRIVATE_PERSO" + "Lionel Messi"


def test_unknown_tag_passes_through():
    filt = StreamingDeanonymizer(PLACEHOLDER_MAP)
    out = collect(filt, ["mystery <PRIVATE_PERSON_99> here"])
    assert out == "mystery <PRIVATE_PERSON_99> here"


def test_unterminated_tag_is_flushed_verbatim():
    filt = StreamingDeanonymizer(PLACEHOLDER_MAP)
    assert filt.feed("tail <PRIVATE_PER") == "tail "
    assert filt.pending == "<PRIVATE_PER"
    assert filt.flush() == "<PRIVATE_PER"
    assert filt.flush() == ""


def test_placeholder_that_never_closes_releases_all_held_text():
    """A tag that diverges before its closing ``>`` must be replayed verbatim.

    Mirrors the accumulated sequence::

        "<" -> "<PRIVATE" -> "<PRIVATE_PERSON_" -> "<PRIVATE_PERSON_12"
        -> "<PRIVATE_PERSON_12 is a good person"

    The first four deltas are held (each is a valid placeholder prefix).  The
    fifth adds a space, which can never appear in a placeholder, so the whole
    held buffer is released in that delta.
    """
    deltas = ["<", "PRIVATE", "_PERSON_", "12", " is a good person"]
    filt = StreamingDeanonymizer(PLACEHOLDER_MAP)

    emitted = [filt.feed(delta) for delta in deltas]

    assert emitted[:4] == ["", "", "", ""]
    assert emitted[4] == "<PRIVATE_PERSON_12 is a good person"
    assert "".join(emitted) == "".join(deltas)
    assert filt.flush() == ""


def test_held_prefix_is_released_as_soon_as_it_is_invalidated():
    filt = StreamingDeanonymizer(PLACEHOLDER_MAP)
    assert filt.feed("<") == ""
    assert filt.feed("PRIVATE_PERSON_12") == ""
    # A space invalidates the prefix: everything held comes out at once.
    assert filt.feed(" is") == "<PRIVATE_PERSON_12 is"
    assert filt.pending == ""


def test_diverging_prefix_still_recovers_a_real_tag_afterwards():
    filt = StreamingDeanonymizer(PLACEHOLDER_MAP)
    assert filt.feed("<PRIVATE_PERSON_99x <PRIV") == "<PRIVATE_PERSON_99x "
    assert filt.pending == "<PRIV"
    assert filt.feed("ATE_PERSON_1>") == "Lionel Messi"


def test_replacement_value_is_not_rescanned():
    placeholder_map = {"<SECRET_1>": "value with <PRIVATE_PERSON_1> inside"}
    filt = StreamingDeanonymizer(placeholder_map)
    out = collect(filt, ["<SECRET_1>"])
    # The placeholder inside the *replacement* must remain untouched.
    assert out == "value with <PRIVATE_PERSON_1> inside"


def test_empty_text_is_noop():
    filt = StreamingDeanonymizer(PLACEHOLDER_MAP)
    assert filt.feed("") == ""
    assert filt.flush() == ""


# ---------------------------------------------------------------------------
# Property-based equivalence with the buffered implementation
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "text",
    [
        "Hello world, no tags here.",
        "Name: <PRIVATE_PERSON_1>!",
        "<PRIVATE_PERSON_1> and <PRIVATE_ADDRESS_1>",
        "<PRIVATE_PERSON_1><PRIVATE_EMAIL_1>",
        "before <SECRET_1> after <PRIVATE_PERSON_10>",
        "a<PRIVATE_PERSON_1>b<PRIVATE_EMAIL_1>c",
        "math 2 < 3 and <PRIVATE_PERSON_1>",
        "unicode \u00a1<PRIVATE_PERSON_1>\u00bf ok",
        "<PRIVATE_PERSO<PRIVATE_PERSON_1>",
        "<PRIVATE_PERSON_1<PRIVATE_PERSON_1>",
    ],
)
def test_property_every_partition_is_equivalent_to_buffered(text):
    expected = deanonymize_text(text, PLACEHOLDER_MAP)
    # Exhaustive for short strings; sampled for longer ones.
    partitions = all_partitions(text) if len(text) <= 12 else random_partitions(
        text, 400, random.Random(0xC0FFEE)
    )
    for chunks in partitions:
        filt = StreamingDeanonymizer(PLACEHOLDER_MAP)
        assert collect(filt, chunks) == expected, chunks


# ---------------------------------------------------------------------------
# StreamResponseFilter: content / reasoning / tool calls / persistence
# ---------------------------------------------------------------------------


def _chunk(index: int, delta: dict, finish_reason=None) -> dict:
    return {
        "id": "chatcmpl-test",
        "object": "chat.completion.chunk",
        "choices": [
            {"index": index, "delta": delta, "finish_reason": finish_reason}
        ],
    }


def test_filter_content_across_chunks():
    stream_filter = StreamResponseFilter(PLACEHOLDER_MAP)
    c1 = _chunk(0, {"role": "assistant", "content": ""})
    c2 = _chunk(0, {"content": "You are <PRIV"})
    c3 = _chunk(0, {"content": "ATE_PERSON_1>."})
    c4 = _chunk(0, {}, finish_reason="stop")
    for chunk in (c1, c2, c3, c4):
        stream_filter.process_chunk(chunk)

    assert c1["choices"][0]["delta"]["content"] == ""
    assert c2["choices"][0]["delta"]["content"] == "You are "
    assert c3["choices"][0]["delta"]["content"] == "Lionel Messi."
    assert c4["choices"][0]["delta"] == {}

    hidden, raw = stream_filter.assistant_history(0)
    assert hidden["content"] == "You are <PRIVATE_PERSON_1>."
    assert raw["content"] == "You are Lionel Messi."


def test_filter_reasoning_field():
    stream_filter = StreamResponseFilter(PLACEHOLDER_MAP)
    stream_filter.process_chunk(_chunk(0, {"reasoning": "think <PRIV"}))
    stream_filter.process_chunk(_chunk(0, {"reasoning": "ATE_PERSON_1> done"}))
    hidden, raw = stream_filter.assistant_history(0)
    # Reasoning is de-anonymised for the client but not stored in the session
    # assistant turn (mirrors the buffered path).
    assert hidden["content"] is None and raw["content"] is None


def test_filter_legacy_reasoning_content_field():
    stream_filter = StreamResponseFilter(PLACEHOLDER_MAP)
    chunk = _chunk(0, {"reasoning_content": "<PRIVATE_ADDRESS_1>"})
    stream_filter.process_chunk(chunk)
    assert chunk["choices"][0]["delta"]["reasoning_content"] == "123 Main St, New York"


def test_filter_streaming_tool_call_arguments():
    stream_filter = StreamResponseFilter(PLACEHOLDER_MAP)
    stream_filter.process_chunk(
        _chunk(
            0,
            {
                "tool_calls": [
                    {
                        "index": 0,
                        "id": "call_1",
                        "type": "function",
                        "function": {"name": "lookup", "arguments": '{"q":"<PRIV'},
                    }
                ]
            },
        )
    )
    stream_filter.process_chunk(
        _chunk(
            0,
            {
                "tool_calls": [
                    {
                        "index": 0,
                        "function": {"arguments": 'ATE_PERSON_1>"}'},
                    }
                ]
            },
        )
    )

    hidden, raw = stream_filter.assistant_history(0)
    assert hidden["tool_calls"][0]["function"]["arguments"] == '{"q":"<PRIVATE_PERSON_1>"}'
    assert raw["tool_calls"][0]["function"]["arguments"] == '{"q":"Lionel Messi"}'


def test_filter_multiple_choices_are_independent():
    stream_filter = StreamResponseFilter(PLACEHOLDER_MAP)
    stream_filter.process_chunk(_chunk(0, {"content": "<PRIVATE_PERSON_1>"}))
    stream_filter.process_chunk(_chunk(1, {"content": "<SECRET_1>"}))
    assert stream_filter.assistant_history(0)[1]["content"] == "Lionel Messi"
    assert stream_filter.assistant_history(1)[1]["content"] == "hunter2"


def test_flush_choice_drains_partial_and_tracks_history():
    stream_filter = StreamResponseFilter(PLACEHOLDER_MAP)
    stream_filter.process_chunk(_chunk(0, {"content": "partial <PRIV"}))
    residuals = stream_filter.flush_choice(0)
    assert residuals == {"content": "<PRIV"}
    # Idempotent.
    assert stream_filter.flush_choice(0) == {}
    hidden, raw = stream_filter.assistant_history(0)
    assert hidden["content"] == "partial <PRIV"
    assert raw["content"] == "partial <PRIV"


def test_custom_labels_via_constructor():
    stream_filter = StreamResponseFilter(
        {"<NAME_1>": "Alice"},
        labels=("NAME",),
    )
    stream_filter.process_chunk(_chunk(0, {"content": "<NA"}))
    stream_filter.process_chunk(_chunk(0, {"content": "ME_1>"}))
    hidden, raw = stream_filter.assistant_history(0)
    assert raw["content"] == "Alice"


def test_detector_also_uses_labels_found_in_placeholder_map():
    # Configured labels omit CUSTOM_ID, but the session map contains one. The
    # detector must still recognise (and hold) it so the placeholder cannot
    # leak to the client.
    stream_filter = StreamResponseFilter(
        {"<CUSTOM_ID_1>": "secret-value"},
        labels=("PRIVATE_PERSON",),
    )
    stream_filter.process_chunk(_chunk(0, {"content": "id <CUST"}))
    stream_filter.process_chunk(_chunk(0, {"content": "OM_ID_1>!"}))
    hidden, raw = stream_filter.assistant_history(0)
    assert raw["content"] == "id secret-value!"
    assert hidden["content"] == "id <CUSTOM_ID_1>!"


def test_reasoning_content_residual_keeps_its_key():
    stream_filter = StreamResponseFilter(PLACEHOLDER_MAP)
    stream_filter.process_chunk(_chunk(0, {"reasoning_content": "think <SEC"}))
    assert stream_filter.flush_choice(0) == {"reasoning_content": "<SEC"}


def test_reasoning_content_residual_on_finish_chunk_keeps_its_key():
    stream_filter = StreamResponseFilter(PLACEHOLDER_MAP)
    stream_filter.process_chunk(_chunk(0, {"reasoning_content": "think <SEC"}))
    finish = _chunk(0, {}, finish_reason="stop")
    stream_filter.process_chunk(finish)
    delta = finish["choices"][0]["delta"]
    assert delta.get("reasoning_content") == "<SEC"
    assert "reasoning" not in delta


def test_default_labels_constant_matches_privacy_filter_taxonomy():
    assert set(DEFAULT_PLACEHOLDER_LABELS) == {
        "ACCOUNT_NUMBER",
        "PRIVATE_ADDRESS",
        "PRIVATE_DATE",
        "PRIVATE_EMAIL",
        "PRIVATE_PERSON",
        "PRIVATE_PHONE",
        "PRIVATE_URL",
        "SECRET",
    }


def test_all_default_labels_are_replaceable():
    placeholder_map = {
        f"<{label}_1>": f"value-{label}" for label in DEFAULT_PLACEHOLDER_LABELS
    }
    for label in DEFAULT_PLACEHOLDER_LABELS:
        filt = StreamingDeanonymizer(placeholder_map, DEFAULT_PLACEHOLDER_LABELS)
        assert collect(filt, [f"<{label}_1>"]) == f"value-{label}"


# ---------------------------------------------------------------------------
# Log views (client_message / raw_message) vs persistence (assistant_history)
# ---------------------------------------------------------------------------


def test_client_message_includes_reasoning_but_assistant_history_does_not():
    stream_filter = StreamResponseFilter(PLACEHOLDER_MAP)
    stream_filter.process_chunk(_chunk(0, {"content": "Answer: <PRIV"}))
    stream_filter.process_chunk(_chunk(0, {"reasoning": "He is <PRIV"}))
    stream_filter.process_chunk(_chunk(0, {"reasoning": "ATE_PERSON_1>."}))
    stream_filter.process_chunk(_chunk(0, {"content": "ATE_PERSON_1> done"}))

    client_msg = stream_filter.client_message(0)
    assert client_msg["content"] == "Answer: Lionel Messi done"
    assert client_msg["reasoning"] == "He is Lionel Messi."

    raw_msg = stream_filter.raw_message(0)
    assert raw_msg["content"] == "Answer: <PRIVATE_PERSON_1> done"
    assert raw_msg["reasoning"] == "He is <PRIVATE_PERSON_1>."

    hidden, raw = stream_filter.assistant_history(0)
    assert "reasoning" not in hidden
    assert "reasoning" not in raw
    assert hidden["content"] == "Answer: <PRIVATE_PERSON_1> done"
    assert raw["content"] == "Answer: Lionel Messi done"


def test_client_and_raw_message_include_flushed_reasoning_residual():
    stream_filter = StreamResponseFilter(PLACEHOLDER_MAP)
    stream_filter.process_chunk(_chunk(0, {"reasoning": "partial <SEC"}))
    stream_filter.flush_choice(0)

    assert stream_filter.client_message(0)["reasoning"] == "partial <SEC"
    assert stream_filter.raw_message(0)["reasoning"] == "partial <SEC"


def test_client_and_raw_message_include_tool_calls():
    stream_filter = StreamResponseFilter(PLACEHOLDER_MAP)
    stream_filter.process_chunk(
        _chunk(
            0,
            {
                "tool_calls": [
                    {
                        "index": 0,
                        "id": "call_1",
                        "type": "function",
                        "function": {"name": "lookup", "arguments": '{"q":"<PRIV'},
                    }
                ]
            },
        )
    )
    stream_filter.process_chunk(
        _chunk(
            0,
            {"tool_calls": [{"index": 0, "function": {"arguments": "ATE_PERSON_1>\"}"}}]},
        )
    )

    client_msg = stream_filter.client_message(0)
    raw_msg = stream_filter.raw_message(0)
    assert client_msg["tool_calls"][0]["function"]["arguments"] == '{"q":"Lionel Messi"}'
    assert raw_msg["tool_calls"][0]["function"]["arguments"] == '{"q":"<PRIVATE_PERSON_1>"}'


def test_log_views_for_absent_choice_are_empty():
    stream_filter = StreamResponseFilter(PLACEHOLDER_MAP)
    assert stream_filter.client_message(0) == {"role": "assistant", "content": None}
    assert stream_filter.raw_message(0) == {"role": "assistant", "content": None}

