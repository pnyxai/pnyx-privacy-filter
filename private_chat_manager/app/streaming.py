"""Real-time (streaming) de-anonymisation of LLM output.

Background
==========

The downstream LLM never sees real PII.  Before a user message is forwarded,
the Triton privacy-filter replaces every detected span with an *indexed
placeholder* of the form ``<LABEL_N>`` (e.g. ``<PRIVATE_PERSON_1>``).  The
originating text is kept in a session-scoped ``placeholder_map``::

    {"<PRIVATE_PERSON_1>": "Lionel Messi",
     "<PRIVATE_ADDRESS_1>": "123 Main St, New York",
     ...}

When the LLM answers it may refer back to those placeholders.  In the
non-streaming path :func:`privacy_manager.deanonymize_text` substitutes them
back with a simple ``str.replace`` over the *complete* response.  A streaming
response, however, arrives one small token at a time, so a placeholder (and
even the true text replacing it) can be split across several chunks::

    chunk 1: "<PRIV"
    chunk 2: "ATE_PERSON_"
    chunk 3: "1> was here"

We cannot emit ``"<PRIV"`` to the client before we know whether it is the
start of a placeholder, otherwise the placeholder would leak.  Equally we
must not buffer the entire stream, otherwise streaming would be pointless.
The state machine below solves exactly that problem.

The detection/replacement algorithm
====================================

The vocabulary of placeholders is *finite and known in advance*.
The model selected for privacy-filtering known in advance.
Every label is a short uppercase token (``PRIVATE_PERSON``, ``SECRET`` …) and
every placeholder is one of::

    <LABEL>              # base form emitted by the privacy-filter
    <LABEL_N>            # indexed form emitted by apply_placeholder_indexing

Two regular expressions are pre-computed (and memoised) from that label set:

``_TAG_RE``
    Matches a *complete* placeholder: ``<(LABEL1|LABEL2|…)(?:_\\d+)?>``.
    The label alternation is sorted longest-first and the index uses a greedy
    ``\\d+`` so ``<PRIVATE_PERSON_10>`` is matched in full rather than being
    mistaken for ``<PRIVATE_PERSON_1>`` followed by a stray ``0``.

``_PREFIX_RE``
    Matches a string that is a *valid prefix* of a placeholder, including the
    bare ``<`` opener::

        <
        <P
        <PRIV
        <PRIVATE_PERSON
        <PRIVATE_PERSON_
        <PRIVATE_PERSON_12
        …

    This is the key to deciding, without look-ahead into future chunks,
    whether the tail of the buffer *could still become* a placeholder.  It is
    deliberately conservative (never accepts a string that cannot become a
    tag) so partial tags are buffered, but ordinary text is never delayed.

Feeding a chunk
---------------

:meth:`StreamingDeanonymizer.feed` concatenates any previously held tail with
the new text and scans left to right:

1. **Fast path.**  If nothing is held and the chunk contains no ``<`` there
   can be no placeholder, so the chunk is returned untouched.  This is the
   common case for the vast majority of tokens and keeps the hot path free of
   regex work and string copies.
2. Everything up to the first ``<`` is emitted verbatim.
3. At the ``<`` we try ``_TAG_RE``:
   * **Match** → replace with ``placeholder_map.get(tag, tag)``.  Unknown tags
     (e.g. a hallucinated ``<PRIVATE_PERSON_99>``) are passed through
     unchanged, mirroring ``deanonymize_text``.  Replacement text is written
     straight to the output and *is never rescanned*, so a redaction value
     that itself contains ``<`` cannot trigger a second replacement.
   * **No match, but the whole remainder is a valid prefix** (``_PREFIX_RE``)
     → hold the remainder in ``pending`` and stop.  It will be prepended to
     the next chunk.
   * **No match and not a valid prefix** → the ``<`` is literal text.  Emit
     ``<`` and resume scanning immediately after it.
4. At end of stream :meth:`flush` returns any residual ``pending`` verbatim
   (an unterminated ``<PRIV`` is, by definition, not a placeholder).  The same
   contract covers a tag that *diverges* before its closing ``>`` (e.g. the
   accumulated ``<PRIVATE_PERSON_12 is a good person``): as soon as a character
   that cannot appear in a placeholder arrives, the held suffix stops being a
   valid prefix and the entire opening is replayed verbatim.  A placeholder is
   therefore only ever *delayed*, never lost or corrupted.

Because we only ever hold the *longest valid-prefix suffix* of the buffer,
every character emitted before it has been proven incapable of combining with
future input to form a placeholder.  For every possible partition of *text*
into *chunks*, the incremental result therefore matches the buffered
:func:`privacy_manager.deanonymize_text` substitution — with one deliberate
difference: a replacement value is never rescanned.  If a true value itself
contains another placeholder-shaped string (``{"<PRIVATE_PERSON_1>": "x
<SECRET_1>", ...}``), the stream emits it verbatim while ``deanonymize_text``
would substitute it on a later pass.  Not rescanning is the safer behaviour,
since data restored for one span must not be re-redacted or cross-substituted.

Field / choice / tool-call multiplexing
---------------------------------------

A single assistant turn can carry several independent text streams:

* ``content`` – one per ``choices[i]`` (``n > 1``), and
* ``reasoning`` – the reasoning parser's side channel,
* ``tool_calls[j].function.arguments`` – one JSON fragment stream per tool
  call index, itself streamed incrementally.

:class:`StreamResponseFilter` owns one :class:`StreamingDeanonymizer` per
``(choice, field)`` pair (tool-call argument filters are created lazily per
tool-call index).  It also accumulates the *raw* (placeholder-bearing) and
*de-anonymised* text so that the assistant turn can be persisted in both the
``hidden_messages`` and ``raw_messages`` session histories exactly like the
non-streaming path.
"""

from __future__ import annotations

import re
from collections.abc import Iterable, Mapping
from functools import lru_cache
from typing import Any

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

#: Labels produced by ``opf._core.runtime._label_placeholder`` for the current
#: ``openai/privacy-filter`` v2 taxonomy (the normalization is
#: ``re.sub(r"[^A-Za-z0-9]+", "_", label.upper()).strip("_")``).  ``PCM``
#: overrides this via ``PCM_PLACEHOLDER_LABELS``.
DEFAULT_PLACEHOLDER_LABELS: tuple[str, ...] = (
    "ACCOUNT_NUMBER",
    "PRIVATE_ADDRESS",
    "PRIVATE_DATE",
    "PRIVATE_EMAIL",
    "PRIVATE_PERSON",
    "PRIVATE_PHONE",
    "PRIVATE_URL",
    "SECRET",
)

#: Matches a complete placeholder and, on a miss, is guaranteed not to match a
#: longer string.
_NEVER_RE = re.compile(r"(?!x)x")

#: Extracts the label from a placeholder key, e.g. ``<PRIVATE_PERSON_12>`` →
#: ``PRIVATE_PERSON`` and ``<SECRET>`` → ``SECRET``.
_PLACEHOLDER_KEY_RE = re.compile(r"^<([A-Z0-9_]+?)(?:_\d+)?>$")


# ---------------------------------------------------------------------------
# Regex construction (memoised per label set)
# ---------------------------------------------------------------------------


@lru_cache(maxsize=32)
def _compile_regexes(labels: frozenset[str]) -> tuple[re.Pattern[str], re.Pattern[str]]:
    """Build ``(complete_tag_re, valid_prefix_re)`` for *labels*.

    Memoised because the label set is effectively constant for the lifetime of
    the process; the compiled objects are shared across every session and
    request.  See the module docstring for the exact semantics.
    """
    if not labels:
        return _NEVER_RE, _NEVER_RE

    # Longest-first so that e.g. PRIVATE_ADDRESS is attempted before a shorter
    # label that happens to share a prefix.
    label_alts = "|".join(
        re.escape(label) for label in sorted(labels, key=len, reverse=True)
    )
    tag_re = re.compile(rf"<(?:(?:{label_alts}))(?:_\d+)?>")

    # Every non-empty prefix of every label, longest-first.
    prefixes: set[str] = set()
    for label in labels:
        for end in range(1, len(label) + 1):
            prefixes.add(label[:end])
    prefix_alts = "|".join(
        re.escape(prefix) for prefix in sorted(prefixes, key=len, reverse=True)
    )
    # The outer group is optional so that the bare "<" opener is itself a valid
    # prefix.  ``_\d*`` accepts an index that is still being streamed.
    prefix_re = re.compile(rf"<(?:(?:{prefix_alts})(?:_\d*)?)?")
    return tag_re, prefix_re


def _labels_from_placeholder_map(placeholder_map: Mapping[str, str]) -> set[str]:
    """Extract labels from the authoritative session placeholder keys.

    The detector is normally built from ``PCM_PLACEHOLDER_LABELS``, but the
    session ``placeholder_map`` is the source of truth for what the
    privacy-filter actually emitted.  Merging its labels in means a taxonomy
    change (or a misconfigured label list) can no longer cause a real
    placeholder to be forwarded to the client as literal text.
    """
    labels: set[str] = set()
    for key in placeholder_map:
        match = _PLACEHOLDER_KEY_RE.match(key)
        if match:
            labels.add(match.group(1))
    return labels


# ---------------------------------------------------------------------------
# Per-field streaming de-anonymiser
# ---------------------------------------------------------------------------


class StreamingDeanonymizer:
    """Incrementally replace ``<LABEL_N>`` placeholders in a token stream.

    One instance handles a single logical text stream (e.g. the ``content`` of
    one choice).  It is intentionally tiny and allocation-light: when the
    session has no placeholders it is a pure pass-through, and when a chunk
    contains no ``<`` it performs no regex work at all.

    Args:
        placeholder_map: Session placeholder → true-text mapping.  The map is
            fixed before the LLM call and never changes mid-stream.
        labels: Normalised privacy-filter labels used to build the detector.
    """

    __slots__ = ("_map", "_tag_re", "_prefix_re", "_pending")

    def __init__(
        self,
        placeholder_map: Mapping[str, str],
        labels: Iterable[str] = DEFAULT_PLACEHOLDER_LABELS,
    ) -> None:
        self._map: Mapping[str, str] = placeholder_map
        self._tag_re, self._prefix_re = _compile_regexes(frozenset(labels))
        self._pending: str = ""

    # -- public API --------------------------------------------------------

    @property
    def pending(self) -> str:
        """The currently buffered, possibly-incomplete tag opener."""
        return self._pending

    def feed(self, text: str) -> str:
        """Consume the next chunk and return the text safe to emit now.

        The return value may be shorter than *text* (a partial tag was held)
        or longer (a held tag completed and was replaced by its true value).
        """
        if not self._map:
            # No placeholders exist for this session: nothing to detect.
            return text

        # Fast path: nothing buffered and no tag opener in this chunk.
        if not self._pending and "<" not in text:
            return text

        buffer = self._pending + text
        self._pending = ""
        if "<" not in buffer:
            return buffer

        out: list[str] = []
        cursor = 0
        length = len(buffer)
        while cursor < length:
            lt = buffer.find("<", cursor)
            if lt < 0:
                out.append(buffer[cursor:])
                break

            if lt > cursor:
                out.append(buffer[cursor:lt])

            match = self._tag_re.match(buffer, lt)
            if match is not None:
                tag = match.group(0)
                # Unknown tags pass through untouched (parity with
                # deanonymize_text, which only substitutes map keys).
                out.append(self._map.get(tag, tag))
                cursor = match.end()
                continue

            tail = buffer[lt:]
            if self._prefix_re.fullmatch(tail) is not None:
                # Could still become a placeholder: hold it for the next chunk.
                self._pending = tail
                break

            # A literal '<' that can never open a placeholder.
            out.append("<")
            cursor = lt + 1

        return "".join(out)

    def flush(self) -> str:
        """Emit any residual buffer at end of stream.

        A residual is always an unterminated prefix (``<``, ``<PRIV``, …), so
        it is returned verbatim rather than being mistaken for a placeholder.
        """
        pending = self._pending
        self._pending = ""
        return pending


# ---------------------------------------------------------------------------
# Multi-field / multi-choice / tool-call orchestration
# ---------------------------------------------------------------------------

_ASSISTANT_ROLE = "assistant"


class _ToolCallAccumulator:
    """Reassembles the hidden and de-anonymised view of one streamed tool call."""

    __slots__ = ("id", "type", "name", "hidden_arguments", "raw_arguments")

    def __init__(self) -> None:
        self.id: str | None = None
        self.type: str = "function"
        self.name: str | None = None
        self.hidden_arguments: list[str] = []
        self.raw_arguments: list[str] = []

    def as_message_dict(self, *, raw: bool) -> dict[str, Any]:
        arguments = self.raw_arguments if raw else self.hidden_arguments
        return {
            "id": self.id,
            "type": self.type,
            "function": {
                "name": self.name,
                "arguments": "".join(arguments),
            },
        }


class _ChoiceState:
    """Per-choice filters and assembled-history buffers."""

    __slots__ = (
        "_map",
        "_labels",
        "content_filter",
        "reasoning_filter",
        "tool_filters",
        "tool_calls",
        "hidden_content_parts",
        "raw_content_parts",
        "hidden_reasoning_parts",
        "raw_reasoning_parts",
        "reasoning_key",
    )

    def __init__(
        self,
        placeholder_map: Mapping[str, str],
        labels: tuple[str, ...],
    ) -> None:
        self._map = placeholder_map
        self._labels = labels
        self.content_filter = StreamingDeanonymizer(placeholder_map, labels)
        self.reasoning_filter = StreamingDeanonymizer(placeholder_map, labels)
        self.tool_filters: dict[int, StreamingDeanonymizer] = {}
        self.tool_calls: dict[int, _ToolCallAccumulator] = {}
        self.hidden_content_parts: list[str] = []
        self.raw_content_parts: list[str] = []
        # Reasoning is de-anonymised for the client and for logging, but it is
        # never written to the session history (matching the buffered path).
        self.hidden_reasoning_parts: list[str] = []
        self.raw_reasoning_parts: list[str] = []
        # Which delta key carried reasoning (``reasoning`` or the legacy
        # ``reasoning_content``), so drained residuals keep the same key.
        self.reasoning_key = "reasoning"

    def tool_filter(self, index: int) -> StreamingDeanonymizer:
        filt = self.tool_filters.get(index)
        if filt is None:
            filt = StreamingDeanonymizer(self._map, self._labels)
            self.tool_filters[index] = filt
        return filt

    def tool_accumulator(self, index: int) -> _ToolCallAccumulator:
        acc = self.tool_calls.get(index)
        if acc is None:
            acc = _ToolCallAccumulator()
            self.tool_calls[index] = acc
        return acc

    def content(self) -> str:
        return "".join(self.hidden_content_parts)

    def raw_content(self) -> str:
        return "".join(self.raw_content_parts)

    def hidden_reasoning(self) -> str:
        return "".join(self.hidden_reasoning_parts)

    def raw_reasoning(self) -> str:
        return "".join(self.raw_reasoning_parts)


class StreamResponseFilter:
    """Applies :class:`StreamingDeanonymizer` to every text field of a stream.

    The filter is stateful for the lifetime of one streaming LLM call.  Feed
    each decoded ``chat.completion.chunk`` object to :meth:`process_chunk`;
    the method mutates it in place so the caller can immediately re-serialise
    it.  At the end of the stream call :meth:`flush_choice` for each choice to
    drain residual buffers, then :meth:`assistant_history` to obtain the
    assembled assistant turn for session persistence.  :meth:`client_message`
    and :meth:`raw_message` provide the same view *including* the reasoning
    channel for logging only — reasoning is never persisted.

    Args:
        placeholder_map: Session placeholder → true-text mapping.
        labels: Normalised privacy-filter labels.
    """

    def __init__(
        self,
        placeholder_map: Mapping[str, str],
        labels: Iterable[str] = DEFAULT_PLACEHOLDER_LABELS,
    ) -> None:
        self._placeholder_map = placeholder_map
        # Freeze to a tuple for cheap reuse by the per-field filters.  Merge in
        # labels found in the session map so detection cannot miss a real tag.
        self._labels: tuple[str, ...] = tuple(
            sorted(set(labels) | _labels_from_placeholder_map(placeholder_map))
        )
        self._choices: dict[int, _ChoiceState] = {}

    # -- helpers -----------------------------------------------------------

    def _choice(self, index: int) -> _ChoiceState:
        state = self._choices.get(index)
        if state is None:
            state = _ChoiceState(self._placeholder_map, self._labels)
            self._choices[index] = state
        return state

    # -- chunk processing --------------------------------------------------

    def process_chunk(self, chunk: dict[str, Any]) -> None:
        """De-anonymise the text fields of *chunk* in place.

        Non-text fields (``role``, ``logprobs``, ``token_ids``,
        ``finish_reason`` …) are left untouched.  Handles ``content``,
        ``reasoning``/``reasoning_content`` and streaming ``tool_calls``.
        """
        for choice in chunk.get("choices") or []:
            index = choice.get("index", 0)
            state = self._choice(index)
            delta = choice.get("delta")
            if not isinstance(delta, dict):
                continue

            raw_content = delta.get("content")
            if isinstance(raw_content, str):
                state.hidden_content_parts.append(raw_content)
                filtered = state.content_filter.feed(raw_content)
                state.raw_content_parts.append(filtered)
                delta["content"] = filtered

            self._process_reasoning(state, delta)

            for tool_call in delta.get("tool_calls") or []:
                self._process_tool_call(state, tool_call)

            if choice.get("finish_reason") is not None:
                # Attach any drained partial tag to the terminal chunk so the
                # client never receives content after ``finish_reason``.
                for field_name, residual in self.flush_choice(index).items():
                    existing = delta.get(field_name)
                    delta[field_name] = (existing or "") + residual

    def _process_reasoning(self, state: _ChoiceState, delta: dict[str, Any]) -> None:
        # vLLM emits ``reasoning`` in current versions and ``reasoning_content``
        # in older ones; handle both transparently.
        for key in ("reasoning", "reasoning_content"):
            raw = delta.get(key)
            if isinstance(raw, str):
                filtered = state.reasoning_filter.feed(raw)
                state.hidden_reasoning_parts.append(raw)
                state.raw_reasoning_parts.append(filtered)
                state.reasoning_key = key
                delta[key] = filtered

    def _process_tool_call(self, state: _ChoiceState, tool_call: Any) -> None:
        if not isinstance(tool_call, dict):
            return
        index = tool_call.get("index", 0)
        accumulator = state.tool_accumulator(index)

        if tool_call.get("id") is not None:
            accumulator.id = tool_call["id"]
        if tool_call.get("type") is not None:
            accumulator.type = tool_call["type"]

        function = tool_call.get("function")
        if not isinstance(function, dict):
            return
        if function.get("name") is not None:
            accumulator.name = function["name"]

        arguments = function.get("arguments")
        if isinstance(arguments, str):
            accumulator.hidden_arguments.append(arguments)
            filtered = state.tool_filter(index).feed(arguments)
            accumulator.raw_arguments.append(filtered)
            function["arguments"] = filtered

    # -- end of stream -----------------------------------------------------

    def flush_choice(self, index: int) -> dict[str, str]:
        """Drain residual buffers for *index* and return text to emit.

        Returns a mapping keyed by field name (``"content"``, ``"reasoning"`` or
        ``"reasoning_content"``) containing only the non-empty residuals.
        Flushing is idempotent.
        """
        state = self._choices.get(index)
        if state is None:
            return {}

        residuals: dict[str, str] = {}

        content = state.content_filter.flush()
        if content:
            # Hidden history already holds the raw fragment; only the
            # de-anonymised view needs the drained tail appended.
            state.raw_content_parts.append(content)
            residuals["content"] = content

        reasoning = state.reasoning_filter.flush()
        if reasoning:
            # Hidden reasoning already holds the raw fragment; only the
            # de-anonymised view needs the drained tail appended.  Keep the
            # same key the stream used so clients assembling
            # ``reasoning_content`` are not surprised by a ``reasoning`` field.
            state.raw_reasoning_parts.append(reasoning)
            residuals[state.reasoning_key] = reasoning

        for index_, filt in state.tool_filters.items():
            residual = filt.flush()
            if residual:
                state.tool_calls[index_].raw_arguments.append(residual)

        return residuals

    def choice_indices(self) -> list[int]:
        """Indices seen so far (used to flush every choice at shutdown)."""
        return list(self._choices)

    # -- persistence + logging views ---------------------------------------

    @staticmethod
    def _content_and_tools(
        state: _ChoiceState, *, raw: bool
    ) -> tuple[str | None, list[dict[str, Any]]]:
        content = (state.raw_content() if raw else state.content()) or None
        tool_calls = [
            accumulator.as_message_dict(raw=raw)
            for _, accumulator in sorted(state.tool_calls.items())
        ]
        return content, tool_calls

    def assistant_history(
        self, index: int = 0
    ) -> tuple[dict[str, Any], dict[str, Any]]:
        """Return ``(hidden_message, raw_message)`` for choice *index*.

        The hidden message is what the LLM produced (placeholders intact) and
        is stored in ``session.hidden_messages``; the raw message is the
        de-anonymised view stored in ``session.raw_messages``.  Neither carries
        ``reasoning`` — that is only exposed through :meth:`client_message` /
        :meth:`raw_message` for logging.  ``None`` values are stripped by the
        caller, matching the non-streaming path.
        """
        state = self._choices.get(index)
        if state is None:
            return (
                {"role": _ASSISTANT_ROLE, "content": None},
                {"role": _ASSISTANT_ROLE, "content": None},
            )

        hidden_content, hidden_tool_calls = self._content_and_tools(state, raw=False)
        raw_content, raw_tool_calls = self._content_and_tools(state, raw=True)

        hidden: dict[str, Any] = {
            "role": _ASSISTANT_ROLE,
            "content": hidden_content,
        }
        raw: dict[str, Any] = {
            "role": _ASSISTANT_ROLE,
            "content": raw_content,
        }
        if hidden_tool_calls:
            hidden["tool_calls"] = hidden_tool_calls
        if raw_tool_calls:
            raw["tool_calls"] = raw_tool_calls
        return hidden, raw

    def client_message(self, index: int = 0) -> dict[str, Any]:
        """Assemble the de-anonymised assistant message the client sees.

        Unlike :meth:`assistant_history` this includes the reasoning side
        channel, so it is suitable for the ``response_body`` debug log.  It is
        never written to the session.
        """
        state = self._choices.get(index)
        if state is None:
            return {"role": _ASSISTANT_ROLE, "content": None}

        content, tool_calls = self._content_and_tools(state, raw=True)
        message: dict[str, Any] = {"role": _ASSISTANT_ROLE, "content": content}
        if state.raw_reasoning_parts:
            message["reasoning"] = state.raw_reasoning()
        if tool_calls:
            message["tool_calls"] = tool_calls
        return message

    def raw_message(self, index: int = 0) -> dict[str, Any]:
        """Like :meth:`client_message` but with placeholders intact.

        Used for the ``llm_raw_response`` debug log (what the LLM actually
        produced), including the placeholder-bearing reasoning channel.
        """
        state = self._choices.get(index)
        if state is None:
            return {"role": _ASSISTANT_ROLE, "content": None}

        content, tool_calls = self._content_and_tools(state, raw=False)
        message: dict[str, Any] = {"role": _ASSISTANT_ROLE, "content": content}
        if state.hidden_reasoning_parts:
            message["reasoning"] = state.hidden_reasoning()
        if tool_calls:
            message["tool_calls"] = tool_calls
        return message
