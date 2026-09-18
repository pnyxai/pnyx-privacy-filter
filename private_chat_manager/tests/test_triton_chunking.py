"""Unit tests for Triton long-input chunking and result merging.

These exercise the client without any network I/O by stubbing the private
``_infer_single`` method.
"""

from __future__ import annotations

import pytest

from app.triton_client import (
    TritonPrivacyFilterClient,
    merge_redaction_results,
    split_text_chunks,
)


def _result(text: str, spans: list | None = None) -> dict:
    return {
        "schema_version": 1,
        "summary": {"output_type": "typed", "span_count": len(spans or [])},
        "text": text,
        "detected_spans": spans or [],
        "redacted_text": text,
    }


# ---------------------------------------------------------------------------
# split_text_chunks
# ---------------------------------------------------------------------------


def test_split_short_text_is_single_chunk():
    assert split_text_chunks("hello world", 100) == ["hello world"]


def test_split_exact_length_is_single_chunk():
    text = "x" * 50
    assert split_text_chunks(text, 50) == [text]


def test_split_respects_max_and_preserves_text():
    text = ("word " * 100).strip()
    chunks = split_text_chunks(text, 50)
    assert len(chunks) > 1
    assert all(len(c) <= 50 for c in chunks)
    assert "".join(chunks) == text


def test_split_prefers_paragraph_boundary():
    text = "A" * 40 + "\n\n" + "B" * 40
    chunks = split_text_chunks(text, 50)
    assert chunks[0] == "A" * 40 + "\n\n"
    assert chunks[1] == "B" * 40


def test_split_ignores_boundary_before_midpoint():
    text = "A" * 10 + " " + "B" * 100
    chunks = split_text_chunks(text, 50)
    # The space at index 10 is before the midpoint, so a hard cut is used.
    assert len(chunks[0]) == 50


def test_split_hard_cut_without_boundaries():
    text = "A" * 120
    chunks = split_text_chunks(text, 50)
    assert [len(c) for c in chunks] == [50, 50, 20]
    assert "".join(chunks) == text


# ---------------------------------------------------------------------------
# merge_redaction_results
# ---------------------------------------------------------------------------


def test_merge_concatenates_and_shifts_offsets():
    r1 = _result(
        "abcde",
        [{"label": "X", "start": 2, "end": 5, "text": "cde", "placeholder": "<X>"}],
    )
    r1["redacted_text"] = "ab<X>"
    r2 = _result(
        "fgh",
        [{"label": "Y", "start": 1, "end": 2, "text": "g", "placeholder": "<Y>"}],
    )
    r2["redacted_text"] = "f<Y>h"

    merged = merge_redaction_results([r1, r2])
    assert merged["text"] == "abcdefgh"
    assert merged["redacted_text"] == "ab<X>f<Y>h"
    assert merged["summary"]["span_count"] == 2
    assert merged["detected_spans"][0]["start"] == 2
    assert merged["detected_spans"][1]["start"] == 6


# ---------------------------------------------------------------------------
# TritonPrivacyFilterClient._sync_infer
# ---------------------------------------------------------------------------


def test_short_text_is_single_inference(monkeypatch):
    client = TritonPrivacyFilterClient("url", "model", max_chars=1000)
    calls: list[str] = []

    def fake(text: str) -> dict:
        calls.append(text)
        return _result(text)

    monkeypatch.setattr(client, "_infer_single", fake)
    out = client._sync_infer("hello")

    assert calls == ["hello"]
    assert out["redacted_text"] == "hello"


def test_long_text_is_chunked_and_merged(monkeypatch):
    client = TritonPrivacyFilterClient("url", "model", max_chars=20)
    calls: list[str] = []

    def fake(text: str) -> dict:
        calls.append(text)
        return _result(text)

    monkeypatch.setattr(client, "_infer_single", fake)
    text = ("word " * 20).strip()
    out = client._sync_infer(text)

    assert len(calls) > 1
    assert out["text"] == text
    assert out["redacted_text"] == text


def test_empty_text_skips_triton(monkeypatch):
    client = TritonPrivacyFilterClient("url", "model")

    def boom(text: str) -> dict:
        raise AssertionError("Triton must not be called for blank input")

    monkeypatch.setattr(client, "_infer_single", boom)
    out = client._sync_infer("   ")
    assert out["detected_spans"] == []


def test_failing_chunk_is_bisected(monkeypatch):
    client = TritonPrivacyFilterClient("url", "model", max_chars=1000)

    def fake(text: str) -> dict:
        if len(text) > 700:
            raise RuntimeError("simulated OOM")
        return _result(text)

    monkeypatch.setattr(client, "_infer_single", fake)
    text = "bbbb " * 300
    out = client._sync_infer(text)

    assert out["text"] == text
    assert out["detected_spans"] == []


def test_persistent_failure_propagates(monkeypatch):
    client = TritonPrivacyFilterClient("url", "model", max_chars=1000)

    def boom(text: str) -> dict:
        raise RuntimeError("triton down")

    monkeypatch.setattr(client, "_infer_single", boom)
    with pytest.raises(RuntimeError):
        client._sync_infer("x" * 500)
