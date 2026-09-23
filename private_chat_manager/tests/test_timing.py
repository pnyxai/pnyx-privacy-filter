"""Unit tests for the per-request phase timing helper."""

from __future__ import annotations

import pytest

import app.timing as timing
from app.timing import RequestTimings


def test_marks_accumulate_and_as_fields(monkeypatch):
    # start, _last, mark(resolve), mark(redact), mark(redact), total
    clock = iter([0.0, 0.0, 0.1, 0.3, 0.35, 0.6])
    monkeypatch.setattr(timing.time, "monotonic", lambda: next(clock))

    t = RequestTimings()
    assert t.mark("resolve") == pytest.approx(100.0)
    # A repeated phase accumulates.
    assert t.mark("redact") == pytest.approx(200.0)
    assert t.mark("redact") == pytest.approx(50.0)

    fields = t.as_fields()
    assert fields["resolve_ms"] == 100.0
    assert fields["redact_ms"] == 250.0
    assert fields["total_ms"] == 600.0


def test_total_is_at_least_sum_of_phases():
    t = RequestTimings()
    a = t.mark("a")
    b = t.mark("b")
    assert t.total_ms >= a + b
