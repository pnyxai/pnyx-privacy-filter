"""Unit tests for ``Settings`` parsing and normalisation."""

from __future__ import annotations

import pytest

from app.config import Settings


@pytest.mark.parametrize(
    "raw, expected",
    [
        ("https://host", "https://host"),
        ("https://host/", "https://host"),
        ("https://host/v1", "https://host"),
        ("https://host/v1/", "https://host"),
        ("https://host/api/v1", "https://host/api"),
        ("  https://host/zen/go/v1  ", "https://host/zen/go"),
    ],
)
def test_llm_url_is_normalised(raw, expected):
    assert Settings(llm_url=raw).llm_url == expected


def test_llm_model_name_is_optional(monkeypatch):
    monkeypatch.delenv("PCM_LLM_MODEL_NAME", raising=False)
    settings = Settings(llm_url="http://llm.local")
    assert settings.llm_model_name is None


def test_llm_session_header_defaults_to_none(monkeypatch):
    monkeypatch.delenv("PCM_LLM_SESSION_HEADER", raising=False)
    assert Settings(llm_url="http://llm.local").llm_session_header is None


def test_llm_session_header_blank_becomes_none():
    settings = Settings(llm_url="http://llm.local", llm_session_header="   ")
    assert settings.llm_session_header is None


def test_llm_session_header_is_preserved():
    settings = Settings(
        llm_url="http://llm.local", llm_session_header="x-opencode-session"
    )
    assert settings.llm_session_header == "x-opencode-session"
