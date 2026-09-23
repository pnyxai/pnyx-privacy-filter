"""Unit tests for ``Settings`` parsing and normalisation."""

from __future__ import annotations

import pytest

from app.config import Settings, parse_duration


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


# ---------------------------------------------------------------------------
# Session TTL duration parsing
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "raw, expected",
    [
        ("1d", 86400),
        ("6h", 21600),
        ("360m", 21600),
        ("2w", 1209600),
        ("30s", 30),
        ("1.5s", 2),
        ("90S", 90),
        ("1.5d", 129600),
        ("0.5h", 1800),
        ("1.5D", 129600),
        (" 2 d ", 172800),
        ("0", 0),
        ("0.0", 0),
        ("", 0),
        (None, 0),
        (0, 0),
        (90, 90),
        (1.5, 2),
    ],
)
def test_parse_duration(raw, expected):
    assert parse_duration(raw, setting="PCM_SESSION_TTL") == expected


@pytest.mark.parametrize(
    "raw",
    ["1.5", "1.5x", "d", "-1d", "1.5.5d", ".5d", "abc", "1d2h"],
)
def test_parse_duration_rejects_invalid(raw):
    with pytest.raises(ValueError):
        parse_duration(raw, setting="PCM_SESSION_TTL")


def test_session_ttl_defaults_to_disabled(monkeypatch):
    monkeypatch.delenv("PCM_SESSION_TTL", raising=False)
    assert Settings(llm_url="http://llm.local").session_ttl == 0


def test_session_ttl_is_parsed():
    settings = Settings(llm_url="http://llm.local", session_ttl="1.5d")
    assert settings.session_ttl == 129600


def test_session_ttl_sweep_default():
    assert Settings(llm_url="http://llm.local").session_ttl_sweep == 600


def test_session_ttl_sweep_is_parsed():
    settings = Settings(llm_url="http://llm.local", session_ttl_sweep="30m")
    assert settings.session_ttl_sweep == 1800
