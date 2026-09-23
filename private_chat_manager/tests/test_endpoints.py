"""Unit tests for the endpoint registry and session-header policy.

Endpoint detection and the ``PCM_LLM_URL`` → session-header mapping live in
:mod:`app.endpoints` (adapted from the Hermes agent project; see the module
docstring there for source files and commits).
"""

from __future__ import annotations

import pytest

from app import endpoints
from app.endpoints import (
    EndpointRule,
    all_of,
    any_of,
    default_session_header_for,
    endpoint_rule_for,
    endpoint_rules,
    endpoint_session_policy,
    generic_validate,
    host_matches,
    host_matches_rule,
    hostname_of,
    is_opencode_endpoint,
    opencode_validate,
    path_prefix_matches,
    path_prefix_rule,
    register_endpoint_rule,
    url_regex,
    url_regex_rule,
)


@pytest.fixture
def restore_registry():
    """Restore the global registry after a test registers extra rules."""
    original = list(endpoints._REGISTERED_RULES)
    try:
        yield
    finally:
        endpoints._REGISTERED_RULES[:] = original


# ---------------------------------------------------------------------------
# URL matchers
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "url, expected",
    [
        ("https://opencode.ai/zen/go", "opencode.ai"),
        ("https://api.opencode.ai/v1", "api.opencode.ai"),
        ("http://opencode.ai:8080/zen", "opencode.ai"),
        ("https://user:pass@opencode.ai/zen", "opencode.ai"),
        ("opencode.ai/zen/go", "opencode.ai"),
        ("OpenCode.AI", "opencode.ai"),
        ("https://opencode.ai./zen", "opencode.ai"),
        ("", ""),
        (None, ""),
    ],
)
def test_hostname_of(url, expected):
    assert hostname_of(url) == expected


@pytest.mark.parametrize(
    "url, expected",
    [
        ("https://opencode.ai/zen/go/v1", True),
        ("https://opencode.ai", True),
        ("https://api.opencode.ai/v1", True),
        ("http://localhost:8080/v1", False),
        ("https://evilopencode.ai/v1", False),
        ("https://opencode.ai.evil.com/v1", False),
        ("https://evil.com/opencode.ai", False),
        ("", False),
        (None, False),
    ],
)
def test_host_matches(url, expected):
    assert host_matches(url, "opencode.ai") is expected


@pytest.mark.parametrize(
    "url, prefix, expected",
    [
        ("https://h/zen/go/v1", "/zen/go", True),
        ("https://h/zen/go", "/zen/go", True),
        ("https://h/zen/go/", "/zen/go", True),
        ("https://h/zen/gopher", "/zen/go", False),
        ("https://h/zen", "/zen/go", False),
        ("https://h/anything", "/", True),
    ],
)
def test_path_prefix_matches(url, prefix, expected):
    assert path_prefix_matches(url, prefix) is expected


def test_url_regex_is_case_insensitive_search():
    matcher = url_regex(r"zen/go")
    assert matcher("https://OpenCode.AI/ZEN/GO/v1")
    assert not matcher("https://opencode.ai/zen")


def test_all_of_and_any_of():
    combined_all = all_of(host_matches_rule("opencode.ai"), path_prefix_rule("/zen/go"))
    assert combined_all("https://opencode.ai/zen/go/v1")
    assert not combined_all("https://opencode.ai/zen")

    combined_any = any_of(host_matches_rule("opencode.ai"), path_prefix_rule("/zen/go"))
    assert combined_any("https://other.example/zen/go")
    assert not combined_any("https://other.example/zen")


def test_rule_factories():
    assert host_matches_rule("opencode.ai")("https://opencode.ai")
    assert path_prefix_rule("/zen")("https://h/zen/go")
    assert url_regex_rule(r"opencode\.ai")("https://opencode.ai")


# ---------------------------------------------------------------------------
# Validators
# ---------------------------------------------------------------------------


def test_generic_validate():
    assert generic_validate("abc")
    assert not generic_validate("")
    assert not generic_validate(None)
    assert not generic_validate(" padded ")


def test_opencode_validate():
    assert opencode_validate("abc-123")
    assert not opencode_validate("has spaces")
    assert not opencode_validate("")
    assert not opencode_validate(None)


# ---------------------------------------------------------------------------
# Policy resolution
# ---------------------------------------------------------------------------


def test_default_session_header_for():
    assert default_session_header_for("https://opencode.ai/zen/go") == (
        "x-opencode-session"
    )
    assert default_session_header_for("http://localhost:8080") is None


@pytest.mark.parametrize(
    "url, expected",
    [
        ("https://opencode.ai/zen/go/v1", True),
        ("https://opencode.ai", True),
        ("https://api.opencode.ai/v1", True),
        ("http://localhost:8080/v1", False),
        ("https://evilopencode.ai/v1", False),
        ("https://opencode.ai.evil.com/v1", False),
        ("", False),
        (None, False),
    ],
)
def test_is_opencode_endpoint(url, expected):
    assert is_opencode_endpoint(url) is expected


def test_endpoint_session_policy_opencode():
    policy = endpoint_session_policy("https://opencode.ai/zen/go")
    assert policy.header_name == "x-opencode-session"
    assert policy.validate("abc-123")
    assert not policy.validate("has spaces")
    assert not policy.validate("")
    assert policy.generate()


def test_endpoint_session_policy_plain_endpoint_has_no_header():
    policy = endpoint_session_policy("http://llm.local")
    assert policy.header_name is None
    # A plain endpoint still gets a working generic policy.
    assert policy.validate("anything")
    assert policy.generate()


def test_explicit_override_wins_over_registry():
    policy = endpoint_session_policy(
        "https://opencode.ai/zen/go", override_header="x-hermes-session"
    )
    assert policy.header_name == "x-hermes-session"
    # The override uses the generic validator, not opencode's.
    assert policy.validate("has spaces")


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------


def test_builtin_opencode_rule_is_registered():
    names = [rule.name for rule in endpoint_rules()]
    assert "opencode" in names
    rule = endpoint_rule_for("https://opencode.ai/zen/go")
    assert rule is not None and rule.name == "opencode"


def test_register_endpoint_rule(restore_registry):
    register_endpoint_rule(
        EndpointRule(
            name="litellm",
            matches=host_matches_rule("litellm.example.com"),
            header_name="x-litellm-session-id",
        )
    )
    rule = endpoint_rule_for("https://litellm.example.com/v1")
    assert rule is not None and rule.name == "litellm"

    policy = endpoint_session_policy("https://litellm.example.com/v1")
    assert policy.header_name == "x-litellm-session-id"
    assert policy.validate("opaque")


def test_first_matching_rule_wins(restore_registry):
    register_endpoint_rule(
        EndpointRule(
            name="specific",
            matches=all_of(
                host_matches_rule("proxy.example.com"), path_prefix_rule("/tenant-a")
            ),
            header_name="x-tenant-a-session",
        )
    )
    register_endpoint_rule(
        EndpointRule(
            name="broad",
            matches=host_matches_rule("proxy.example.com"),
            header_name="x-broad-session",
        )
    )

    assert (
        endpoint_session_policy("https://proxy.example.com/tenant-a/v1").header_name
        == "x-tenant-a-session"
    )
    assert (
        endpoint_session_policy("https://proxy.example.com/tenant-b/v1").header_name
        == "x-broad-session"
    )
