"""Endpoint detection and session-header policy derived from ``PCM_LLM_URL``.

A downstream LLM is addressed by a single string (``Settings.llm_url``).  Some
endpoints require a session header (and define what a well-formed session id
looks like); most do not.  Rather than hardcoding one host, this module keeps a
small **registry** of :class:`EndpointRule` entries, each pairing a URL matcher
with the session-header policy for that endpoint.  Adding a new endpoint is a
one-line :func:`register_endpoint_rule` call.

Resolution order (see :func:`endpoint_session_policy`):

1. An explicit ``PCM_LLM_SESSION_HEADER`` (operator override) wins.
2. Otherwise the first matching registry rule (insertion order) applies.
3. Otherwise the endpoint gets no session header.

URL-matching helpers are adapted from the Hermes agent project
(``/home/giskard/Documents/repos/hermes-agent``):

* ``utils.py::base_url_host_matches`` — commit
  ``dbb7e00e7eb51bc614f6cd1bb6b53716af9072b5`` (repo HEAD
  ``c7c2df1a536d62b45fda8907bb1898981721794d``); the subdomain-safe host match.
* ``agent/anthropic_endpoints.py`` — commit
  ``7cbffdd125477fb1628cfeb9f299ad3fec02d924``; the pattern of a dedicated
  module of pure endpoint predicates decided from the configured base URL.
* ``agent/opencode_affinity.py::opencode_session_headers`` — commit
  ``139396995a8dcfd767dbdb64e15ed591e3c66d37``; the endpoint → session-header
  mapping concept (``opencode.ai`` → ``x-opencode-session``).
"""

from __future__ import annotations

import re
import uuid
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from urllib.parse import urlparse

# Session header required by the OpenCode Zen/Go relay (opencode.ai).
OPENCODE_SESSION_HEADER = "x-opencode-session"
OPENCODE_HOST = "opencode.ai"

# Request header that selects an alternate downstream LLM base URL for a single
# request.  Only honoured when the URL is in ``PCM_LLM_URL_ALLOWLIST`` (a
# testing aid: try several engines without restarting PCM).
LLM_URL_HEADER = "x-pcm-llm-url"


def normalize_llm_url(url: str | None) -> str:
    """Normalise a downstream base URL (strip trailing slash and ``/v1``).

    PCM always appends ``/v1/...`` itself, so a base ending in ``/v1`` (a common
    mistake) is de-duplicated.  Used for both ``PCM_LLM_URL`` and the allowlist.
    """
    text = str(url or "").strip().rstrip("/")
    if text.lower().endswith("/v1"):
        text = text[:-3].rstrip("/")
    return text


def requested_llm_url(headers: Mapping[str, str] | None) -> str | None:
    """Return the ``X-PCM-LLM-URL`` header value, if any (case-insensitive)."""
    for name, value in (headers or {}).items():
        if name.lower() == LLM_URL_HEADER:
            return value
    return None


def resolve_llm_url(
    default_url: str,
    allowlist: frozenset[str] | set[str],
    requested: str | None,
) -> str:
    """Return the downstream base URL to use for this request.

    When *requested* is set and *allowlist* is non-empty, the URL must normalise
    to an allowlisted entry (else :class:`ValueError`).  With no allowlist the
    override is disabled and *default_url* is used; a stray header is ignored.
    """
    if not requested or not allowlist:
        return default_url
    normalized = normalize_llm_url(requested)
    if normalized not in allowlist:
        raise ValueError(
            "Requested X-PCM-LLM-URL is not in PCM_LLM_URL_ALLOWLIST"
        )
    return normalized



# ---------------------------------------------------------------------------
# URL matchers (pure functions over the base-URL string)
# ---------------------------------------------------------------------------


def _normalized(url: str | None) -> str:
    """Coerce *url* to a stripped string, tolerating ``None``/``httpx.URL``."""
    return str(url).strip() if url else ""


def _with_scheme(url: str | None) -> str:
    """Return *url* with a scheme so ``urlparse`` treats a bare host as netloc.

    ``"opencode.ai/zen/go"`` parses as a path; ``"//opencode.ai/zen/go"`` parses
    with ``hostname == "opencode.ai"``.  Full URLs are returned unchanged.
    """
    text = _normalized(url)
    if text and "://" not in text:
        return "//" + text
    return text


def hostname_of(url: str | None) -> str:
    """Return the lowercased hostname of *url*, or ``""`` when unparseable."""
    try:
        host = urlparse(_with_scheme(url)).hostname
    except ValueError:  # malformed IPv6 bracket / port
        return ""
    return (host or "").lower().rstrip(".")


def host_matches(url: str | None, domain: str) -> bool:
    """True when *url*'s hostname is *domain* or a subdomain of it.

    Safer than ``domain in url``: ``evil.com/opencode.ai`` and
    ``opencode.ai.evil.com`` must not match.  Accepts bare hosts, full URLs and
    URLs with paths.

    Adapted from Hermes ``utils.py::base_url_host_matches`` (commit
    ``dbb7e00e7eb51bc614f6cd1bb6b53716af9072b5``).
    """
    host = hostname_of(url)
    domain = (domain or "").strip().lower().rstrip(".")
    return bool(host and domain) and (host == domain or host.endswith("." + domain))


def path_prefix_matches(url: str | None, prefix: str) -> bool:
    """True when *url*'s path equals *prefix* or lives under it.

    ``path_prefix_matches("https://h/zen/go/v1", "/zen/go")`` is true while
    ``.../zen/gopher`` is not (segment-aware, not a raw string prefix).
    """
    path = urlparse(_with_scheme(url)).path.rstrip("/")
    normalized = "/" + str(prefix or "").strip().strip("/")
    if normalized == "/":
        return True
    return path == normalized or path.startswith(normalized + "/")


def url_regex(pattern: str) -> Callable[[str | None], bool]:
    """Return a matcher that searches the lowercased URL with *pattern*."""
    compiled = re.compile(pattern)

    def _matches(url: str | None) -> bool:
        return bool(compiled.search(_normalized(url).lower()))

    return _matches


def all_of(*matchers: Callable[[str | None], bool]) -> Callable[[str | None], bool]:
    """Combine matchers that must all hold."""

    def _matches(url: str | None) -> bool:
        return all(matcher(url) for matcher in matchers)

    return _matches


def any_of(*matchers: Callable[[str | None], bool]) -> Callable[[str | None], bool]:
    """Combine matchers of which at least one must hold."""

    def _matches(url: str | None) -> bool:
        return any(matcher(url) for matcher in matchers)

    return _matches


def host_matches_rule(domain: str) -> Callable[[str | None], bool]:
    """Rule factory: match a host (or its subdomains)."""
    return lambda url: host_matches(url, domain)


def path_prefix_rule(prefix: str) -> Callable[[str | None], bool]:
    """Rule factory: match a URL path prefix."""
    return lambda url: path_prefix_matches(url, prefix)


def url_regex_rule(pattern: str) -> Callable[[str | None], bool]:
    """Rule factory: match a regular expression against the URL."""
    return url_regex(pattern)


# ---------------------------------------------------------------------------
# Session-id validation / generation
# ---------------------------------------------------------------------------


def generic_validate(value: str | None) -> bool:
    """A session id is well-formed when non-empty with no surrounding space."""
    return bool(value) and value == value.strip()


# Opaque tokens (no internal whitespace) are accepted as OpenCode session ids.
_OPENCODE_SESSION_RE = re.compile(r"^\S{1,256}$")


def opencode_validate(value: str | None) -> bool:
    """Permissive check for an OpenCode/Zen session id.

    The relay accepts opaque tokens; we only reject empty or whitespace-bearing
    values so that an obviously malformed client value is replaced rather than
    forwarded verbatim.
    """
    return bool(value) and bool(_OPENCODE_SESSION_RE.match(value))


def generate_uuid() -> str:
    """Generate a fresh, opaque endpoint session id."""
    return uuid.uuid4().hex


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class EndpointSessionPolicy:
    """How session identity is represented for a given upstream endpoint.

    * ``header_name`` — the header the endpoint expects (or ``None`` when it
      needs no session header).
    * ``validate`` — whether a client-supplied value is well-formed for this
      endpoint and may therefore be reused as the endpoint session id.
    * ``generate`` — produce a fresh, well-formed endpoint session id.
    """

    header_name: str | None
    validate: Callable[[str | None], bool] = generic_validate
    generate: Callable[[], str] = generate_uuid


@dataclass(frozen=True)
class EndpointRule:
    """A URL matcher paired with the session-header policy for that endpoint."""

    name: str
    matches: Callable[[str | None], bool]
    header_name: str | None
    validate: Callable[[str | None], bool] = generic_validate
    generate: Callable[[], str] = generate_uuid


_REGISTERED_RULES: list[EndpointRule] = []


def register_endpoint_rule(rule: EndpointRule) -> None:
    """Append *rule* to the registry (checked in insertion order).

    Built-in rules are registered at import time; callers may register more to
    teach PCM about additional endpoints.
    """
    _REGISTERED_RULES.append(rule)


def endpoint_rules() -> tuple[EndpointRule, ...]:
    """Return the registered rules in match order."""
    return tuple(_REGISTERED_RULES)


def endpoint_rule_for(url: str | None) -> EndpointRule | None:
    """Return the first rule whose matcher accepts *url*, or ``None``."""
    return next((rule for rule in _REGISTERED_RULES if rule.matches(url)), None)


# Built-in endpoints.  Add new ones as one-liners here.
register_endpoint_rule(
    EndpointRule(
        name="opencode",
        matches=host_matches_rule(OPENCODE_HOST),
        header_name=OPENCODE_SESSION_HEADER,
        validate=opencode_validate,
    )
)


# ---------------------------------------------------------------------------
# Public resolution API
# ---------------------------------------------------------------------------


def endpoint_session_policy(
    url: str | None, override_header: str | None = None
) -> EndpointSessionPolicy:
    """Resolve the endpoint session policy for *url*.

    An explicit ``PCM_LLM_SESSION_HEADER`` (*override_header*) wins; otherwise
    the first matching registry rule applies; otherwise no session header.
    """
    header = (override_header or "").strip()
    if header:
        return EndpointSessionPolicy(header_name=header)
    rule = endpoint_rule_for(url)
    if rule is None:
        return EndpointSessionPolicy(header_name=None)
    return EndpointSessionPolicy(
        header_name=rule.header_name,
        validate=rule.validate,
        generate=rule.generate,
    )


def default_session_header_for(url: str | None) -> str | None:
    """Return the session header the endpoint requires by convention, if any."""
    rule = endpoint_rule_for(url)
    return rule.header_name if rule is not None else None


def is_opencode_endpoint(url: str | None) -> bool:
    """True when *url*'s host is ``opencode.ai`` (including subdomains)."""
    return host_matches(url, OPENCODE_HOST)
