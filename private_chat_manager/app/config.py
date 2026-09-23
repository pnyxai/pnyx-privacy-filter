import re
from typing import Any

from pydantic import field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

from .endpoints import normalize_llm_url
from .streaming import DEFAULT_PLACEHOLDER_LABELS

# Roles/fields are redacted by default (default-deny).  The pass-through
# settings below are the only way to opt a role or message field out; both
# default to empty.
def _parse_name_set(v: Any) -> frozenset[str]:
    """Parse a comma-separated (or iterable) list of lowercased names."""
    if isinstance(v, str):
        return frozenset(name.strip().lower() for name in v.split(",") if name.strip())
    if isinstance(v, (list, tuple, set, frozenset)):
        return frozenset(str(name).strip().lower() for name in v if str(name).strip())
    return frozenset()

# Duration units accepted by the session-TTL settings (seconds per unit).
_DURATION_UNITS = {"s": 1, "m": 60, "h": 3600, "d": 86400, "w": 604800}
# A non-negative number (digit before any dot) followed by a single unit.
_DURATION_RE = re.compile(r"^\s*(\d+(?:\.\d+)?)\s*([smhdw])\s*$", re.IGNORECASE)


def parse_duration(value: Any, *, setting: str) -> int:
    """Parse a human duration like ``30s``/``360m``/``6h``/``1.5d``/``2w``.

    Units: ``s`` seconds, ``m`` minutes, ``h`` hours, ``d`` days, ``w`` weeks.
    The number may be an integer or a float (e.g. ``1.5d`` = 1.5 days) and must
    have a digit before any dot.  Empty/``None``/``0`` disable the feature
    (``0``).  Any other shape raises :class:`ValueError` so misconfiguration
    fails fast.
    """
    if value is None:
        return 0
    if isinstance(value, bool):
        raise ValueError(f"{setting} must be a duration, not a boolean")
    if isinstance(value, (int, float)):
        if value < 0:
            raise ValueError(f"{setting} must not be negative")
        return int(round(float(value)))

    text = str(value).strip()
    if not text:
        return 0
    try:
        if float(text) == 0:
            return 0
    except ValueError:
        pass

    match = _DURATION_RE.match(text)
    if not match:
        raise ValueError(
            f"{setting} must be a duration like '30s', '360m', '6h', '1.5d' or '2w' "
            f"(units: s=seconds, m=minutes, h=hours, d=days, w=weeks); got {value!r}"
        )
    amount = float(match.group(1))
    return int(round(amount * _DURATION_UNITS[match.group(2).lower()]))


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_prefix="PCM_",
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # Downstream LLM endpoint
    llm_url: str
    llm_api_key: str = ""
    # Optional fallback model used when the client does not send one.
    llm_model_name: str | None = None
    # Optional upstream-specific session header (e.g. "x-opencode-session" for
    # the OpenCode Zen gateway, "X-Hermes-Session-Id" for Hermes).  When set,
    # the resolved client session id is also emitted under this name.  Leave
    # empty for upstreams that need no session header (vLLM, OpenRouter, …).
    llm_session_header: str | None = None

    @field_validator("llm_url", mode="after")
    @classmethod
    def _normalize_llm_url(cls, v: str) -> str:
        """Normalise the downstream base URL (strip trailing slash and ``/v1``)."""
        return normalize_llm_url(v)

    # Optional allowlist of *additional* downstream LLM base URLs a request may
    # select via the ``X-PCM-LLM-URL`` header (comma-separated).  Empty disables
    # the override entirely (the header is ignored).  Intended as a testing aid
    # so several engines can be exercised without restarting PCM; the allowlist
    # is the security boundary against arbitrary (SSRF) targets.
    llm_url_allowlist: Any = frozenset()

    @field_validator("llm_url_allowlist", mode="before")
    @classmethod
    def _parse_llm_url_allowlist(cls, v: Any) -> frozenset[str]:
        if isinstance(v, str):
            raw = [url for url in v.split(",") if url.strip()]
        elif isinstance(v, (list, tuple, set, frozenset)):
            raw = [str(url) for url in v if str(url).strip()]
        else:
            return frozenset()
        return frozenset(n for n in (normalize_llm_url(url) for url in raw) if n)

    @field_validator("llm_session_header", mode="before")
    @classmethod
    def _parse_llm_session_header(cls, v: Any) -> str | None:
        if v is None:
            return None
        name = str(v).strip()
        return name or None

    # Triton privacy-filter server
    triton_url: str = "localhost:8000"
    triton_model_name: str = "ensemble_model"
    # Maximum characters sent to Triton in one inference request.  Longer
    # messages are split at natural boundaries and redacted chunk by chunk
    # (the ensemble OOMs on very long inputs).  <=0 disables the limit.
    triton_max_chars: int = 8000

    # Embedded session database
    db_path: str = "./sessions.db"

    # Session time-to-live, counted from the last activity (``updated_at``). A
    # duration like "360m", "6h", "1.5d" or "2w" (m/h/d/w = minutes/hours/days/
    # weeks; integer or float). Empty/0 disables expiry (the default).
    session_ttl: int = 0

    @field_validator("session_ttl", mode="before")
    @classmethod
    def _parse_session_ttl(cls, v: Any) -> int:
        return parse_duration(v, setting="PCM_SESSION_TTL")

    # Grace added to the TTL before the background sweeper physically deletes a
    # session.  `updated_at` is refreshed when a session is resolved, but a very
    # long redaction/upstream/stream can still outlive the TTL; this margin
    # keeps the sweeper from deleting an in-flight session's row and history.
    session_ttl_grace: int = 120

    @field_validator("session_ttl_grace", mode="before")
    @classmethod
    def _parse_session_ttl_grace(cls, v: Any) -> int:
        return parse_duration(v, setting="PCM_SESSION_TTL_GRACE")

    # How often the background sweeper purges expired sessions (same duration
    # syntax). Only used when PCM_SESSION_TTL is enabled.
    session_ttl_sweep: int = 600

    @field_validator("session_ttl_sweep", mode="before")
    @classmethod
    def _parse_session_ttl_sweep(cls, v: Any) -> int:
        return parse_duration(v, setting="PCM_SESSION_TTL_SWEEP")

    # Uvicorn bind settings (used by the startup script / Docker CMD)
    host: str = "0.0.0.0"
    port: int = 8080

    # Logging
    log_level: str = "INFO"

    # ------------------------------------------------------------------
    # Privacy filtering scope (default-deny).
    #
    # Client-supplied history is untrusted: every message role and every known
    # text field is redacted unless explicitly listed here as a pass-through.
    # Fresh conversations (cursor == 0) use the *_FRESH sets, which do NOT
    # inherit the resumed sets, so loosening the resumed policy cannot leak PII
    # from a replayed conversation.  Retired: PCM_FILTERABLE_ROLES.
    # ------------------------------------------------------------------
    passthrough_roles: Any = frozenset()
    passthrough_roles_fresh: Any = frozenset()
    passthrough_fields: Any = frozenset()
    passthrough_fields_fresh: Any = frozenset()

    @field_validator(
        "passthrough_roles",
        "passthrough_roles_fresh",
        "passthrough_fields",
        "passthrough_fields_fresh",
        mode="before",
    )
    @classmethod
    def _parse_passthrough(cls, v: Any) -> frozenset[str]:
        return _parse_name_set(v)

    # Labels used by the streaming de-anonymiser to recognise placeholder tags.
    # They must match the labels emitted by the privacy-filter (after the
    # ``_label_placeholder`` normalisation: uppercase, non-alphanumerics → "_").
    # The default covers the current openai/privacy-filter v2 taxonomy.
    placeholder_labels: Any = frozenset(DEFAULT_PLACEHOLDER_LABELS)

    @field_validator("placeholder_labels", mode="before")
    @classmethod
    def _parse_placeholder_labels(cls, v: Any) -> frozenset[str]:
        def _normalize(label: str) -> str:
            return re.sub(r"[^A-Za-z0-9]+", "_", label.upper()).strip("_")

        if isinstance(v, str):
            raw_labels: list[str] = v.split(",")
        elif isinstance(v, (list, tuple, set, frozenset)):
            raw_labels = [str(raw) for raw in v]
        else:
            return frozenset(DEFAULT_PLACEHOLDER_LABELS)

        labels = frozenset(n for n in (_normalize(raw) for raw in raw_labels) if n)
        return labels if labels else frozenset(DEFAULT_PLACEHOLDER_LABELS)

    # Text appended to the client's system prompt so the downstream LLM knows
    # how to treat placeholder tags (e.g. "<PRIVATE_PERSON_1>"). Empty disables
    # the feature. In a .env file, use "\n" for newlines (or a quoted,
    # multi-line value); literal "\n"/"\t" escapes are converted here.
    system_prompt_pii_instruction: str = ""

    @field_validator("system_prompt_pii_instruction", mode="before")
    @classmethod
    def _parse_system_prompt_pii_instruction(cls, v: Any) -> str:
        if v is None:
            return ""
        text = str(v)
        if "\\" not in text:
            return text
        return text.replace("\\n", "\n").replace("\\t", "\t")

    # Comma-separated list of verbose event names to emit at DEBUG level.
    # Default: empty — none of the heavy payloads are logged.
    # Available events:
    #   request_body      — incoming PrivateChatRequest + X-Session-ID header
    #   redaction_result  — full Triton response including original PII spans (⚠ contains PII)
    #   llm_payload       — redacted payload + forwarded headers sent to the LLM
    #   llm_raw_response  — raw LLM response before de-anonymisation
    #   session_state     — full session (raw + hidden messages + privacy state) after save (⚠ contains PII)
    #   response_body     — final de-anonymised response returned to the client (⚠ contains PII)
    # Use "*" to enable all events.
    # Example: PCM_VERBOSE_LOG_EVENTS=llm_payload,llm_raw_response
    verbose_log_events: Any = frozenset()

    @field_validator("verbose_log_events", mode="before")
    @classmethod
    def _parse_verbose_log_events(cls, v: Any) -> frozenset[str]:
        if isinstance(v, str):
            v = v.strip()
            if v == "*":
                return frozenset({
                    "request_body", "redaction_result", "llm_payload",
                    "llm_raw_response", "session_state", "response_body",
                })
            return frozenset(e.strip().lower() for e in v.split(",") if e.strip())
        if isinstance(v, (list, tuple, set, frozenset)):
            return frozenset(str(e).strip().lower() for e in v if str(e).strip())
        return frozenset()
