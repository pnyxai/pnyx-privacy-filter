import re
from typing import Any

from pydantic import field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

from .streaming import DEFAULT_PLACEHOLDER_LABELS

# Roles filtered by default when PCM_FILTERABLE_ROLES is not set.
_DEFAULT_FILTERABLE_ROLES: frozenset[str] = frozenset({"user", "tool", "function"})


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
        """Normalise the downstream base URL.

        PCM always appends ``/v1/...`` itself, so a base that already ends in
        ``/v1`` (a common user mistake) is de-duplicated here.  Trailing
        slashes are stripped so the join never produces ``//``.
        """
        url = v.strip().rstrip("/")
        if url.lower().endswith("/v1"):
            url = url[:-3].rstrip("/")
        return url

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

    # Uvicorn bind settings (used by the startup script / Docker CMD)
    host: str = "0.0.0.0"
    port: int = 8080

    # Logging
    log_level: str = "INFO"

    # Comma-separated list of message roles whose content is sent through the
    # Triton privacy filter before being stored in the hidden session history.
    filterable_roles: Any = _DEFAULT_FILTERABLE_ROLES

    @field_validator("filterable_roles", mode="before")
    @classmethod
    def _parse_filterable_roles(cls, v: Any) -> frozenset[str]:
        if isinstance(v, str):
            roles = frozenset(r.strip().lower() for r in v.split(",") if r.strip())
            return roles if roles else _DEFAULT_FILTERABLE_ROLES
        if isinstance(v, (list, tuple, set, frozenset)):
            return frozenset(str(r).strip().lower() for r in v if str(r).strip())
        return _DEFAULT_FILTERABLE_ROLES

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
