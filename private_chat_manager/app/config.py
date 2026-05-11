from typing import Any

from pydantic import field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

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
    llm_model_name: str

    # Triton privacy-filter server
    triton_url: str = "localhost:8000"
    triton_model_name: str = "ensemble_model"

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

    # Comma-separated list of verbose event names to emit at DEBUG level.
    # Default: empty — none of the heavy payloads are logged.
    # Available events:
    #   request_body      — incoming PrivateChatRequest + X-Session-ID header
    #   redaction_result  — full Triton response including original PII spans (⚠ contains PII)
    #   llm_payload       — full payload sent to the LLM (hidden/redacted messages)
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
