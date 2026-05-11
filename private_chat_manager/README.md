# PrivateChatManager (PCM)

OpenAI-compatible privacy mid-layer that sits between a client and any downstream LLM. PII is redacted before the LLM sees the request and de-anonymised before the response reaches the client. State is maintained per conversation across multiple turns.

## Request / response flow

```
Client
  │  POST /v1/chat/completions
  │  X-Session-ID: <uuid>   (optional)
  ▼
PCM
  ├─ 1. Resolve session ID (header → existing session, absent → new UUID)
  ├─ 2. Load session from SQLite (or create empty)
  ├─ 3. Identify new messages (everything past the stored history cursor)
  ├─ 4. For each new filterable message:
  │       → Triton: plain text in, RedactionResult out
  │       → apply_placeholder_indexing: base placeholders → <LABEL_N>
  │       → store redacted copy in hidden_messages
  ├─ 5. Forward hidden_messages to the downstream LLM
  ├─ 6. De-anonymise the LLM response (replace <LABEL_N> → original text)
  ├─ 7. Persist raw_messages + hidden_messages + PrivacyFilterState to SQLite
  └─ 8. Return de-anonymised response + X-Session-ID header to the client
```

Non-filterable roles (`system` by default) and `assistant` messages pass through without touching Triton.

## Session state

Each session stores three independent things in SQLite:

| Field | Contains | Sent to LLM |
|---|---|---|
| `raw_messages` | Original messages with real PII | Never |
| `hidden_messages` | Redacted messages with `<LABEL_N>` placeholders | Yes |
| `privacy_state` | Placeholder map, per-type counters, redaction audit log | Never |

The **cursor** mechanism allows clients to follow standard OpenAI conventions (send the full history every turn). PCM slices from `len(hidden_messages)` onward to find only the new messages, so previously processed turns are never re-sent to Triton.

## Placeholder indexing

Triton returns base placeholders like `<PRIVATE_PERSON>` that are not unique across spans. PCM converts them to globally-unique, session-scoped tokens:

```
"My name is Alice and my colleague is Bob."
→ Triton:  "My name is <PRIVATE_PERSON> and my colleague is <PRIVATE_PERSON>."
→ PCM:     "My name is <PRIVATE_PERSON_1> and my colleague is <PRIVATE_PERSON_2>."
```

Counters are **never reset** between turns — if `Alice` appears again in turn 3 she is still `<PRIVATE_PERSON_1>`, not `<PRIVATE_PERSON_3>`. This guarantees the LLM always refers to the same person by the same token.

## Bypass mode

Setting `bypass_privacy_filter: true` in the request body skips Triton entirely for that turn. The raw message is appended to both `raw_messages` and `hidden_messages` unchanged. De-anonymisation of the LLM response still runs using the existing placeholder map from earlier turns.

## API endpoints

| Method | Path | Description |
|---|---|---|
| `POST` | `/v1/chat/completions` | Privacy-aware chat completions |
| `GET` | `/v1/sessions/{session_id}` | Inspect raw messages, hidden messages, and privacy state |
| `GET` | `/health` | Liveness probe |
| `GET` | `/health/triton` | Readiness probe — checks Triton model is ready |

### Request body (`POST /v1/chat/completions`)

Standard OpenAI `ChatCompletion` fields plus:

| Field | Type | Default | Description |
|---|---|---|---|
| `bypass_privacy_filter` | bool | `false` | Skip Triton redaction for this turn |

Streaming (`stream: true`) is **currently not supported** and returns HTTP 400.

### Response

Standard OpenAI `ChatCompletion` response — no extra fields. The resolved session ID is returned in the `X-Session-ID` **response header**. Pass it back as `X-Session-ID` on the next request to continue the same session.

## Configuration

All settings are read from environment variables with the `PCM_` prefix.

| Variable | Required | Default | Description |
|---|---|---|---|
| `PCM_LLM_URL` | yes | — | Base URL of the downstream LLM (e.g. `http://vllm:8000`) |
| `PCM_LLM_MODEL_NAME` | yes | — | Model name forwarded in every LLM request |
| `PCM_LLM_API_KEY` | no | `""` | Bearer token for the LLM endpoint |
| `PCM_TRITON_URL` | no | `localhost:8000` | Host and port of the Triton server |
| `PCM_TRITON_MODEL_NAME` | no | `ensemble_model` | Triton model name |
| `PCM_DB_PATH` | no | `./sessions.db` | SQLite database path inside the container |
| `PCM_HOST` | no | `0.0.0.0` | Uvicorn bind address |
| `PCM_PORT` | no | `8080` | Uvicorn bind port |
| `PCM_LOG_LEVEL` | no | `INFO` | Log verbosity: `DEBUG` \| `INFO` \| `WARNING` \| `ERROR` |
| `PCM_FILTERABLE_ROLES` | no | `user,tool,function` | Comma-separated roles sent through Triton |
| `PCM_VERBOSE_LOG_EVENTS` | no | `""` | Comma-separated debug event names (see below) |

### PCM_FILTERABLE_ROLES

Controls which message roles are sent through the Triton privacy filter. `assistant` messages are always excluded (they are produced with placeholders already in place or predate PCM). Add `system` if your system prompt contains personal data.

### PCM_VERBOSE_LOG_EVENTS

Emits additional payloads at `DEBUG` level (requires `PCM_LOG_LEVEL=DEBUG`). Use `*` to enable all.

> **IMPORTANT**: Debug logs may contain sensitive information, including raw messages with PII and Triton outputs. Only enable in a secure environment and never log to a persistent or shared location.

| Event | Contains PII | Description |
|---|---|---|
| `request_body` | yes | Incoming request + `X-Session-ID` header |
| `redaction_result` | yes | Full Triton output including original span text |
| `llm_payload` | no | Redacted payload forwarded to the LLM |
| `llm_raw_response` | no | Raw LLM response before de-anonymisation |
| `session_state` | yes | Full session after save (raw messages + privacy state) |
| `response_body` | yes | Final de-anonymised response returned to the client |

## Source layout

| File | Responsibility |
|---|---|
| `app/main.py` | FastAPI app factory, routes, lifespan |
| `app/config.py` | `Settings` (pydantic-settings, env vars) |
| `app/models.py` | Pydantic models: request, response, session state |
| `app/privacy_manager.py` | Core request handler, placeholder indexing, de-anonymisation |
| `app/session_store.py` | SQLite read/write via aiosqlite |
| `app/triton_client.py` | HTTP client for the Triton inference endpoint |
| `app/_logging.py` | Structured logging configuration (structlog) |
