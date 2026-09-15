# PrivateChatManager (PCM)

OpenAI-compatible privacy mid-layer that sits between a client and any downstream LLM. PII is redacted before the LLM sees the request and de-anonymised before the response reaches the client. State is maintained per conversation across multiple turns. Both buffered and streaming responses are supported; streaming de-anonymisation happens in real time, token by token.

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
  │       • stream=false: buffer the full response, then substitute
  │       • stream=true:  substitute each SSE delta in real time
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

## Real-time de-anonymisation (streaming)

With `stream: true` the LLM produces one small token per SSE chunk, so a single
placeholder — or the true text that replaces it — can be split across chunk
boundaries:

```
chunk 1: "You are <PRIV"
chunk 2: "ATE_PERSON_"
chunk 3: "1>."
```

PCM may not forward `<PRIV` before it knows whether a placeholder is coming,
otherwise the placeholder would leak; and it may not buffer the whole stream.
The algorithm in [`app/streaming.py`](app/streaming.py) solves this with a
small state machine built on two pre-computed regular expressions derived from
the finite label set (`PCM_PLACEHOLDER_LABELS`):

* **`_TAG_RE`** matches a *complete* placeholder `<LABEL_N>` (or the unindexed
  `<LABEL>`), with a greedy index so `<PRIVATE_PERSON_10>` is matched in full
  rather than as `<PRIVATE_PERSON_1>` + `0`.
* **`_PREFIX_RE`** matches any string that is a *valid prefix* of a
  placeholder, including the bare `<` opener.

For every incoming delta the filter:

1. Returns the chunk untouched on a fast path when nothing is buffered and the
   chunk contains no `<` (the overwhelmingly common case).
2. Emits everything up to the first `<`.
3. At the `<`, tries `_TAG_RE`:
   * **match** → substitute the true value from the session `placeholder_map`
     (unknown tags pass through unchanged, matching the buffered behaviour);
   * **no match but a valid prefix** → buffer the tail and wait for the next
     chunk;
   * **no match and not a valid prefix** → the `<` is literal text.
4. Drains any residual buffer at end of stream (an unterminated `<PRIV` is not
   a placeholder, so it is emitted verbatim).

A placeholder is only ever *delayed*, never lost or corrupted. If the LLM
opens a tag but diverges before closing it — e.g. the accumulated
`<PRIVATE_PERSON_12 is a good person` — the held prefix stops matching
`_PREFIX_RE` as soon as the space arrives and the whole opening is replayed to
the client unchanged. If the stream simply ends mid-tag, `flush()` replays the
remaining prefix verbatim.

This yields an exact equivalence guarantee: for **any** partitioning of a text
into chunks, the concatenation of the emitted deltas equals the buffered
`deanonymize_text(text, placeholder_map)`. Substituted text is never rescanned.

De-anonymisation is applied independently to each streamed field:

* `delta.content` (per choice, so `n > 1` works),
* `delta.reasoning` / `delta.reasoning_content`, and
* `delta.tool_calls[i].function.arguments` (assembled across chunks).

Non-text fields (`role`, `logprobs`, `token_ids`, `finish_reason`, `usage`) are
forwarded unchanged, and the terminal `data: [DONE]` is always re-emitted.
The completed assistant turn is persisted to the session when the stream ends
(normally, on upstream error, or on client disconnect), keeping
`hidden_messages` (placeholder-bearing) and `raw_messages` (de-anonymised) in
sync for the next turn.

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
| `stream` | bool | `false` | Stream SSE deltas, de-anonymised in real time |
| `stream_options` | object | `null` | Forwarded to the LLM (e.g. `{"include_usage": true}`) |

### Response

* `stream=false`: a standard OpenAI `ChatCompletion` JSON object.
* `stream=true`: a `text/event-stream` SSE response of
  `chat.completion.chunk` frames terminated by `data: [DONE]`. Delta
  `content`/`reasoning`/tool-call `arguments` are already de-anonymised; the
  raw upstream bytes for a placeholder are never sent to the client. Delimiter
  text around a partial tag may be delayed by at most one placeholder length.

In both modes the resolved session ID is returned in the `X-Session-ID`
**response header**. Pass it back as `X-Session-ID` on the next request to
continue the same session.

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
| `PCM_PLACEHOLDER_LABELS` | no | the 8 privacy-filter labels | Labels the streaming de-anonymiser recognises |
| `PCM_SYSTEM_PROMPT_PII_INSTRUCTION` | no | `""` | Text appended to the system prompt explaining how to handle placeholder tags |
| `PCM_VERBOSE_LOG_EVENTS` | no | `""` | Comma-separated debug event names (see below) |

### PCM_FILTERABLE_ROLES

Controls which message roles are sent through the Triton privacy filter. `assistant` messages are always excluded (they are produced with placeholders already in place or predate PCM). Add `system` if your system prompt contains personal data.

### PCM_PLACEHOLDER_LABELS

Comma-separated labels used by the streaming de-anonymiser to recognise
placeholder tags. They must match the labels emitted by the privacy-filter
after the `_label_placeholder` normalisation (uppercase, non-alphanumerics
replaced by `_`). The default is the current `openai/privacy-filter` v2
taxonomy:

```
ACCOUNT_NUMBER, PRIVATE_ADDRESS, PRIVATE_DATE, PRIVATE_EMAIL,
PRIVATE_PERSON, PRIVATE_PHONE, PRIVATE_URL, SECRET
```

Override this when the upstream model's taxonomy changes so PCM keeps matching
`<LABEL_N>` tokens. It only affects detection of *streams*; buffered responses
substitute whatever keys are in the session `placeholder_map`.

### PCM_SYSTEM_PROMPT_PII_INSTRUCTION

When non-empty, this text is appended (after a blank line) to the client's
**system prompt** before the request is forwarded to the LLM, so the model
knows how to treat placeholder tags. It is injected *after* redaction, so the
instruction itself never passes through Triton.

```
PCM_SYSTEM_PROMPT_PII_INSTRUCTION="# Privacy Tags:\nThe user has a privacy filter active.\n- <PRIVATE_PERSON_i>"
```

Behaviour:

- The instruction is appended to the **first** `system` message in the request.
- On the first turn it is stored in `hidden_messages` (the redacted history the
  LLM sees); `raw_messages` and the client never see it. Later turns reuse the
  stored copy, so it is not duplicated.
- If the client sends **no** system message, a synthetic system message
  containing only the instruction is prepended to the outgoing payload. It is
  not persisted (this keeps PCM's new-message cursor aligned).
- In a `.env` file, either use a quoted multi-line value or the escapes `\n`
  and `\t`, which PCM expands.

### PCM_VERBOSE_LOG_EVENTS

Emits additional payloads at `DEBUG` level (requires `PCM_LOG_LEVEL=DEBUG`). Use `*` to enable all.

> **IMPORTANT**: Debug logs may contain sensitive information, including raw messages with PII and Triton outputs. Only enable in a secure environment and never log to a persistent or shared location.

| Event | Contains PII | Description |
|---|---|---|
| `request_body` | yes | Incoming request + `X-Session-ID` header |
| `redaction_result` | yes | Full Triton output including original span text |
| `llm_payload` | no | Redacted payload forwarded to the LLM |
| `llm_raw_response` | no | Raw LLM response before de-anonymisation (buffered body, or the assembled stream with placeholders) |
| `session_state` | yes | Full session after save (raw messages + privacy state) |
| `response_body` | yes | Final de-anonymised response returned to the client (buffered body, or the assembled stream including `reasoning`) |

For `stream: true`, `llm_raw_response` and `response_body` are logged **once**
when the stream finishes, using the fully assembled message (after any buffered
partial tag is flushed), per choice. `response_body` includes the reasoning
side channel, which is logged but never persisted to the session.

## Source layout

| File | Responsibility |
|---|---|
| `app/main.py` | FastAPI app factory, routes, lifespan |
| `app/config.py` | `Settings` (pydantic-settings, env vars) |
| `app/models.py` | Pydantic models: request, response, session state |
| `app/privacy_manager.py` | Session preparation, placeholder indexing, buffered + streaming handlers |
| `app/streaming.py` | Real-time placeholder detection/replacement state machine (see its module docstring) |
| `app/session_store.py` | SQLite read/write via aiosqlite |
| `app/triton_client.py` | HTTP client for the Triton inference endpoint |
| `app/_logging.py` | Structured logging configuration (structlog) |
| `tests/` | Unit tests (`test_streaming.py`) and SSE endpoint tests (`test_streaming_endpoint.py`) |

### Running the tests

```bash
cd private_chat_manager
pip install -e ".[test]"
pytest
```

The streaming unit tests include an exhaustive property-based check that
incremental de-anonymisation matches the buffered implementation for every
possible chunk partition.
