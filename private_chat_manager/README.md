# PrivateChatManager (PCM)

OpenAI-compatible privacy mid-layer that sits between a client and any downstream LLM. PII is redacted before the LLM sees the request and de-anonymised before the response reaches the client. State is maintained per conversation across multiple turns. Both buffered and streaming responses are supported; streaming de-anonymisation happens in real time, token by token.

## Request / response flow

```
Client
  │  POST /v1/chat/completions
  │  model: <client-chosen>                     (authoritative)
  │  X-Session-ID / x-opencode-session / …      (any x…session… header, optional)
  ▼
PCM
  ├─ 1. Resolve the session from the message history (the client header is a hint)
  ├─ 2. Load session from SQLite (or create empty)
  ├─ 3. Identify new messages (everything past the stored history cursor)
  ├─ 4. For each new message (default-deny: every role and text field):
  │       → Triton: plain text in, RedactionResult out
  │       → apply_placeholder_indexing: base placeholders → <LABEL_N>
  │       → store redacted copy in hidden_messages
  ├─ 5. Forward hidden_messages to the downstream LLM, using the client's
  │       model and the client's headers (minus hop-by-hop/technical ones);
  │       re-emit the session id under the endpoint's session header
  ├─ 6. De-anonymise the LLM response (replace <LABEL_N> → original text)
  │       • stream=false: buffer the full response, then substitute
  │       • stream=true:  substitute each SSE delta in real time
  ├─ 7. Persist raw_messages + hidden_messages + PrivacyFilterState to SQLite
  └─ 8. Return de-anonymised response + X-Session-ID header to the client
```

Client-supplied history is treated as **untrusted**: every message role and
every known text field is redacted unless explicitly passed through (see
[Filtering scope](#filtering-scope-default-deny)). This closes the leak where a
client replays a prior conversation (after the PCM session expired or was reset)
and the de-anonymised `assistant`/`tool` turns would otherwise reach the LLM
verbatim.

## Session state

Each session stores three independent things in SQLite, plus its identity
columns (`session_id`, `client_x_session_header`, `endpoint_x_session_header`,
`user_hash` — see [Conversation identity](#conversation-identity)):

| Field | Contains | Sent to LLM |
|---|---|---|
| `raw_messages` | Original messages with real PII | Never |
| `hidden_messages` | Redacted messages with `<LABEL_N>` placeholders | Yes |
| `privacy_state` | Placeholder map, per-type counters, redaction audit log | Never |

The **cursor** mechanism allows clients to follow standard OpenAI conventions
(send the full history every turn). PCM aligns the client's replayed history
against `raw_messages` (the exact messages the client sent/received) and
processes only what follows the shared prefix, so previously processed turns are
never re-sent to Triton. If the client's history diverges — it drops, edits or
reorders a message, **or** renders an assistant/tool turn differently — PCM
**forks a new session** seeded with the shared prefix and re-redacts the tail;
the parent is left intact (see
[Cursor alignment, undo and branching](#cursor-alignment-undo-and-branching)).
An assistant turn with no `content` and no `tool_calls` (e.g. a reasoning-only
reply) is not persisted; the user turn is, so continuity is preserved.

### Conversation identity

Each session stores two session ids plus a conversation hash:

| Field | Meaning |
| --- | --- |
| `client_x_session_header` | The id in the client↔PCM namespace — what the client sent, or the `auto-<hash>` id PCM assigned and returned in `X-Session-ID`. |
| `endpoint_x_session_header` | The id in the PCM↔endpoint namespace — sent upstream under the endpoint's session header (e.g. `x-opencode-session`). PCM owns this value and overrides any client-supplied one. |
| `user_hash` | Merkle root over the *answered* user messages, updated after each turn. Each turn's root is also appended to the `session_hashes` table (see [Cursor alignment](#cursor-alignment-undo-and-branching)) so an undo can be matched back to its session. |

A conversation is identified by **message content**, not by the client id:

* On every request PCM hashes the user messages *excluding the last* (the
  not-yet-answered turn) and uses that hash, together with the replayed messages
  and the historical roots, to find the owning session.
* A client `x…session…` header is only a hint: it can select a session, but the
  messages decide. A header that points at a session whose history disagrees is
  ignored.
* When no client header is sent (e.g. Hermes on a custom endpoint), the hash and
  the historical roots alone recover the conversation, so turns still accumulate
  history.
* When several sessions share the same history (identical prefixes, such as an
  agent's title generator and its main chat), the one that shares the **most
  messages** wins; a true tie is broken at random.

The endpoint session id is derived by PCM from the **effective per-request
endpoint policy**: a well-formed client value is reused, otherwise a fresh one
is generated (and a malformed client value is replaced). This is evaluated on
every request, so an existing session routed to a different endpoint by an
`X-PCM-LLM-URL` override still emits the endpoint's required session header. The
client-facing id is echoed in the `X-Session-ID` response header. Requests for
one conversation are serialised on the **answered-message hash** (the header is
only a hint), so headerless, stale-header and rotated-header representations of
the same conversation share one lock.

> Caveat: two unrelated conversations whose replayed history is byte-identical
> (the same longest shared prefix) collide until they diverge; one is picked at
> random. A colliding side-channel with a *shorter* shared prefix — e.g. a title
> generator whose single user message is the opening — no longer shadows the
> real conversation.

### Cursor alignment, undo and branching

The cursor is not assumed from the stored length. PCM computes the **longest
common prefix (LCP)** between the client's replayed `messages` and the stored
`raw_messages`. `_same_message` compares role, content and **every** semantic
tool-call field (`tool_calls`, `tool_call_id`, `name`, and the legacy
`function_call` / `function_calls`), so an edit to a call's arguments or function
name — not only its id — is detected as divergence and re-redacted. The
`reasoning`/`refusal` side channels are intentionally ignored (the client may or
may not echo them). The LCP length is the cursor; everything after it is new.

The governing rule:

- **Continuation (append-only)** — the client's replayed history contains the
  whole stored history as its prefix (`cursor == stored length`); PCM keeps the
  same session and only the new message(s) are processed. This also covers a
  replayed reasoning-only or injected assistant turn (it just lands in the new
  region and is redacted).
- **Divergence (fork)** — the shared prefix is shorter than the stored history:
  the client dropped/edited/reordered a message, **or** the assistant/tool
  rendering differs (e.g. a client that doesn't replay assistant turns). PCM
  **branches**: it copies the shared prefix into a **new session** (`session_id`)
  and continues there, while the **parent is left intact**. The new tail is
  treated as *fresh* (strict redaction). Because the parent survives, a later redo
  (replaying the original history) resolves back to it.
- **Undo (strict-prefix rewind)** — `/undo` in Hermes and opencode drops the
  trailing assistant/tool reply and re-submits the same user turn, so the body
  is a **strict prefix** of the stored history. One rule covers it: the cursor is
  `min(lcp, turn_start)`, where `turn_start` is the last message that is not an
  `assistant` (the turn awaiting a reply). PCM therefore forks at that turn,
  reusing the shared prefix's placeholder state and re-redacting **only the
  re-asked turn** — instead of starting a new conversation and re-running Triton
  over everything. This holds for any cut depth (`/undo N` in Hermes, repeated
  `/undo` in opencode) because the cut's answered-user prefix is a recorded
  historical root. A strict prefix with no answered-user prefix (a title
  generator's shortened history) is **not** treated as a rewind, so it cannot
  hijack the session.

There is **no in-place rewrite**: every divergence forks, so a session's stored
history never changes after the fact (append-only) and the conversation forms a
git-like tree. Lineage is recorded in `parent_session_id` / `root_session_id` /
`origin_message_count`.

Session lookup gathers candidates from three sources — the client session header,
the current `user_hash`, and the per-turn historical prefix roots
(`session_hashes`) — and picks the one that shares the **longest run of raw
messages** with the request; an exact `user_hash` match only breaks ties (ties on
byte-identical prefixes are broken at random). Scoring by the shared messages
keeps a side-channel session — for example a title generator whose single user
message collides with the conversation's opening, so its `user_hash` equals the
conversation's first answered-user prefix — from shadowing the real conversation,
which shares the actual messages.

- **With a session header**, that header is only a hint: it never overrides the
  messages. A strict-prefix undo is honoured only when the incoming history has a
  non-empty answered-user prefix, so a title-generator prefix cannot hijack it.
- **Without a session header**, the historical roots still let a rewind map back
  to its conversation, so only the changed tail is re-redacted. A history with no
  shared prefix starts a new conversation.

Forked sessions inherit the parent's client-facing id, so a fixed-header client
keeps resolving the same family (the shared prefix disambiguates which branch).
The branch's `privacy_state` is **pruned to the retained prefix**: only
placeholder-map entries whose token actually appears in the inherited hidden
messages are kept (counters are rebuilt from those tokens), and audit results
whose span text is not present in the inherited raw messages are dropped. This
ensures a stale token from a discarded turn can never de-anonymise discarded PII
on the new branch.

**Lineage (sessions are append-only).** Every divergence forks; no session row is
ever rewritten. Each session records its lineage so the conversation tree is
queryable:

| Column | Meaning |
| --- | --- |
| `parent_session_id` | The session this one was forked from (`NULL` for a root). Indexed. |
| `root_session_id` | The shared root of the whole conversation tree (a root session points at itself). Indexed. |
| `origin_message_count` | How many messages were inherited from the parent at the fork point. |

`session_hashes` also stores the `message_count` at each recorded root (the turn
boundary), so a matched historical root maps directly to a cursor. This is
groundwork for future **forking** (list branches of a conversation, fork at
message N, rebuild the tree) — those are query/API work over these columns.

> Future stretch (not implemented): giving each stored message a stable identity
> (message id / content hash) would enable message-level **diff/merge** between
> branches. See the `_same_message` note in `app/privacy_manager.py`.

LCP cost is a few string comparisons over the history (sub-millisecond) versus a
Triton call (~hundreds of ms), so it is not a performance concern. The historical
root lookup is a single indexed query (`session_hashes.user_hash`); expired
sessions' history rows are removed by the TTL sweep.

### Session lifetime (TTL)

Set `PCM_SESSION_TTL` to expire conversations after a period of inactivity
(measured from `updated_at`). The value uses `s`/`m`/`h`/`d`/`w` units (seconds,
minutes, hours, days, weeks) with an integer or float number — e.g. `30s`,
`360m`, `6h`, `1.5d`, `2w`. Empty or `0` disables expiry (the default).

When enabled, PCM purges expired sessions once at startup and then periodically
in the background (every `PCM_SESSION_TTL_SWEEP`, default `10m`). Resolution
also treats an expired session as absent, so a request arriving between sweeps
starts a fresh conversation instead of reviving the old one.

Resolution **touches** the session (bumps `updated_at`) before the potentially
long redaction/upstream work, and the sweeper deletes a row only once it is idle
for `PCM_SESSION_TTL + PCM_SESSION_TTL_GRACE` (`PCM_SESSION_TTL_GRACE` defaults
to `120s`). Together these prevent the sweeper from deleting an in-flight
request's session — and its `session_hashes` history — mid-request.


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
| `GET` | `/v1/sessions/{session_id}` | Inspect raw messages, hidden messages, privacy state, identity and lineage |
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

In both modes the client-facing session ID is returned in the `X-Session-ID`
**response header** (and, when the client used a different session header name
such as `x-session-affinity`, echoed under that name too). Pass it back as
`X-Session-ID` (or any `x…session…` header) on the next request to continue the
same session; the message history is the source of truth, so the id is only a
hint. If the client sends a different session header — e.g.
`x-session-affinity`, `x-opencode-session` or `X-Hermes-Session-Id` — its value
is used instead, which lets agents that manage their own session ids resume
conversations transparently. The value PCM sends *upstream* under
`PCM_LLM_SESSION_HEADER` is derived from the endpoint and always replaces any
client-supplied value.

## Configuration

All settings are read from environment variables with the `PCM_` prefix.

| Variable | Required | Default | Description |
|---|---|---|---|
| `PCM_LLM_URL` | yes | — | Base URL of the downstream LLM (e.g. `http://vllm:8000`). PCM appends `/v1/…`; a trailing `/v1` is stripped automatically |
| `PCM_LLM_MODEL_NAME` | no | `""` | Fallback model used only when the client omits `model`. The client-supplied model always wins |
| `PCM_LLM_API_KEY` | no | `""` | Bearer token for the LLM endpoint |
| `PCM_LLM_SESSION_HEADER` | no | auto | Upstream session header. Resolved by the endpoint registry (`app/endpoints.py`), whose built-in rule maps an `opencode.ai` host to `x-opencode-session`; set explicitly to override for other endpoints (e.g. `X-Hermes-Session-Id`). PCM derives the endpoint session id and emits it under this name, overriding any client value |
| `PCM_LLM_URL_ALLOWLIST` | no | `""` | Comma-separated base URLs a request may select for a single call via the `X-PCM-LLM-URL` header (testing aid; empty disables the override). The header is stripped before forwarding upstream |
| `PCM_TRITON_URL` | no | `localhost:8000` | Host and port of the Triton server |
| `PCM_TRITON_MODEL_NAME` | no | `ensemble_model` | Triton model name |
| `PCM_TRITON_MAX_CHARS` | no | `8000` | Max characters per Triton call; longer messages are split at natural boundaries and redacted chunk by chunk |
| `PCM_DB_PATH` | no | `./sessions.db` | SQLite database path inside the container |
| `PCM_SESSION_TTL` | no | `0` (disabled) | Session time-to-live, counted from the last activity (`updated_at`). Duration with `s`/`m`/`h`/`d`/`w` units, integer or float (e.g. `30s`, `360m`, `6h`, `1.5d`, `2w`). Empty/`0` disables expiry |
| `PCM_SESSION_TTL_SWEEP` | no | `10m` | How often the background sweeper purges expired sessions (same duration syntax). Only used when `PCM_SESSION_TTL` is enabled |
| `PCM_SESSION_TTL_GRACE` | no | `120s` | Extra margin added to the TTL before physical deletion (same duration syntax), so an in-flight redaction/upstream/stream cannot have its session and history swept mid-request |
| `PCM_HOST` | no | `0.0.0.0` | Uvicorn bind address |
| `PCM_PORT` | no | `8080` | Uvicorn bind port |
| `PCM_LOG_LEVEL` | no | `INFO` | Log verbosity: `DEBUG` \| `INFO` \| `WARNING` \| `ERROR` |
| `PCM_PASSTHROUGH_ROLES` | no | `""` | Roles to skip redaction on **resumed** sessions. Default empty → redact every role |
| `PCM_PASSTHROUGH_ROLES_FRESH` | no | `""` | Roles to skip redaction on **fresh** conversations (`cursor == 0`). Does **not** inherit the resumed list |
| `PCM_PASSTHROUGH_FIELDS` | no | `""` | Message text fields to skip redaction on **resumed** sessions |
| `PCM_PASSTHROUGH_FIELDS_FRESH` | no | `""` | Message text fields to skip redaction on **fresh** conversations |
| `PCM_PLACEHOLDER_LABELS` | no | the 8 privacy-filter labels | Labels the streaming de-anonymiser recognises |
| `PCM_SYSTEM_PROMPT_PII_INSTRUCTION` | no | `""` | Text appended to the system prompt explaining how to handle placeholder tags |
| `PCM_VERBOSE_LOG_EVENTS` | no | `""` | Comma-separated debug event names (see below) |

### Selecting a downstream per request

`PCM_LLM_URL` is fixed at startup. To exercise several engines without
restarting PCM, list their base URLs in `PCM_LLM_URL_ALLOWLIST` and send the
desired one per request in the `X-PCM-LLM-URL` header (chat completions and the
pass-through proxy honour it). The URL is normalised and must be allowlisted,
otherwise the request is rejected with `403`; when the allowlist is empty the
header is ignored. PCM strips `X-PCM-LLM-URL` before forwarding upstream.

```
PCM_LLM_URL_ALLOWLIST=http://localhost:9087,http://localhost:10080
# curl -H 'X-PCM-LLM-URL: http://localhost:9087' ... /v1/chat/completions
```

The endpoint session-header policy (see [Conversation identity](#conversation-identity))
is derived from the effective URL, so a request routed to `opencode.ai` still
gets `x-opencode-session`.

### Filtering scope (default-deny)

Client-supplied history is untrusted, so PCM redacts **every** message role and
**every** known text field unless it is explicitly passed through. (The former
`PCM_FILTERABLE_ROLES` allowlist is retired.)

| Setting | Applies to | Default |
|---|---|---|
| `PCM_PASSTHROUGH_ROLES` | resumed sessions | empty (redact all) |
| `PCM_PASSTHROUGH_ROLES_FRESH` | fresh conversations (`cursor == 0`) | empty (redact all) |
| `PCM_PASSTHROUGH_FIELDS` | resumed sessions | empty (redact all) |
| `PCM_PASSTHROUGH_FIELDS_FRESH` | fresh conversations | empty (redact all) |

The `*_FRESH` sets are **independent** — an empty fresh list does **not** inherit
the resumed list — so loosening the resumed policy (e.g. skipping `system` for
speed) can never leak PII from a replayed conversation.

**Safeguard (edge-case only):** PCM's own assistant turns are stored in
`hidden_messages` and covered by the cursor, so they are *never* re-processed —
a resumed turn normally only carries a new `user`/`tool` message. The only time
an `assistant`/`system` message reaches the redaction loop is when the client
supplies one PCM did not store: a replayed reasoning-only turn, an injected
prefill, or a branch/undo. Those are untrusted, so the pass-through is
**ignored** for roles `user`, `assistant`, `tool`, `function`, `developer` and
fields `content`, `reasoning`, `reasoning_content`, `name` and the tool/function
call arguments. Only the `system` role and the `refusal` field can still be
passed through. A warning is logged when a configured entry is ignored. This is
not a per-turn cost: the check only runs for messages in the new region.

The text fields covered are the registry in
[`app/message_fields.py`](app/message_fields.py): `content` (including each
`type=="text"` part of a multimodal message, redacted in place), `reasoning`,
`reasoning_content`, `refusal`, `name`, `tool_calls.arguments`,
`function_calls.arguments` and `function_call.arguments`.

Why this matters: when a session is lost (TTL expiry, DB reset, or a
user-message-hash miss) the client replays the whole conversation, including the
assistant/tool turns PCM had de-anonymised. Those turns must be re-redacted —
they were never sent to the upstream model in cleartext.

> Residual risk: keys PCM does not know about inside a message dict are
> forwarded untouched, and non-text multimodal parts (images/audio) cannot be
> redacted by the text-only filter.

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
| `llm_payload` | no | Redacted payload + forwarded headers sent to the LLM (auth/cookies redacted) |
| `llm_raw_response` | no | Raw LLM response before de-anonymisation (buffered body, or the assembled stream with placeholders) |
| `session_state` | yes | Full session after save (raw messages + privacy state) |
| `response_body` | yes | Final de-anonymised response returned to the client (buffered body, or the assembled stream including `reasoning`) |

For `stream: true`, `llm_raw_response` and `response_body` are logged **once**
when the stream finishes, using the fully assembled message (after any buffered
partial tag is flushed), per choice. `response_body` includes the reasoning
side channel, which is logged but never persisted to the session.

### Request timing

Every request emits one `INFO` event `request timing` with the end-to-end
duration and the intermediate phase times (all in milliseconds). The phases are
sequential slices of the request:

| Field | What it covers |
|---|---|
| `resolve_ms` | Everything up to and including picking the session: header parsing, the per-request URL/lock-key computation, **time waiting for the per-conversation lock**, opening the DB connection, and `resolve_session` (DB lookups, user-message hash match, and any branch/fork). No Triton, no LLM. |
| `redact_ms` | **All Triton calls for this request, end-to-end**, plus the surrounding work: sending each new message/field through the privacy filter (one or more inference calls per field, chunked for long inputs), `apply_placeholder_indexing`, and storing the redacted copies in `hidden_messages`. Empty/whitespace fields skip Triton and cost ~0. |
| `upstream_ms` | The downstream LLM call: the buffered `POST`, or — for `stream: true` — the whole SSE stream (reading it and forwarding chunks to the client, so it also includes client transfer time). Also covers payload/header assembly between the two phases. |
| `finalize_ms` | Post-LLM work: de-anonymising the response for the client, and persisting the session (`save_session`). For streaming this is the `_finish` step (flush buffered partial tags, optional debug logging, persist). |
| `total_ms` | End-to-end, from when PCM received the request to when the response completed. |

So the **Triton cost is `redact_ms`** (`resolve_ms` and the others do not call
Triton). For a finer breakdown, the `message redacted` events logged during
processing carry a per-field `elapsed_ms` (one per Triton call).

Every request emits the event **exactly once**, including failures. A request
that exits before completing the pipeline — empty body, rejected `X-PCM-LLM-URL`
override, Triton error, upstream HTTP/transport error, response-parse or
persistence failure — is still accounted for by a fallback emit with an
`error=True` field. A guard on the timer makes the success emit and the fallback
mutually exclusive, so a request never logs two timing events.

Example:

```
event='request timing' session_id=… model='deepseek-v4.1-flash' stream=False \
  resolve_ms=1.4 redact_ms=781.7 upstream_ms=1855.3 finalize_ms=23.2 total_ms=2661.7
```

For streaming, the timings are emitted when the stream finishes (in the same
finalize step that persists the session). The proxy path emits the same event
with `method`/`path` instead of `session_id`/`model`.

## Source layout

| File | Responsibility |
|---|---|
| `app/main.py` | FastAPI app factory, routes, lifespan |
| `app/config.py` | `Settings` (pydantic-settings, env vars) |
| `app/models.py` | Pydantic models: request, response, session state |
| `app/privacy_manager.py` | Session preparation, placeholder indexing, buffered + streaming handlers |
| `app/message_fields.py` | Registry of the text-bearing fields inside a chat message (default-deny) |
| `app/endpoints.py` | Endpoint registry: `PCM_LLM_URL` → session-header policy (matchers, validate/generate) |
| `app/headers.py` | Generic HTTP header helpers (session-header resolution, forwarding, redaction) |
| `app/streaming.py` | Real-time placeholder detection/replacement state machine (see its module docstring) |
| `app/session_store.py` | SQLite read/write via aiosqlite (schema create + in-place column migration, TTL touch/purge) |
| `app/session_lock.py` | Per-conversation async lock |
| `app/triton_client.py` | HTTP client for the Triton inference endpoint |
| `app/timing.py` | Per-request phase timer (`RequestTimings`, idempotent once-only `log_timing`) |
| `app/_logging.py` | Structured logging configuration (structlog) |
| `tests/` | Unit + endpoint tests (session resolution, cursor/branching/undo, fork state pruning, schema migration, redaction, streaming, timing incl. failure paths, locks, endpoints, …) |

### Running the tests

```bash
cd private_chat_manager
pip install -e ".[test]"
pytest
```

The streaming unit tests include an exhaustive property-based check that
incremental de-anonymisation matches the buffered implementation for every
possible chunk partition.
