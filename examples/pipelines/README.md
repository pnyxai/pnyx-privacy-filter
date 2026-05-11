# PCM Privacy-Filtered Pipeline

An [Open WebUI Pipelines](https://github.com/open-webui/pipelines) **manifold** that proxies
chat requests to the PrivateChatManager (PCM) backend and forwards the Open WebUI
`chat_id` as the `X-Session-ID` HTTP header so PCM can maintain per-conversation
privacy state (placeholder map, de-anonymisation) across turns.

---

## How it works

### The two-request problem

Every time a user sends a message, Open WebUI makes **two separate, independent
HTTP requests** to the Pipelines server:

| # | Endpoint | Pipelines calls | Body contains |
|---|----------|-----------------|---------------|
| 1 | `POST /filter/inlet/{id}` | `pipeline.inlet(full_owu_body)` | Full Open WebUI body including `metadata.chat_id` |
| 2 | `POST /v1/chat/completions` | `pipeline.pipe(...)` | Clean OpenAI-format body — **`chat_id` has been stripped** |

The Pipelines server parses the second request through `OpenAIChatCompletionForm`
before calling `pipe()`, which only knows about standard OpenAI fields.
`chat_id` is an Open WebUI internal field and is never included in the forwarded body.

### Why `inlet` is necessary

`inlet` is the **only hook** that ever receives the full Open WebUI body with
`body["metadata"]["chat_id"]`. There is no other point in the pipeline lifecycle
where `chat_id` is available to `pipe()`.

> **Note:** `chat_id` lives at `body["metadata"]["chat_id"]`, **not** at the
> top-level `body["chat_id"]`. The top-level key does not exist.

### Why the stash mechanism is necessary

Because `inlet` and `pipe` are called in separate HTTP requests, no argument
is shared between them. The only available bridge is **instance state** on the
`Pipeline` object (`self._pending_chat_ids`).

The flow is:

```
Open WebUI
  │
  ├─► POST /filter/inlet  ─► inlet()  reads body["metadata"]["chat_id"]
  │                                   stashes it in self._pending_chat_ids[key]
  │
  └─► POST /chat/completions ─► pipe()  pops self._pending_chat_ids[key]
                                        injects it as X-Session-ID header
                                        forwards request to PCM
```

### Why `_request_key` is necessary

`_request_key(messages)` generates a **collision-resistant correlation key** that
is identical for the same logical request in both `inlet` and `pipe`:

```python
payload = json.dumps(messages, sort_keys=True, ensure_ascii=False)
return hashlib.sha256(payload.encode()).hexdigest()
```

The full messages list is the only piece of information that is present in **both**
the `inlet` body and the `pipe` arguments, making it a reliable join key.
SHA-256 over the entire serialised list makes cross-user collision
cryptographically negligible and handles multimodal content, long messages, and
duplicate prefixes correctly.

---

## Configuration

Valves are configurable via **Admin Panel → Pipelines** or environment variables:

| Valve | Env var | Default | Description |
|-------|---------|---------|-------------|
| `PCM_URL` | `PCM_URL` | `http://pnyx-pcm:8080` | Base URL of the PCM service |
| `PCM_API_KEY` | `PCM_API_KEY` | _(empty)_ | Bearer token for PCM (optional) |
| `PCM_LLM_MODEL_NAME` | `PCM_LLM_MODEL_NAME` | `meta-llama/Llama-3-8B-Instruct` | Model name forwarded to PCM |

---

## Limitations

- PCM does not support streaming; `stream` is forced to `false`.
- `_pending_chat_ids` is in-memory and not shared across multiple Pipelines
  replicas. Single-instance deployments are assumed.
- The correlation key can theoretically collide if two concurrent requests share
  the exact same message count and last user message. In practice this is
  extremely unlikely in single-user or low-concurrency scenarios.
