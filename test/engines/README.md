# Testing downstream engines with PCM

Live integration test that validates the OpenAI-compatible engines behind PCM
(llama.cpp, vLLM, …) and the PCM privacy pipeline against the configured
downstream.

The endpoints are **operator-defined** in `engines.yaml` (gitignored). Copy the
template once:

```bash
cp test/engines/engines.example.yaml test/engines/engines.yaml
# then edit pcm.url and the engines
```

Schema:

```yaml
pcm:
  url: http://localhost:8080
engines:
  llama_cpp:
    base_url: http://localhost:10080
    pcm_url: http://host.docker.internal:10080  # container-facing (optional)
    # model: llama.cpp       # optional; omit to auto-detect from /v1/models
  vllm:
    base_url: http://localhost:9087
    pcm_url: http://host.docker.internal:9087
    # model: vllm  # optional; must match the id from /v1/models
```

`model` is optional — when omitted the first id from the engine's `/v1/models`
is used, so the template works for any deployment. Pin it only to select a
specific model; a value that the engine does not advertise fails the
"model is advertised" check. `pcm_url` is optional — it is the URL PCM (inside
its container) uses to reach the engine, sent in `X-PCM-LLM-URL` with
`--via-override`; it defaults to `base_url`. Set it when PCM runs in Docker and
the engine is on the host.

## Prerequisites

1. The engine(s) reachable at the `base_url`s in `engines.yaml`.
2. A running PCM exposing `pcm.url` (e.g. `docker compose up -d pnyx-pf-triton pnyx-pcm`).
3. `httpx` and `pyyaml` (both present in `private_chat_manager/.venv`).

## What it checks

1. **Per engine (direct):** `/v1/models` is reachable and advertises the model;
   a buffered completion returns a choice; a streaming completion is
   `text/event-stream` and ends with `[DONE]`.
2. **PCM health:** `/health` is ok and `/health/triton` is ready.
3. **PCM privacy pipeline** against PCM's configured downstream (detected via the
   proxied `/v1/models`), for buffered and streaming:
   - placeholders are created and the hidden history contains `<PRIVATE_…>`;
   - the hidden history does **not** contain any of the PII strings;
   - the raw history keeps the original message;
   - no placeholder leaks back to the client.
4. **PCM conversation flow** against the same downstream:
   - buffered multi-turn continuity (same session, placeholder map accumulates);
   - a streaming turn on the same session (no placeholder leak);
   - headerless continuation recovered by the user-message hash;
   - a custom session header (`X-Hermes-Session-Id`) stored and echoed back;
   - **branching**: rewriting an answered user message forks a new session with
     `parent_session_id`/`root_session_id`/`origin_message_count` set, while the
     parent stays intact;
   - **redo**: replaying the original history resolves back to the parent;
   - structural divergence (assistant turns omitted) also forks;
   - a headerless rewind matched via a historical user-hash root stays in the
     same conversation tree;
   - a **strict-prefix undo** (`/undo`: re-submitting the last turn without its
     reply) forks and reuses the prefix — run only when the engine persisted the
     reply, so reasoning-only engines skip it.

## Usage

Run from the repository root with the PCM venv:

```bash
./private_chat_manager/.venv/bin/python ./test/engines/test.py
```

Useful flags:

```bash
# only one engine
./private_chat_manager/.venv/bin/python ./test/engines/test.py --engine vllm

# skip the PCM round-trip, only probe the engines
./private_chat_manager/.venv/bin/python ./test/engines/test.py --skip-pcm

# skip the multi-turn/header/branching conversation-flow stage
./private_chat_manager/.venv/bin/python ./test/engines/test.py --skip-conversation

# custom config location
./private_chat_manager/.venv/bin/python ./test/engines/test.py --config /path/to/engines.yaml
```

The script exits non-zero if any check fails (or `2` when `engines.yaml` is
missing).

## Testing PCM against a specific local engine

By default PCM talks to whatever `PCM_LLM_URL` points at. To exercise a local
engine end-to-end and assert the match:

```bash
# examples/.env: PCM_LLM_URL=http://localhost:9087
cd examples
docker compose up -d --force-recreate --no-deps pnyx-pcm
cd ..
./private_chat_manager/.venv/bin/python ./test/engines/test.py --require-pcm-engine vllm
```

`--require-pcm-engine NAME` fails unless PCM's downstream model matches that
engine. Repeat with `PCM_LLM_URL=http://localhost:10080` for llama.cpp.

## Testing several engines without restarting PCM

Point PCM at one engine (or anything), then list the engine URLs in
`PCM_LLM_URL_ALLOWLIST` once and restart PCM a single time. The allowlist must
contain the **same URLs the runner sends** (`pcm_url`), because PCM matches the
`X-PCM-LLM-URL` value against it exactly. When PCM runs in Docker and the
engines are on the host, that is `host.docker.internal` (not `localhost`, which
inside the PCM container is the container itself):

```bash
# examples/.env
# PCM_LLM_URL_ALLOWLIST=http://host.docker.internal:9087,http://host.docker.internal:10080
cd examples && docker compose up -d --force-recreate --no-deps pnyx-pcm && cd ..
```

Now the runner can route each request to a chosen engine via the
`X-PCM-LLM-URL` header, so both engines are exercised through PCM with no
further restarts:

```bash
./private_chat_manager/.venv/bin/python ./test/engines/test.py --via-override
./private_chat_manager/.venv/bin/python ./test/engines/test.py --via-override --engine vllm
```

If an engine URL is not in the allowlist the PCM checks report a `403` and tell
you to add it. When `PCM_LLM_URL_ALLOWLIST` is empty the override is ignored and
PCM uses its configured `PCM_LLM_URL`.
