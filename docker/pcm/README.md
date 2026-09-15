# PrivateChatManager — Docker

This folder contains the `Dockerfile` and `build.sh` to build and run the
**PrivateChatManager** (PCM) service.  PCM is a privacy-aware mid-layer that
sits in front of any OpenAI-compatible LLM endpoint, redacting PII via the
Triton privacy-filter before forwarding requests and de-anonymising responses
before returning them to the client.

## Prerequisites

- The **Triton privacy-filter** server must be running and reachable (see
  `docker/triton/`).
- A running **OpenAI-compatible LLM** endpoint (e.g. vLLM, OpenAI API).

## Build

The build context must be the **`pnyx/` parent directory** so that
`pnyx-privacy-filter/` is accessible to the `COPY` instructions.

Run from inside `pnyx-privacy-filter/docker/pcm/`:
```bash
bash build.sh
```
## Environment Variables

All variables use the `PCM_` prefix.

| Variable | Required | Default | Description |
|---|---|---|---|
| `PCM_LLM_URL` | **yes** | — | Base URL of the downstream LLM, e.g. `http://vllm:8000` |
| `PCM_LLM_MODEL_NAME` | **yes** | — | Model name forwarded in every LLM request, e.g. `meta-llama/Llama-3-8B-Instruct` |
| `PCM_LLM_API_KEY` | no | `""` | Bearer token for the LLM endpoint (leave empty if not required) |
| `PCM_TRITON_URL` | no | `localhost:8000` | Host and port of the Triton inference server |
| `PCM_TRITON_MODEL_NAME` | no | `ensemble_model` | Triton model name to call for privacy filtering |
| `PCM_DB_PATH` | no | `./sessions.db` | Path inside the container for the SQLite session database |
| `PCM_HOST` | no | `0.0.0.0` | Bind address for the uvicorn server |
| `PCM_PORT` | no | `8080` | Bind port for the uvicorn server |
| `PCM_LOG_LEVEL` | no | `INFO` | Log verbosity: `DEBUG` \| `INFO` \| `WARNING` \| `ERROR` |
| `PCM_FILTERABLE_ROLES` | no | `user,tool,function` | Comma-separated message roles sent through Triton |
| `PCM_VERBOSE_LOG_EVENTS` | no | `""` | Comma-separated debug event names (see the PCM README) |
| `PCM_PLACEHOLDER_LABELS` | no | the 8 privacy-filter labels | Comma-separated labels the streaming de-anonymiser recognises as placeholder tags |
| `PCM_SYSTEM_PROMPT_PII_INSTRUCTION` | no | `""` | Text appended to the system prompt explaining how to handle placeholder tags |

`PCM_PLACEHOLDER_LABELS` only affects `stream: true` responses. It defaults to
the `openai/privacy-filter` v2 taxonomy (`ACCOUNT_NUMBER`, `PRIVATE_ADDRESS`,
`PRIVATE_DATE`, `PRIVATE_EMAIL`, `PRIVATE_PERSON`, `PRIVATE_PHONE`,
`PRIVATE_URL`, `SECRET`); override it if the upstream model's labels change.

## Run

### Recommended: Docker Compose (Triton + PCM together)

See `examples/` at the repository root for a ready-to-use `docker-compose.yaml`:

```bash
cd examples/
cp sample.env .env   # fill in TRITON_MODELS_PATH, PCM_LLM_URL, PCM_LLM_MODEL_NAME …
docker compose up -d
```

### Standalone: Triton and LLM on the same Docker network

```bash
docker run -d \
  --name pnyx-pcm \
  --network pnyx-net-pf \
  -p 8080:8080 \
  -e PCM_LLM_URL="http://vllm:8000" \
  -e PCM_LLM_MODEL_NAME="meta-llama/Llama-3-8B-Instruct" \
  -e PCM_TRITON_URL="pnyx-pf-triton:8000" \
  pnyx-pcm:latest
```

### Standalone: with an API key and a persistent session database

```bash
docker run -d \
  --name pnyx-pcm \
  --network pnyx-net-pf \
  -p 8080:8080 \
  -v /path/to/data:/data \
  -e PCM_LLM_URL="http://vllm:8000" \
  -e PCM_LLM_MODEL_NAME="meta-llama/Llama-3-8B-Instruct" \
  -e PCM_LLM_API_KEY="sk-..." \
  -e PCM_TRITON_URL="pnyx-pf-triton:8000" \
  -e PCM_DB_PATH="/data/sessions.db" \
  pnyx-pcm:latest
```

## Ports

| Container port | Description |
|---|---|
| `8080` | PCM HTTP API (`/v1/chat/completions`, `/v1/sessions/{id}`, `/health`) |

## API Endpoints

| Method | Path | Description |
|---|---|---|
| `POST` | `/v1/chat/completions` | Privacy-aware chat completions (OpenAI-compatible + `bypass_privacy_filter` field and `X-Session-ID` header) |
| `GET` | `/v1/sessions/{session_id}` | Inspect raw messages, hidden messages, and privacy state for a session |
| `GET` | `/health` | Liveness probe |
| `GET` | `/health/triton` | Readiness probe — checks that the Triton model is ready |

Both buffered (`stream: false`) and streaming (`stream: true`) responses are
supported. Streaming returns `text/event-stream` SSE frames whose
`content`/`reasoning`/tool-call deltas are de-anonymised in real time. The
resolved session ID is returned in the `X-Session-ID` response header; send it
back on the next request to continue the same session.

## Quick test

```bash
curl http://localhost:8080/health

curl http://localhost:8080/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "messages": [
      {"role": "user", "content": "My name is Alice Smith and my email is alice@example.com. Summarise this."}
    ]
  }'

# Streaming
curl -N http://localhost:8080/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "stream": true,
    "messages": [
      {"role": "user", "content": "My name is Alice Smith and my email is alice@example.com. Summarise this."}
    ]
  }'
```
