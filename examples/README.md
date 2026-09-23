# Examples

Docker Compose stack that runs the full pnyx-privacy-filter system:
**Triton** (privacy-filter model) + **PrivateChatManager** (PCM) + **Open WebUI** with the PCM pipeline.

## Prerequisites

1. **Build the images** — from the `pnyx/` parent directory:
   1. `./docker/triton/build.sh` → `pnyx-pf-triton:latest` (see [`docker/triton/README.md`](../docker/triton/README.md) for details)
   2. `./docker/pcm/build.sh` → `pnyx-pcm:latest`
2. A running **OpenAI-compatible LLM** endpoint reachable from the host.

## Configuration

```bash
cp sample.env .env
```

Edit `.env` and fill in at minimum:

| Variable | Description |
|---|---|
| `TRITON_MODELS_PATH` | Absolute host path to the model weights directory |
| `TRITON_WEIGHT_MAPPINGS` | ONNX symlink mappings — see [`docker/triton/README.md`](../docker/triton/README.md) |
| `PCM_LLM_URL` | Base URL of the downstream LLM (e.g. `http://host.docker.internal:8000`); PCM appends `/v1/…` |
| `PCM_LLM_MODEL_NAME` | Optional fallback model used only when the client omits `model` |
| `PCM_LLM_SESSION_HEADER` | Optional upstream session header; overrides the endpoint registry (`app/endpoints.py`), whose built-in rule maps an `opencode.ai` host to `x-opencode-session` |
| `PCM_SESSION_TTL` | Optional session time-to-live from last activity (e.g. `30s`, `360m`, `6h`, `1.5d`, `2w`); empty/`0` disables expiry |
| `PCM_SESSION_TTL_SWEEP` | Background purge interval (same syntax; default `10m`), used only when `PCM_SESSION_TTL` is enabled |
| `PCM_SESSION_TTL_GRACE` | Optional extra margin (same syntax; default `120s`) added to the TTL before physical deletion, so an in-flight request's session is not swept |

All variables are documented inline in `sample.env`.

## Start / Stop

```bash
docker compose up -d      # start all services
docker compose logs -f    # follow logs
docker compose down       # stop and remove containers
```

## Services

| Service | Host port | Description |
|---|---|---|
| `pnyx-pf-triton` | — | Triton privacy-filter (internal only; PCM reaches it on the Docker network) |
| `pnyx-pcm` | `8080` | PrivateChatManager — OpenAI-compatible endpoint with privacy filtering |
| `open-webui` | `3000` | Chat interface, pre-configured to route requests through the PCM pipeline |
| `pnyx-pipelines` | `9099` | Open WebUI Pipelines — attaches session tracking to chat requests (see `pipelines/README.md`) |

## Calling PCM directly

PCM is a drop-in OpenAI-compatible endpoint:

```bash
# Health check
curl http://localhost:8080/health

# Chat completion
curl http://localhost:8080/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{"messages": [{"role": "user", "content": "My name is Alice. What is 2+2?"}]}'
```
