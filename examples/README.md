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
| `PCM_LLM_SESSION_HEADER` | Optional upstream session header; auto-defaults to `x-opencode-session` for `opencode.ai` upstreams |

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
