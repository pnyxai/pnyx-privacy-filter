# Triton Inference Server with Custom Conda Environment

This folder contains the `Dockerfile` and requirements needed to build a custom Triton Inference Server image. The image is optimized for deployment using a multi-stage build process.

## Prerequisites

Before building this image, you must create a base `tritonserver:latest` image using the Triton [Model Navigator](https://github.com/triton-inference-server/server) (specifically the `compose.py` tool). This allows for a leaner base image containing only the required backends.

Run the following command from the root of the [triton-inference-server/server](https://github.com/triton-inference-server/server) repository:

```bash
python3 compose.py \
  --backend pytorch \
  --backend python \
  --backend onnxruntime \
  --repoagent checksum \
  --image min,nvcr.io/nvidia/tritonserver:26.03-py3-min \
  --image full,nvcr.io/nvidia/tritonserver:26.03-py3
```

This will generate a `tritonserver:latest` image which is used as the base in our `Dockerfile`.

> Update the version tag (`26.03`) as needed.

## Dockerfile Overview

The `Dockerfile` follows a two-stage build process:

### 1. Builder Stage (Conda Environment Setup)
- **Base Image**: Uses `python:3.12-slim` for a lightweight build process.
- **Miniconda Installation**: Installs Miniconda to manage Python environments.
- **Environment Creation**: Creates a dedicated Conda environment named `pnyx_pf` with Python 3.12.
- **Dependency Management**: Installs CPU-only `torch` and other packages listed in `requirements.txt`. Installing CPU-only torch significantly reduces the final image size.
- **Packaging**: Uses `conda-pack` to compress the environment into a tarball (`pnyx_pf.tar.gz`). This allows the environment to be portable and used by the Triton Python backend.

### 2. Final Execution Stage
- **Base Image**: Uses the custom `tritonserver:latest` built in the prerequisites step.
- **System Utilities**: Installs `gettext-base` for `envsubst` support.
- **Artifact Transfer**: Copies the packed environment (`pnyx_pf.tar.gz`) from the builder stage to `/opt/`.
- **Custom Entrypoint**: Sets `custom_entrypoint.sh` as the entrypoint. This script automates:
    - Initializing the model repository based on `$MODEL_TYPE`.
    - Substituting environment variables (e.g., `$MAX_BATCH_SIZE`, `$MAX_SEQUENCE_LENGTH`) in `.pbtxt` files.
    - Creating symlinks for model weights from `/mnt/models` based on the `$WEIGHT_MAPPINGS` variable.

## Build Instructions

The build context must be the **`pnyx/` parent directory** (one level above this project) so that
both `pnyx-privacy-filter/` and `privacy-filter/` are accessible to the Dockerfile.

Run from inside `pnyx-privacy-filter/docker/triton/`:
```bash
bash build.sh
```

Or manually from the `pnyx/` parent directory:
```bash
docker build \
  -f pnyx-privacy-filter/docker/triton/Dockerfile \
  -t pnyx-pf-triton:latest \
  .
```

## Runtime Environment Variables

| Variable | Required | Default | Description |
|---|---|---|---|
| `MODEL_TYPE` | yes | — | Model subfolder to activate, e.g. `pnyx-pf` |
| `MAX_BATCH_SIZE` | yes | — | Max batch size for the ensemble + GPU model |
| `INSTANCE_COUNT` | yes | — | Number of GPU model instances |
| `MAX_SEQUENCE_LENGTH` | no | `4096` | Tokenizer max sequence length |
| `WEIGHT_MAPPINGS` | yes | — | Comma-separated `src=dest` pairs mapping model weight files to their serving paths |
| `HF_TOKEN` | no | — | Hugging Face token, required only for private model repositories |

### WEIGHT_MAPPINGS by ONNX variant

`src` is relative to `/mnt/models`; `dest` is relative to the model repository inside the container. The ONNX graph file must always map to `main_model/1/model.onnx`.

| Variant | Notes |
|---|---|
| `model_fp16.onnx` | GPU supports MoE kernels (fastest) |
| `model.onnx` | Recommended fallback for broader compatibility |
| `model_q4f16.onnx` / `model_q4.onnx` / `model_quantized.onnx` | Quantised variants for reduced VRAM |

Example for `model.onnx`:
```bash
WEIGHT_MAPPINGS="onnx/model.onnx=main_model/1/model.onnx,onnx/model.onnx_data=main_model/1/model.onnx_data,onnx/model.onnx_data_1=main_model/1/model.onnx_data_1,onnx/model.onnx_data_2=main_model/1/model.onnx_data_2"
```

## Running Standalone

```bash
docker run --gpus all --rm -it \
  --shm-size=2g \
  -p 8000:8000 -p 8001:8001 -p 8002:8002 \
  -v /path/to/model-weights:/mnt/models \
  -e MODEL_TYPE="pnyx-pf" \
  -e MAX_BATCH_SIZE=64 \
  -e INSTANCE_COUNT=1 \
  -e MAX_SEQUENCE_LENGTH=8192 \
  -e WEIGHT_MAPPINGS="onnx/model.onnx=main_model/1/model.onnx,onnx/model.onnx_data=main_model/1/model.onnx_data,onnx/model.onnx_data_1=main_model/1/model.onnx_data_1,onnx/model.onnx_data_2=main_model/1/model.onnx_data_2" \
  pnyx-pf-triton:latest
```

### Health & inference check

```bash
# Server ready
curl http://localhost:8000/v2/health/ready

# Model ready
curl http://localhost:8000/v2/models/ensemble_model/ready

# Inference
curl http://localhost:8000/v2/models/ensemble_model/infer \
  -H "Content-Type: application/json" \
  -d '{
    "inputs": [{
      "name": "PROMPT",
      "shape": [1, 1],
      "datatype": "BYTES",
      "data": ["Alice Smith lives at 123 Main St."]
    }]
  }'
```

For production use, run Triton together with PCM via Docker Compose — see [`examples/`](../../examples/README.md).
```