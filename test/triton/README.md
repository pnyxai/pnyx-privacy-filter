# Testing Triton Privacy Filter

This folder contains integration and benchmark testing for the Triton ensemble endpoint.

## Prerequisites

1. A running Triton server exposing port 8000.
2. The ensemble model loaded and ready (`ensemble_model`).
3. Python dependencies installed locally:

```bash
pip install numpy "tritonclient[http]"
```

## Test Script

- Script: `test.py`
- Default model: `ensemble_model`
- Expected output type: JSON string in `LABELS` (postprocessed redaction result)

The script performs:

1. Model readiness check.
2. One diagnostic inference request.
3. A small concurrent benchmark (configurable).

## Basic Usage

Run from the repository root (`pnyx-privacy-filter/`):

```bash
python ./test/test.py --verbose
```

## Common Examples

### 1) One diagnostic request only

```bash
python ./test/test.py --requests 1 --verbose
```

### 2) Custom input text

```bash
python ./test/test.py \
  --text "Alice Smith lives at 123 Main St and email is alice@example.com" \
  --verbose
```

### 3) Benchmark settings

```bash
python ./test/test.py --requests 20 --concurrency 4
```

### 4) Non-default endpoint or model

```bash
python ./test/test.py --url localhost:8000 --model ensemble_model --verbose
```

## Expected Verbose Output

- `Model 'ensemble_model' is READY.`
- A JSON payload decoded from `LABELS`.
- `Redacted Text` and `Detections found` summary.
- Benchmark stats (throughput, avg latency, p95, success rate).