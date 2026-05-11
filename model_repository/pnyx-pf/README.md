# pnyx-pf Model Repository

This directory contains the Triton Inference Server model repository for the `openai/privacy-filter` serving pipeline.

## Ensemble Pipeline

The ensemble is a 3-stage sequential pipeline:

```
PROMPT (string)
    │
    ▼
[tokenizer_model]  ─── Python backend (CPU)
    │  INPUT_IDS [B, seq_len]  int64
    │  ATTN_MASK [B, seq_len]  int64
    ▼
[main_model]       ─── ONNX Runtime backend (GPU)
    │  LOGITS [B, seq_len, 33]  float32
    ▼
[postprocessing_model]  ─── Python backend (CPU)
    │  LABELS [B, 1]  string (JSON)
    ▼
[ensemble_model]   ─── routes I/O between the three stages
```

The equivalent Python API call is:
```python
from opf import OPF
result = OPF().redact(text)
```

---

## Correspondence with `opf._core.runtime.predict_text`

The `postprocessing_model` is the Triton equivalent of the post-model logic inside
`opf._core.runtime.predict_text`. The table below documents each step.

### Step-by-step mapping

| Step | `predict_text` | `postprocessing_model/_process_sample` | Notes |
|------|---------------|---------------------------------------|-------|
| Tokenization | `runtime.encoding.encode(text)` | Performed upstream in `tokenizer_model` | Decoupled into a separate Triton stage |
| Forward pass | `runtime.model(window_tokens, attention_mask)` | Performed upstream in `main_model` | Decoupled into a separate Triton stage |
| Remove padding | Implicit — model only receives real tokens | `pad_len = shape[0] - attn_mask.sum()` | Left-padding is stripped from logits and input_ids |
| Logit → log-prob | `F.log_softmax(logits.float(), dim=-1)` | Identical | |
| Viterbi decode | `decoder.decode(stacked_scores)` | `self.decoder.decode(log_probs)` | Identical |
| Viterbi fallback | `if len(decoded_labels) != len(token_positions): argmax(...)` | `if len(decoded_labels) != seq_len: argmax(...)` | Identical safety net |
| Token → span | `labels_to_spans(predicted_labels_by_index, ...)` | `labels_to_spans(labels_by_index, ...)` | Identical |
| Char offsets | `decode_text_with_offsets(token_ids, runtime.encoding)` | `_decode_text_and_token_char_ranges_agnostic(token_bytes)` | Agnostic version; tokenizer-decoupled (see below) |
| Whitespace trim | `trim_char_spans_whitespace(...)` (conditional on `runtime.trim_span_whitespace`) | Always applied | opf default is `True`; we match the default |
| Per-label overlap | `discard_overlapping_spans_by_label(...)` (conditional on `runtime.discard_overlapping_predicted_spans`) | **Not applied** | opf default is `False`; we match the default |
| Cross-label overlap | `_select_non_overlapping_spans(detected)` | Identical | |
| Output mode | `_apply_output_mode_to_detected_spans(..., output_mode=...)` | Hardcoded `"typed"` | Only `"typed"` is supported in serving |
| Serialization | `RedactionResult` returned as Python object | `RedactionResult.to_dict()` serialized to JSON | Schema-identical output |

### Intentional behavioral differences

1. **Long-text truncation**: `predict_text` uses `example_to_windows` to split texts longer than
   `n_ctx` tokens into non-overlapping chunks and processes each chunk with a separate forward pass.
   The Triton pipeline instead **truncates** at `MAX_SEQUENCE_LENGTH` tokens in `tokenizer_model`.
   For texts within `MAX_SEQUENCE_LENGTH`, the behavior is identical.

2. **`discard_overlapping_spans_by_label`**: Not applied (matching `opf.OPF()` default of
   `discard_overlapping_predicted_spans=False`). Cross-label overlaps are still resolved by
   `_select_non_overlapping_spans`.

---

## Tokenizer-Agnostic Char-Offset Decoding

`opf._core.spans.decode_text_with_offsets` is tied to `tiktoken.Encoding` via
`encoding.decode_single_token_bytes`. The Triton post-processor uses a local
`_decode_text_and_token_char_ranges_agnostic` that operates on pre-extracted `bytes` objects.

The `openai/privacy-filter` HuggingFace tokenizer is used exclusively. Token bytes are
extracted via `AutoTokenizer.convert_tokens_to_string([convert_ids_to_tokens(id)]).encode("utf-8")`.

---

## Environment Variables

| Variable | Stage | Default | Description |
|----------|-------|---------|-------------|
| `MAX_BATCH_SIZE` | all | required | Triton max batch size |
| `INSTANCE_COUNT` | `main_model` | required | Number of ONNX GPU instances |
| `MAX_SEQUENCE_LENGTH` | `tokenizer_model` | `4096` | Hard truncation limit (tokens) |
| `WEIGHT_MAPPINGS` | entrypoint | required | NFS blob → model path symlinks |
