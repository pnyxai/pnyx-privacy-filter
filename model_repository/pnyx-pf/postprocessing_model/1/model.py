"""Triton postprocessing model for openai/privacy-filter.

Receives batched logits [B, seq_len, 33], left-padded token ids [B, seq_len],
attention masks [B, seq_len], and original prompt strings [B, 1].

For each sample:
1. Skips left-padding tokens using attention mask.
2. Applies log-softmax and Viterbi CRF decoding to produce per-token labels.
3. Converts token-level labels to character-level spans via tiktoken byte offsets.
4. Builds a RedactionResult-compatible JSON string.

Returns LABELS [B, 1] as serialized JSON strings.

The output is byte-identical to what opf.OPF().redact() produces for the same
input, subject to the tokenizer truncation limit (MAX_SEQUENCE_LENGTH).
"""

import json
import os
from bisect import bisect_left, bisect_right
from typing import Sequence

import numpy as np
import torch
import torch.nn.functional as F

import triton_python_backend_utils as pb_utils

from opf._api import RedactionResult, _redact_text
from opf._common.label_space import NER_CLASS_NAMES_BY_CATEGORY_VERSION
from opf._core.decoding import ViterbiCRFDecoder, resolve_viterbi_transition_biases
from opf._core.runtime import (
    DetectedSpan,
    _label_placeholder,
    _select_non_overlapping_spans,
    build_detection_summary,
)
from opf._core.sequence_labeling import build_label_info
from opf._core.spans import (
    labels_to_spans,
    token_spans_to_char_spans,
    trim_char_spans_whitespace,
)

from transformers import AutoTokenizer

# v2 taxonomy: 8 span categories × 4 BIOES tags + 1 background = 33 total classes
_NER_CLASS_NAMES_V2 = NER_CLASS_NAMES_BY_CATEGORY_VERSION["v2"]


def _decode_text_and_token_char_ranges_agnostic(
    token_bytes: Sequence[bytes],
) -> tuple[str, list[int], list[int]]:
    """Decode token bytes and compute per-token character ranges in the decoded text."""
    decoded_text = b"".join(token_bytes).decode("utf-8", errors="replace")
    if not token_bytes:
        return decoded_text, [], []

    char_byte_starts: list[int] = []
    char_byte_ends: list[int] = []
    byte_cursor = 0
    for ch in decoded_text:
        char_byte_starts.append(byte_cursor)
        byte_cursor += len(ch.encode("utf-8"))
        char_byte_ends.append(byte_cursor)

    char_starts: list[int] = []
    char_ends: list[int] = []
    token_byte_cursor = 0
    for raw_bytes in token_bytes:
        token_byte_start = token_byte_cursor
        token_byte_end = token_byte_start + len(raw_bytes)
        token_byte_cursor = token_byte_end
        start_idx = bisect_right(char_byte_ends, token_byte_start)
        end_idx = bisect_left(char_byte_starts, token_byte_end)
        if end_idx < start_idx:
            end_idx = start_idx
        char_starts.append(start_idx)
        char_ends.append(end_idx)

    return decoded_text, char_starts, char_ends


class TritonPythonModel:
    def initialize(self, args):
        # Build label info and Viterbi CRF decoder from the v2 label space.
        self.label_info = build_label_info(_NER_CLASS_NAMES_V2)

        # Auto-discover viterbi_calibration.json from the versioned model dir.
        # e.g. /models/postprocessing_model/1/viterbi_calibration.json
        checkpoint_dir = os.path.join(args["model_repository"], args["model_version"])
        biases = resolve_viterbi_transition_biases(
            viterbi_calibration_path=None,
            checkpoint_dir=checkpoint_dir,
        )
        self.decoder = ViterbiCRFDecoder(label_info=self.label_info, **biases)

        self.tokenizer = AutoTokenizer.from_pretrained("openai/privacy-filter")
        self._get_token_bytes = lambda ids: [
            self.tokenizer.convert_tokens_to_string(
                [self.tokenizer.convert_ids_to_tokens(tid)]
            ).encode("utf-8")
            for tid in ids
        ]

    def execute(self, requests):
        responses = []
        for request in requests:
            try:
                logits_np = pb_utils.get_input_tensor_by_name(
                    request, "LOGITS"
                ).as_numpy()  # [B, seq_padded, 33]
                input_ids_np = pb_utils.get_input_tensor_by_name(
                    request, "INPUT_IDS"
                ).as_numpy()  # [B, seq_padded]
                attn_mask_np = pb_utils.get_input_tensor_by_name(
                    request, "ATTN_MASK"
                ).as_numpy()  # [B, seq_padded]
                prompts_np = pb_utils.get_input_tensor_by_name(
                    request, "PROMPT"
                ).as_numpy()  # [B, 1]

                batch_size = logits_np.shape[0]
                results: list[str] = []

                for i in range(batch_size):
                    result_json = self._process_sample(
                        logits=logits_np[i],
                        input_ids=input_ids_np[i],
                        attn_mask=attn_mask_np[i],
                        prompt=prompts_np[i, 0],
                    )
                    results.append(result_json)

                out_tensor = pb_utils.Tensor(
                    "LABELS",
                    np.array(results, dtype=object).reshape(-1, 1),
                )
                responses.append(
                    pb_utils.InferenceResponse(output_tensors=[out_tensor])
                )

            except Exception as e:
                responses.append(
                    pb_utils.InferenceResponse(
                        error=pb_utils.TritonError(str(e))
                    )
                )
        return responses

    def _process_sample(
        self,
        logits: np.ndarray,
        input_ids: np.ndarray,
        attn_mask: np.ndarray,
        prompt: bytes | str,
    ) -> str:
        """Decode one sample and return a RedactionResult-compatible JSON string.

        The output schema is identical to opf.OPF().redact() (RedactionResult.to_dict()).
        """
        prompt_str = (
            prompt.decode("utf-8") if isinstance(prompt, (bytes, bytearray)) else str(prompt)
        )

        # Remove left-padding: real tokens are at positions [pad_len:]
        seq_len = int(attn_mask.sum())
        pad_len = attn_mask.shape[0] - seq_len

        real_token_ids: list[int] = input_ids[pad_len:].tolist()
        real_logits: np.ndarray = logits[pad_len:, :]  # [seq_len, 33]

        if seq_len == 0 or real_logits.shape[0] == 0:
            return json.dumps(self._empty_result(prompt_str))

        # Viterbi CRF decode: logits → log-probs → label sequence
        logits_tensor = torch.from_numpy(real_logits.astype(np.float32))
        log_probs = F.log_softmax(logits_tensor, dim=-1)  # [seq_len, 33]
        decoded_labels: list[int] = self.decoder.decode(log_probs)
        # Safety net: if the decoder returns a length mismatch, fall back to argmax.
        # Mirrors the same guard in opf._core.runtime.predict_text.
        if len(decoded_labels) != seq_len:
            decoded_labels = log_probs.argmax(dim=-1).tolist()

        # Convert token label ids → token-level spans
        labels_by_index = {j: lbl for j, lbl in enumerate(decoded_labels)}
        token_spans = labels_to_spans(labels_by_index, self.label_info)

        # Get token bytes for decoding logic using the mapped function
        token_bytes = self._get_token_bytes(real_token_ids)

        # Map token indices → character offsets using agnostic logic
        decoded_text, char_starts, char_ends = _decode_text_and_token_char_ranges_agnostic(
            token_bytes
        )
        decoded_mismatch = decoded_text != prompt_str
        source_text = decoded_text if decoded_mismatch else prompt_str

        # Convert token spans → char spans and trim whitespace.
        # NOTE: discard_overlapping_spans_by_label is intentionally NOT called here.
        # opf.OPF() defaults to discard_overlapping_predicted_spans=False, so we
        # match that default. Cross-label overlaps are handled by _select_non_overlapping_spans below.
        char_spans = token_spans_to_char_spans(token_spans, char_starts, char_ends)
        char_spans = trim_char_spans_whitespace(char_spans, source_text)

        # Build DetectedSpan objects (same dataclass used by opf runtime)
        n_span_classes = len(self.label_info.span_class_names)
        detected: list[DetectedSpan] = []
        for label_idx, start, end in char_spans:
            if not (0 <= start < end <= len(source_text)):
                continue
            label = (
                str(self.label_info.span_class_names[label_idx])
                if 0 <= label_idx < n_span_classes
                else f"label_{label_idx}"
            )
            detected.append(
                DetectedSpan(
                    label=label,
                    start=start,
                    end=end,
                    text=source_text[start:end],
                    placeholder=_label_placeholder(label),  # from opf._core.runtime
                )
            )

        # Drop cross-label overlaps with greedy left-to-right selection.
        # _select_non_overlapping_spans is imported from opf._core.runtime.
        display_spans = tuple(_select_non_overlapping_spans(detected))

        # Build RedactionResult and return its dictionary representation.
        redacted_text = _redact_text(source_text, display_spans)
        summary = build_detection_summary(
            output_mode="typed",
            labels=[span.label for span in display_spans],
            decoded_mismatch=decoded_mismatch,
        )

        warning = None
        if decoded_mismatch:
            warning = (
                "Input text did not exactly match tokenizer round-trip decode; "
                "spans are based on decoded token text."
            )

        result = RedactionResult(
            schema_version=1,
            summary=summary,
            text=source_text,
            detected_spans=display_spans,
            redacted_text=redacted_text,
            warning=warning,
        )
        return json.dumps(result.to_dict(), ensure_ascii=False)

    @staticmethod
    def _empty_result(text: str) -> dict:
        return {
            "schema_version": 1,
            "summary": {
                "output_mode": "typed",
                "span_count": 0,
                "by_label": {},
                "decoded_mismatch": False,
            },
            "text": text,
            "detected_spans": [],
            "redacted_text": text,
        }

    def finalize(self):
        pass
