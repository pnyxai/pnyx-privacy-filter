from __future__ import annotations

import asyncio
import json
import time
from typing import Any

import numpy as np
import tritonclient.http as httpclient

from ._logging import get_logger

logger = get_logger(__name__)

# Default cap per Triton request.  The privacy-filter ensemble OOMs on very
# long inputs (attention buffers), so messages above this length are split and
# redacted chunk by chunk.  Overridable via ``PCM_TRITON_MAX_CHARS``.
DEFAULT_MAX_CHARS = 8000

# Never bisect below this size: if even a small chunk fails, the error is real
# and should surface rather than be hidden.
_MIN_CHUNK_CHARS = 256

# Preferred split points, most-wanted first.  A candidate is only used when it
# falls past the midpoint of the window so chunks stay close to ``max_chars``.
_CHUNK_BOUNDARIES = ("\n\n", "\n", ". ", "! ", "? ", " ")


def split_text_chunks(text: str, max_chars: int) -> list[str]:
    """Split *text* into chunks of at most *max_chars*, preferring boundaries.

    Cutting at paragraph/line/sentence/word boundaries avoids splitting an
    entity in half.  A hard cut is used only when no boundary exists past the
    midpoint.  The concatenation of the returned chunks is exactly *text*.
    """
    if max_chars <= 0 or len(text) <= max_chars:
        return [text]

    chunks: list[str] = []
    remaining = text
    half = max_chars // 2

    while len(remaining) > max_chars:
        window = remaining[:max_chars]
        cut = -1
        for separator in _CHUNK_BOUNDARIES:
            index = window.rfind(separator)
            if index >= half:
                cut = index + len(separator)
                break
        if cut <= 0:
            cut = max_chars
        chunks.append(remaining[:cut])
        remaining = remaining[cut:]

    if remaining:
        chunks.append(remaining)
    return chunks


def _empty_result(text: str) -> dict[str, Any]:
    return {
        "schema_version": 1,
        "summary": {"output_type": "typed", "span_count": 0},
        "text": text,
        "detected_spans": [],
        "redacted_text": text,
    }


def merge_redaction_results(results: list[dict[str, Any]]) -> dict[str, Any]:
    """Merge per-chunk RedactionResults into a single result.

    Reconstructs the full ``text`` and ``redacted_text`` by concatenation and
    shifts each span's ``start``/``end`` by the cumulative chunk offset so the
    audit log stays correct.
    """
    merged_spans: list[dict[str, Any]] = []
    redacted_parts: list[str] = []
    text_parts: list[str] = []
    offset = 0

    for result in results:
        chunk_text = result.get("text", "")
        redacted_parts.append(result.get("redacted_text", chunk_text))
        text_parts.append(chunk_text)
        for span in result.get("detected_spans", []):
            adjusted = dict(span)
            if isinstance(adjusted.get("start"), int):
                adjusted["start"] += offset
            if isinstance(adjusted.get("end"), int):
                adjusted["end"] += offset
            merged_spans.append(adjusted)
        offset += len(chunk_text)

    return {
        "schema_version": results[0].get("schema_version", 1) if results else 1,
        "summary": {"output_type": "typed", "span_count": len(merged_spans)},
        "text": "".join(text_parts),
        "detected_spans": merged_spans,
        "redacted_text": "".join(redacted_parts),
    }


class TritonPrivacyFilterClient:
    """Async wrapper around the synchronous Triton HTTP inference client.

    The underlying ``tritonclient.http`` library is synchronous, so all
    network I/O is dispatched to a thread-pool executor to avoid blocking
    the asyncio event loop.

    Inputs longer than ``max_chars`` are split and redacted chunk by chunk;
    the per-chunk results are merged back into one RedactionResult.
    """

    def __init__(
        self,
        url: str,
        model_name: str,
        max_chars: int = DEFAULT_MAX_CHARS,
    ) -> None:
        self._url = url
        self._model_name = model_name
        self._max_chars = max_chars if max_chars > 0 else DEFAULT_MAX_CHARS

    # ------------------------------------------------------------------
    # Synchronous helpers (run inside executor)
    # ------------------------------------------------------------------

    def _infer_single(self, text: str) -> dict:
        """Send one (possibly small) *text* to Triton and return its result."""
        logger.debug(
            "Triton inference request",
            model=self._model_name,
            char_count=len(text),
            text=text,
        )
        _t0 = time.monotonic()
        client = httpclient.InferenceServerClient(url=self._url)

        prompt_data = np.array([[text.encode("utf-8")]], dtype=object)
        input_tensor = httpclient.InferInput("PROMPT", [1, 1], "BYTES")
        input_tensor.set_data_from_numpy(prompt_data)

        outputs = [httpclient.InferRequestedOutput("LABELS")]
        result = client.infer(
            model_name=self._model_name,
            inputs=[input_tensor],
            outputs=outputs,
        )

        raw = result.as_numpy("LABELS").flatten()[0]
        if isinstance(raw, (bytes, bytearray)):
            raw = raw.decode("utf-8")
        parsed = json.loads(raw)
        _elapsed_ms = round((time.monotonic() - _t0) * 1000)
        logger.debug(
            "Triton inference response",
            model=self._model_name,
            span_count=len(parsed.get("detected_spans", [])),
            elapsed_ms=_elapsed_ms,
        )
        return parsed

    def _infer_chunk(self, text: str) -> list[dict]:
        """Infer *text*, bisecting on failure down to ``_MIN_CHUNK_CHARS``."""
        try:
            return [self._infer_single(text)]
        except Exception:
            if len(text) <= _MIN_CHUNK_CHARS:
                raise
            smaller = max(_MIN_CHUNK_CHARS, len(text) // 2)
            pieces = split_text_chunks(text, smaller)
            if len(pieces) <= 1:
                raise
            logger.warning(
                "Triton inference failed; retrying with smaller chunks",
                model=self._model_name,
                char_count=len(text),
                retry_char_count=smaller,
            )
            results: list[dict] = []
            for piece in pieces:
                results.extend(self._infer_chunk(piece))
            return results

    def _sync_infer(self, text: str) -> dict:
        """Redact *text*, splitting it when it exceeds ``max_chars``."""
        if not text or not text.strip():
            return _empty_result(text)

        chunks = split_text_chunks(text, self._max_chars)
        if len(chunks) > 1:
            logger.debug(
                "Triton input split into chunks",
                model=self._model_name,
                char_count=len(text),
                chunk_count=len(chunks),
                max_chars=self._max_chars,
            )
        results: list[dict] = []
        for chunk in chunks:
            results.extend(self._infer_chunk(chunk))
        if len(results) == 1:
            return results[0]
        return merge_redaction_results(results)

    def _sync_is_ready(self) -> bool:
        client = httpclient.InferenceServerClient(url=self._url)
        ready = client.is_model_ready(self._model_name)
        logger.debug("Triton readiness check", model=self._model_name, ready=ready)
        return ready

    # ------------------------------------------------------------------
    # Public async API
    # ------------------------------------------------------------------

    async def infer(self, text: str) -> dict:
        """Async redaction: returns a RedactionResult-compatible dict."""
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(None, self._sync_infer, text)

    async def is_ready(self) -> bool:
        """Return True if the Triton model is loaded and ready."""
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(None, self._sync_is_ready)
