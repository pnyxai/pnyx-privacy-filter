from __future__ import annotations

import asyncio
import json
import time

import numpy as np
import tritonclient.http as httpclient

from ._logging import get_logger

logger = get_logger(__name__)


class TritonPrivacyFilterClient:
    """Async wrapper around the synchronous Triton HTTP inference client.

    The underlying ``tritonclient.http`` library is synchronous, so all
    network I/O is dispatched to a thread-pool executor to avoid blocking
    the asyncio event loop.
    """

    def __init__(self, url: str, model_name: str) -> None:
        self._url = url
        self._model_name = model_name

    # ------------------------------------------------------------------
    # Synchronous helpers (run inside executor)
    # ------------------------------------------------------------------

    def _sync_infer(self, text: str) -> dict:
        """Send *text* to the Triton ensemble model and return a RedactionResult dict."""
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
