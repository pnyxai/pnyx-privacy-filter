"""Per-request phase timing.

One :class:`RequestTimings` instance is created per inbound request and threaded
through the pipeline.  Each stage calls :meth:`mark`; the handler emits a single
summary log at the end with every intermediate phase and the end-to-end total.

Phases used by the chat completion path (sequential slices of the request)::

    resolve   -> header parsing, per-conversation lock (incl. waiting), DB
                 connection, and session resolution (hash match, branch/fork)
    redact    -> ALL Triton calls for the new messages, end-to-end (per field,
                 chunked), plus placeholder indexing and storing hidden copies
    upstream  -> the downstream LLM call: buffered POST, or the whole stream
                 (reading it and forwarding chunks to the client)
    finalize  -> de-anonymise the response + persist the session

So the Triton cost is ``redact``; ``resolve`` performs no Triton. All values are
milliseconds.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any

from ._logging import get_logger

logger = get_logger(__name__)


@dataclass
class RequestTimings:
    start: float = field(default_factory=lambda: time.monotonic())
    _last: float = field(default_factory=lambda: time.monotonic())
    phases: dict[str, float] = field(default_factory=dict)
    # Set once a summary has been logged so failure-path fallbacks cannot emit a
    # second, partial event for the same request.
    emitted: bool = False

    def mark(self, name: str) -> float:
        """Record the elapsed ms since the previous mark under *name*.

        Repeated marks for the same *name* accumulate, so a phase can be built
        from several disjoint segments.
        """
        now = time.monotonic()
        elapsed = (now - self._last) * 1000.0
        self._last = now
        self.phases[name] = self.phases.get(name, 0.0) + elapsed
        return elapsed

    @property
    def total_ms(self) -> float:
        """End-to-end elapsed ms since the instance was created."""
        return (time.monotonic() - self.start) * 1000.0

    def as_fields(self) -> dict[str, Any]:
        """Structured-log fields: ``<phase>_ms`` for each phase + ``total_ms``."""
        fields: dict[str, Any] = {
            f"{name}_ms": round(value, 1) for name, value in self.phases.items()
        }
        fields["total_ms"] = round(self.total_ms, 1)
        return fields

    def log_timing(self, **fields: Any) -> None:
        """Emit the single ``request timing`` event for this request, once.

        Success paths call this after marking every phase; failure paths call it
        from a ``finally``/``except`` so every request is still accounted for.
        The :attr:`emitted` guard makes repeated calls a no-op, so a later
        failure fallback cannot overwrite a completed summary.
        """
        if self.emitted:
            return
        self.emitted = True
        logger.info("request timing", **fields, **self.as_fields())
