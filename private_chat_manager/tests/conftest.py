from __future__ import annotations

import os
import re

# ``app.main`` instantiates a module-level FastAPI app at import time, which
# requires the mandatory settings to be resolvable.  Provide harmless defaults
# before that import so the test suite does not depend on the environment.
os.environ.setdefault("PCM_LLM_URL", "http://llm.local")
os.environ.setdefault("PCM_LLM_MODEL_NAME", "pocket_network")

import httpx  # noqa: E402
import pytest  # noqa: E402
import pytest_asyncio  # noqa: E402

from app.config import Settings  # noqa: E402
from app.main import create_app  # noqa: E402
from app.session_store import ensure_schema  # noqa: E402


def _label_placeholder(label: str) -> str:
    normalized = re.sub(r"[^A-Za-z0-9]+", "_", label.upper()).strip("_")
    return f"<{normalized or 'REDACTED'}>"


class FakeTritonClient:
    """Minimal stand-in for ``TritonPrivacyFilterClient``.

    ``detections`` is an ordered list of ``(needle, label)`` pairs.  ``infer``
    replaces the first occurrence of each needle with the base placeholder and
    returns a RedactionResult-compatible dict, which is exactly the shape
    ``apply_placeholder_indexing`` consumes.
    """

    def __init__(self, detections: list[tuple[str, str]] | None = None) -> None:
        self._detections = detections or []
        self.calls: list[str] = []

    async def infer(self, text: str) -> dict:
        self.calls.append(text)
        spans: list[dict] = []
        redacted = text
        for needle, label in self._detections:
            if needle not in redacted:
                continue
            placeholder = _label_placeholder(label)
            start = redacted.index(needle)
            redacted = redacted.replace(needle, placeholder, 1)
            spans.append(
                {
                    "label": label,
                    "start": start,
                    "end": start + len(needle),
                    "text": needle,
                    "placeholder": placeholder,
                }
            )
        return {
            "schema_version": 1,
            "summary": {"output_type": "typed", "span_count": len(spans)},
            "text": text,
            "detected_spans": spans,
            "redacted_text": redacted,
        }

    async def is_ready(self) -> bool:
        return True


@pytest.fixture
def settings(tmp_path) -> Settings:
    return Settings(
        llm_url="http://llm.local",
        llm_model_name="pocket_network",
        triton_url="triton:8000",
        triton_model_name="ensemble_model",
        db_path=str(tmp_path / "sessions.db"),
    )


def _fake_triton() -> FakeTritonClient:
    return FakeTritonClient(
        [("Lionel Messi", "private_person"), ("555-0123", "private_phone")]
    )


async def _build_client(settings: Settings) -> httpx.AsyncClient:
    application = create_app(settings)
    await ensure_schema(settings.db_path)
    application.state.triton_client = _fake_triton()
    transport = httpx.ASGITransport(app=application)
    return httpx.AsyncClient(transport=transport, base_url="http://test")


@pytest_asyncio.fixture
async def app(settings: Settings):
    application = create_app(settings)
    await ensure_schema(settings.db_path)
    application.state.triton_client = _fake_triton()
    return application


@pytest_asyncio.fixture
async def client(app):
    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as c:
        yield c


@pytest_asyncio.fixture
async def debug_client(settings: Settings):
    """App at DEBUG level with the streaming response debug events enabled."""
    debug_settings = settings.model_copy(
        update={
            "log_level": "DEBUG",
            "verbose_log_events": frozenset(
                {"llm_raw_response", "response_body"}
            ),
        }
    )
    client = await _build_client(debug_settings)
    async with client:
        yield client


@pytest_asyncio.fixture
async def debug_silent_client(settings: Settings):
    """App at DEBUG level with no verbose events enabled."""
    debug_settings = settings.model_copy(update={"log_level": "DEBUG"})
    client = await _build_client(debug_settings)
    async with client:
        yield client


@pytest_asyncio.fixture
async def make_client(settings: Settings):
    """Factory for app clients with arbitrary ``Settings`` overrides."""
    clients: list[httpx.AsyncClient] = []

    async def _make(**overrides):
        overridden = settings.model_copy(update=overrides)
        client = await _build_client(overridden)
        clients.append(client)
        return client

    yield _make

    for client in clients:
        await client.aclose()
