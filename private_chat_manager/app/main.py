from __future__ import annotations

from contextlib import asynccontextmanager
from typing import Any

import httpx
import aiosqlite
from fastapi import APIRouter, Depends, FastAPI, Header, HTTPException, Request
from fastapi.responses import JSONResponse, Response, StreamingResponse

from ._logging import configure_logging, get_logger
from .config import Settings
from .models import PrivateChatRequest, SessionInspectResponse
from .privacy_manager import (
    handle_request,
    handle_stream_request,
    prepare_request,
)
from .session_store import ensure_schema, get_session
from .triton_client import TritonPrivacyFilterClient

logger = get_logger(__name__)


# ---------------------------------------------------------------------------
# Application lifespan: initialise shared resources once at startup
# ---------------------------------------------------------------------------


@asynccontextmanager
async def lifespan(app: FastAPI):
    settings: Settings = app.state.settings  # injected before startup (see factory)

    # Ensure the SQLite schema exists
    await ensure_schema(settings.db_path)
    logger.info("SQLite session store ready", db_path=settings.db_path)

    # Warm up the Triton client reference (connection is per-call, not pooled)
    triton_client = TritonPrivacyFilterClient(
        url=settings.triton_url,
        model_name=settings.triton_model_name,
    )
    app.state.triton_client = triton_client
    logger.info(
        "Triton client configured",
        triton_url=settings.triton_url,
        triton_model=settings.triton_model_name,
    )

    yield
    # No explicit cleanup needed: SQLite connections are opened per-request.


# ---------------------------------------------------------------------------
# Application factory
# ---------------------------------------------------------------------------


def create_app(settings: Settings | None = None) -> FastAPI:
    """Create and return the FastAPI application.

    Accepting *settings* as a parameter makes the app trivially testable
    without environment-variable side-effects.
    """
    if settings is None:
        settings = Settings()

    configure_logging(settings.log_level)
    logger.info("Starting PrivateChatManager", log_level=settings.log_level)

    app = FastAPI(
        title="PrivateChatManager",
        description=(
            "Privacy-preserving mid-layer for /v1/chat/completions. "
            "Redacts PII via the Triton privacy-filter before forwarding "
            "requests to the downstream LLM, then de-anonymises responses."
        ),
        version="0.1.0",
        lifespan=lifespan,
    )
    app.state.settings = settings
    app.include_router(router)
    return app


# ---------------------------------------------------------------------------
# Dependencies
# ---------------------------------------------------------------------------


def get_settings(request: Request) -> Settings:
    return request.app.state.settings


def get_triton_client(request: Request) -> TritonPrivacyFilterClient:
    return request.app.state.triton_client


# ---------------------------------------------------------------------------
# Routes
#
# Defined on a router so that ``create_app()`` returns a fully routed app and
# tests can build one with custom settings.  ``create_app`` is invoked at the
# bottom of this module for ``uvicorn app.main:app``.
# ---------------------------------------------------------------------------


router = APIRouter()


@router.get("/health")
async def health() -> dict[str, str]:
    """Liveness probe."""
    return {"status": "ok"}


@router.get("/health/triton")
async def health_triton(
    triton_client: TritonPrivacyFilterClient = Depends(get_triton_client),
) -> dict[str, Any]:
    """Readiness probe: checks whether the Triton model is ready."""
    ready = await triton_client.is_ready()
    if not ready:
        logger.warning("Triton model not ready")
        raise HTTPException(
            status_code=503,
            detail="Triton model is not ready",
        )
    logger.debug("Triton model ready")
    return {"status": "ok", "triton": "ready"}


@router.get("/v1/sessions/{session_id}", response_model=SessionInspectResponse)
async def inspect_session(
    session_id: str,
    settings: Settings = Depends(get_settings),
) -> SessionInspectResponse:
    """Inspect the stored state of a session.

    Returns the raw messages (original user content with PII), the hidden
    messages (redacted content forwarded to the LLM), and the
    ``PrivacyFilterState`` (placeholder map, per-label counters, and per-turn
    RedactionResult audit log).

    This endpoint is intended for debugging and auditing.  In production you
    may want to restrict access to it via a reverse-proxy or API-key check.
    """
    async with aiosqlite.connect(settings.db_path) as conn:
        await conn.execute("PRAGMA journal_mode=WAL")
        session = await get_session(conn, session_id)

    if session is None:
        logger.debug("session not found", session_id=session_id)
        raise HTTPException(status_code=404, detail=f"Session '{session_id}' not found.")

    # A "turn" = one user/tool request + one assistant response.
    # Count assistant messages in raw_messages as a proxy for completed turns.
    turn_count = sum(
        1 for m in session.raw_messages if m.get("role") == "assistant"
    )
    filtered_turn_count = len(session.privacy_state.redaction_results)
    logger.debug(
        "session inspected",
        session_id=session_id,
        turn_count=turn_count,
        filtered_turn_count=filtered_turn_count,
    )

    return SessionInspectResponse(
        session_id=session.session_id,
        created_at=session.created_at,
        updated_at=session.updated_at,
        raw_messages=session.raw_messages,
        hidden_messages=session.hidden_messages,
        privacy_state=session.privacy_state,
        turn_count=turn_count,
        filtered_turn_count=filtered_turn_count,
    )


@router.post("/v1/chat/completions")
async def chat_completions(
    request: PrivateChatRequest,
    x_session_id: str | None = Header(default=None, alias="X-Session-ID"),
    settings: Settings = Depends(get_settings),
    triton_client: TritonPrivacyFilterClient = Depends(get_triton_client),
) -> Response:
    """Privacy-aware /v1/chat/completions endpoint.

    Accepts the same body as the OpenAI Chat Completions API with one extra
    field:

    * ``bypass_privacy_filter`` *(bool, default false)*: When ``true``, the
      last user message is forwarded to the LLM without any PII redaction.
      Placeholders from *previous* turns in the session are still
      de-anonymised in the response.

    Both buffered (``stream=false``) and streaming (``stream=true``) modes are
    supported.  Streaming returns ``text/event-stream`` SSE frames whose
    deltas have been de-anonymised in real time; the resolved session ID is
    always returned in the ``X-Session-ID`` response header.
    """
    logger.info(
        "chat completion request",
        session_id_hint=x_session_id or "new",
        msg_count=len(request.messages),
        bypass=request.bypass_privacy_filter,
        stream=bool(request.stream),
    )

    if request.stream:
        async with aiosqlite.connect(settings.db_path) as conn:
            await conn.execute("PRAGMA journal_mode=WAL")
            prepared = await prepare_request(
                request=request,
                session_id_header=x_session_id,
                settings=settings,
                triton_client=triton_client,
                conn=conn,
            )

        logger.info(
            "streaming chat completion started",
            session_id=prepared.session_id,
        )
        return StreamingResponse(
            handle_stream_request(prepared, settings),
            media_type="text/event-stream",
            headers={
                "X-Session-ID": prepared.session_id,
                "Cache-Control": "no-cache",
                "Connection": "keep-alive",
                "X-Accel-Buffering": "no",
            },
        )

    async with aiosqlite.connect(settings.db_path) as conn:
        await conn.execute("PRAGMA journal_mode=WAL")
        response_data, session_id = await handle_request(
            request=request,
            session_id_header=x_session_id,
            settings=settings,
            triton_client=triton_client,
            conn=conn,
        )

    logger.info("chat completion response sent", session_id=session_id)
    if "response_body" in settings.verbose_log_events:
        logger.debug("chat completion response body", session_id=session_id, response=response_data)
    return JSONResponse(
        content=response_data,
        headers={"X-Session-ID": session_id},
    )


# ---------------------------------------------------------------------------
# Catch-all: proxy any unrecognised request to the downstream LLM as-is.
# MUST be declared last so the routes above take priority.
# ---------------------------------------------------------------------------

# Headers that must not be forwarded (hop-by-hop or host-specific).
_HOP_BY_HOP = frozenset({
    "host", "transfer-encoding", "te", "trailers",
    "connection", "keep-alive", "upgrade", "proxy-authorization",
})


@router.api_route("/{path:path}", methods=["GET", "POST", "PUT", "PATCH", "DELETE", "OPTIONS", "HEAD"])
async def llm_proxy(
    path: str,
    raw_request: Request,
    settings: Settings = Depends(get_settings),
) -> Response:
    """Transparent reverse-proxy for all paths not handled by PCM.

    Forwards the original method, path, query string, body, and headers
    (minus hop-by-hop headers) to the downstream LLM endpoint and streams
    the response back unchanged.  The LLM API key is injected if configured.
    """
    target_url = f"{settings.llm_url.rstrip('/')}/{path}"
    if raw_request.url.query:
        target_url = f"{target_url}?{raw_request.url.query}"

    # Build forwarded headers: drop hop-by-hop, inject auth if present.
    forward_headers = {
        k: v
        for k, v in raw_request.headers.items()
        if k.lower() not in _HOP_BY_HOP
    }
    if settings.llm_api_key:
        forward_headers["authorization"] = f"Bearer {settings.llm_api_key}"

    body = await raw_request.body()

    logger.info(
        "proxy passthrough",
        method=raw_request.method,
        path=f"/{path}",
        target=target_url,
    )

    async with httpx.AsyncClient(timeout=120.0) as client:
        llm_resp = await client.request(
            method=raw_request.method,
            url=target_url,
            headers=forward_headers,
            content=body,
        )

    # Strip hop-by-hop headers from the response before forwarding.
    response_headers = {
        k: v
        for k, v in llm_resp.headers.items()
        if k.lower() not in _HOP_BY_HOP
    }

    logger.info(
        "proxy passthrough response",
        method=raw_request.method,
        path=f"/{path}",
        status_code=llm_resp.status_code,
    )

    return Response(
        content=llm_resp.content,
        status_code=llm_resp.status_code,
        headers=response_headers,
        media_type=llm_resp.headers.get("content-type"),
    )


# Module-level application for ``uvicorn app.main:app``.
app = create_app()

