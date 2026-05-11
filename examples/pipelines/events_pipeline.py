"""
title: PCM Privacy-Filtered
author: pnyx
version: 0.1.0
license: MIT
description: >
  Manifold pipeline that proxies /v1/chat/completions to the
  PrivateChatManager (PCM) backend, injecting the Open WebUI chat_id as
  the X-Session-ID header so PCM can maintain per-conversation privacy state
  (placeholder map, de-anonymisation) across message turns.

  The request body is forwarded to PCM exactly as received from Open WebUI.
  Only the X-Session-ID header is added.
requirements: requests
"""

from typing import List, Optional, Union

import hashlib
import json
import os
import requests
from pydantic import BaseModel


class Pipeline:
    class Valves(BaseModel):
        # Base URL of the PCM service reachable from the Pipelines container.
        # When running in the same Docker network as PCM use the service name.
        PCM_URL: str = "http://pnyx-pcm:8080"

        # Bearer token sent to PCM.  PCM ignores auth by default; set a value
        # if you expose PCM behind a reverse proxy that enforces it.
        PCM_API_KEY: Optional[str] = ""

        # Model name forwarded in the request body to PCM (and on to the LLM).
        # Override per-deployment via Admin Panel → Pipelines.
        PCM_LLM_MODEL_NAME: str = "meta-llama/Llama-3-8B-Instruct"

    # ──────────────────────────────────────────────────────────────────────────
    def __init__(self):
        self.type = "manifold"
        self.name = "PCM: "

        self.valves = self.Valves(
            PCM_URL=os.getenv("PCM_URL", "http://pnyx-pcm:8080"),
            PCM_API_KEY=os.getenv("PCM_API_KEY", ""),
            PCM_LLM_MODEL_NAME=os.getenv("PCM_LLM_MODEL_NAME", "meta-llama/Llama-3-8B-Instruct"),
        )

        # Open WebUI sends chat_id only in filter/inlet calls, not in the LLM
        # API call body.  We capture it here so pipe() can use it.
        self._pending_chat_ids: dict = {}

    def pipelines(self) -> List[dict]:
        """Expose a single named model in the Open WebUI model selector."""
        return [
            {
                "id": "privacy-filtered",
                "name": "Privacy-Filtered",
            }
        ]

    async def on_startup(self):
        print(f"on_startup: {__name__}")

    async def on_shutdown(self):
        print(f"on_shutdown: {__name__}")

    async def on_valves_updated(self):
        print(f"on_valves_updated: {__name__}")

    # ──────────────────────────────────────────────────────────────────────────
    def _request_key(self, messages: list) -> str:
        """Collision-resistant correlation key: SHA-256 of the full messages list."""
        payload = json.dumps(messages, sort_keys=True, ensure_ascii=False)
        return hashlib.sha256(payload.encode()).hexdigest()

    async def inlet(self, body: dict, user=None) -> dict:
        """Capture chat_id from the full Open WebUI body before it is stripped
        for the downstream OpenAI-compatible LLM call."""
        # Open WebUI places chat_id inside body["metadata"]["chat_id"], not at the top level.
        chat_id = (body.get("metadata") or {}).get("chat_id", "")
        if chat_id:
            key = self._request_key(body.get("messages", []))
            if key:
                self._pending_chat_ids[key] = chat_id
        return body

    # ──────────────────────────────────────────────────────────────────────────
    def pipe(
        self,
        user_message: str,
        model_id: str,
        messages: List[dict],
        body: dict,
        user=None,
    ) -> Union[str, dict]:
        """Forward the request to PCM, injecting X-Session-ID from chat_id."""

        # ── 1. Extract chat_id ─────────────────────────────────────────────
        # Open WebUI strips body["metadata"] before calling pipe(); the chat_id
        # was stashed in inlet() and is retrieved here.
        chat_id: str = self._pending_chat_ids.pop(
            self._request_key(messages), ""
        )

        # ── 2. Build headers ───────────────────────────────────────────────
        headers = {"Content-Type": "application/json"}
        if self.valves.PCM_API_KEY:
            headers["Authorization"] = f"Bearer {self.valves.PCM_API_KEY}"
        if chat_id:
            headers["X-Session-ID"] = chat_id

        # ── 3. Build payload ───────────────────────────────────────────────
        payload = {**body}
        # Override model with the operator-configured value.
        payload["model"] = self.valves.PCM_LLM_MODEL_NAME
        # PCM does not support streaming.
        payload["stream"] = False
        # Open WebUI sends 'user' as a dict {name, id, email, role}; the
        # OpenAI spec (and PCM) expect it to be a string or absent.
        payload.pop("user", None)

        # ── 4. Forward to PCM ─────────────────────────────────────────────
        url = f"{self.valves.PCM_URL.rstrip('/')}/v1/chat/completions"
        print(
            f"[pcm-pipeline] POST {url} | X-Session-ID={chat_id!r} "
            f"| model={payload['model']!r} | msgs={len(messages)}"
        )

        try:
            r = requests.post(url, json=payload, headers=headers, timeout=120)
            r.raise_for_status()
            return r.json()
        except requests.HTTPError as exc:
            detail = ""
            try:
                detail = exc.response.json()
            except Exception:
                detail = exc.response.text
            return f"Error {exc.response.status_code}: {detail}"
        except Exception as exc:
            return f"Error: {exc}"
