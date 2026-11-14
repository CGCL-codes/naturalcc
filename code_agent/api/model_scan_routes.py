"""Streaming, stateless model detection for existing platform scan clients."""
from __future__ import annotations

import asyncio
import json
from threading import Event
from typing import Any

from fastapi import APIRouter, HTTPException, Request
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field
from starlette.concurrency import iterate_in_threadpool

from code_agent.api.agent_routes import get_default_runtime_model_config, normalize_runtime_model_config
from code_agent.agent_core.model_gateway import RuntimeModelConfig
from code_agent.agent_core.tool_registry import redact_sensitive_value
from code_agent.model_vulnerability_scan import ModelScan


class ModelScanRequest(BaseModel):
    project_dir: str
    target_files: list[str] = Field(default_factory=list)
    feature_config: dict[str, Any] = Field(default_factory=dict)
    runtime_model_config: dict[str, Any] | None = None
    api_key: str | None = None


router = APIRouter()


async def model_scan_response(body: ModelScanRequest, request: Request) -> StreamingResponse:
    # Validate paths, source sizes and GT before opening the NDJSON response.
    # File I/O and model iteration both run off the ASGI event loop.
    try:
        config = RuntimeModelConfig.from_dict(normalize_runtime_model_config(
            body.runtime_model_config, get_default_runtime_model_config()))
        scanner = await asyncio.to_thread(ModelScan, body.project_dir, body.target_files,
                                         body.feature_config, config, body.api_key)
    except (ValueError, OSError, TypeError) as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    cancelled = Event()

    async def generate():
        try:
            async for event in iterate_in_threadpool(scanner.events(cancelled)):
                if await request.is_disconnected():
                    break
                # Keep legacy start/log/done framing and add typed model deltas.
                # Credentials are request-scoped and never included in artifacts.
                encoded = json.dumps(redact_sensitive_value(event), ensure_ascii=False)
                if body.api_key:
                    encoded = encoded.replace(json.dumps(body.api_key, ensure_ascii=False)[1:-1], "[REDACTED]")
                yield encoded + "\n"
        finally:
            # The pending provider read can take until its timeout to unblock;
            # no subsequent chunk/file calls may start after disconnection.
            cancelled.set()

    return StreamingResponse(generate(), media_type="application/x-ndjson",
                             headers={"Cache-Control": "no-cache, no-transform", "X-Accel-Buffering": "no"})


@router.post("/api/vulnerability/model")
async def model_scan(body: ModelScanRequest, request: Request) -> StreamingResponse:
    return await model_scan_response(body, request)
