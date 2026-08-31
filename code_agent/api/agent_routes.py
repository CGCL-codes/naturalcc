from __future__ import annotations

import asyncio
from dataclasses import dataclass
import json
import logging
import os
import sqlite3
from pathlib import Path
from typing import Any, AsyncIterator

from fastapi import APIRouter, HTTPException, Query
from fastapi.responses import FileResponse, StreamingResponse
from pydantic import BaseModel, Field

from code_agent.agent_core.contracts import RiskLevel, RunBudget
from code_agent.agent_core.compaction import CompactionService
from code_agent.agent_core.context_builder import ContextPlanner
from code_agent.agent_core.event_store import EventStore, ThreadActiveConflict, VersionConflict
from code_agent.agent_core.memory_proposals import (
    MemoryProposalGenerationError,
    MemoryProposalService,
    ProposalValidationError,
)
from code_agent.agent_core.memory_store import MemoryProposalConflict, MemoryStore
from code_agent.agent_core.model_gateway import (
    DeepSeekRequestSerializer,
    OpenAICompatibleGateway,
)
from code_agent.agent_core.run_engine import RunEngine
from code_agent.agent_core.tool_registry import build_default_registry
from code_agent.agent_core.tools.codegraph import (
    CodeGraphClient,
    codegraph_artifact_name,
    normalize_codegraph_capabilities,
)
from code_agent.agent_core.token_budget import (
    ContextHardLimitExceeded,
    DeepSeekModelProfile,
    DeepSeekTokenCounter,
)


logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class DefaultContextRuntime:
    profile: DeepSeekModelProfile
    counter: DeepSeekTokenCounter
    serializer: DeepSeekRequestSerializer


def build_default_context_runtime(code_agent_root: str | Path) -> DefaultContextRuntime:
    root = Path(code_agent_root).expanduser().resolve()
    profile = DeepSeekModelProfile(
        model=os.environ.get("CODE_AGENT_MODEL", "deepseek-chat"),
        context_window_tokens=int(
            os.environ.get("CODE_AGENT_CONTEXT_WINDOW_TOKENS", "65536")
        ),
        default_output_reserve_tokens=int(
            os.environ.get("CODE_AGENT_OUTPUT_RESERVE_TOKENS", "4096")
        ),
        safety_margin_tokens=int(
            os.environ.get("CODE_AGENT_CONTEXT_SAFETY_MARGIN_TOKENS", "512")
        ),
        provider_framing_tokens=int(
            os.environ.get("CODE_AGENT_PROVIDER_FRAMING_TOKENS", "256")
        ),
        compaction_trigger_ratio=float(
            os.environ.get("CODE_AGENT_COMPACTION_TRIGGER_RATIO", "0.72")
        ),
        compaction_target_ratio=float(
            os.environ.get("CODE_AGENT_COMPACTION_TARGET_RATIO", "0.50")
        ),
        analyzer_output_tokens=int(
            os.environ.get("CODE_AGENT_ANALYZER_OUTPUT_TOKENS", "4096")
        ),
        summarizer_output_tokens=int(
            os.environ.get("CODE_AGENT_SUMMARIZER_OUTPUT_TOKENS", "2048")
        ),
    )
    tokenizer_directory = Path(
        os.environ.get(
            "CODE_AGENT_TOKENIZER_DIR",
            str(root / "resources" / "deepseek_v3_tokenizer"),
        )
    )
    serializer = DeepSeekRequestSerializer()
    counter = DeepSeekTokenCounter.from_directory(tokenizer_directory)
    return DefaultContextRuntime(profile, counter, serializer)


class CreateRunRequest(BaseModel):
    workspace: str
    goal: str = Field(min_length=1)
    target_files: list[str] = Field(default_factory=list)
    authorized_paths: list[str] = Field(default_factory=list)
    budget: dict[str, Any] = Field(default_factory=dict)
    thread_id: str | None = None
    capabilities: dict[str, Any] = Field(default_factory=dict)
    api_key: str | None = None


class CreateThreadRequest(BaseModel):
    title: str = Field(min_length=1, max_length=200)
    workspace: str | None = None
    model: str = ""
    runtime_mode: str = "agent"
    budget: dict[str, Any] = Field(default_factory=dict)
    authorized_paths: list[str] = Field(default_factory=list)
    context_items: list[dict[str, Any]] = Field(default_factory=list)
    capabilities: dict[str, Any] = Field(default_factory=dict)


class UpdateThreadRequest(BaseModel):
    title: str | None = Field(default=None, min_length=1, max_length=200)
    workspace: str | None = None
    model: str | None = None
    runtime_mode: str | None = None
    budget: dict[str, Any] | None = None
    authorized_paths: list[str] | None = None
    context_items: list[dict[str, Any]] | None = None
    capabilities: dict[str, Any] | None = None
    summary: str | None = None
    expected_version: int | None = Field(default=None, ge=0)


class CreateConversationMessageRequest(BaseModel):
    content: str = Field(min_length=1)
    target_files: list[str] = Field(default_factory=list)
    authorized_paths: list[str] = Field(default_factory=list)
    context_items: list[dict[str, Any]] = Field(default_factory=list)
    budget: dict[str, Any] | None = None
    capabilities: dict[str, Any] | None = None
    api_key: str | None = None


class CodeGraphVisualizeRequest(BaseModel):
    keyword: str | None = Field(default=None, max_length=500)
    depth: int = Field(default=2, ge=0, le=5)
    max_nodes: int = Field(default=150, ge=10, le=500)


class ContextResolveRequest(BaseModel):
    thread_id: str
    value: str = Field(min_length=1)
    limit: int = Field(default=20, ge=1, le=100)


class BudgetPatchRequest(BaseModel):
    budget: dict[str, Any]


class ApprovalRequest(BaseModel):
    risk: RiskLevel


class MemoryCandidateRequest(BaseModel):
    scope: str
    kind: str
    project_id: str | None = None
    thread_id: str | None = None
    subject: str
    content: str
    source_run_id: str | None = None
    source_revision: str | None = None
    confidence: float = Field(ge=0, le=1)


class MemoryUpdateRequest(BaseModel):
    subject: str = Field(min_length=1)
    content: str = Field(min_length=1)
    confidence: float = Field(ge=0, le=1)


class MemoryProposalSelectionRequest(BaseModel):
    thread_id: str = Field(min_length=1)
    project_id: str | None = None
    evidence_refs: list[dict[str, Any]] = Field(min_length=1, max_length=50)


class MemoryProposalUpdateRequest(BaseModel):
    expected_version: int = Field(ge=0)
    subject: str | None = Field(default=None, min_length=1, max_length=200)
    canonical_content: str | None = Field(default=None, min_length=1, max_length=4_000)
    scope: str | None = None
    kind: str | None = None
    expires_at: str | None = None


class MemoryProposalActionRequest(BaseModel):
    expected_version: int = Field(ge=0)
    reason: str = Field(default="", max_length=2_000)


def create_agent_router(engine: RunEngine, memory_store: MemoryStore | None = None) -> APIRouter:
    router = APIRouter(prefix="/api/agent", tags=["agent"])
    engine.store.migrate_legacy_runs()
    proposal_service = (
        MemoryProposalService(
            event_store=engine.store,
            memory_store=memory_store,
            gateway=engine.model,
            context_planner=engine.context_planner,
        )
        if memory_store is not None
        else None
    )

    @router.post("/threads")
    async def create_thread(request: CreateThreadRequest) -> dict[str, Any]:
        thread_id = await asyncio.to_thread(
            engine.store.create_thread,
            request.title,
            workspace=request.workspace,
            model=request.model,
            runtime_mode=request.runtime_mode,
            budget=request.budget,
            authorized_paths=request.authorized_paths,
            context_items=request.context_items,
            capabilities=normalize_codegraph_capabilities(request.capabilities),
        )
        return {"thread_id": thread_id, "thread": engine.store.get_thread(thread_id)}

    @router.get("/threads")
    async def list_threads(limit: int = Query(default=100, ge=1, le=500)) -> dict[str, Any]:
        return {"threads": await asyncio.to_thread(engine.store.list_threads, limit)}

    @router.get("/threads/{thread_id}")
    async def get_thread(thread_id: str) -> dict[str, Any]:
        try:
            thread = await asyncio.to_thread(engine.store.get_thread, thread_id)
            messages = await asyncio.to_thread(
                engine.store.list_conversation_messages,
                thread_id,
            )
            last_run = (
                await asyncio.to_thread(engine.get_state, thread["last_run_id"])
                if thread.get("last_run_id")
                else None
            )
        except KeyError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc
        return {"thread": thread, "messages": messages, "last_run": last_run}

    @router.patch("/threads/{thread_id}")
    async def update_thread(thread_id: str, request: UpdateThreadRequest) -> dict[str, Any]:
        values = request.model_dump(exclude_unset=True)
        expected_version = values.pop("expected_version", None)
        if values.get("capabilities") is not None:
            values["capabilities"] = normalize_codegraph_capabilities(values["capabilities"])
        try:
            thread = await asyncio.to_thread(
                engine.store.update_thread,
                thread_id,
                expected_version=expected_version,
                **values,
            )
        except KeyError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc
        except VersionConflict as exc:
            raise HTTPException(status_code=409, detail=str(exc)) from exc
        return {"thread": thread}

    @router.delete("/threads/{thread_id}")
    async def delete_thread(thread_id: str) -> dict[str, str]:
        try:
            deleted = await asyncio.to_thread(engine.store.delete_thread, thread_id)
        except KeyError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc
        except ThreadActiveConflict as exc:
            raise HTTPException(status_code=409, detail=str(exc)) from exc
        return {"deleted": deleted}

    def codegraph_client_for_thread(thread_id: str) -> tuple[dict[str, Any], CodeGraphClient]:
        thread = engine.store.get_thread(thread_id)
        if not thread["workspace"]:
            raise ValueError("thread workspace is not configured")
        return thread, CodeGraphClient(thread["workspace"])

    @router.get("/threads/{thread_id}/codegraph/status")
    async def codegraph_status(thread_id: str) -> dict[str, Any]:
        try:
            thread, client = await asyncio.to_thread(codegraph_client_for_thread, thread_id)
            status = await asyncio.to_thread(client.status)
        except KeyError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc
        except ValueError as exc:
            raise HTTPException(status_code=409, detail=str(exc)) from exc
        return {
            "status": status,
            "capabilities": normalize_codegraph_capabilities(thread.get("capabilities")),
        }

    @router.post("/threads/{thread_id}/codegraph/init")
    async def codegraph_init(thread_id: str) -> dict[str, Any]:
        try:
            _thread, client = await asyncio.to_thread(codegraph_client_for_thread, thread_id)
            result = await asyncio.to_thread(client.init)
            status = await asyncio.to_thread(client.status)
        except KeyError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc
        except ValueError as exc:
            raise HTTPException(status_code=409, detail=str(exc)) from exc
        if result.returncode != 0:
            raise HTTPException(status_code=503, detail=result.output.strip() or "CodeGraph init failed")
        return {"status": status, "output": result.output.strip()}

    @router.post("/threads/{thread_id}/codegraph/sync")
    async def codegraph_sync(thread_id: str) -> dict[str, Any]:
        try:
            _thread, client = await asyncio.to_thread(codegraph_client_for_thread, thread_id)
            result = await asyncio.to_thread(client.sync)
            status = await asyncio.to_thread(client.status)
        except KeyError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc
        except ValueError as exc:
            raise HTTPException(status_code=409, detail=str(exc)) from exc
        if result.returncode != 0:
            raise HTTPException(status_code=503, detail=result.output.strip() or "CodeGraph sync failed")
        return {"status": status, "output": result.output.strip()}

    @router.post("/threads/{thread_id}/codegraph/visualize")
    async def codegraph_visualize(
        thread_id: str,
        request: CodeGraphVisualizeRequest,
    ) -> dict[str, Any]:
        try:
            thread, client = await asyncio.to_thread(codegraph_client_for_thread, thread_id)
            artifact_root = Path(thread["workspace"]) / ".code-agent" / "codegraph"
            output = artifact_root / codegraph_artifact_name(thread_id)
            data = await asyncio.to_thread(
                client.visualize,
                output,
                keyword=request.keyword,
                depth=request.depth,
                max_nodes=request.max_nodes,
            )
        except KeyError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc
        except (FileNotFoundError, ValueError, sqlite3.Error) as exc:
            raise HTTPException(status_code=409, detail=str(exc)) from exc
        return {
            "node_count": data["node_count"],
            "edge_count": data["edge_count"],
            "url": f"/api/agent/threads/{thread_id}/codegraph/visualization",
        }

    @router.get("/threads/{thread_id}/codegraph/visualization")
    async def codegraph_visualization(thread_id: str) -> FileResponse:
        try:
            thread, _client = await asyncio.to_thread(codegraph_client_for_thread, thread_id)
        except KeyError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc
        except ValueError as exc:
            raise HTTPException(status_code=409, detail=str(exc)) from exc
        output = (
            Path(thread["workspace"])
            / ".code-agent"
            / "codegraph"
            / codegraph_artifact_name(thread_id)
        )
        if not output.is_file():
            raise HTTPException(status_code=404, detail="CodeGraph visualization has not been generated")
        return FileResponse(output, media_type="text/html")

    @router.get("/threads/{thread_id}/messages")
    async def list_thread_messages(
        thread_id: str,
        limit: int = Query(default=500, ge=1, le=2_000),
    ) -> dict[str, Any]:
        try:
            messages = await asyncio.to_thread(
                engine.store.list_conversation_messages,
                thread_id,
                limit=limit,
            )
        except KeyError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc
        return {"messages": messages}

    @router.post("/threads/{thread_id}/messages")
    async def create_thread_message(
        thread_id: str,
        request: CreateConversationMessageRequest,
    ) -> dict[str, Any]:
        try:
            thread = await asyncio.to_thread(engine.store.get_thread, thread_id)
            if not thread["workspace"]:
                raise ValueError("thread workspace is not configured")
            authorized_paths = list(
                dict.fromkeys([*thread["authorized_paths"], *request.authorized_paths])
            )
            if authorized_paths != thread["authorized_paths"]:
                thread = await asyncio.to_thread(
                    engine.store.update_thread,
                    thread_id,
                    authorized_paths=authorized_paths,
                    expected_version=thread["version"],
                )
            if request.context_items != thread["context_items"]:
                thread = await asyncio.to_thread(
                    engine.store.update_thread,
                    thread_id,
                    context_items=request.context_items,
                    expected_version=thread["version"],
                )
            if request.capabilities is not None:
                capabilities = normalize_codegraph_capabilities(request.capabilities)
                if capabilities != normalize_codegraph_capabilities(thread.get("capabilities")):
                    thread = await asyncio.to_thread(
                        engine.store.update_thread,
                        thread_id,
                        capabilities=capabilities,
                        expected_version=thread["version"],
                    )
            budget = RunBudget.from_dict(request.budget or thread["budget"])
            run_id = await asyncio.to_thread(
                engine.create_run,
                thread["workspace"],
                request.content,
                budget,
                thread_id,
                request.target_files,
                authorized_paths,
                thread["capabilities"],
                request.api_key,
            )
        except KeyError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc
        except (ValueError, VersionConflict) as exc:
            raise HTTPException(status_code=409, detail=str(exc)) from exc
        return {"run_id": run_id, "state": engine.get_state(run_id)}

    @router.post("/context/resolve")
    async def resolve_context(request: ContextResolveRequest) -> dict[str, Any]:
        try:
            thread = await asyncio.to_thread(engine.store.get_thread, request.thread_id)
        except KeyError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc
        if not thread["workspace"]:
            raise HTTPException(status_code=409, detail="thread workspace is not configured")
        workspace = Path(thread["workspace"]).resolve()
        raw = request.value.strip()
        candidate = Path(raw).expanduser()
        matches: list[dict[str, Any]] = []
        if candidate.is_absolute():
            resolved = candidate.resolve()
            if not resolved.exists():
                raise HTTPException(status_code=404, detail=f"path does not exist: {raw}")
            try:
                display = resolved.relative_to(workspace).as_posix()
                external = False
            except ValueError:
                display = str(resolved)
                external = True
                authorized = list(dict.fromkeys([*thread["authorized_paths"], str(resolved)]))
                try:
                    thread = await asyncio.to_thread(
                        engine.store.update_thread,
                        request.thread_id,
                        authorized_paths=authorized,
                        expected_version=thread["version"],
                    )
                except VersionConflict as exc:
                    raise HTTPException(status_code=409, detail=str(exc)) from exc
            matches.append(
                {
                    "path": display,
                    "absolute_path": str(resolved),
                    "external": external,
                    "type": "directory" if resolved.is_dir() else "file",
                }
            )
        else:
            query = raw.removeprefix("@").casefold()
            if not query:
                return {"matches": []}
            ignored = {".git", ".venv", "venv", "__pycache__", "node_modules", "dist", "build"}
            for path in sorted(workspace.rglob("*"), key=lambda item: item.as_posix().casefold()):
                relative = path.relative_to(workspace)
                if any(part in ignored for part in relative.parts):
                    continue
                if query not in relative.as_posix().casefold():
                    continue
                matches.append(
                    {
                        "path": relative.as_posix(),
                        "absolute_path": str(path.resolve()),
                        "external": False,
                        "type": "directory" if path.is_dir() else "file",
                    }
                )
                if len(matches) >= request.limit:
                    break
        return {"matches": matches, "thread": thread}

    @router.post("/runs")
    async def create_run(request: CreateRunRequest) -> dict[str, Any]:
        try:
            run_id = await asyncio.to_thread(
                engine.create_run,
                request.workspace,
                request.goal,
                RunBudget.from_dict(request.budget),
                request.thread_id,
                request.target_files,
                request.authorized_paths,
                request.capabilities,
                request.api_key,
            )
        except (ValueError, FileNotFoundError, KeyError) as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        return {"run_id": run_id, "state": engine.get_state(run_id)}

    @router.get("/runs")
    async def list_runs(limit: int = Query(default=100, ge=1, le=500)) -> dict[str, Any]:
        return {"runs": await asyncio.to_thread(engine.store.list_runs, limit)}

    @router.get("/runs/{run_id}")
    async def get_run(run_id: str) -> dict[str, Any]:
        try:
            return await asyncio.to_thread(engine.get_state, run_id)
        except KeyError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc

    @router.patch("/runs/{run_id}/budget")
    async def update_run_budget(run_id: str, request: BudgetPatchRequest) -> dict[str, Any]:
        try:
            updated = await asyncio.to_thread(engine.update_budget, run_id, request.budget)
            run = await asyncio.to_thread(engine.store.get_run, run_id)
            if run.get("thread_id"):
                thread = await asyncio.to_thread(engine.store.get_thread, run["thread_id"])
                await asyncio.to_thread(
                    engine.store.update_thread,
                    run["thread_id"],
                    budget=updated["budget"],
                    expected_version=thread["version"],
                )
        except KeyError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc
        except (ValueError, VersionConflict) as exc:
            raise HTTPException(status_code=409, detail=str(exc)) from exc
        return updated

    @router.post("/runs/{run_id}/step")
    async def step_run(run_id: str) -> dict[str, Any]:
        try:
            return await asyncio.to_thread(engine.step, run_id)
        except KeyError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc

    @router.post("/runs/{run_id}/run")
    async def run_until_pause(run_id: str) -> dict[str, Any]:
        try:
            return await asyncio.to_thread(engine.run, run_id)
        except KeyError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc

    @router.post("/runs/{run_id}/approve")
    async def approve_run(run_id: str, request: ApprovalRequest) -> dict[str, Any]:
        try:
            return await asyncio.to_thread(engine.approve, run_id, request.risk)
        except KeyError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc

    @router.post("/runs/{run_id}/reject")
    async def reject_run(run_id: str) -> dict[str, Any]:
        try:
            return await asyncio.to_thread(engine.reject, run_id)
        except KeyError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc
        except ValueError as exc:
            raise HTTPException(status_code=409, detail=str(exc)) from exc

    @router.post("/runs/{run_id}/pause")
    async def pause_run(run_id: str) -> dict[str, Any]:
        try:
            return await asyncio.to_thread(engine.pause, run_id)
        except KeyError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc
        except ValueError as exc:
            raise HTTPException(status_code=409, detail=str(exc)) from exc

    @router.post("/runs/{run_id}/resume")
    async def resume_run(run_id: str) -> dict[str, Any]:
        try:
            return await asyncio.to_thread(engine.resume, run_id)
        except KeyError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc
        except ValueError as exc:
            raise HTTPException(status_code=409, detail=str(exc)) from exc

    @router.post("/runs/{run_id}/cancel")
    async def cancel_run(run_id: str) -> dict[str, Any]:
        try:
            return await asyncio.to_thread(engine.cancel, run_id)
        except KeyError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc
        except ValueError as exc:
            raise HTTPException(status_code=409, detail=str(exc)) from exc

    @router.get("/runs/{run_id}/events")
    async def get_events(run_id: str, after: int = Query(default=0, ge=0)) -> dict[str, Any]:
        try:
            engine.store.get_run(run_id)
        except KeyError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc
        events = await asyncio.to_thread(engine.store.list_events, run_id, after)
        return {"events": [event.to_dict() for event in events]}

    @router.get("/runs/{run_id}/events.ndjson")
    async def stream_events(run_id: str, after: int = Query(default=0, ge=0)) -> StreamingResponse:
        try:
            engine.store.get_run(run_id)
        except KeyError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc

        async def generate() -> AsyncIterator[str]:
            events = await asyncio.to_thread(engine.store.list_events, run_id, after)
            for event in events:
                yield json.dumps(event.to_dict(), ensure_ascii=False) + "\n"

        return StreamingResponse(generate(), media_type="application/x-ndjson")

    if memory_store is not None:
        @router.post("/memory-proposals/from-selection")
        async def create_memory_proposals_from_selection(
            request: MemoryProposalSelectionRequest,
        ) -> dict[str, Any]:
            assert proposal_service is not None
            try:
                reviews = await asyncio.to_thread(
                    proposal_service.create_from_selection,
                    **request.model_dump(),
                )
            except KeyError as exc:
                raise HTTPException(status_code=404, detail=str(exc)) from exc
            except ProposalValidationError as exc:
                raise HTTPException(status_code=422, detail=str(exc)) from exc
            except ContextHardLimitExceeded as exc:
                raise HTTPException(status_code=413, detail=str(exc)) from exc
            except MemoryProposalGenerationError as exc:
                logger.warning("memory proposal generation failed: %s", exc)
                raise HTTPException(status_code=502, detail=str(exc)) from exc
            return {"proposals": reviews, "count": len(reviews)}

        @router.get("/memory-proposals")
        async def list_memory_proposals(
            status: str | None = None,
            project_id: str | None = None,
            thread_id: str | None = None,
        ) -> dict[str, Any]:
            assert proposal_service is not None
            reviews = await asyncio.to_thread(
                proposal_service.list_reviews,
                status=status,
                project_id=project_id,
                thread_id=thread_id,
            )
            return {"proposals": reviews}

        @router.get("/memory-proposals/{proposal_id}/review")
        async def get_memory_proposal_review(proposal_id: str) -> dict[str, Any]:
            assert proposal_service is not None
            try:
                review = await asyncio.to_thread(proposal_service.get_review, proposal_id)
            except KeyError as exc:
                raise HTTPException(status_code=404, detail=str(exc)) from exc
            return {"proposal": review}

        @router.get("/memory-proposals/{proposal_id}/evidence")
        async def get_memory_proposal_evidence(proposal_id: str) -> dict[str, Any]:
            try:
                evidence = await asyncio.to_thread(
                    memory_store.get_proposal_evidence, proposal_id
                )
            except KeyError as exc:
                raise HTTPException(status_code=404, detail=str(exc)) from exc
            return {"evidence": evidence}

        @router.patch("/memory-proposals/{proposal_id}")
        async def update_memory_proposal(
            proposal_id: str,
            request: MemoryProposalUpdateRequest,
        ) -> dict[str, Any]:
            assert proposal_service is not None
            values = request.model_dump(exclude_unset=True)
            expected_version = values.pop("expected_version")
            try:
                review = await asyncio.to_thread(
                    proposal_service.update_review,
                    proposal_id,
                    expected_version=expected_version,
                    **values,
                )
            except KeyError as exc:
                raise HTTPException(status_code=404, detail=str(exc)) from exc
            except (MemoryProposalConflict, ValueError) as exc:
                raise HTTPException(status_code=409, detail=str(exc)) from exc
            return {"proposal": review}

        @router.post("/memory-proposals/{proposal_id}/approve")
        async def approve_memory_proposal(
            proposal_id: str,
            request: MemoryProposalActionRequest,
        ) -> dict[str, Any]:
            assert proposal_service is not None
            try:
                review = await asyncio.to_thread(
                    proposal_service.approve,
                    proposal_id,
                    expected_version=request.expected_version,
                )
            except KeyError as exc:
                raise HTTPException(status_code=404, detail=str(exc)) from exc
            except MemoryProposalConflict as exc:
                raise HTTPException(status_code=409, detail=str(exc)) from exc
            return {"proposal": review}

        @router.post("/memory-proposals/{proposal_id}/reject")
        async def reject_memory_proposal(
            proposal_id: str,
            request: MemoryProposalActionRequest,
        ) -> dict[str, Any]:
            assert proposal_service is not None
            try:
                review = await asyncio.to_thread(
                    proposal_service.reject,
                    proposal_id,
                    expected_version=request.expected_version,
                    reason=request.reason,
                )
            except KeyError as exc:
                raise HTTPException(status_code=404, detail=str(exc)) from exc
            except MemoryProposalConflict as exc:
                raise HTTPException(status_code=409, detail=str(exc)) from exc
            return {"proposal": review}

        @router.post("/memory-proposals/{proposal_id}/defer")
        async def defer_memory_proposal(
            proposal_id: str,
            request: MemoryProposalActionRequest,
        ) -> dict[str, Any]:
            assert proposal_service is not None
            try:
                review = await asyncio.to_thread(
                    proposal_service.defer,
                    proposal_id,
                    expected_version=request.expected_version,
                    reason=request.reason,
                )
            except KeyError as exc:
                raise HTTPException(status_code=404, detail=str(exc)) from exc
            except MemoryProposalConflict as exc:
                raise HTTPException(status_code=409, detail=str(exc)) from exc
            return {"proposal": review}

        @router.get("/memories")
        async def list_memories(
            status: str | None = None,
            project_id: str | None = None,
            thread_id: str | None = None,
        ) -> dict[str, Any]:
            return {
                "memories": await asyncio.to_thread(
                    memory_store.list,
                    status=status,
                    project_id=project_id,
                    thread_id=thread_id,
                )
            }

        @router.post("/memories")
        async def create_memory(request: MemoryCandidateRequest) -> dict[str, Any]:
            memory_id = await asyncio.to_thread(memory_store.create_candidate, **request.model_dump())
            return {"memory_id": memory_id, "memory": memory_store.get(memory_id)}

        @router.post("/memories/{memory_id}/activate")
        async def activate_memory(memory_id: str) -> dict[str, Any]:
            try:
                await asyncio.to_thread(memory_store.activate, memory_id)
            except KeyError as exc:
                raise HTTPException(status_code=404, detail=str(exc)) from exc
            return {"memory": memory_store.get(memory_id)}

        @router.post("/memories/{memory_id}/reject")
        async def reject_memory(memory_id: str) -> dict[str, Any]:
            try:
                await asyncio.to_thread(memory_store.reject, memory_id)
            except KeyError as exc:
                raise HTTPException(status_code=404, detail=str(exc)) from exc
            return {"memory": memory_store.get(memory_id)}

        @router.put("/memories/{memory_id}")
        async def update_memory(memory_id: str, request: MemoryUpdateRequest) -> dict[str, Any]:
            try:
                await asyncio.to_thread(memory_store.update, memory_id, **request.model_dump())
            except KeyError as exc:
                raise HTTPException(status_code=404, detail=str(exc)) from exc
            return {"memory": memory_store.get(memory_id)}

        @router.delete("/memories/{memory_id}")
        async def delete_memory(memory_id: str) -> dict[str, Any]:
            if not await asyncio.to_thread(memory_store.delete, memory_id):
                raise HTTPException(status_code=404, detail="memory not found")
            return {"deleted": memory_id}

    return router


def build_default_agent_router() -> APIRouter:
    code_agent_root = Path(__file__).resolve().parents[1]
    db_path = Path(os.environ.get("CODE_AGENT_DB", code_agent_root / "outputs" / "agent_runtime.db"))
    store = EventStore(db_path)
    memory = MemoryStore(db_path)
    context_runtime = build_default_context_runtime(code_agent_root)
    gateway = OpenAICompatibleGateway(
        model=context_runtime.profile.model,
        base_url=os.environ.get("CODE_AGENT_API_BASE", "https://api.deepseek.com/v1"),
        serializer=context_runtime.serializer,
    )
    planner = ContextPlanner(
        context_runtime.counter,
        context_runtime.serializer,
        context_runtime.profile,
    )
    compaction_service = CompactionService(
        store=store,
        gateway=gateway,
        counter=context_runtime.counter,
        serializer=context_runtime.serializer,
        profile=context_runtime.profile,
    )
    engine = RunEngine(
        store,
        build_default_registry(include_mutating=True),
        gateway,
        memory_store=memory,
        context_planner=planner,
        compaction_service=compaction_service,
    )
    return create_agent_router(engine, memory)
