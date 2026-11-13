from __future__ import annotations

import asyncio
from dataclasses import dataclass
import json
import logging
import os
import sqlite3
from pathlib import Path
from typing import Any, AsyncIterator

import psutil
from fastapi import APIRouter, HTTPException, Query, Request
from fastapi.responses import FileResponse, StreamingResponse
from pydantic import BaseModel, Field

from code_agent.agent_core.contracts import RiskLevel, RunBudget
from code_agent.agent_core.compaction import CompactionService
from code_agent.agent_core.context_builder import ContextPlanner
from code_agent.agent_core.event_store import EventStore, ThreadActiveConflict, VersionConflict
from code_agent.agent_core.memory_proposals import (
    AutoMemoryWorker,
    MemoryProposalGenerationError,
    MemoryProposalService,
    ProposalValidationError,
)
from code_agent.agent_core.memory_store import MemoryProposalConflict, MemoryStore
from code_agent.agent_core.model_gateway import (
    DeepSeekRequestSerializer,
    RoutedModelGateway,
    RuntimeModelConfig,
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


# [maint-note:380c997e72a6]
# 数据与行为导航：DefaultContextRuntime。
# 字段 profile：DeepSeekModelProfile。
# 字段 counter：DeepSeekTokenCounter。
# 字段 serializer：DeepSeekRequestSerializer。
# 字段 default_model_config：RuntimeModelConfig。
# [/maint-note:380c997e72a6]
@dataclass(frozen=True)
class DefaultContextRuntime:
    profile: DeepSeekModelProfile
    counter: DeepSeekTokenCounter
    serializer: DeepSeekRequestSerializer
    default_model_config: RuntimeModelConfig


# [maint-note:0a4b89a9d199]
# 维护导航：get_default_runtime_model_config；下列为本函数源码中的接口与路径索引。
# 返回契约：RuntimeModelConfig。
# 分支条件：provider.strip().lower() in {'openrouter', 'ollama'} and configured_base_url == 'https://api.deepseek.com/v1'。
# 分支条件：'CODE_AGENT_CONTEXT_WINDOW_TOKENS' in os.environ。
# 分支条件：configured_model。
# 分支条件：'CODE_AGENT_CONTEXT_SAFETY_MARGIN_TOKENS' in os.environ。
# 返回路径：RuntimeModelConfig.from_dict(config_source)。
# 写入目标：config_source['context_window_tokens']。
# 写入目标：config_source['model']。
# 写入目标：config_source['safety_margin_tokens']。
# 直接调用：os.environ.get。
# 直接调用：os.environ.get('CODE_AGENT_API_BASE', '').strip。
# 直接调用：provider.strip().lower。
# 直接调用：provider.strip。
# 直接调用：os.environ.get('CODE_AGENT_MODEL', '').strip。
# 直接调用：int。
# 直接调用：RuntimeModelConfig.from_dict。
# [/maint-note:0a4b89a9d199]
def get_default_runtime_model_config() -> RuntimeModelConfig:
    """Read the process-wide default without constructing tokenizer state.

    The web UI calls this for its bootstrap response, while the Agent Runtime
    snapshots the same configuration when a Thread or Run is created.
    """
    provider = os.environ.get("CODE_AGENT_PROVIDER", "deepseek")
    configured_base_url = os.environ.get("CODE_AGENT_API_BASE", "").strip()
    if provider.strip().lower() in {"openrouter", "ollama"} and configured_base_url == "https://api.deepseek.com/v1":
        configured_base_url = ""
    configured_model = os.environ.get("CODE_AGENT_MODEL", "").strip()
    config_source: dict[str, Any] = {
        "provider": provider,
        "base_url": configured_base_url,
    }
    if "CODE_AGENT_CONTEXT_WINDOW_TOKENS" in os.environ:
        config_source["context_window_tokens"] = int(os.environ["CODE_AGENT_CONTEXT_WINDOW_TOKENS"])
    if configured_model:
        config_source["model"] = configured_model
    if "CODE_AGENT_CONTEXT_SAFETY_MARGIN_TOKENS" in os.environ:
        config_source["safety_margin_tokens"] = int(
            os.environ["CODE_AGENT_CONTEXT_SAFETY_MARGIN_TOKENS"]
        )
    return RuntimeModelConfig.from_dict(config_source)


# [maint-note:2dbba1c6aa76]
# 维护导航：get_runtime_provider_defaults；下列为本函数源码中的接口与路径索引。
# 返回契约：dict[str, dict[str, Any]]。
# 返回路径：{provider: RuntimeModelConfig.from_dict({'provider': provider}).to_dict() for provider in ('deepseek', 'openrouter', 'ollama')}。
# 直接调用：RuntimeModelConfig.from_dict({'provider': provider}).to_dict。
# 直接调用：RuntimeModelConfig.from_dict。
# 集合派生：provider 来自 ('deepseek', 'openrouter', 'ollama')。
# [/maint-note:2dbba1c6aa76]
def get_runtime_provider_defaults() -> dict[str, dict[str, Any]]:
    """Return safe, key-free model defaults used to populate the UI."""
    return {
        provider: RuntimeModelConfig.from_dict({"provider": provider}).to_dict()
        for provider in ("deepseek", "openrouter", "ollama")
    }


# [maint-note:a3b80adada6e]
# 维护导航：build_default_context_runtime；下列为本函数源码中的接口与路径索引。
# 输入 code_agent_root：str | Path。
# 返回契约：DefaultContextRuntime。
# 返回路径：DefaultContextRuntime(profile, counter, serializer, default_model_config)。
# 直接调用：Path(code_agent_root).expanduser().resolve。
# 直接调用：Path(code_agent_root).expanduser。
# 直接调用：Path。
# 直接调用：get_default_runtime_model_config。
# 直接调用：DeepSeekModelProfile。
# 直接调用：int。
# 直接调用：os.environ.get。
# 直接调用：float。
# 直接调用：str。
# 直接调用：DeepSeekRequestSerializer。
# 直接调用：DeepSeekTokenCounter.from_directory。
# 直接调用：DefaultContextRuntime。
# [/maint-note:a3b80adada6e]
def build_default_context_runtime(code_agent_root: str | Path) -> DefaultContextRuntime:
    root = Path(code_agent_root).expanduser().resolve()
    default_model_config = get_default_runtime_model_config()
    profile = DeepSeekModelProfile(
        model=default_model_config.model,
        context_window_tokens=default_model_config.context_window_tokens,
        default_output_reserve_tokens=int(
            os.environ.get("CODE_AGENT_OUTPUT_RESERVE_TOKENS", "4096")
        ),
        safety_margin_tokens=default_model_config.safety_margin_tokens,
        provider_framing_tokens=int(
            os.environ.get("CODE_AGENT_PROVIDER_FRAMING_TOKENS", "256")
        ),
        compaction_trigger_ratio=float(
            os.environ.get("CODE_AGENT_COMPACTION_TRIGGER_RATIO", "0.65")
        ),
        compaction_target_ratio=float(
            os.environ.get("CODE_AGENT_COMPACTION_TARGET_RATIO", "0.45")
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
    return DefaultContextRuntime(profile, counter, serializer, default_model_config)


# [maint-note:b678a1014827]
# 维护导航：normalize_runtime_model_config；下列为本函数源码中的接口与路径索引。
# 输入 value：dict[str, Any] | None。
# 输入 default：RuntimeModelConfig。
# 输入 legacy_model：str；缺省表达式 ''。
# 返回契约：dict[str, Any]。
# 分支条件：legacy_model and (not source.get('model')) and (not (source.get('provider') == 'ollama' and default.provider != 'ollama'))。
# 返回路径：config.to_dict()。
# 写入目标：source['model']。
# 直接调用：dict。
# 直接调用：source.get。
# 直接调用：RuntimeModelConfig.from_dict。
# 直接调用：config.to_dict。
# [/maint-note:b678a1014827]
def normalize_runtime_model_config(
    value: dict[str, Any] | None,
    default: RuntimeModelConfig,
    *,
    legacy_model: str = "",
) -> dict[str, Any]:
    source = dict(value or {})
    if (legacy_model and not source.get("model")
            and not (source.get("provider") == "ollama" and default.provider != "ollama")):
        source["model"] = legacy_model
    config = RuntimeModelConfig.from_dict(
        source,
        default_provider=default.provider,
        default_model=default.model,
        default_base_url=default.base_url,
        default_context_window_tokens=default.context_window_tokens,
        default_safety_margin_tokens=default.safety_margin_tokens,
    )
    return config.to_dict()


# [maint-note:692dff015ee4]
# 数据与行为导航：CreateRunRequest。
# 继承接口：BaseModel。
# 字段 workspace：str。
# 字段 goal：str；初始表达式 Field(min_length=1)。
# 字段 target_files：list[str]；初始表达式 Field(default_factory=list)。
# 字段 authorized_paths：list[str]；初始表达式 Field(default_factory=list)。
# 字段 budget：dict[str, Any]；初始表达式 Field(default_factory=dict)。
# 字段 thread_id：str | None；初始表达式 None。
# 字段 capabilities：dict[str, Any]；初始表达式 Field(default_factory=dict)。
# 字段 api_key：str | None；初始表达式 None。
# 字段 runtime_model_config：dict[str, Any] | None；初始表达式 None。
# [/maint-note:692dff015ee4]
class CreateRunRequest(BaseModel):
    workspace: str
    goal: str = Field(min_length=1)
    target_files: list[str] = Field(default_factory=list)
    authorized_paths: list[str] = Field(default_factory=list)
    budget: dict[str, Any] = Field(default_factory=dict)
    thread_id: str | None = None
    capabilities: dict[str, Any] = Field(default_factory=dict)
    api_key: str | None = None
    runtime_model_config: dict[str, Any] | None = None


# [maint-note:d159e7542362]
# 数据与行为导航：CreateThreadRequest。
# 继承接口：BaseModel。
# 字段 title：str；初始表达式 Field(min_length=1, max_length=200)。
# 字段 workspace：str | None；初始表达式 None。
# 字段 model：str；初始表达式 ''。
# 字段 runtime_model_config：dict[str, Any] | None；初始表达式 None。
# 字段 runtime_mode：str；初始表达式 'agent'。
# 字段 budget：dict[str, Any]；初始表达式 Field(default_factory=dict)。
# 字段 authorized_paths：list[str]；初始表达式 Field(default_factory=list)。
# 字段 context_items：list[dict[str, Any]]；初始表达式 Field(default_factory=list)。
# 字段 capabilities：dict[str, Any]；初始表达式 Field(default_factory=dict)。
# [/maint-note:d159e7542362]
class CreateThreadRequest(BaseModel):
    title: str = Field(min_length=1, max_length=200)
    workspace: str | None = None
    model: str = ""
    runtime_model_config: dict[str, Any] | None = None
    runtime_mode: str = "agent"
    budget: dict[str, Any] = Field(default_factory=dict)
    authorized_paths: list[str] = Field(default_factory=list)
    context_items: list[dict[str, Any]] = Field(default_factory=list)
    capabilities: dict[str, Any] = Field(default_factory=dict)


# [maint-note:36621066f080]
# 数据与行为导航：UpdateThreadRequest。
# 继承接口：BaseModel。
# 字段 title：str | None；初始表达式 Field(default=None, min_length=1, max_length=200)。
# 字段 workspace：str | None；初始表达式 None。
# 字段 model：str | None；初始表达式 None。
# 字段 runtime_model_config：dict[str, Any] | None；初始表达式 None。
# 字段 runtime_mode：str | None；初始表达式 None。
# 字段 budget：dict[str, Any] | None；初始表达式 None。
# 字段 authorized_paths：list[str] | None；初始表达式 None。
# 字段 context_items：list[dict[str, Any]] | None；初始表达式 None。
# 字段 capabilities：dict[str, Any] | None；初始表达式 None。
# 字段 summary：str | None；初始表达式 None。
# 字段 expected_version：int | None；初始表达式 Field(default=None, ge=0)。
# [/maint-note:36621066f080]
class UpdateThreadRequest(BaseModel):
    title: str | None = Field(default=None, min_length=1, max_length=200)
    workspace: str | None = None
    model: str | None = None
    runtime_model_config: dict[str, Any] | None = None
    runtime_mode: str | None = None
    budget: dict[str, Any] | None = None
    authorized_paths: list[str] | None = None
    context_items: list[dict[str, Any]] | None = None
    capabilities: dict[str, Any] | None = None
    summary: str | None = None
    expected_version: int | None = Field(default=None, ge=0)


# [maint-note:86a88de98449]
# 数据与行为导航：CreateConversationMessageRequest。
# 继承接口：BaseModel。
# 字段 content：str；初始表达式 Field(min_length=1)。
# 字段 target_files：list[str]；初始表达式 Field(default_factory=list)。
# 字段 authorized_paths：list[str]；初始表达式 Field(default_factory=list)。
# 字段 context_items：list[dict[str, Any]]；初始表达式 Field(default_factory=list)。
# 字段 budget：dict[str, Any] | None；初始表达式 None。
# 字段 capabilities：dict[str, Any] | None；初始表达式 None。
# 字段 api_key：str | None；初始表达式 None。
# [/maint-note:86a88de98449]
class CreateConversationMessageRequest(BaseModel):
    content: str = Field(min_length=1)
    target_files: list[str] = Field(default_factory=list)
    authorized_paths: list[str] = Field(default_factory=list)
    context_items: list[dict[str, Any]] = Field(default_factory=list)
    budget: dict[str, Any] | None = None
    capabilities: dict[str, Any] | None = None
    api_key: str | None = None


# [maint-note:559b65bb14c8]
# 数据与行为导航：CodeGraphVisualizeRequest。
# 继承接口：BaseModel。
# 字段 keyword：str | None；初始表达式 Field(default=None, max_length=500)。
# 字段 depth：int；初始表达式 Field(default=2, ge=0, le=5)。
# 字段 max_nodes：int；初始表达式 Field(default=150, ge=10, le=500)。
# [/maint-note:559b65bb14c8]
class CodeGraphVisualizeRequest(BaseModel):
    keyword: str | None = Field(default=None, max_length=500)
    depth: int = Field(default=2, ge=0, le=5)
    max_nodes: int = Field(default=150, ge=10, le=500)


# [maint-note:cf1edcc1142c]
# 数据与行为导航：ContextResolveRequest。
# 继承接口：BaseModel。
# 字段 thread_id：str。
# 字段 value：str；初始表达式 Field(min_length=1)。
# 字段 limit：int；初始表达式 Field(default=20, ge=1, le=100)。
# [/maint-note:cf1edcc1142c]
class ContextResolveRequest(BaseModel):
    thread_id: str
    value: str = Field(min_length=1)
    limit: int = Field(default=20, ge=1, le=100)


# [maint-note:534f167f9fe7]
# 数据与行为导航：BudgetPatchRequest。
# 继承接口：BaseModel。
# 字段 budget：dict[str, Any]。
# [/maint-note:534f167f9fe7]
class BudgetPatchRequest(BaseModel):
    budget: dict[str, Any]


# [maint-note:9ea6f091b3e3]
# 数据与行为导航：ApprovalRequest。
# 继承接口：BaseModel。
# 字段 risk：RiskLevel。
# 字段 tool_call_id：str | None；初始表达式 None。
# [/maint-note:9ea6f091b3e3]
class ApprovalRequest(BaseModel):
    risk: RiskLevel
    tool_call_id: str | None = None


# [maint-note:24ba3027434b]
# 数据与行为导航：MemoryCandidateRequest。
# 继承接口：BaseModel。
# 字段 scope：str。
# 字段 kind：str。
# 字段 project_id：str | None；初始表达式 None。
# 字段 thread_id：str | None；初始表达式 None。
# 字段 subject：str。
# 字段 content：str。
# 字段 source_run_id：str | None；初始表达式 None。
# 字段 source_revision：str | None；初始表达式 None。
# 字段 confidence：float；初始表达式 Field(ge=0, le=1)。
# [/maint-note:24ba3027434b]
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


# [maint-note:9ab45e7b4c08]
# 数据与行为导航：MemoryUpdateRequest。
# 继承接口：BaseModel。
# 字段 subject：str；初始表达式 Field(min_length=1)。
# 字段 content：str；初始表达式 Field(min_length=1)。
# 字段 confidence：float；初始表达式 Field(ge=0, le=1)。
# [/maint-note:9ab45e7b4c08]
class MemoryUpdateRequest(BaseModel):
    subject: str = Field(min_length=1)
    content: str = Field(min_length=1)
    confidence: float = Field(ge=0, le=1)


# [maint-note:1c33e4ee42e9]
# 数据与行为导航：MemoryProposalSelectionRequest。
# 继承接口：BaseModel。
# 字段 thread_id：str；初始表达式 Field(min_length=1)。
# 字段 project_id：str | None；初始表达式 None。
# 字段 evidence_refs：list[dict[str, Any]]；初始表达式 Field(min_length=1, max_length=50)。
# [/maint-note:1c33e4ee42e9]
class MemoryProposalSelectionRequest(BaseModel):
    thread_id: str = Field(min_length=1)
    project_id: str | None = None
    evidence_refs: list[dict[str, Any]] = Field(min_length=1, max_length=50)


# [maint-note:37f2c2504304]
# 数据与行为导航：MemoryProposalUpdateRequest。
# 继承接口：BaseModel。
# 字段 expected_version：int；初始表达式 Field(ge=0)。
# 字段 subject：str | None；初始表达式 Field(default=None, min_length=1, max_length=200)。
# 字段 canonical_content：str | None；初始表达式 Field(default=None, min_length=1, max_length=4000)。
# 字段 scope：str | None；初始表达式 None。
# 字段 kind：str | None；初始表达式 None。
# 字段 expires_at：str | None；初始表达式 None。
# [/maint-note:37f2c2504304]
class MemoryProposalUpdateRequest(BaseModel):
    expected_version: int = Field(ge=0)
    subject: str | None = Field(default=None, min_length=1, max_length=200)
    canonical_content: str | None = Field(default=None, min_length=1, max_length=4_000)
    scope: str | None = None
    kind: str | None = None
    expires_at: str | None = None


# [maint-note:a5a216f969f6]
# 数据与行为导航：MemoryProposalActionRequest。
# 继承接口：BaseModel。
# 字段 expected_version：int；初始表达式 Field(ge=0)。
# 字段 reason：str；初始表达式 Field(default='', max_length=2000)。
# [/maint-note:a5a216f969f6]
class MemoryProposalActionRequest(BaseModel):
    expected_version: int = Field(ge=0)
    reason: str = Field(default="", max_length=2_000)


# [maint-note:fb9f0847a0ee]
# 维护导航：create_agent_router；下列为本函数源码中的接口与路径索引。
# 输入 engine：RunEngine。
# 输入 memory_store：MemoryStore | None；缺省表达式 None。
# 输入 auto_memory_mode：str；缺省表达式 'off'。
# 返回契约：APIRouter。
# 分支条件：auto_memory_mode not in {'off', 'review', 'auto'}。
# 分支条件：proposal_service is not None and auto_memory_mode != 'off'。
# 分支条件：values.get('capabilities') is not None。
# 分支条件：'runtime_model_config' in values。
# 分支条件：not thread['workspace']。
# 分支条件：result.returncode != 0。
# 分支条件：not output.is_file()。
# 分支条件：authorized_paths != thread['authorized_paths']。
# 分支条件：request.context_items != thread['context_items']。
# 分支条件：request.capabilities is not None。
# 分支条件：capabilities != normalize_codegraph_capabilities(thread.get('capabilities'))。
# 分支条件：candidate.is_absolute()。
# 分支条件：not resolved.exists()。
# 分支条件：not query。
# 分支条件：any((part in ignored for part in relative.parts))。
# 分支条件：query not in relative.as_posix().casefold()。
# 分支条件：len(matches) >= request.limit。
# 分支条件：run.get('thread_id')。
# 分支条件：outcome is None。
# 分支条件：outcome['status'] in {'queued', 'running'} and (not psutil.pid_exists(worker_pid))。
# 分支条件：memory_store is not None。
# 遍历边界：path ← sorted(workspace.rglob('*'), key=lambda item: item.as_posix().casefold())。
# 返回路径：{'thread_id': thread_id, 'thread': engine.store.get_thread(thread_id)}。
# 返回路径：{'threads': await asyncio.to_thread(engine.store.list_threads, limit)}。
# 返回路径：{'thread': thread, 'messages': messages, 'last_run': last_run}。
# 返回路径：{'thread': thread}。
# 返回路径：{'deleted': deleted}。
# 返回路径：(thread, CodeGraphClient(thread['workspace']))。
# 返回路径：{'status': status, 'capabilities': normalize_codegraph_capabilities(thread.get('capabilities'))}。
# 返回路径：{'status': status, 'output': result.output.strip()}。
# 返回路径：{'node_count': data['node_count'], 'edge_count': data['edge_count'], 'url': f'/api/agent/threads/{thread_id}/codegraph/visualization'}。
# 返回路径：FileResponse(output, media_type='text/html')。
# 返回路径：{'messages': messages}。
# 返回路径：{'run_id': run_id, 'state': engine.get_state(run_id)}。
# 返回路径：{'matches': []}。
# 返回路径：{'matches': matches, 'thread': thread}。
# 返回路径：{'runs': await asyncio.to_thread(engine.store.list_runs, limit)}。
# 返回路径：await asyncio.to_thread(engine.get_state, run_id)。
# 返回路径：updated。
# 返回路径：await asyncio.to_thread(engine.step, run_id)。
# 返回路径：await asyncio.to_thread(engine.run, run_id)。
# 返回路径：await asyncio.to_thread(engine.approve, run_id, request.risk, request.tool_call_id)。
# 返回路径：await asyncio.to_thread(engine.reject, run_id)。
# 返回路径：await asyncio.to_thread(engine.pause, run_id)。
# 返回路径：await asyncio.to_thread(engine.resume, run_id)。
# 返回路径：await asyncio.to_thread(engine.cancel, run_id)。
# 返回路径：{'events': [event.to_dict() for event in events]}。
# 返回路径：{'run_id': run_id, 'run_status': run['status'], 'status': status}。
# 返回路径：{'run_id': run_id, 'run_status': run['status'], **outcome}。
# 返回路径：StreamingResponse(generate(), media_type='application/x-ndjson', headers={'Cache-Control': 'no-cache', 'X-Accel-Buffering': 'no'})。
# 返回路径：router。
# 显式抛出：ValueError('CODE_AGENT_AUTO_MEMORY_MODE must be off, review, or auto')。
# 显式抛出：HTTPException(status_code=422, detail=str(exc))。
# 显式抛出：HTTPException(status_code=404, detail=str(exc))。
# 显式抛出：HTTPException(status_code=409, detail=str(exc))。
# 显式抛出：ValueError('thread workspace is not configured')。
# 显式抛出：HTTPException(status_code=503, detail=result.output.strip() or 'CodeGraph init failed')。
# 显式抛出：HTTPException(status_code=503, detail=result.output.strip() or 'CodeGraph sync failed')。
# 显式抛出：HTTPException(status_code=404, detail='CodeGraph visualization has not been generated')。
# 显式抛出：HTTPException(status_code=409, detail='thread workspace is not configured')。
# 显式抛出：HTTPException(status_code=404, detail=f'path does not exist: {raw}')。
# 显式抛出：HTTPException(status_code=400, detail=str(exc))。
# 捕获异常：ValueError。
# 捕获异常：KeyError。
# 捕获异常：(ValueError, VersionConflict)。
# 捕获异常：ThreadActiveConflict。
# 捕获异常：(FileNotFoundError, ValueError, sqlite3.Error)。
# 捕获异常：VersionConflict。
# 捕获异常：(ValueError, FileNotFoundError, KeyError)。
# 写入目标：engine.auto_memory。
# 写入目标：values['capabilities']。
# 写入目标：values['runtime_model_config']。
# 直接调用：ValueError。
# 直接调用：APIRouter。
# 直接调用：engine.store.migrate_legacy_runs。
# 直接调用：MemoryProposalService。
# 直接调用：AutoMemoryWorker。
# 直接调用：normalize_runtime_model_config。
# 直接调用：asyncio.to_thread。
# 直接调用：normalize_codegraph_capabilities。
# 直接调用：HTTPException。
# 直接调用：str。
# 直接调用：engine.store.get_thread。
# 直接调用：router.post。
# 直接调用：Query。
# 直接调用：router.get。
# 直接调用：thread.get。
# 直接调用：request.model_dump。
# 直接调用：values.pop。
# 直接调用：values.get。
# 直接调用：current.get。
# 直接调用：router.patch。
# 直接调用：router.delete。
# 直接调用：CodeGraphClient。
# 直接调用：result.output.strip。
# 直接调用：Path。
# 直接调用：codegraph_artifact_name。
# 直接调用：output.is_file。
# 直接调用：FileResponse。
# 直接调用：list。
# 直接调用：dict.fromkeys。
# 直接调用：RunBudget.from_dict。
# 直接调用：engine.get_state。
# 直接调用：Path(thread['workspace']).resolve。
# 直接调用：request.value.strip。
# 直接调用：Path(raw).expanduser。
# 直接调用：candidate.is_absolute。
# 直接调用：candidate.resolve。
# 直接调用：resolved.exists。
# 直接调用：resolved.relative_to(workspace).as_posix。
# 直接调用：resolved.relative_to。
# 直接调用：matches.append。
# 直接调用：resolved.is_dir。
# 直接调用：raw.removeprefix('@').casefold。
# 直接调用：raw.removeprefix。
# 直接调用：sorted。
# 直接调用：workspace.rglob。
# 直接调用：path.relative_to。
# 直接调用：any。
# 直接调用：relative.as_posix().casefold。
# 直接调用：relative.as_posix。
# 直接调用：path.resolve。
# 直接调用：path.is_dir。
# 直接调用：len。
# 直接调用：run.get。
# 直接调用：engine.store.get_run。
# 直接调用：event.to_dict。
# 直接调用：outcome.pop。
# 直接调用：psutil.pid_exists。
# 直接调用：outcome.update。
# 直接调用：StreamingResponse。
# 直接调用：generate。
# 表达式分流：memory_store is not None；真值与假值分支分别计算，不同时执行。
# [/maint-note:fb9f0847a0ee]
def create_agent_router(
    engine: RunEngine,
    memory_store: MemoryStore | None = None,
    *,
    auto_memory_mode: str = "off",
) -> APIRouter:
    if auto_memory_mode not in {"off", "review", "auto"}:
        raise ValueError("CODE_AGENT_AUTO_MEMORY_MODE must be off, review, or auto")
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
    if proposal_service is not None and auto_memory_mode != "off":
        engine.auto_memory = AutoMemoryWorker(
            proposal_service, auto_apply=auto_memory_mode == "auto"
        )

    # [maint-note:715f423c1839]
    # 维护导航：create_thread；下列为本函数源码中的接口与路径索引。
    # 输入 request：CreateThreadRequest。
    # 返回契约：dict[str, Any]。
    # 异步边界：调用方需等待协程；异常沿 await 链传播或被上层捕获。
    # 返回路径：{'thread_id': thread_id, 'thread': engine.store.get_thread(thread_id)}。
    # 显式抛出：HTTPException(status_code=422, detail=str(exc))。
    # 捕获异常：ValueError。
    # 直接调用：normalize_runtime_model_config。
    # 直接调用：asyncio.to_thread。
    # 直接调用：normalize_codegraph_capabilities。
    # 直接调用：HTTPException。
    # 直接调用：str。
    # 直接调用：engine.store.get_thread。
    # 等待边界：asyncio.to_thread(engine.store.create_thread, request.title, workspace=request.workspace, model=request.model or runtime_model_config['model'], runtime_model_config=runtime_model_config, runtime_mode=request.runtime_mode, budget=request.bud；异常在此 await 返回给当前协程。
    # [/maint-note:715f423c1839]
    @router.post("/threads")
    async def create_thread(request: CreateThreadRequest) -> dict[str, Any]:
        try:
            runtime_model_config = normalize_runtime_model_config(
                request.runtime_model_config,
                engine.default_runtime_model_config,
                legacy_model=request.model,
            )
            thread_id = await asyncio.to_thread(
                engine.store.create_thread,
                request.title,
                workspace=request.workspace,
                model=request.model or runtime_model_config["model"],
                runtime_model_config=runtime_model_config,
                runtime_mode=request.runtime_mode,
                budget=request.budget,
                authorized_paths=request.authorized_paths,
                context_items=request.context_items,
                capabilities=normalize_codegraph_capabilities(request.capabilities),
            )
        except ValueError as exc:
            raise HTTPException(status_code=422, detail=str(exc)) from exc
        return {"thread_id": thread_id, "thread": engine.store.get_thread(thread_id)}

    # [maint-note:1033a0e73d16]
    # 维护导航：list_threads；下列为本函数源码中的接口与路径索引。
    # 输入 limit：int；缺省表达式 Query(default=100, ge=1, le=500)。
    # 返回契约：dict[str, Any]。
    # 异步边界：调用方需等待协程；异常沿 await 链传播或被上层捕获。
    # 返回路径：{'threads': await asyncio.to_thread(engine.store.list_threads, limit)}。
    # 直接调用：asyncio.to_thread。
    # 等待边界：asyncio.to_thread(engine.store.list_threads, limit)；异常在此 await 返回给当前协程。
    # [/maint-note:1033a0e73d16]
    @router.get("/threads")
    async def list_threads(limit: int = Query(default=100, ge=1, le=500)) -> dict[str, Any]:
        return {"threads": await asyncio.to_thread(engine.store.list_threads, limit)}

    # [maint-note:bbcc459868d6]
    # 维护导航：get_thread；下列为本函数源码中的接口与路径索引。
    # 输入 thread_id：str。
    # 返回契约：dict[str, Any]。
    # 异步边界：调用方需等待协程；异常沿 await 链传播或被上层捕获。
    # 返回路径：{'thread': thread, 'messages': messages, 'last_run': last_run}。
    # 显式抛出：HTTPException(status_code=404, detail=str(exc))。
    # 显式抛出：HTTPException(status_code=409, detail=str(exc))。
    # 捕获异常：KeyError。
    # 捕获异常：ValueError。
    # 直接调用：asyncio.to_thread。
    # 直接调用：thread.get。
    # 直接调用：HTTPException。
    # 直接调用：str。
    # 等待边界：asyncio.to_thread(engine.store.get_thread, thread_id)；异常在此 await 返回给当前协程。
    # 等待边界：asyncio.to_thread(engine.store.list_conversation_messages, thread_id)；异常在此 await 返回给当前协程。
    # 表达式分流：thread.get('last_run_id')；真值与假值分支分别计算，不同时执行。
    # 等待边界：asyncio.to_thread(engine.get_state, thread['last_run_id'])；异常在此 await 返回给当前协程。
    # [/maint-note:bbcc459868d6]
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
        except ValueError as exc:
            raise HTTPException(status_code=409, detail=str(exc)) from exc
        return {"thread": thread, "messages": messages, "last_run": last_run}

    # [maint-note:e099212a8877]
    # 维护导航：update_thread；下列为本函数源码中的接口与路径索引。
    # 输入 thread_id：str。
    # 输入 request：UpdateThreadRequest。
    # 返回契约：dict[str, Any]。
    # 异步边界：调用方需等待协程；异常沿 await 链传播或被上层捕获。
    # 分支条件：values.get('capabilities') is not None。
    # 分支条件：'runtime_model_config' in values。
    # 返回路径：{'thread': thread}。
    # 显式抛出：HTTPException(status_code=404, detail=str(exc))。
    # 显式抛出：HTTPException(status_code=409, detail=str(exc))。
    # 捕获异常：KeyError。
    # 捕获异常：(ValueError, VersionConflict)。
    # 写入目标：values['capabilities']。
    # 写入目标：values['runtime_model_config']。
    # 直接调用：request.model_dump。
    # 直接调用：values.pop。
    # 直接调用：values.get。
    # 直接调用：normalize_codegraph_capabilities。
    # 直接调用：asyncio.to_thread。
    # 直接调用：normalize_runtime_model_config。
    # 直接调用：current.get。
    # 直接调用：HTTPException。
    # 直接调用：str。
    # 等待边界：asyncio.to_thread(engine.store.get_thread, thread_id)；异常在此 await 返回给当前协程。
    # 等待边界：asyncio.to_thread(engine.store.update_thread, thread_id, expected_version=expected_version, **values)；异常在此 await 返回给当前协程。
    # [/maint-note:e099212a8877]
    @router.patch("/threads/{thread_id}")
    async def update_thread(thread_id: str, request: UpdateThreadRequest) -> dict[str, Any]:
        values = request.model_dump(exclude_unset=True)
        expected_version = values.pop("expected_version", None)
        if values.get("capabilities") is not None:
            values["capabilities"] = normalize_codegraph_capabilities(values["capabilities"])
        try:
            if "runtime_model_config" in values:
                current = await asyncio.to_thread(engine.store.get_thread, thread_id)
                values["runtime_model_config"] = normalize_runtime_model_config(
                    values["runtime_model_config"],
                    engine.default_runtime_model_config,
                    legacy_model=values.get("model") or current.get("model", ""),
                )
            thread = await asyncio.to_thread(
                engine.store.update_thread,
                thread_id,
                expected_version=expected_version,
                **values,
            )
        except KeyError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc
        except (ValueError, VersionConflict) as exc:
            raise HTTPException(status_code=409, detail=str(exc)) from exc
        return {"thread": thread}

    # [maint-note:e26f35e7632b]
    # 维护导航：delete_thread；下列为本函数源码中的接口与路径索引。
    # 输入 thread_id：str。
    # 返回契约：dict[str, str]。
    # 异步边界：调用方需等待协程；异常沿 await 链传播或被上层捕获。
    # 返回路径：{'deleted': deleted}。
    # 显式抛出：HTTPException(status_code=404, detail=str(exc))。
    # 显式抛出：HTTPException(status_code=409, detail=str(exc))。
    # 捕获异常：KeyError。
    # 捕获异常：ThreadActiveConflict。
    # 直接调用：asyncio.to_thread。
    # 直接调用：HTTPException。
    # 直接调用：str。
    # 等待边界：asyncio.to_thread(engine.store.delete_thread, thread_id)；异常在此 await 返回给当前协程。
    # [/maint-note:e26f35e7632b]
    @router.delete("/threads/{thread_id}")
    async def delete_thread(thread_id: str) -> dict[str, str]:
        try:
            deleted = await asyncio.to_thread(engine.store.delete_thread, thread_id)
        except KeyError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc
        except ThreadActiveConflict as exc:
            raise HTTPException(status_code=409, detail=str(exc)) from exc
        return {"deleted": deleted}

    # [maint-note:6afa3aa61439]
    # 维护导航：codegraph_client_for_thread；下列为本函数源码中的接口与路径索引。
    # 输入 thread_id：str。
    # 返回契约：tuple[dict[str, Any], CodeGraphClient]。
    # 分支条件：not thread['workspace']。
    # 返回路径：(thread, CodeGraphClient(thread['workspace']))。
    # 显式抛出：ValueError('thread workspace is not configured')。
    # 直接调用：engine.store.get_thread。
    # 直接调用：ValueError。
    # 直接调用：CodeGraphClient。
    # [/maint-note:6afa3aa61439]
    def codegraph_client_for_thread(thread_id: str) -> tuple[dict[str, Any], CodeGraphClient]:
        thread = engine.store.get_thread(thread_id)
        if not thread["workspace"]:
            raise ValueError("thread workspace is not configured")
        return thread, CodeGraphClient(thread["workspace"])

    # [maint-note:c3fd16f5e6af]
    # 维护导航：codegraph_status；下列为本函数源码中的接口与路径索引。
    # 输入 thread_id：str。
    # 返回契约：dict[str, Any]。
    # 异步边界：调用方需等待协程；异常沿 await 链传播或被上层捕获。
    # 返回路径：{'status': status, 'capabilities': normalize_codegraph_capabilities(thread.get('capabilities'))}。
    # 显式抛出：HTTPException(status_code=404, detail=str(exc))。
    # 显式抛出：HTTPException(status_code=409, detail=str(exc))。
    # 捕获异常：KeyError。
    # 捕获异常：ValueError。
    # 直接调用：asyncio.to_thread。
    # 直接调用：HTTPException。
    # 直接调用：str。
    # 直接调用：normalize_codegraph_capabilities。
    # 直接调用：thread.get。
    # 等待边界：asyncio.to_thread(codegraph_client_for_thread, thread_id)；异常在此 await 返回给当前协程。
    # 等待边界：asyncio.to_thread(client.status)；异常在此 await 返回给当前协程。
    # [/maint-note:c3fd16f5e6af]
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

    # [maint-note:9b191466622a]
    # 维护导航：codegraph_init；下列为本函数源码中的接口与路径索引。
    # 输入 thread_id：str。
    # 返回契约：dict[str, Any]。
    # 异步边界：调用方需等待协程；异常沿 await 链传播或被上层捕获。
    # 分支条件：result.returncode != 0。
    # 返回路径：{'status': status, 'output': result.output.strip()}。
    # 显式抛出：HTTPException(status_code=404, detail=str(exc))。
    # 显式抛出：HTTPException(status_code=409, detail=str(exc))。
    # 显式抛出：HTTPException(status_code=503, detail=result.output.strip() or 'CodeGraph init failed')。
    # 捕获异常：KeyError。
    # 捕获异常：ValueError。
    # 直接调用：asyncio.to_thread。
    # 直接调用：HTTPException。
    # 直接调用：str。
    # 直接调用：result.output.strip。
    # 等待边界：asyncio.to_thread(codegraph_client_for_thread, thread_id)；异常在此 await 返回给当前协程。
    # 等待边界：asyncio.to_thread(client.init)；异常在此 await 返回给当前协程。
    # 等待边界：asyncio.to_thread(client.status)；异常在此 await 返回给当前协程。
    # [/maint-note:9b191466622a]
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

    # [maint-note:16789ef9e7bf]
    # 维护导航：codegraph_sync；下列为本函数源码中的接口与路径索引。
    # 输入 thread_id：str。
    # 返回契约：dict[str, Any]。
    # 异步边界：调用方需等待协程；异常沿 await 链传播或被上层捕获。
    # 分支条件：result.returncode != 0。
    # 返回路径：{'status': status, 'output': result.output.strip()}。
    # 显式抛出：HTTPException(status_code=404, detail=str(exc))。
    # 显式抛出：HTTPException(status_code=409, detail=str(exc))。
    # 显式抛出：HTTPException(status_code=503, detail=result.output.strip() or 'CodeGraph sync failed')。
    # 捕获异常：KeyError。
    # 捕获异常：ValueError。
    # 直接调用：asyncio.to_thread。
    # 直接调用：HTTPException。
    # 直接调用：str。
    # 直接调用：result.output.strip。
    # 等待边界：asyncio.to_thread(codegraph_client_for_thread, thread_id)；异常在此 await 返回给当前协程。
    # 等待边界：asyncio.to_thread(client.sync)；异常在此 await 返回给当前协程。
    # 等待边界：asyncio.to_thread(client.status)；异常在此 await 返回给当前协程。
    # [/maint-note:16789ef9e7bf]
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

    # [maint-note:0a4cc62d5b22]
    # 维护导航：codegraph_visualize；下列为本函数源码中的接口与路径索引。
    # 输入 thread_id：str。
    # 输入 request：CodeGraphVisualizeRequest。
    # 返回契约：dict[str, Any]。
    # 异步边界：调用方需等待协程；异常沿 await 链传播或被上层捕获。
    # 返回路径：{'node_count': data['node_count'], 'edge_count': data['edge_count'], 'url': f'/api/agent/threads/{thread_id}/codegraph/visualization'}。
    # 显式抛出：HTTPException(status_code=404, detail=str(exc))。
    # 显式抛出：HTTPException(status_code=409, detail=str(exc))。
    # 捕获异常：KeyError。
    # 捕获异常：(FileNotFoundError, ValueError, sqlite3.Error)。
    # 直接调用：asyncio.to_thread。
    # 直接调用：Path。
    # 直接调用：codegraph_artifact_name。
    # 直接调用：HTTPException。
    # 直接调用：str。
    # 等待边界：asyncio.to_thread(codegraph_client_for_thread, thread_id)；异常在此 await 返回给当前协程。
    # 等待边界：asyncio.to_thread(client.visualize, output, keyword=request.keyword, depth=request.depth, max_nodes=request.max_nodes)；异常在此 await 返回给当前协程。
    # [/maint-note:0a4cc62d5b22]
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

    # [maint-note:c6eca6748e02]
    # 维护导航：codegraph_visualization；下列为本函数源码中的接口与路径索引。
    # 输入 thread_id：str。
    # 返回契约：FileResponse。
    # 异步边界：调用方需等待协程；异常沿 await 链传播或被上层捕获。
    # 分支条件：not output.is_file()。
    # 返回路径：FileResponse(output, media_type='text/html')。
    # 显式抛出：HTTPException(status_code=404, detail=str(exc))。
    # 显式抛出：HTTPException(status_code=409, detail=str(exc))。
    # 显式抛出：HTTPException(status_code=404, detail='CodeGraph visualization has not been generated')。
    # 捕获异常：KeyError。
    # 捕获异常：ValueError。
    # 直接调用：asyncio.to_thread。
    # 直接调用：HTTPException。
    # 直接调用：str。
    # 直接调用：Path。
    # 直接调用：codegraph_artifact_name。
    # 直接调用：output.is_file。
    # 直接调用：FileResponse。
    # 等待边界：asyncio.to_thread(codegraph_client_for_thread, thread_id)；异常在此 await 返回给当前协程。
    # [/maint-note:c6eca6748e02]
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

    # [maint-note:8253cf5417a7]
    # 维护导航：list_thread_messages；下列为本函数源码中的接口与路径索引。
    # 输入 thread_id：str。
    # 输入 limit：int；缺省表达式 Query(default=500, ge=1, le=2000)。
    # 返回契约：dict[str, Any]。
    # 异步边界：调用方需等待协程；异常沿 await 链传播或被上层捕获。
    # 返回路径：{'messages': messages}。
    # 显式抛出：HTTPException(status_code=404, detail=str(exc))。
    # 捕获异常：KeyError。
    # 直接调用：asyncio.to_thread。
    # 直接调用：HTTPException。
    # 直接调用：str。
    # 等待边界：asyncio.to_thread(engine.store.list_conversation_messages, thread_id, limit=limit)；异常在此 await 返回给当前协程。
    # [/maint-note:8253cf5417a7]
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

    # [maint-note:4dd250f1b029]
    # 维护导航：create_thread_message；下列为本函数源码中的接口与路径索引。
    # 输入 thread_id：str。
    # 输入 request：CreateConversationMessageRequest。
    # 返回契约：dict[str, Any]。
    # 异步边界：调用方需等待协程；异常沿 await 链传播或被上层捕获。
    # 分支条件：not thread['workspace']。
    # 分支条件：authorized_paths != thread['authorized_paths']。
    # 分支条件：request.context_items != thread['context_items']。
    # 分支条件：request.capabilities is not None。
    # 分支条件：capabilities != normalize_codegraph_capabilities(thread.get('capabilities'))。
    # 返回路径：{'run_id': run_id, 'state': engine.get_state(run_id)}。
    # 显式抛出：ValueError('thread workspace is not configured')。
    # 显式抛出：HTTPException(status_code=404, detail=str(exc))。
    # 显式抛出：HTTPException(status_code=409, detail=str(exc))。
    # 捕获异常：KeyError。
    # 捕获异常：(ValueError, VersionConflict)。
    # 直接调用：asyncio.to_thread。
    # 直接调用：ValueError。
    # 直接调用：list。
    # 直接调用：dict.fromkeys。
    # 直接调用：normalize_codegraph_capabilities。
    # 直接调用：thread.get。
    # 直接调用：RunBudget.from_dict。
    # 直接调用：HTTPException。
    # 直接调用：str。
    # 直接调用：engine.get_state。
    # 等待边界：asyncio.to_thread(engine.store.get_thread, thread_id)；异常在此 await 返回给当前协程。
    # 等待边界：asyncio.to_thread(engine.store.update_thread, thread_id, authorized_paths=authorized_paths, expected_version=thread['version'])；异常在此 await 返回给当前协程。
    # 等待边界：asyncio.to_thread(engine.store.update_thread, thread_id, context_items=request.context_items, expected_version=thread['version'])；异常在此 await 返回给当前协程。
    # 等待边界：asyncio.to_thread(engine.store.update_thread, thread_id, capabilities=capabilities, expected_version=thread['version'])；异常在此 await 返回给当前协程。
    # 等待边界：asyncio.to_thread(engine.create_run, thread['workspace'], request.content, budget, thread_id, request.target_files, authorized_paths, thread['capabilities'], request.api_key)；异常在此 await 返回给当前协程。
    # [/maint-note:4dd250f1b029]
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

    # [maint-note:64e33e125cfb]
    # 维护导航：resolve_context；下列为本函数源码中的接口与路径索引。
    # 输入 request：ContextResolveRequest。
    # 返回契约：dict[str, Any]。
    # 异步边界：调用方需等待协程；异常沿 await 链传播或被上层捕获。
    # 分支条件：not thread['workspace']。
    # 分支条件：candidate.is_absolute()。
    # 分支条件：not resolved.exists()。
    # 分支条件：not query。
    # 分支条件：any((part in ignored for part in relative.parts))。
    # 分支条件：query not in relative.as_posix().casefold()。
    # 分支条件：len(matches) >= request.limit。
    # 遍历边界：path ← sorted(workspace.rglob('*'), key=lambda item: item.as_posix().casefold())。
    # 返回路径：{'matches': []}。
    # 返回路径：{'matches': matches, 'thread': thread}。
    # 显式抛出：HTTPException(status_code=404, detail=str(exc))。
    # 显式抛出：HTTPException(status_code=409, detail='thread workspace is not configured')。
    # 显式抛出：HTTPException(status_code=404, detail=f'path does not exist: {raw}')。
    # 显式抛出：HTTPException(status_code=409, detail=str(exc))。
    # 捕获异常：KeyError。
    # 捕获异常：ValueError。
    # 捕获异常：VersionConflict。
    # 直接调用：asyncio.to_thread。
    # 直接调用：HTTPException。
    # 直接调用：str。
    # 直接调用：Path(thread['workspace']).resolve。
    # 直接调用：Path。
    # 直接调用：request.value.strip。
    # 直接调用：Path(raw).expanduser。
    # 直接调用：candidate.is_absolute。
    # 直接调用：candidate.resolve。
    # 直接调用：resolved.exists。
    # 直接调用：resolved.relative_to(workspace).as_posix。
    # 直接调用：resolved.relative_to。
    # 直接调用：list。
    # 直接调用：dict.fromkeys。
    # 直接调用：matches.append。
    # 直接调用：resolved.is_dir。
    # 直接调用：raw.removeprefix('@').casefold。
    # 直接调用：raw.removeprefix。
    # 直接调用：sorted。
    # 直接调用：workspace.rglob。
    # 直接调用：path.relative_to。
    # 直接调用：any。
    # 直接调用：relative.as_posix().casefold。
    # 直接调用：relative.as_posix。
    # 直接调用：path.resolve。
    # 直接调用：path.is_dir。
    # 直接调用：len。
    # 等待边界：asyncio.to_thread(engine.store.get_thread, request.thread_id)；异常在此 await 返回给当前协程。
    # 等待边界：asyncio.to_thread(engine.store.update_thread, request.thread_id, authorized_paths=authorized, expected_version=thread['version'])；异常在此 await 返回给当前协程。
    # 表达式分流：resolved.is_dir()；真值与假值分支分别计算，不同时执行。
    # 集合派生：part 来自 relative.parts。
    # 表达式分流：path.is_dir()；真值与假值分支分别计算，不同时执行。
    # [/maint-note:64e33e125cfb]
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

    # [maint-note:3212b215caf5]
    # 维护导航：create_run；下列为本函数源码中的接口与路径索引。
    # 输入 request：CreateRunRequest。
    # 返回契约：dict[str, Any]。
    # 异步边界：调用方需等待协程；异常沿 await 链传播或被上层捕获。
    # 返回路径：{'run_id': run_id, 'state': engine.get_state(run_id)}。
    # 显式抛出：HTTPException(status_code=400, detail=str(exc))。
    # 捕获异常：(ValueError, FileNotFoundError, KeyError)。
    # 直接调用：asyncio.to_thread。
    # 直接调用：RunBudget.from_dict。
    # 直接调用：HTTPException。
    # 直接调用：str。
    # 直接调用：engine.get_state。
    # 等待边界：asyncio.to_thread(engine.create_run, request.workspace, request.goal, RunBudget.from_dict(request.budget), request.thread_id, request.target_files, request.authorized_paths, request.capabilities, request.api_key, request.runtime_model_confi；异常在此 await 返回给当前协程。
    # [/maint-note:3212b215caf5]
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
                request.runtime_model_config,
            )
        except (ValueError, FileNotFoundError, KeyError) as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        return {"run_id": run_id, "state": engine.get_state(run_id)}

    # [maint-note:a418f8db83ce]
    # 维护导航：list_runs；下列为本函数源码中的接口与路径索引。
    # 输入 limit：int；缺省表达式 Query(default=100, ge=1, le=500)。
    # 返回契约：dict[str, Any]。
    # 异步边界：调用方需等待协程；异常沿 await 链传播或被上层捕获。
    # 返回路径：{'runs': await asyncio.to_thread(engine.store.list_runs, limit)}。
    # 直接调用：asyncio.to_thread。
    # 等待边界：asyncio.to_thread(engine.store.list_runs, limit)；异常在此 await 返回给当前协程。
    # [/maint-note:a418f8db83ce]
    @router.get("/runs")
    async def list_runs(limit: int = Query(default=100, ge=1, le=500)) -> dict[str, Any]:
        return {"runs": await asyncio.to_thread(engine.store.list_runs, limit)}

    # [maint-note:ce3ab10ab173]
    # 维护导航：get_run；下列为本函数源码中的接口与路径索引。
    # 输入 run_id：str。
    # 返回契约：dict[str, Any]。
    # 异步边界：调用方需等待协程；异常沿 await 链传播或被上层捕获。
    # 返回路径：await asyncio.to_thread(engine.get_state, run_id)。
    # 显式抛出：HTTPException(status_code=404, detail=str(exc))。
    # 显式抛出：HTTPException(status_code=409, detail=str(exc))。
    # 捕获异常：KeyError。
    # 捕获异常：ValueError。
    # 直接调用：asyncio.to_thread。
    # 直接调用：HTTPException。
    # 直接调用：str。
    # 等待边界：asyncio.to_thread(engine.get_state, run_id)；异常在此 await 返回给当前协程。
    # [/maint-note:ce3ab10ab173]
    @router.get("/runs/{run_id}")
    async def get_run(run_id: str) -> dict[str, Any]:
        try:
            return await asyncio.to_thread(engine.get_state, run_id)
        except KeyError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc
        except ValueError as exc:
            raise HTTPException(status_code=409, detail=str(exc)) from exc

    # [maint-note:ec4ada4e1580]
    # 维护导航：update_run_budget；下列为本函数源码中的接口与路径索引。
    # 输入 run_id：str。
    # 输入 request：BudgetPatchRequest。
    # 返回契约：dict[str, Any]。
    # 异步边界：调用方需等待协程；异常沿 await 链传播或被上层捕获。
    # 分支条件：run.get('thread_id')。
    # 返回路径：updated。
    # 显式抛出：HTTPException(status_code=404, detail=str(exc))。
    # 显式抛出：HTTPException(status_code=409, detail=str(exc))。
    # 捕获异常：KeyError。
    # 捕获异常：(ValueError, VersionConflict)。
    # 直接调用：asyncio.to_thread。
    # 直接调用：run.get。
    # 直接调用：HTTPException。
    # 直接调用：str。
    # 等待边界：asyncio.to_thread(engine.update_budget, run_id, request.budget)；异常在此 await 返回给当前协程。
    # 等待边界：asyncio.to_thread(engine.store.get_run, run_id)；异常在此 await 返回给当前协程。
    # 等待边界：asyncio.to_thread(engine.store.get_thread, run['thread_id'])；异常在此 await 返回给当前协程。
    # 等待边界：asyncio.to_thread(engine.store.update_thread, run['thread_id'], budget=updated['budget'], expected_version=thread['version'])；异常在此 await 返回给当前协程。
    # [/maint-note:ec4ada4e1580]
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

    # [maint-note:b6fb5487770b]
    # 维护导航：step_run；下列为本函数源码中的接口与路径索引。
    # 输入 run_id：str。
    # 返回契约：dict[str, Any]。
    # 异步边界：调用方需等待协程；异常沿 await 链传播或被上层捕获。
    # 返回路径：await asyncio.to_thread(engine.step, run_id)。
    # 显式抛出：HTTPException(status_code=404, detail=str(exc))。
    # 显式抛出：HTTPException(status_code=409, detail=str(exc))。
    # 捕获异常：KeyError。
    # 捕获异常：(ValueError, VersionConflict)。
    # 直接调用：asyncio.to_thread。
    # 直接调用：HTTPException。
    # 直接调用：str。
    # 等待边界：asyncio.to_thread(engine.step, run_id)；异常在此 await 返回给当前协程。
    # [/maint-note:b6fb5487770b]
    @router.post("/runs/{run_id}/step")
    async def step_run(run_id: str) -> dict[str, Any]:
        try:
            return await asyncio.to_thread(engine.step, run_id)
        except KeyError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc
        except (ValueError, VersionConflict) as exc:
            raise HTTPException(status_code=409, detail=str(exc)) from exc

    # [maint-note:ab0f93f90b13]
    # 维护导航：run_until_pause；下列为本函数源码中的接口与路径索引。
    # 输入 run_id：str。
    # 返回契约：dict[str, Any]。
    # 异步边界：调用方需等待协程；异常沿 await 链传播或被上层捕获。
    # 返回路径：await asyncio.to_thread(engine.run, run_id)。
    # 显式抛出：HTTPException(status_code=404, detail=str(exc))。
    # 显式抛出：HTTPException(status_code=409, detail=str(exc))。
    # 捕获异常：KeyError。
    # 捕获异常：(ValueError, VersionConflict)。
    # 直接调用：asyncio.to_thread。
    # 直接调用：HTTPException。
    # 直接调用：str。
    # 等待边界：asyncio.to_thread(engine.run, run_id)；异常在此 await 返回给当前协程。
    # [/maint-note:ab0f93f90b13]
    @router.post("/runs/{run_id}/run")
    async def run_until_pause(run_id: str) -> dict[str, Any]:
        try:
            return await asyncio.to_thread(engine.run, run_id)
        except KeyError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc
        except (ValueError, VersionConflict) as exc:
            raise HTTPException(status_code=409, detail=str(exc)) from exc

    # [maint-note:e2d90472c7f0]
    # 维护导航：approve_run；下列为本函数源码中的接口与路径索引。
    # 输入 run_id：str。
    # 输入 request：ApprovalRequest。
    # 返回契约：dict[str, Any]。
    # 异步边界：调用方需等待协程；异常沿 await 链传播或被上层捕获。
    # 返回路径：await asyncio.to_thread(engine.approve, run_id, request.risk, request.tool_call_id)。
    # 显式抛出：HTTPException(status_code=404, detail=str(exc))。
    # 显式抛出：HTTPException(status_code=409, detail=str(exc))。
    # 捕获异常：KeyError。
    # 捕获异常：(ValueError, VersionConflict)。
    # 直接调用：asyncio.to_thread。
    # 直接调用：HTTPException。
    # 直接调用：str。
    # 等待边界：asyncio.to_thread(engine.approve, run_id, request.risk, request.tool_call_id)；异常在此 await 返回给当前协程。
    # [/maint-note:e2d90472c7f0]
    @router.post("/runs/{run_id}/approve")
    async def approve_run(run_id: str, request: ApprovalRequest) -> dict[str, Any]:
        try:
            return await asyncio.to_thread(engine.approve, run_id, request.risk, request.tool_call_id)
        except KeyError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc
        except (ValueError, VersionConflict) as exc:
            raise HTTPException(status_code=409, detail=str(exc)) from exc

    # [maint-note:1448848d5b12]
    # 维护导航：reject_run；下列为本函数源码中的接口与路径索引。
    # 输入 run_id：str。
    # 返回契约：dict[str, Any]。
    # 异步边界：调用方需等待协程；异常沿 await 链传播或被上层捕获。
    # 返回路径：await asyncio.to_thread(engine.reject, run_id)。
    # 显式抛出：HTTPException(status_code=404, detail=str(exc))。
    # 显式抛出：HTTPException(status_code=409, detail=str(exc))。
    # 捕获异常：KeyError。
    # 捕获异常：ValueError。
    # 直接调用：asyncio.to_thread。
    # 直接调用：HTTPException。
    # 直接调用：str。
    # 等待边界：asyncio.to_thread(engine.reject, run_id)；异常在此 await 返回给当前协程。
    # [/maint-note:1448848d5b12]
    @router.post("/runs/{run_id}/reject")
    async def reject_run(run_id: str) -> dict[str, Any]:
        try:
            return await asyncio.to_thread(engine.reject, run_id)
        except KeyError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc
        except ValueError as exc:
            raise HTTPException(status_code=409, detail=str(exc)) from exc

    # [maint-note:3ad5c8a05fbf]
    # 维护导航：pause_run；下列为本函数源码中的接口与路径索引。
    # 输入 run_id：str。
    # 返回契约：dict[str, Any]。
    # 异步边界：调用方需等待协程；异常沿 await 链传播或被上层捕获。
    # 返回路径：await asyncio.to_thread(engine.pause, run_id)。
    # 显式抛出：HTTPException(status_code=404, detail=str(exc))。
    # 显式抛出：HTTPException(status_code=409, detail=str(exc))。
    # 捕获异常：KeyError。
    # 捕获异常：ValueError。
    # 直接调用：asyncio.to_thread。
    # 直接调用：HTTPException。
    # 直接调用：str。
    # 等待边界：asyncio.to_thread(engine.pause, run_id)；异常在此 await 返回给当前协程。
    # [/maint-note:3ad5c8a05fbf]
    @router.post("/runs/{run_id}/pause")
    async def pause_run(run_id: str) -> dict[str, Any]:
        try:
            return await asyncio.to_thread(engine.pause, run_id)
        except KeyError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc
        except ValueError as exc:
            raise HTTPException(status_code=409, detail=str(exc)) from exc

    # [maint-note:b98bcd1c8c2e]
    # 维护导航：resume_run；下列为本函数源码中的接口与路径索引。
    # 输入 run_id：str。
    # 返回契约：dict[str, Any]。
    # 异步边界：调用方需等待协程；异常沿 await 链传播或被上层捕获。
    # 返回路径：await asyncio.to_thread(engine.resume, run_id)。
    # 显式抛出：HTTPException(status_code=404, detail=str(exc))。
    # 显式抛出：HTTPException(status_code=409, detail=str(exc))。
    # 捕获异常：KeyError。
    # 捕获异常：ValueError。
    # 直接调用：asyncio.to_thread。
    # 直接调用：HTTPException。
    # 直接调用：str。
    # 等待边界：asyncio.to_thread(engine.resume, run_id)；异常在此 await 返回给当前协程。
    # [/maint-note:b98bcd1c8c2e]
    @router.post("/runs/{run_id}/resume")
    async def resume_run(run_id: str) -> dict[str, Any]:
        try:
            return await asyncio.to_thread(engine.resume, run_id)
        except KeyError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc
        except ValueError as exc:
            raise HTTPException(status_code=409, detail=str(exc)) from exc

    # [maint-note:912bbd4a684e]
    # 维护导航：cancel_run；下列为本函数源码中的接口与路径索引。
    # 输入 run_id：str。
    # 返回契约：dict[str, Any]。
    # 异步边界：调用方需等待协程；异常沿 await 链传播或被上层捕获。
    # 返回路径：await asyncio.to_thread(engine.cancel, run_id)。
    # 显式抛出：HTTPException(status_code=404, detail=str(exc))。
    # 显式抛出：HTTPException(status_code=409, detail=str(exc))。
    # 捕获异常：KeyError。
    # 捕获异常：ValueError。
    # 直接调用：asyncio.to_thread。
    # 直接调用：HTTPException。
    # 直接调用：str。
    # 等待边界：asyncio.to_thread(engine.cancel, run_id)；异常在此 await 返回给当前协程。
    # [/maint-note:912bbd4a684e]
    @router.post("/runs/{run_id}/cancel")
    async def cancel_run(run_id: str) -> dict[str, Any]:
        try:
            return await asyncio.to_thread(engine.cancel, run_id)
        except KeyError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc
        except ValueError as exc:
            raise HTTPException(status_code=409, detail=str(exc)) from exc

    # [maint-note:7a0a1b136ad8]
    # 维护导航：get_events；下列为本函数源码中的接口与路径索引。
    # 输入 run_id：str。
    # 输入 after：int；缺省表达式 Query(default=0, ge=0)。
    # 返回契约：dict[str, Any]。
    # 异步边界：调用方需等待协程；异常沿 await 链传播或被上层捕获。
    # 返回路径：{'events': [event.to_dict() for event in events]}。
    # 显式抛出：HTTPException(status_code=404, detail=str(exc))。
    # 捕获异常：KeyError。
    # 直接调用：engine.store.get_run。
    # 直接调用：HTTPException。
    # 直接调用：str。
    # 直接调用：asyncio.to_thread。
    # 直接调用：event.to_dict。
    # 等待边界：asyncio.to_thread(engine.store.list_events, run_id, after)；异常在此 await 返回给当前协程。
    # 集合派生：event 来自 events。
    # [/maint-note:7a0a1b136ad8]
    @router.get("/runs/{run_id}/events")
    async def get_events(run_id: str, after: int = Query(default=0, ge=0)) -> dict[str, Any]:
        try:
            engine.store.get_run(run_id)
        except KeyError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc
        events = await asyncio.to_thread(engine.store.list_events, run_id, after)
        return {"events": [event.to_dict() for event in events]}

    # [maint-note:446cf9719e3f]
    # 维护导航：get_memory_extraction；下列为本函数源码中的接口与路径索引。
    # 输入 run_id：str。
    # 返回契约：dict[str, Any]。
    # 异步边界：调用方需等待协程；异常沿 await 链传播或被上层捕获。
    # 分支条件：outcome is None。
    # 分支条件：outcome['status'] in {'queued', 'running'} and (not psutil.pid_exists(worker_pid))。
    # 返回路径：{'run_id': run_id, 'run_status': run['status'], 'status': status}。
    # 返回路径：{'run_id': run_id, 'run_status': run['status'], **outcome}。
    # 显式抛出：HTTPException(status_code=404, detail=str(exc))。
    # 捕获异常：KeyError。
    # 直接调用：asyncio.to_thread。
    # 直接调用：HTTPException。
    # 直接调用：str。
    # 直接调用：outcome.pop。
    # 直接调用：psutil.pid_exists。
    # 直接调用：outcome.update。
    # 等待边界：asyncio.to_thread(engine.store.get_run, run_id)；异常在此 await 返回给当前协程。
    # 等待边界：asyncio.to_thread(engine.store.get_memory_extraction, run_id)；异常在此 await 返回给当前协程。
    # 表达式分流：engine.auto_memory is None；真值与假值分支分别计算，不同时执行。
    # [/maint-note:446cf9719e3f]
    @router.get("/runs/{run_id}/memory-extraction")
    async def get_memory_extraction(run_id: str) -> dict[str, Any]:
        try:
            run = await asyncio.to_thread(engine.store.get_run, run_id)
        except KeyError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc
        outcome = await asyncio.to_thread(engine.store.get_memory_extraction, run_id)
        if outcome is None:
            status = "disabled" if engine.auto_memory is None else "not_scheduled"
            return {"run_id": run_id, "run_status": run["status"], "status": status}
        worker_pid = outcome.pop("worker_pid")
        if outcome["status"] in {"queued", "running"} and not psutil.pid_exists(worker_pid):
            outcome.update(status="interrupted", reason="worker_stopped")
        return {"run_id": run_id, "run_status": run["status"], **outcome}

    # [maint-note:8a65cd78f0bc]
    # 维护导航：stream_events；下列为本函数源码中的接口与路径索引。
    # 输入 request：Request。
    # 输入 run_id：str。
    # 输入 after：int；缺省表达式 Query(default=0, ge=0)。
    # 返回契约：StreamingResponse。
    # 异步边界：调用方需等待协程；异常沿 await 链传播或被上层捕获。
    # 分支条件：run['status'] in {'completed', 'failed', 'cancelled', 'budget_exhausted', 'waiting_approval', 'paused'}。
    # 分支条件：now - last_heartbeat >= 10。
    # 遍历边界：not await request.is_disconnected()。
    # 遍历边界：event ← events。
    # 返回路径：StreamingResponse(generate(), media_type='application/x-ndjson', headers={'Cache-Control': 'no-cache', 'X-Accel-Buffering': 'no'})。
    # 显式抛出：HTTPException(status_code=404, detail=str(exc))。
    # 捕获异常：KeyError。
    # 直接调用：engine.store.get_run。
    # 直接调用：HTTPException。
    # 直接调用：str。
    # 直接调用：asyncio.get_running_loop().time。
    # 直接调用：asyncio.get_running_loop。
    # 直接调用：request.is_disconnected。
    # 直接调用：asyncio.to_thread。
    # 直接调用：json.dumps。
    # 直接调用：event.to_dict。
    # 直接调用：asyncio.sleep。
    # 直接调用：StreamingResponse。
    # 直接调用：generate。
    # [/maint-note:8a65cd78f0bc]
    @router.get("/runs/{run_id}/events.ndjson")
    async def stream_events(request: Request, run_id: str, after: int = Query(default=0, ge=0)) -> StreamingResponse:
        try:
            engine.store.get_run(run_id)
        except KeyError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc

        # [maint-note:48eb4698c84f]
        # 维护导航：generate；下列为本函数源码中的接口与路径索引。
        # 返回契约：AsyncIterator[str]。
        # 异步边界：调用方需等待协程；异常沿 await 链传播或被上层捕获。
        # 分支条件：run['status'] in {'completed', 'failed', 'cancelled', 'budget_exhausted', 'waiting_approval', 'paused'}。
        # 分支条件：now - last_heartbeat >= 10。
        # 遍历边界：not await request.is_disconnected()。
        # 遍历边界：event ← events。
        # 直接调用：asyncio.get_running_loop().time。
        # 直接调用：asyncio.get_running_loop。
        # 直接调用：request.is_disconnected。
        # 直接调用：asyncio.to_thread。
        # 直接调用：json.dumps。
        # 直接调用：event.to_dict。
        # 直接调用：asyncio.sleep。
        # 等待边界：request.is_disconnected()；异常在此 await 返回给当前协程。
        # 等待边界：asyncio.to_thread(engine.store.list_events, run_id, cursor)；异常在此 await 返回给当前协程。
        # 生成器交付：json.dumps(event.to_dict(), ensure_ascii=False) + '\n'。
        # 等待边界：asyncio.to_thread(engine.store.get_run, run_id)；异常在此 await 返回给当前协程。
        # 生成器交付：'\n'。
        # 等待边界：asyncio.sleep(0.15)；异常在此 await 返回给当前协程。
        # [/maint-note:48eb4698c84f]
        async def generate() -> AsyncIterator[str]:
            cursor = after
            last_heartbeat = asyncio.get_running_loop().time()
            while not await request.is_disconnected():
                events = await asyncio.to_thread(engine.store.list_events, run_id, cursor)
                for event in events:
                    cursor = event.sequence
                    yield json.dumps(event.to_dict(), ensure_ascii=False) + "\n"
                run = await asyncio.to_thread(engine.store.get_run, run_id)
                if run["status"] in {"completed", "failed", "cancelled", "budget_exhausted", "waiting_approval", "paused"}:
                    break
                now = asyncio.get_running_loop().time()
                if now - last_heartbeat >= 10:
                    yield "\n"
                    last_heartbeat = now
                await asyncio.sleep(0.15)

        return StreamingResponse(
            generate(),
            media_type="application/x-ndjson",
            headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
        )

    if memory_store is not None:
        # [maint-note:863d234c1b48]
        # 维护导航：create_memory_proposals_from_selection；下列为本函数源码中的接口与路径索引。
        # 输入 request：MemoryProposalSelectionRequest。
        # 返回契约：dict[str, Any]。
        # 异步边界：调用方需等待协程；异常沿 await 链传播或被上层捕获。
        # 返回路径：{'proposals': reviews, 'count': len(reviews)}。
        # 显式抛出：HTTPException(status_code=404, detail=str(exc))。
        # 显式抛出：HTTPException(status_code=422, detail=str(exc))。
        # 显式抛出：HTTPException(status_code=413, detail=str(exc))。
        # 显式抛出：HTTPException(status_code=502, detail=str(exc))。
        # 捕获异常：KeyError。
        # 捕获异常：ProposalValidationError。
        # 捕获异常：ContextHardLimitExceeded。
        # 捕获异常：MemoryProposalGenerationError。
        # 直接调用：asyncio.to_thread。
        # 直接调用：request.model_dump。
        # 直接调用：HTTPException。
        # 直接调用：str。
        # 直接调用：logger.warning。
        # 直接调用：len。
        # 等待边界：asyncio.to_thread(proposal_service.create_from_selection, **request.model_dump())；异常在此 await 返回给当前协程。
        # [/maint-note:863d234c1b48]
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

        # [maint-note:9dd25724bc9a]
        # 维护导航：list_memory_proposals；下列为本函数源码中的接口与路径索引。
        # 输入 status：str | None；缺省表达式 None。
        # 输入 project_id：str | None；缺省表达式 None。
        # 输入 thread_id：str | None；缺省表达式 None。
        # 返回契约：dict[str, Any]。
        # 异步边界：调用方需等待协程；异常沿 await 链传播或被上层捕获。
        # 返回路径：{'proposals': reviews}。
        # 直接调用：asyncio.to_thread。
        # 等待边界：asyncio.to_thread(proposal_service.list_reviews, status=status, project_id=project_id, thread_id=thread_id)；异常在此 await 返回给当前协程。
        # [/maint-note:9dd25724bc9a]
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

        # [maint-note:06c361f22cf3]
        # 维护导航：get_memory_proposal_review；下列为本函数源码中的接口与路径索引。
        # 输入 proposal_id：str。
        # 返回契约：dict[str, Any]。
        # 异步边界：调用方需等待协程；异常沿 await 链传播或被上层捕获。
        # 返回路径：{'proposal': review}。
        # 显式抛出：HTTPException(status_code=404, detail=str(exc))。
        # 捕获异常：KeyError。
        # 直接调用：asyncio.to_thread。
        # 直接调用：HTTPException。
        # 直接调用：str。
        # 等待边界：asyncio.to_thread(proposal_service.get_review, proposal_id)；异常在此 await 返回给当前协程。
        # [/maint-note:06c361f22cf3]
        @router.get("/memory-proposals/{proposal_id}/review")
        async def get_memory_proposal_review(proposal_id: str) -> dict[str, Any]:
            assert proposal_service is not None
            try:
                review = await asyncio.to_thread(proposal_service.get_review, proposal_id)
            except KeyError as exc:
                raise HTTPException(status_code=404, detail=str(exc)) from exc
            return {"proposal": review}

        # [maint-note:9b1b62ac6c2c]
        # 维护导航：get_memory_proposal_evidence；下列为本函数源码中的接口与路径索引。
        # 输入 proposal_id：str。
        # 返回契约：dict[str, Any]。
        # 异步边界：调用方需等待协程；异常沿 await 链传播或被上层捕获。
        # 返回路径：{'evidence': evidence}。
        # 显式抛出：HTTPException(status_code=404, detail=str(exc))。
        # 捕获异常：KeyError。
        # 直接调用：asyncio.to_thread。
        # 直接调用：HTTPException。
        # 直接调用：str。
        # 等待边界：asyncio.to_thread(memory_store.get_proposal_evidence, proposal_id)；异常在此 await 返回给当前协程。
        # [/maint-note:9b1b62ac6c2c]
        @router.get("/memory-proposals/{proposal_id}/evidence")
        async def get_memory_proposal_evidence(proposal_id: str) -> dict[str, Any]:
            try:
                evidence = await asyncio.to_thread(
                    memory_store.get_proposal_evidence, proposal_id
                )
            except KeyError as exc:
                raise HTTPException(status_code=404, detail=str(exc)) from exc
            return {"evidence": evidence}

        # [maint-note:eea360b6c30f]
        # 维护导航：update_memory_proposal；下列为本函数源码中的接口与路径索引。
        # 输入 proposal_id：str。
        # 输入 request：MemoryProposalUpdateRequest。
        # 返回契约：dict[str, Any]。
        # 异步边界：调用方需等待协程；异常沿 await 链传播或被上层捕获。
        # 返回路径：{'proposal': review}。
        # 显式抛出：HTTPException(status_code=404, detail=str(exc))。
        # 显式抛出：HTTPException(status_code=409, detail=str(exc))。
        # 捕获异常：KeyError。
        # 捕获异常：(MemoryProposalConflict, ValueError)。
        # 直接调用：request.model_dump。
        # 直接调用：values.pop。
        # 直接调用：asyncio.to_thread。
        # 直接调用：HTTPException。
        # 直接调用：str。
        # 等待边界：asyncio.to_thread(proposal_service.update_review, proposal_id, expected_version=expected_version, **values)；异常在此 await 返回给当前协程。
        # [/maint-note:eea360b6c30f]
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

        # [maint-note:4f29721b8566]
        # 维护导航：approve_memory_proposal；下列为本函数源码中的接口与路径索引。
        # 输入 proposal_id：str。
        # 输入 request：MemoryProposalActionRequest。
        # 返回契约：dict[str, Any]。
        # 异步边界：调用方需等待协程；异常沿 await 链传播或被上层捕获。
        # 返回路径：{'proposal': review}。
        # 显式抛出：HTTPException(status_code=404, detail=str(exc))。
        # 显式抛出：HTTPException(status_code=409, detail=str(exc))。
        # 捕获异常：KeyError。
        # 捕获异常：MemoryProposalConflict。
        # 直接调用：asyncio.to_thread。
        # 直接调用：HTTPException。
        # 直接调用：str。
        # 等待边界：asyncio.to_thread(proposal_service.approve, proposal_id, expected_version=request.expected_version)；异常在此 await 返回给当前协程。
        # [/maint-note:4f29721b8566]
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

        # [maint-note:1d1a8be53111]
        # 维护导航：reject_memory_proposal；下列为本函数源码中的接口与路径索引。
        # 输入 proposal_id：str。
        # 输入 request：MemoryProposalActionRequest。
        # 返回契约：dict[str, Any]。
        # 异步边界：调用方需等待协程；异常沿 await 链传播或被上层捕获。
        # 返回路径：{'proposal': review}。
        # 显式抛出：HTTPException(status_code=404, detail=str(exc))。
        # 显式抛出：HTTPException(status_code=409, detail=str(exc))。
        # 捕获异常：KeyError。
        # 捕获异常：MemoryProposalConflict。
        # 直接调用：asyncio.to_thread。
        # 直接调用：HTTPException。
        # 直接调用：str。
        # 等待边界：asyncio.to_thread(proposal_service.reject, proposal_id, expected_version=request.expected_version, reason=request.reason)；异常在此 await 返回给当前协程。
        # [/maint-note:1d1a8be53111]
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

        # [maint-note:6d7d47267cf8]
        # 维护导航：defer_memory_proposal；下列为本函数源码中的接口与路径索引。
        # 输入 proposal_id：str。
        # 输入 request：MemoryProposalActionRequest。
        # 返回契约：dict[str, Any]。
        # 异步边界：调用方需等待协程；异常沿 await 链传播或被上层捕获。
        # 返回路径：{'proposal': review}。
        # 显式抛出：HTTPException(status_code=404, detail=str(exc))。
        # 显式抛出：HTTPException(status_code=409, detail=str(exc))。
        # 捕获异常：KeyError。
        # 捕获异常：MemoryProposalConflict。
        # 直接调用：asyncio.to_thread。
        # 直接调用：HTTPException。
        # 直接调用：str。
        # 等待边界：asyncio.to_thread(proposal_service.defer, proposal_id, expected_version=request.expected_version, reason=request.reason)；异常在此 await 返回给当前协程。
        # [/maint-note:6d7d47267cf8]
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

        # [maint-note:b17101422a65]
        # 维护导航：list_memories；下列为本函数源码中的接口与路径索引。
        # 输入 status：str | None；缺省表达式 None。
        # 输入 project_id：str | None；缺省表达式 None。
        # 输入 thread_id：str | None；缺省表达式 None。
        # 返回契约：dict[str, Any]。
        # 异步边界：调用方需等待协程；异常沿 await 链传播或被上层捕获。
        # 返回路径：{'memories': await asyncio.to_thread(memory_store.list, status=status, project_id=project_id, thread_id=thread_id)}。
        # 直接调用：asyncio.to_thread。
        # 等待边界：asyncio.to_thread(memory_store.list, status=status, project_id=project_id, thread_id=thread_id)；异常在此 await 返回给当前协程。
        # [/maint-note:b17101422a65]
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

        # [maint-note:4ebbf9a98f6f]
        # 维护导航：create_memory；下列为本函数源码中的接口与路径索引。
        # 输入 request：MemoryCandidateRequest。
        # 返回契约：dict[str, Any]。
        # 异步边界：调用方需等待协程；异常沿 await 链传播或被上层捕获。
        # 返回路径：{'memory_id': memory_id, 'memory': memory_store.get(memory_id)}。
        # 直接调用：asyncio.to_thread。
        # 直接调用：request.model_dump。
        # 直接调用：memory_store.get。
        # 等待边界：asyncio.to_thread(memory_store.create_candidate, **request.model_dump())；异常在此 await 返回给当前协程。
        # [/maint-note:4ebbf9a98f6f]
        @router.post("/memories")
        async def create_memory(request: MemoryCandidateRequest) -> dict[str, Any]:
            memory_id = await asyncio.to_thread(memory_store.create_candidate, **request.model_dump())
            return {"memory_id": memory_id, "memory": memory_store.get(memory_id)}

        # [maint-note:509936369116]
        # 维护导航：activate_memory；下列为本函数源码中的接口与路径索引。
        # 输入 memory_id：str。
        # 返回契约：dict[str, Any]。
        # 异步边界：调用方需等待协程；异常沿 await 链传播或被上层捕获。
        # 返回路径：{'memory': memory_store.get(memory_id)}。
        # 显式抛出：HTTPException(status_code=404, detail=str(exc))。
        # 捕获异常：KeyError。
        # 直接调用：asyncio.to_thread。
        # 直接调用：HTTPException。
        # 直接调用：str。
        # 直接调用：memory_store.get。
        # 等待边界：asyncio.to_thread(memory_store.activate, memory_id)；异常在此 await 返回给当前协程。
        # [/maint-note:509936369116]
        @router.post("/memories/{memory_id}/activate")
        async def activate_memory(memory_id: str) -> dict[str, Any]:
            try:
                await asyncio.to_thread(memory_store.activate, memory_id)
            except KeyError as exc:
                raise HTTPException(status_code=404, detail=str(exc)) from exc
            return {"memory": memory_store.get(memory_id)}

        # [maint-note:0ade44daeb6b]
        # 维护导航：reject_memory；下列为本函数源码中的接口与路径索引。
        # 输入 memory_id：str。
        # 返回契约：dict[str, Any]。
        # 异步边界：调用方需等待协程；异常沿 await 链传播或被上层捕获。
        # 返回路径：{'memory': memory_store.get(memory_id)}。
        # 显式抛出：HTTPException(status_code=404, detail=str(exc))。
        # 捕获异常：KeyError。
        # 直接调用：asyncio.to_thread。
        # 直接调用：HTTPException。
        # 直接调用：str。
        # 直接调用：memory_store.get。
        # 等待边界：asyncio.to_thread(memory_store.reject, memory_id)；异常在此 await 返回给当前协程。
        # [/maint-note:0ade44daeb6b]
        @router.post("/memories/{memory_id}/reject")
        async def reject_memory(memory_id: str) -> dict[str, Any]:
            try:
                await asyncio.to_thread(memory_store.reject, memory_id)
            except KeyError as exc:
                raise HTTPException(status_code=404, detail=str(exc)) from exc
            return {"memory": memory_store.get(memory_id)}

        # [maint-note:81f2b7a5c982]
        # 维护导航：update_memory；下列为本函数源码中的接口与路径索引。
        # 输入 memory_id：str。
        # 输入 request：MemoryUpdateRequest。
        # 返回契约：dict[str, Any]。
        # 异步边界：调用方需等待协程；异常沿 await 链传播或被上层捕获。
        # 返回路径：{'memory': memory_store.get(memory_id)}。
        # 显式抛出：HTTPException(status_code=404, detail=str(exc))。
        # 捕获异常：KeyError。
        # 直接调用：asyncio.to_thread。
        # 直接调用：request.model_dump。
        # 直接调用：HTTPException。
        # 直接调用：str。
        # 直接调用：memory_store.get。
        # 等待边界：asyncio.to_thread(memory_store.update, memory_id, **request.model_dump())；异常在此 await 返回给当前协程。
        # [/maint-note:81f2b7a5c982]
        @router.put("/memories/{memory_id}")
        async def update_memory(memory_id: str, request: MemoryUpdateRequest) -> dict[str, Any]:
            try:
                await asyncio.to_thread(memory_store.update, memory_id, **request.model_dump())
            except KeyError as exc:
                raise HTTPException(status_code=404, detail=str(exc)) from exc
            return {"memory": memory_store.get(memory_id)}

        # [maint-note:13a0717c013f]
        # 维护导航：delete_memory；下列为本函数源码中的接口与路径索引。
        # 输入 memory_id：str。
        # 返回契约：dict[str, Any]。
        # 异步边界：调用方需等待协程；异常沿 await 链传播或被上层捕获。
        # 分支条件：not await asyncio.to_thread(memory_store.delete, memory_id)。
        # 返回路径：{'deleted': memory_id}。
        # 显式抛出：HTTPException(status_code=404, detail='memory not found')。
        # 直接调用：asyncio.to_thread。
        # 直接调用：HTTPException。
        # 等待边界：asyncio.to_thread(memory_store.delete, memory_id)；异常在此 await 返回给当前协程。
        # [/maint-note:13a0717c013f]
        @router.delete("/memories/{memory_id}")
        async def delete_memory(memory_id: str) -> dict[str, Any]:
            if not await asyncio.to_thread(memory_store.delete, memory_id):
                raise HTTPException(status_code=404, detail="memory not found")
            return {"deleted": memory_id}

    return router


# [maint-note:d44faab00e08]
# 维护导航：build_default_agent_router；下列为本函数源码中的接口与路径索引。
# 返回契约：APIRouter。
# 返回路径：create_agent_router(engine, memory, auto_memory_mode=os.environ.get('CODE_AGENT_AUTO_MEMORY_MODE', 'auto').strip().lower())。
# 直接调用：Path(__file__).resolve。
# 直接调用：Path。
# 直接调用：os.environ.get。
# 直接调用：EventStore。
# 直接调用：MemoryStore。
# 直接调用：build_default_context_runtime。
# 直接调用：RoutedModelGateway。
# 直接调用：ContextPlanner。
# 直接调用：CompactionService。
# 直接调用：RunEngine。
# 直接调用：build_default_registry。
# 直接调用：create_agent_router。
# 直接调用：os.environ.get('CODE_AGENT_AUTO_MEMORY_MODE', 'auto').strip().lower。
# 直接调用：os.environ.get('CODE_AGENT_AUTO_MEMORY_MODE', 'auto').strip。
# [/maint-note:d44faab00e08]
def build_default_agent_router() -> APIRouter:
    code_agent_root = Path(__file__).resolve().parents[1]
    db_path = Path(os.environ.get("CODE_AGENT_DB", code_agent_root / "outputs" / "agent_runtime.db"))
    store = EventStore(db_path)
    memory = MemoryStore(db_path)
    context_runtime = build_default_context_runtime(code_agent_root)
    gateway = RoutedModelGateway(serializer=context_runtime.serializer)
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
        default_runtime_model_config=context_runtime.default_model_config,
    )
    return create_agent_router(
        engine, memory,
        auto_memory_mode=os.environ.get("CODE_AGENT_AUTO_MEMORY_MODE", "auto").strip().lower(),
    )
