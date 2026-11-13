from __future__ import annotations

from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path
from threading import Event
from typing import Any, Callable


# [maint-note:bc72d6e34268]
# 维护导航：now_iso；下列为本函数源码中的接口与路径索引。
# 返回契约：str。
# 返回路径：datetime.now(timezone.utc).astimezone().isoformat()。
# 直接调用：datetime.now(timezone.utc).astimezone().isoformat。
# 直接调用：datetime.now(timezone.utc).astimezone。
# 直接调用：datetime.now。
# [/maint-note:bc72d6e34268]
def now_iso() -> str:
    return datetime.now(timezone.utc).astimezone().isoformat()


# [maint-note:00291e38f250]
# 数据与行为导航：RiskLevel。
# 继承接口：str, Enum。
# [/maint-note:00291e38f250]
class RiskLevel(str, Enum):
    READ = "read"
    WRITE = "write"
    EXECUTE = "execute"
    NETWORK = "network"
    GIT_WRITE = "git_write"


# [maint-note:b26b65d1bad0]
# 数据与行为导航：RunStatus。
# 继承接口：str, Enum。
# [/maint-note:b26b65d1bad0]
class RunStatus(str, Enum):
    QUEUED = "queued"
    RUNNING = "running"
    WAITING_APPROVAL = "waiting_approval"
    PAUSED = "paused"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"
    BUDGET_EXHAUSTED = "budget_exhausted"


# [maint-note:9e329cdf1715]
# 数据与行为导航：StepStatus。
# 继承接口：str, Enum。
# [/maint-note:9e329cdf1715]
class StepStatus(str, Enum):
    PROPOSED = "proposed"
    APPROVED = "approved"
    RUNNING = "running"
    SUCCEEDED = "succeeded"
    FAILED = "failed"
    CANCELLED = "cancelled"
    UNCERTAIN = "uncertain"


# [maint-note:697bd5ffeeec]
# 数据与行为导航：ToolCall。
# 字段 id：str。
# 字段 name：str。
# 字段 args：dict[str, Any]；初始表达式 field(default_factory=dict)。
# [/maint-note:697bd5ffeeec]
@dataclass(frozen=True)
class ToolCall:
    id: str
    name: str
    args: dict[str, Any] = field(default_factory=dict)

    # [maint-note:53c6bac7e7a8]
    # 维护导航：to_dict；下列为本函数源码中的接口与路径索引。
    # 返回契约：dict[str, Any]。
    # 返回路径：{'id': self.id, 'name': self.name, 'args': self.args}。
    # [/maint-note:53c6bac7e7a8]
    def to_dict(self) -> dict[str, Any]:
        return {"id": self.id, "name": self.name, "args": self.args}


# [maint-note:5c730cfe2667]
# 数据与行为导航：ModelRequest。
# 字段 messages：list[dict[str, Any]]。
# 字段 tools：list[dict[str, Any]]；初始表达式 field(default_factory=list)。
# 字段 purpose：str；初始表达式 'agent'。
# 字段 max_output_tokens：int；初始表达式 4096。
# 字段 response_format：dict[str, Any] | None；初始表达式 None。
# 字段 metadata：dict[str, Any]；初始表达式 field(default_factory=dict)。
# [/maint-note:5c730cfe2667]
@dataclass(frozen=True)
class ModelRequest:
    messages: list[dict[str, Any]]
    tools: list[dict[str, Any]] = field(default_factory=list)
    purpose: str = "agent"
    max_output_tokens: int = 4096
    response_format: dict[str, Any] | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


# [maint-note:aa228dc9fc7d]
# 数据与行为导航：ModelResponse。
# 字段 content：str；初始表达式 ''。
# 字段 tool_calls：list[ToolCall]；初始表达式 field(default_factory=list)。
# 字段 input_tokens：int；初始表达式 0。
# 字段 output_tokens：int；初始表达式 0。
# 字段 prompt_cache_hit_tokens：int；初始表达式 0。
# 字段 prompt_cache_miss_tokens：int；初始表达式 0。
# 字段 cost_usd：float；初始表达式 0.0。
# 字段 model：str | None；初始表达式 None。
# 字段 reasoning：str；初始表达式 ''。
# 字段 reasoning_details：list[dict[str, Any]]；初始表达式 field(default_factory=list)。
# [/maint-note:aa228dc9fc7d]
@dataclass
class ModelResponse:
    content: str = ""
    tool_calls: list[ToolCall] = field(default_factory=list)
    input_tokens: int = 0
    output_tokens: int = 0
    prompt_cache_hit_tokens: int = 0
    prompt_cache_miss_tokens: int = 0
    cost_usd: float = 0.0
    model: str | None = None
    reasoning: str = ""
    reasoning_details: list[dict[str, Any]] = field(default_factory=list)

    # [maint-note:9a228e831551]
    # 维护导航：to_message；下列为本函数源码中的接口与路径索引。
    # 返回契约：dict[str, Any]。
    # 分支条件：self.reasoning_details。
    # 分支条件：self.reasoning。
    # 返回路径：message。
    # 写入目标：message['reasoning_details']。
    # 写入目标：message['reasoning_content']。
    # 直接调用：call.to_dict。
    # 集合派生：call 来自 self.tool_calls。
    # [/maint-note:9a228e831551]
    def to_message(self) -> dict[str, Any]:
        message = {
            "role": "assistant",
            "content": self.content,
            "tool_calls": [call.to_dict() for call in self.tool_calls],
        }
        if self.reasoning_details:
            message["reasoning_details"] = self.reasoning_details
        elif self.reasoning:
            message["reasoning_content"] = self.reasoning
        return message


# [maint-note:36ed4b8023de]
# 数据与行为导航：ModelStreamEvent。
# 字段 kind：str。
# 字段 text：str；初始表达式 ''。
# 字段 response：ModelResponse | None；初始表达式 None。
# [/maint-note:36ed4b8023de]
@dataclass(frozen=True)
class ModelStreamEvent:
    kind: str
    text: str = ""
    response: ModelResponse | None = None


# [maint-note:0f7a6723500c]
# 数据与行为导航：RunBudget。
# 字段 max_llm_calls：int；初始表达式 100。
# 字段 max_compaction_calls：int；初始表达式 16。
# 字段 max_tool_calls：int；初始表达式 100。
# 字段 max_input_tokens：int；初始表达式 120000。
# 字段 max_output_tokens：int；初始表达式 24000。
# 字段 max_seconds：int；初始表达式 1800。
# 字段 max_cost_usd：float | None；初始表达式 None。
# [/maint-note:0f7a6723500c]
@dataclass
class RunBudget:
    max_llm_calls: int = 100
    max_compaction_calls: int = 16
    max_tool_calls: int = 100
    max_input_tokens: int = 120_000
    max_output_tokens: int = 24_000
    max_seconds: int = 1800
    max_cost_usd: float | None = None

    # [maint-note:4c903027057f]
    # 维护导航：to_dict；下列为本函数源码中的接口与路径索引。
    # 返回契约：dict[str, Any]。
    # 返回路径：asdict(self)。
    # 直接调用：asdict。
    # [/maint-note:4c903027057f]
    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    # [maint-note:31be5234dd84]
    # 维护导航：from_dict；下列为本函数源码中的接口与路径索引。
    # 输入 value：dict[str, Any] | None。
    # 返回契约：'RunBudget'。
    # 返回路径：cls(**{key: val for (key, val) in source.items() if key in allowed})。
    # 直接调用：__import__('dataclasses').fields。
    # 直接调用：__import__。
    # 直接调用：cls。
    # 直接调用：source.items。
    # 集合派生：field 来自 __import__('dataclasses').fields(cls)。
    # 集合派生：(key, val) 来自 source.items()；过滤 key in allowed。
    # [/maint-note:31be5234dd84]
    @classmethod
    def from_dict(cls, value: dict[str, Any] | None) -> "RunBudget":
        source = value or {}
        allowed = {field.name for field in __import__("dataclasses").fields(cls)}
        return cls(**{key: val for key, val in source.items() if key in allowed})


# [maint-note:5678b4ee5abd]
# 数据与行为导航：ToolContext。
# 字段 run_id：str。
# 字段 workspace：Path。
# 字段 artifact_root：Path。
# 字段 approved_risks：set[RiskLevel]；初始表达式 field(default_factory=set)。
# 字段 authorized_paths：set[Path]；初始表达式 field(default_factory=set)。
# 字段 cancellation：Event；初始表达式 field(default_factory=Event)。
# 字段 metadata：dict[str, Any]；初始表达式 field(default_factory=dict)。
# [/maint-note:5678b4ee5abd]
@dataclass
class ToolContext:
    run_id: str
    workspace: Path
    artifact_root: Path
    approved_risks: set[RiskLevel] = field(default_factory=set)
    authorized_paths: set[Path] = field(default_factory=set)
    cancellation: Event = field(default_factory=Event)
    metadata: dict[str, Any] = field(default_factory=dict)

    # [maint-note:1ce1fd7e251d]
    # 维护导航：__post_init__；下列为本函数源码中的接口与路径索引。
    # 返回契约：None。
    # 写入目标：self.workspace。
    # 写入目标：self.artifact_root。
    # 写入目标：self.authorized_paths。
    # 直接调用：Path(self.workspace).expanduser().resolve。
    # 直接调用：Path(self.workspace).expanduser。
    # 直接调用：Path。
    # 直接调用：Path(self.artifact_root).expanduser().resolve。
    # 直接调用：Path(self.artifact_root).expanduser。
    # 直接调用：Path(path).expanduser().resolve。
    # 直接调用：Path(path).expanduser。
    # 直接调用：self.artifact_root.mkdir。
    # 集合派生：path 来自 self.authorized_paths。
    # [/maint-note:1ce1fd7e251d]
    def __post_init__(self) -> None:
        self.workspace = Path(self.workspace).expanduser().resolve()
        self.artifact_root = Path(self.artifact_root).expanduser().resolve()
        self.authorized_paths = {
            Path(path).expanduser().resolve()
            for path in self.authorized_paths
        }
        self.artifact_root.mkdir(parents=True, exist_ok=True)


# [maint-note:9ee697fcf1e4]
# 数据与行为导航：ToolResult。
# 字段 status：str。
# 字段 summary：str。
# 字段 data：dict[str, Any]；初始表达式 field(default_factory=dict)。
# 字段 artifacts：list[str]；初始表达式 field(default_factory=list)。
# 字段 changed_files：list[str]；初始表达式 field(default_factory=list)。
# 字段 exit_code：int | None；初始表达式 None。
# 字段 truncated：bool；初始表达式 False。
# 字段 error：dict[str, Any] | None；初始表达式 None。
# 字段 started_at：str；初始表达式 field(default_factory=now_iso)。
# 字段 finished_at：str；初始表达式 field(default_factory=now_iso)。
# [/maint-note:9ee697fcf1e4]
@dataclass
class ToolResult:
    status: str
    summary: str
    data: dict[str, Any] = field(default_factory=dict)
    artifacts: list[str] = field(default_factory=list)
    changed_files: list[str] = field(default_factory=list)
    exit_code: int | None = None
    truncated: bool = False
    error: dict[str, Any] | None = None
    started_at: str = field(default_factory=now_iso)
    finished_at: str = field(default_factory=now_iso)

    # [maint-note:6fc946103fe7]
    # 维护导航：success；下列为本函数源码中的接口与路径索引。
    # 输入 summary：str。
    # 输入 data：dict[str, Any] | None；缺省表达式 None。
    # 输入 artifacts：list[str] | None；缺省表达式 None。
    # 输入 changed_files：list[str] | None；缺省表达式 None。
    # 输入 exit_code：int | None；缺省表达式 None。
    # 返回契约：'ToolResult'。
    # 返回路径：cls('success', summary, data or {}, artifacts or [], changed_files or [], exit_code)。
    # 直接调用：cls。
    # [/maint-note:6fc946103fe7]
    @classmethod
    def success(
        cls,
        summary: str,
        *,
        data: dict[str, Any] | None = None,
        artifacts: list[str] | None = None,
        changed_files: list[str] | None = None,
        exit_code: int | None = None,
    ) -> "ToolResult":
        return cls(
            "success",
            summary,
            data or {},
            artifacts or [],
            changed_files or [],
            exit_code,
        )

    # [maint-note:0f2b2a240c20]
    # 维护导航：failure；下列为本函数源码中的接口与路径索引。
    # 输入 message：str。
    # 输入 error_type：str；缺省表达式 'ToolError'。
    # 返回契约：'ToolResult'。
    # 返回路径：cls('error', message, error={'type': error_type, 'message': message})。
    # 直接调用：cls。
    # [/maint-note:0f2b2a240c20]
    @classmethod
    def failure(cls, message: str, error_type: str = "ToolError") -> "ToolResult":
        return cls("error", message, error={"type": error_type, "message": message})

    # [maint-note:25ed8fe85fdd]
    # 维护导航：to_dict；下列为本函数源码中的接口与路径索引。
    # 返回契约：dict[str, Any]。
    # 返回路径：asdict(self)。
    # 直接调用：asdict。
    # [/maint-note:25ed8fe85fdd]
    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


ToolExecutor = Callable[[ToolContext, dict[str, Any]], ToolResult]


# [maint-note:9a5ad2ee5021]
# 数据与行为导航：ToolSpec。
# 字段 name：str。
# 字段 description：str。
# 字段 input_schema：dict[str, Any]。
# 字段 risk_level：RiskLevel。
# 字段 execute：ToolExecutor。
# 字段 output_schema：dict[str, Any]；初始表达式 field(default_factory=dict)。
# 字段 requires_approval：bool | None；初始表达式 None。
# 字段 idempotent：bool；初始表达式 True。
# 字段 parallel_safe：bool；初始表达式 True。
# 字段 default_timeout_seconds：int；初始表达式 30。
# 字段 max_output_chars：int；初始表达式 12000。
# [/maint-note:9a5ad2ee5021]
@dataclass(frozen=True)
class ToolSpec:
    name: str
    description: str
    input_schema: dict[str, Any]
    risk_level: RiskLevel
    execute: ToolExecutor
    output_schema: dict[str, Any] = field(default_factory=dict)
    requires_approval: bool | None = None
    idempotent: bool = True
    parallel_safe: bool = True
    default_timeout_seconds: int = 30
    max_output_chars: int = 12_000

    # [maint-note:055fb167a961]
    # 维护导航：__post_init__；下列为本函数源码中的接口与路径索引。
    # 返回契约：None。
    # 分支条件：not self.name or ' ' in self.name。
    # 分支条件：self.max_output_chars <= 0。
    # 显式抛出：ValueError('tool name must be non-empty and contain no spaces')。
    # 显式抛出：ValueError('max_output_chars must be positive')。
    # 直接调用：ValueError。
    # [/maint-note:055fb167a961]
    def __post_init__(self) -> None:
        if not self.name or " " in self.name:
            raise ValueError("tool name must be non-empty and contain no spaces")
        if self.max_output_chars <= 0:
            raise ValueError("max_output_chars must be positive")

    # [maint-note:e12bda7117f7]
    # 维护导航：approval_required；下列为本函数源码中的接口与路径索引。
    # 返回契约：bool。
    # 分支条件：self.requires_approval is not None。
    # 返回路径：self.requires_approval。
    # 返回路径：self.risk_level != RiskLevel.READ。
    # [/maint-note:e12bda7117f7]
    @property
    def approval_required(self) -> bool:
        if self.requires_approval is not None:
            return self.requires_approval
        return self.risk_level != RiskLevel.READ

    # [maint-note:7062e42e276a]
    # 维护导航：model_definition；下列为本函数源码中的接口与路径索引。
    # 返回契约：dict[str, Any]。
    # 返回路径：{'type': 'function', 'function': {'name': self.name, 'description': self.description, 'parameters': self.input_schema}}。
    # [/maint-note:7062e42e276a]
    def model_definition(self) -> dict[str, Any]:
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": self.input_schema,
            },
        }


# [maint-note:585e8aeb4a7d]
# 数据与行为导航：AgentEvent。
# 字段 run_id：str。
# 字段 sequence：int。
# 字段 type：str。
# 字段 payload：dict[str, Any]。
# 字段 created_at：str。
# 字段 idempotency_key：str | None；初始表达式 None。
# [/maint-note:585e8aeb4a7d]
@dataclass(frozen=True)
class AgentEvent:
    run_id: str
    sequence: int
    type: str
    payload: dict[str, Any]
    created_at: str
    idempotency_key: str | None = None

    # [maint-note:50d177bb8865]
    # 维护导航：to_dict；下列为本函数源码中的接口与路径索引。
    # 返回契约：dict[str, Any]。
    # 返回路径：asdict(self)。
    # 直接调用：asdict。
    # [/maint-note:50d177bb8865]
    def to_dict(self) -> dict[str, Any]:
        return asdict(self)
