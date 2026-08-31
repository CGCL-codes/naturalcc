from __future__ import annotations

from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path
from threading import Event
from typing import Any, Callable


def now_iso() -> str:
    return datetime.now(timezone.utc).astimezone().isoformat()


class RiskLevel(str, Enum):
    READ = "read"
    WRITE = "write"
    EXECUTE = "execute"
    NETWORK = "network"
    GIT_WRITE = "git_write"


class RunStatus(str, Enum):
    QUEUED = "queued"
    RUNNING = "running"
    WAITING_APPROVAL = "waiting_approval"
    PAUSED = "paused"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"
    BUDGET_EXHAUSTED = "budget_exhausted"


class StepStatus(str, Enum):
    PROPOSED = "proposed"
    APPROVED = "approved"
    RUNNING = "running"
    SUCCEEDED = "succeeded"
    FAILED = "failed"
    CANCELLED = "cancelled"
    UNCERTAIN = "uncertain"


@dataclass(frozen=True)
class ToolCall:
    id: str
    name: str
    args: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {"id": self.id, "name": self.name, "args": self.args}


@dataclass(frozen=True)
class ModelRequest:
    messages: list[dict[str, Any]]
    tools: list[dict[str, Any]] = field(default_factory=list)
    purpose: str = "agent"
    max_output_tokens: int = 4096
    response_format: dict[str, Any] | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


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

    def to_message(self) -> dict[str, Any]:
        return {
            "role": "assistant",
            "content": self.content,
            "tool_calls": [call.to_dict() for call in self.tool_calls],
        }


@dataclass
class RunBudget:
    max_llm_calls: int = 100
    max_compaction_calls: int = 8
    max_tool_calls: int = 100
    max_input_tokens: int = 120_000
    max_output_tokens: int = 24_000
    max_seconds: int = 1800
    max_cost_usd: float | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, value: dict[str, Any] | None) -> "RunBudget":
        source = value or {}
        allowed = {field.name for field in __import__("dataclasses").fields(cls)}
        return cls(**{key: val for key, val in source.items() if key in allowed})


@dataclass
class ToolContext:
    run_id: str
    workspace: Path
    artifact_root: Path
    approved_risks: set[RiskLevel] = field(default_factory=set)
    authorized_paths: set[Path] = field(default_factory=set)
    cancellation: Event = field(default_factory=Event)
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.workspace = Path(self.workspace).expanduser().resolve()
        self.artifact_root = Path(self.artifact_root).expanduser().resolve()
        self.authorized_paths = {
            Path(path).expanduser().resolve()
            for path in self.authorized_paths
        }
        self.artifact_root.mkdir(parents=True, exist_ok=True)


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

    @classmethod
    def failure(cls, message: str, error_type: str = "ToolError") -> "ToolResult":
        return cls("error", message, error={"type": error_type, "message": message})

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


ToolExecutor = Callable[[ToolContext, dict[str, Any]], ToolResult]


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

    def __post_init__(self) -> None:
        if not self.name or " " in self.name:
            raise ValueError("tool name must be non-empty and contain no spaces")
        if self.max_output_chars <= 0:
            raise ValueError("max_output_chars must be positive")

    @property
    def approval_required(self) -> bool:
        if self.requires_approval is not None:
            return self.requires_approval
        return self.risk_level != RiskLevel.READ

    def model_definition(self) -> dict[str, Any]:
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": self.input_schema,
            },
        }


@dataclass(frozen=True)
class AgentEvent:
    run_id: str
    sequence: int
    type: str
    payload: dict[str, Any]
    created_at: str
    idempotency_key: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)
