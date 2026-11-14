"""Durable, policy-controlled runtime for the NaturalCC coding agent."""

# 包入口只重导出轻量协议类型，导入协议时不创建运行引擎或数据库连接。
# 运行引擎、网关与存储由应用装配，避免包导入承担服务初始化副作用。
from .contracts import (
    AgentEvent,
    ModelResponse,
    RiskLevel,
    RunBudget,
    RunStatus,
    ToolCall,
    ToolContext,
    ToolResult,
    ToolSpec,
)

# 固定公开类型清单，调用方无需依赖 contracts 模块的内部辅助符号。
# ToolCall 表达请求，ToolResult 表达执行结果，授权状态由运行时另行管理。
# RunBudget 和 RunStatus 共享同一协议，供 API 与持久化层交换任务状态。
__all__ = [
    "AgentEvent",
    "ModelResponse",
    "RiskLevel",
    "RunBudget",
    "RunStatus",
    "ToolCall",
    "ToolContext",
    "ToolResult",
    "ToolSpec",
]
