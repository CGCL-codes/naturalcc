from __future__ import annotations

import os
from enum import Enum

from .contracts import RiskLevel


# [maint-note:a1bd8670732b]
# 数据与行为导航：PolicyDecision。
# 继承接口：str, Enum。
# [/maint-note:a1bd8670732b]
class PolicyDecision(str, Enum):
    ALLOW = "allow"
    REQUIRE_APPROVAL = "require_approval"
    DENY = "deny"


class PolicyEngine:
    # [maint-note:393a8c4baff2]
    # 维护导航：__init__；下列为本函数源码中的接口与路径索引。
    # 输入 denied_tools：set[str] | None；缺省表达式 None。
    # 返回契约：None。
    # 分支条件：denied_tools is None。
    # 写入目标：self.denied_tools。
    # 写入目标：self._grants。
    # 直接调用：name.strip。
    # 直接调用：os.environ.get('CODE_AGENT_DENIED_TOOLS', '').split。
    # 直接调用：os.environ.get。
    # 直接调用：set。
    # 直接调用：name.replace。
    # 集合派生：name 来自 os.environ.get('CODE_AGENT_DENIED_TOOLS', '').split(',')；过滤 name.strip()。
    # 集合派生：name 来自 denied_tools。
    # [/maint-note:393a8c4baff2]
    def __init__(self, denied_tools: set[str] | None = None) -> None:
        if denied_tools is None:
            # Opt-in hard deny list (canonical names, comma-separated), e.g. for unattended
            # benchmarks that approve one execute tool: approvals are granted per risk level.
            denied_tools = {name.strip() for name in os.environ.get("CODE_AGENT_DENIED_TOOLS", "").split(",") if name.strip()}
        # Model-facing names replace dots with underscores; deny both spellings.
        self.denied_tools = set(denied_tools) | {name.replace(".", "_") for name in denied_tools}
        self._grants: dict[str, set[RiskLevel]] = {}

    # [maint-note:243b5dbce813]
    # 维护导航：grant；下列为本函数源码中的接口与路径索引。
    # 输入 run_id：str。
    # 输入 risk：RiskLevel。
    # 返回契约：None。
    # 直接调用：self._grants.setdefault(run_id, set()).add。
    # 直接调用：self._grants.setdefault。
    # 直接调用：set。
    # [/maint-note:243b5dbce813]
    def grant(self, run_id: str, risk: RiskLevel) -> None:
        self._grants.setdefault(run_id, set()).add(risk)

    # [maint-note:4d4dd0c3074f]
    # 维护导航：revoke；下列为本函数源码中的接口与路径索引。
    # 输入 run_id：str。
    # 输入 risk：RiskLevel。
    # 返回契约：None。
    # 直接调用：self._grants.setdefault(run_id, set()).discard。
    # 直接调用：self._grants.setdefault。
    # 直接调用：set。
    # [/maint-note:4d4dd0c3074f]
    def revoke(self, run_id: str, risk: RiskLevel) -> None:
        self._grants.setdefault(run_id, set()).discard(risk)

    # [maint-note:5cc53fec9666]
    # 维护导航：grants_for；下列为本函数源码中的接口与路径索引。
    # 输入 run_id：str。
    # 返回契约：set[RiskLevel]。
    # 返回路径：set(self._grants.get(run_id, set()))。
    # 直接调用：set。
    # 直接调用：self._grants.get。
    # [/maint-note:5cc53fec9666]
    def grants_for(self, run_id: str) -> set[RiskLevel]:
        return set(self._grants.get(run_id, set()))

    # [maint-note:f4b8f5dc37f9]
    # 权限顺序：显式拒绝优先于已授予的风险级别。
    # 只读操作可直接进行；写入与执行需要当前 Run 的授权。
    # 模型生成的参数不会自动扩大用户授予的权限。
    # 维护导航：decide；下列为本函数源码中的接口与路径索引。
    # 输入 run_id：str。
    # 输入 tool_name：str。
    # 输入 risk：RiskLevel。
    # 返回契约：PolicyDecision。
    # 分支条件：tool_name in self.denied_tools。
    # 分支条件：risk == RiskLevel.READ or risk in self._grants.get(run_id, set())。
    # 返回路径：PolicyDecision.DENY。
    # 返回路径：PolicyDecision.ALLOW。
    # 返回路径：PolicyDecision.REQUIRE_APPROVAL。
    # 直接调用：self._grants.get。
    # 直接调用：set。
    # [/maint-note:f4b8f5dc37f9]
    def decide(self, run_id: str, tool_name: str, risk: RiskLevel) -> PolicyDecision:
        if tool_name in self.denied_tools:
            return PolicyDecision.DENY
        if risk == RiskLevel.READ or risk in self._grants.get(run_id, set()):
            return PolicyDecision.ALLOW
        return PolicyDecision.REQUIRE_APPROVAL
