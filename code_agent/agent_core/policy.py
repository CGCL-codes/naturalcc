from __future__ import annotations

from enum import Enum

from .contracts import RiskLevel


class PolicyDecision(str, Enum):
    ALLOW = "allow"
    REQUIRE_APPROVAL = "require_approval"
    DENY = "deny"


class PolicyEngine:
    def __init__(self, denied_tools: set[str] | None = None) -> None:
        self.denied_tools = denied_tools or set()
        self._grants: dict[str, set[RiskLevel]] = {}

    def grant(self, run_id: str, risk: RiskLevel) -> None:
        self._grants.setdefault(run_id, set()).add(risk)

    def revoke(self, run_id: str, risk: RiskLevel) -> None:
        self._grants.setdefault(run_id, set()).discard(risk)

    def grants_for(self, run_id: str) -> set[RiskLevel]:
        return set(self._grants.get(run_id, set()))

    def decide(self, run_id: str, tool_name: str, risk: RiskLevel) -> PolicyDecision:
        if tool_name in self.denied_tools:
            return PolicyDecision.DENY
        if risk == RiskLevel.READ or risk in self._grants.get(run_id, set()):
            return PolicyDecision.ALLOW
        return PolicyDecision.REQUIRE_APPROVAL
