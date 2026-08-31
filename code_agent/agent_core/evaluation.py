from __future__ import annotations

from dataclasses import dataclass, field
from statistics import fmean


@dataclass
class EvaluationReport:
    records: list[dict] = field(default_factory=list)

    def record(
        self,
        *,
        success: bool,
        unsafe_escape: bool,
        duplicate_side_effect: bool,
        recovered: bool,
        tool_rounds: int,
    ) -> None:
        self.records.append(
            {
                "success": bool(success),
                "unsafe_escape": bool(unsafe_escape),
                "duplicate_side_effect": bool(duplicate_side_effect),
                "recovered": bool(recovered),
                "tool_rounds": int(tool_rounds),
            }
        )

    def summary(self) -> dict:
        if not self.records:
            return {
                "cases": 0,
                "task_success_rate": 0.0,
                "unsafe_escape_rate": 0.0,
                "duplicate_side_effect_rate": 0.0,
                "recovery_rate": 0.0,
                "average_tool_rounds": 0.0,
            }
        count = len(self.records)
        return {
            "cases": count,
            "task_success_rate": sum(record["success"] for record in self.records) / count,
            "unsafe_escape_rate": sum(record["unsafe_escape"] for record in self.records) / count,
            "duplicate_side_effect_rate": sum(record["duplicate_side_effect"] for record in self.records) / count,
            "recovery_rate": sum(record["recovered"] for record in self.records) / count,
            "average_tool_rounds": fmean(record["tool_rounds"] for record in self.records),
        }
