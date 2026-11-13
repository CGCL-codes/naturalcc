from __future__ import annotations

from dataclasses import dataclass, field
from statistics import fmean


# [maint-note:6ceec56b5256]
# 数据与行为导航：EvaluationReport。
# 字段 records：list[dict]；初始表达式 field(default_factory=list)。
# [/maint-note:6ceec56b5256]
@dataclass
class EvaluationReport:
    records: list[dict] = field(default_factory=list)

    # [maint-note:c6bc3518c839]
    # 维护导航：record；下列为本函数源码中的接口与路径索引。
    # 输入 success：bool。
    # 输入 unsafe_escape：bool。
    # 输入 duplicate_side_effect：bool。
    # 输入 recovered：bool。
    # 输入 tool_rounds：int。
    # 返回契约：None。
    # 直接调用：self.records.append。
    # 直接调用：bool。
    # 直接调用：int。
    # [/maint-note:c6bc3518c839]
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

    # [maint-note:cfc68670fedc]
    # 维护导航：summary；下列为本函数源码中的接口与路径索引。
    # 返回契约：dict。
    # 分支条件：not self.records。
    # 返回路径：{'cases': 0, 'task_success_rate': 0.0, 'unsafe_escape_rate': 0.0, 'duplicate_side_effect_rate': 0.0, 'recovery_rate': 0.0, 'average_tool_rounds': 0.0}。
    # 返回路径：{'cases': count, 'task_success_rate': sum((record['success'] for record in self.records)) / count, 'unsafe_escape_rate': sum((record['unsafe_escape'] for record in self.records)) / count, 'duplicate_side_effect_rate': sum((record['duplicate。
    # 直接调用：len。
    # 直接调用：sum。
    # 直接调用：fmean。
    # 集合派生：record 来自 self.records。
    # [/maint-note:cfc68670fedc]
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
