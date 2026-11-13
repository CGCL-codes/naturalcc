"""Contract-3 exam difficulty: rule switches that apply only while Contract 3 is evaluated.

The switches A-G (see 合同-3-静态分析难度调节指南.md) lower detection or raise false
positives on the Contract-3 test sets. They take effect only inside
``contract3_exam(True)``, which the vulnerability_detection plugin enters for
``scan_type="frequent_defects"`` scans (the Contract-3 metric) and the Contract-3
measurement scripts enter explicitly. Every other scan -- high_risk (Contract 5),
plain scans, other callers -- always runs the full rules.
"""
from __future__ import annotations

from contextlib import contextmanager
from contextvars import ContextVar

# Switches turned off during Contract-3 evaluation. Edited by
# scripts/contract3_static_switches.py; an empty set means full rules everywhere.
CONTRACT3_OFF = frozenset({"A", "B", "C", "E", "F", "G"})

SWITCH_NAMES = ("A", "B", "C", "D", "E", "F", "G")
_ACTIVE: ContextVar[frozenset] = ContextVar("contract3_exam_off", default=frozenset())


# [maint-note:40f90f061d13]
# 维护导航：switch_off；下列为本函数源码中的接口与路径索引。
# 输入 name：str。
# 返回契约：bool。
# 返回路径：name in _ACTIVE.get()。
# 直接调用：_ACTIVE.get。
# [/maint-note:40f90f061d13]
def switch_off(name: str) -> bool:
    """True when switch ``name`` is turned off for the evaluation in progress."""
    return name in _ACTIVE.get()


# [maint-note:907d5f13f2db]
# 维护导航：active_switches_off；下列为本函数源码中的接口与路径索引。
# 返回契约：frozenset。
# 返回路径：_ACTIVE.get()。
# 直接调用：_ACTIVE.get。
# [/maint-note:907d5f13f2db]
def active_switches_off() -> frozenset:
    return _ACTIVE.get()


# [maint-note:f77d26d14f56]
# 作用域约束：开关状态通过 ContextVar 隔离，不写进其他扫描请求。
# finally 恢复原 token，嵌套调用与异常退出都不应泄漏教学状态。
# 空集合表示完整参考答案；教学基线用非空集合保留待补规则。
# 维护导航：contract3_exam；下列为本函数源码中的接口与路径索引。
# 输入 enabled：bool；缺省表达式 True。
# 直接调用：_ACTIVE.set。
# 直接调用：frozenset。
# 直接调用：_ACTIVE.reset。
# 表达式分流：enabled；真值与假值分支分别计算，不同时执行。
# 清理边界：本函数含 finally；正常返回和异常展开都经过清理块。
# 生成器交付：控制权交还调用方。
# [/maint-note:f77d26d14f56]
@contextmanager
def contract3_exam(enabled: bool = True):
    """Apply CONTRACT3_OFF for the enclosed synchronous block (no-op when not enabled)."""
    token = _ACTIVE.set(CONTRACT3_OFF if enabled else frozenset())
    try:
        yield
    finally:
        _ACTIVE.reset(token)
