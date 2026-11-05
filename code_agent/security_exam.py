"""Contract-3 exam difficulty: rule switches that apply only while Contract 3 is evaluated.

The switches A-F (see 合同-3-静态分析难度调节指南.md) lower detection or raise false
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
CONTRACT3_OFF = frozenset({"A", "C", "E", "F"})

SWITCH_NAMES = ("A", "B", "C", "D", "E", "F")
_ACTIVE: ContextVar[frozenset] = ContextVar("contract3_exam_off", default=frozenset())


def switch_off(name: str) -> bool:
    """True when switch ``name`` is turned off for the evaluation in progress."""
    return name in _ACTIVE.get()


def active_switches_off() -> frozenset:
    return _ACTIVE.get()


@contextmanager
def contract3_exam(enabled: bool = True):
    """Apply CONTRACT3_OFF for the enclosed synchronous block (no-op when not enabled)."""
    token = _ACTIVE.set(CONTRACT3_OFF if enabled else frozenset())
    try:
        yield
    finally:
        _ACTIVE.reset(token)
