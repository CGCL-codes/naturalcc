from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

from .tools.verification import discover_test_commands


# [maint-note:ada3cde578b2]
# 数据与行为导航：TaskNode。
# 字段 id：str。
# 字段 dependencies：list[str]；初始表达式 field(default_factory=list)。
# 字段 parallel_safe：bool；初始表达式 False。
# [/maint-note:ada3cde578b2]
@dataclass(frozen=True)
class TaskNode:
    id: str
    dependencies: list[str] = field(default_factory=list)
    parallel_safe: bool = False


class TaskGraph:
    # [maint-note:a8dc0ce37792]
    # 维护导航：__init__；下列为本函数源码中的接口与路径索引。
    # 输入 nodes：list[TaskNode]。
    # 返回契约：None。
    # 分支条件：len(self.nodes) != len(nodes)。
    # 分支条件：unknown。
    # 遍历边界：node ← nodes。
    # 显式抛出：ValueError('duplicate task node id')。
    # 显式抛出：ValueError(f'unknown dependencies for {node.id}: {sorted(unknown)}')。
    # 写入目标：self.nodes。
    # 直接调用：len。
    # 直接调用：ValueError。
    # 直接调用：set。
    # 直接调用：sorted。
    # 直接调用：self._assert_acyclic。
    # 集合派生：node 来自 nodes。
    # [/maint-note:a8dc0ce37792]
    def __init__(self, nodes: list[TaskNode]) -> None:
        self.nodes = {node.id: node for node in nodes}
        if len(self.nodes) != len(nodes):
            raise ValueError("duplicate task node id")
        for node in nodes:
            unknown = set(node.dependencies) - set(self.nodes)
            if unknown:
                raise ValueError(f"unknown dependencies for {node.id}: {sorted(unknown)}")
        self._assert_acyclic()

    # [maint-note:37e324641bb4]
    # 维护导航：_assert_acyclic；下列为本函数源码中的接口与路径索引。
    # 返回契约：None。
    # 分支条件：node_id in visiting。
    # 分支条件：node_id in visited。
    # 遍历边界：dependency ← self.nodes[node_id].dependencies。
    # 遍历边界：node_id ← self.nodes。
    # 返回路径：无返回值。
    # 显式抛出：ValueError('task graph contains a cycle')。
    # 直接调用：set。
    # 直接调用：ValueError。
    # 直接调用：visiting.add。
    # 直接调用：visit。
    # 直接调用：visiting.remove。
    # 直接调用：visited.add。
    # [/maint-note:37e324641bb4]
    def _assert_acyclic(self) -> None:
        visiting: set[str] = set()
        visited: set[str] = set()

        # [maint-note:6c76979f6bcd]
        # 维护导航：visit；下列为本函数源码中的接口与路径索引。
        # 输入 node_id：str。
        # 返回契约：None。
        # 分支条件：node_id in visiting。
        # 分支条件：node_id in visited。
        # 遍历边界：dependency ← self.nodes[node_id].dependencies。
        # 返回路径：无返回值。
        # 显式抛出：ValueError('task graph contains a cycle')。
        # 直接调用：ValueError。
        # 直接调用：visiting.add。
        # 直接调用：visit。
        # 直接调用：visiting.remove。
        # 直接调用：visited.add。
        # [/maint-note:6c76979f6bcd]
        def visit(node_id: str) -> None:
            if node_id in visiting:
                raise ValueError("task graph contains a cycle")
            if node_id in visited:
                return
            visiting.add(node_id)
            for dependency in self.nodes[node_id].dependencies:
                visit(dependency)
            visiting.remove(node_id)
            visited.add(node_id)

        for node_id in self.nodes:
            visit(node_id)

    # [maint-note:6d757676d5ed]
    # 维护导航：ready；下列为本函数源码中的接口与路径索引。
    # 输入 completed：set[str]。
    # 输入 running：set[str] | None；缺省表达式 None。
    # 返回契约：list[TaskNode]。
    # 返回路径：[node for node in self.nodes.values() if node.id not in completed and node.id not in active and set(node.dependencies).issubset(completed)]。
    # 直接调用：set。
    # 直接调用：self.nodes.values。
    # 直接调用：set(node.dependencies).issubset。
    # 集合派生：node 来自 self.nodes.values()；过滤 node.id not in completed and node.id not in active and set(node.dependencies).issubset(completed)。
    # [/maint-note:6d757676d5ed]
    def ready(self, completed: set[str], running: set[str] | None = None) -> list[TaskNode]:
        active = running or set()
        return [
            node
            for node in self.nodes.values()
            if node.id not in completed
            and node.id not in active
            and set(node.dependencies).issubset(completed)
        ]

    # [maint-note:5bcd510eb7e1]
    # 维护导航：parallel_read_batch；下列为本函数源码中的接口与路径索引。
    # 输入 completed：set[str]。
    # 返回契约：list[TaskNode]。
    # 返回路径：[node for node in self.ready(completed) if node.parallel_safe]。
    # 直接调用：self.ready。
    # 集合派生：node 来自 self.ready(completed)；过滤 node.parallel_safe。
    # [/maint-note:5bcd510eb7e1]
    def parallel_read_batch(self, completed: set[str]) -> list[TaskNode]:
        return [node for node in self.ready(completed) if node.parallel_safe]


__all__ = ["TaskGraph", "TaskNode", "discover_test_commands"]
