from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

from .tools.verification import discover_test_commands


@dataclass(frozen=True)
class TaskNode:
    id: str
    dependencies: list[str] = field(default_factory=list)
    parallel_safe: bool = False


class TaskGraph:
    def __init__(self, nodes: list[TaskNode]) -> None:
        self.nodes = {node.id: node for node in nodes}
        if len(self.nodes) != len(nodes):
            raise ValueError("duplicate task node id")
        for node in nodes:
            unknown = set(node.dependencies) - set(self.nodes)
            if unknown:
                raise ValueError(f"unknown dependencies for {node.id}: {sorted(unknown)}")
        self._assert_acyclic()

    def _assert_acyclic(self) -> None:
        visiting: set[str] = set()
        visited: set[str] = set()

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

    def ready(self, completed: set[str], running: set[str] | None = None) -> list[TaskNode]:
        active = running or set()
        return [
            node
            for node in self.nodes.values()
            if node.id not in completed
            and node.id not in active
            and set(node.dependencies).issubset(completed)
        ]

    def parallel_read_batch(self, completed: set[str]) -> list[TaskNode]:
        return [node for node in self.ready(completed) if node.parallel_safe]


__all__ = ["TaskGraph", "TaskNode", "discover_test_commands"]
