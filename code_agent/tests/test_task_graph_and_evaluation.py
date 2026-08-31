from pathlib import Path

import pytest

from code_agent.agent_core.evaluation import EvaluationReport
from code_agent.agent_core.task_graph import TaskGraph, TaskNode, discover_test_commands


def test_task_graph_ready_nodes_and_cycle_rejection():
    graph = TaskGraph(
        [
            TaskNode("read", [], True),
            TaskNode("search", [], True),
            TaskNode("edit", ["read", "search"], False),
        ]
    )
    assert {node.id for node in graph.ready(set())} == {"read", "search"}
    assert [node.id for node in graph.ready({"read", "search"})] == ["edit"]
    with pytest.raises(ValueError, match="cycle"):
        TaskGraph([TaskNode("a", ["b"], True), TaskNode("b", ["a"], True)])


def test_command_discovery_and_metrics(tmp_path: Path):
    (tmp_path / "pyproject.toml").write_text("[project]\nname='x'\n", encoding="utf-8")
    (tmp_path / "package.json").write_text('{"scripts":{"test":"vitest"}}', encoding="utf-8")
    commands = discover_test_commands(tmp_path)
    assert ["python", "-m", "pytest"] in commands
    assert ["npm", "test"] in commands

    report = EvaluationReport()
    report.record(success=True, unsafe_escape=False, duplicate_side_effect=False, recovered=True, tool_rounds=2)
    assert report.summary()["task_success_rate"] == 1.0
    assert report.summary()["unsafe_escape_rate"] == 0.0
