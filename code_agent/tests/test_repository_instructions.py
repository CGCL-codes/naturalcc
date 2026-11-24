from pathlib import Path

import pytest

from code_agent.agent_core.repository_instructions import load_repository_instructions


def test_loads_root_and_target_ancestors_in_stable_order(tmp_path: Path):
    workspace = tmp_path / "workspace"
    rules = {
        "AGENTS.md": "Workspace rules",
        "CLAUDE.md": "Compatibility rules",
        "app/AGENTS.md": "App rules",
        "app/core/AGENTS.md": "Core rules",
        "web/AGENTS.md": "Web rules",
        "unselected/AGENTS.md": "Unselected rules",
    }
    for path, body in rules.items():
        file = workspace / path
        file.parent.mkdir(parents=True, exist_ok=True)
        file.write_text(body, encoding="utf-8")
    (tmp_path / "AGENTS.md").write_text("Outside rules", encoding="utf-8")

    def selected(targets: list[str]) -> list[str]:
        return [item["path"] for item in load_repository_instructions(workspace, targets)]

    root_rules = ["CLAUDE.md", "AGENTS.md"]
    target_rules = [*root_rules, "app/AGENTS.md", "web/AGENTS.md", "app/core/AGENTS.md"]
    targets = ["web/index.py", "app/core/main.py", str(tmp_path / "outside.py"), "../outside.py"]
    assert selected([]) == root_rules
    assert selected(targets) == target_rules
    assert selected(list(reversed(targets))) == target_rules


def test_loader_bounds_rule_file_and_keeps_complete_lines(tmp_path: Path):
    (tmp_path / "AGENTS.md").write_text("Keep this complete line.\n" + "测试" * 2_000, encoding="utf-8")

    instructions = load_repository_instructions(tmp_path)
    assert instructions == [{
        "path": "AGENTS.md",
        "content": "Keep this complete line.\n\n[Repository instructions truncated by the runtime.]",
    }]


def test_large_root_rules_leave_room_for_selected_nested_rules(tmp_path: Path):
    for filename in ("CLAUDE.md", "AGENTS.md"):
        (tmp_path / filename).write_text("A root rule.\n" * 500, encoding="utf-8")
    directory = tmp_path / "src"
    directory.mkdir()
    (directory / "AGENTS.md").write_text("Apply the source-tree rule.", encoding="utf-8")

    rules = load_repository_instructions(tmp_path, ["src/main.py"])
    assert [item["path"] for item in rules] == ["CLAUDE.md", "AGENTS.md", "src/AGENTS.md"]
    assert rules[-1]["content"] == "Apply the source-tree rule."


def test_loader_does_not_read_external_link(tmp_path: Path):
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    outside = tmp_path / "outside.md"
    outside.write_text("External content", encoding="utf-8")
    try:
        (workspace / "AGENTS.md").symlink_to(outside)
    except OSError:
        pytest.skip("symbolic links unavailable")

    assert load_repository_instructions(workspace) == []
