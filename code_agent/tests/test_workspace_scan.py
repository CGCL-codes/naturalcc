from __future__ import annotations

import os
from pathlib import Path

from code_agent import agent_web_api


def test_windows_device_names_are_ignored() -> None:
    assert agent_web_api.is_windows_device_name("nul")
    assert agent_web_api.is_windows_device_name("NUL.txt")
    assert agent_web_api.is_windows_device_name("com1.log")
    assert not agent_web_api.is_windows_device_name("null.py")


def test_workspace_scan_skips_unrelativizable_entries(
    tmp_path: Path,
    monkeypatch,
) -> None:
    root = str(tmp_path.resolve())
    monkeypatch.setattr(
        agent_web_api.os,
        "walk",
        lambda _root: iter([(root, [], ["valid.py", "broken.py"])]),
    )
    original_relpath = os.path.relpath

    def relpath(path: str, start: str) -> str:
        if path.endswith("broken.py"):
            raise ValueError("path is on a different mount")
        return original_relpath(path, start)

    monkeypatch.setattr(agent_web_api.os.path, "relpath", relpath)

    assert agent_web_api.get_local_files(root) == ["valid.py"]
