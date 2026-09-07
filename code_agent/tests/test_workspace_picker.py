from __future__ import annotations

import sys
from pathlib import Path
from types import SimpleNamespace

from code_agent import agent_web_api


class FakeTkRoot:
    def __init__(self) -> None:
        self.destroyed = False

    def withdraw(self) -> None:
        pass

    def attributes(self, *_args) -> None:
        pass

    def update(self) -> None:
        pass

    def destroy(self) -> None:
        self.destroyed = True


def install_fake_picker(monkeypatch, selected: str) -> FakeTkRoot:
    root = FakeTkRoot()
    filedialog = SimpleNamespace(askdirectory=lambda **_kwargs: selected)
    tkinter = SimpleNamespace(Tk=lambda: root, filedialog=filedialog)
    monkeypatch.setitem(sys.modules, "tkinter", tkinter)
    monkeypatch.setitem(sys.modules, "tkinter.filedialog", filedialog)
    return root


def test_select_local_directory_returns_selected_workspace(
    tmp_path: Path,
    monkeypatch,
) -> None:
    workspace = tmp_path / "repo"
    workspace.mkdir()
    root = install_fake_picker(monkeypatch, str(workspace))

    result = agent_web_api.select_local_directory(str(tmp_path))

    assert result == {"selected": True, "path": str(workspace.resolve())}
    assert root.destroyed is True


def test_select_local_directory_returns_cancelled_selection(
    tmp_path: Path,
    monkeypatch,
) -> None:
    root = install_fake_picker(monkeypatch, "")

    result = agent_web_api.select_local_directory(str(tmp_path))

    assert result == {"selected": False, "path": ""}
    assert root.destroyed is True
