from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

from code_agent.agent_core.contracts import ToolContext
from code_agent.agent_core.tools import verification


def test_git_status_uses_utf8_replacement_decoding(
    tmp_path: Path,
    monkeypatch,
) -> None:
    calls = []

    def fake_run(argv, **kwargs):
        calls.append((argv, kwargs))
        stdout = "true\n" if argv[1] == "rev-parse" else " M src/main.py …\n"
        return SimpleNamespace(returncode=0, stdout=stdout, stderr="")

    monkeypatch.setattr(verification.shutil, "which", lambda _name: "git")
    monkeypatch.setattr(verification.subprocess, "run", fake_run)

    result = verification._git_status(ToolContext("run", tmp_path, tmp_path / "artifacts"), {})

    assert result.data["is_repo"] is True
    assert "src/main.py" in result.summary
    assert calls
    assert all(kwargs["encoding"] == "utf-8" for _argv, kwargs in calls)
    assert all(kwargs["errors"] == "replace" for _argv, kwargs in calls)


def test_git_diff_uses_utf8_replacement_decoding(
    tmp_path: Path,
    monkeypatch,
) -> None:
    calls = []

    def fake_run(argv, **kwargs):
        calls.append((argv, kwargs))
        return SimpleNamespace(returncode=0, stdout="diff --git a/x b/x\n+…\n", stderr="")

    monkeypatch.setattr(verification.subprocess, "run", fake_run)

    result = verification._git_diff(ToolContext("run", tmp_path, tmp_path / "artifacts"), {})

    assert "diff --git" in result.summary
    assert calls[0][1]["encoding"] == "utf-8"
    assert calls[0][1]["errors"] == "replace"
