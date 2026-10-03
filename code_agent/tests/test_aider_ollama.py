from pathlib import Path

import pytest

from code_agent import aider_runner
from code_agent.plugins.base import ExecutionContext
from code_agent.plugins.code_completion import CodeCompletionPlugin
from code_agent.plugins.code_repair import CodeRepairPlugin
from code_agent.plugins.code_summary import CodeSummaryPlugin


def test_local_aider_command_needs_no_cloud_key(tmp_path, monkeypatch):
    monkeypatch.setattr(aider_runner, "ensure_aider_installed", lambda: None)
    monkeypatch.setattr(aider_runner, "generate_completion_prompt", lambda **kwargs: "project context")
    monkeypatch.setenv("DEEPSEEK_API_KEY", "deepseek-secret")
    command, prompt_file, log, _ = aider_runner.build_aider_context_and_command(
        target_files=[], user_instruction="complete", model="ollama_chat/qwen2.5-coder:1.5b",
        api_key="request-cloud-secret", project_dir=str(tmp_path))
    try:
        assert command[command.index("--model") + 1] == "ollama_chat/qwen2.5-coder:1.5b"
        assert "--api-key" not in command
        assert {"--no-check-update", "--no-analytics", "--no-show-release-notes"} <= set(command)
        assert "未提供 API Key" not in log
        assert "deepseek-secret" not in " ".join(command)
        assert "request-cloud-secret" not in " ".join(command)
    finally:
        Path(prompt_file).unlink()
    cloud_command = aider_runner.build_aider_command([], "deepseek/deepseek-chat", "cloud-key", "prompt.txt")
    assert cloud_command[-2:] == ["--api-key", "deepseek=cloud-key"]
    assert "--no-check-update" not in cloud_command


def test_local_aider_env_uses_unversioned_loopback_base_and_excludes_cloud_keys(monkeypatch):
    for name in ("DEEPSEEK_API_KEY", "OPENROUTER_API_KEY", "OPENAI_API_KEY"):
        monkeypatch.setenv(name, name + "-secret")
    monkeypatch.setenv("HTTPS_PROXY", "http://remote-proxy.example:8080")
    monkeypatch.setenv("HTTP_PROXY", "http://remote-proxy.example:8080")
    local = aider_runner.build_subprocess_env("http://127.0.0.1:11434/v1")
    assert local["OLLAMA_API_BASE"] == "http://127.0.0.1:11434"
    assert all(name not in local for name in ("DEEPSEEK_API_KEY", "OPENROUTER_API_KEY", "OPENAI_API_KEY"))
    assert "HTTPS_PROXY" not in local and "HTTP_PROXY" not in local
    with pytest.raises(ValueError, match="loopback"):
        aider_runner.build_subprocess_env("https://remote.example/v1")
    cloud = aider_runner.build_subprocess_env()
    assert cloud["DEEPSEEK_API_KEY"] == "DEEPSEEK_API_KEY-secret"


def test_completion_plugin_starts_aider_with_local_base_and_no_cloud_key(tmp_path, monkeypatch):
    (tmp_path / "main.c").write_text("int f() {}\n", encoding="utf-8")
    monkeypatch.setattr(aider_runner, "ensure_aider_installed", lambda: None)
    monkeypatch.setattr(aider_runner, "generate_completion_prompt", lambda **kwargs: "project context")
    monkeypatch.setenv("DEEPSEEK_API_KEY", "deepseek-secret")
    monkeypatch.setenv("OPENROUTER_API_KEY", "openrouter-secret")
    monkeypatch.setenv("OPENAI_API_KEY", "openai-secret")
    started = {}
    class FakeProcess:
        stdout = iter([b"completed\n"])
        returncode = 0
        def wait(self):
            return 0
    def popen(argv, **kwargs):
        started.update(argv=argv, env=kwargs["env"])
        return FakeProcess()
    monkeypatch.setattr(aider_runner.subprocess, "Popen", popen)
    context = ExecutionContext(str(tmp_path), ["main.c"], "complete f", "ollama_chat/qwen2.5-coder:1.5b",
        "cloud-secret", base_url="http://localhost:11500/v1")
    logs = list(CodeCompletionPlugin().execute(context))
    assert "任务圆满完成" in logs[-1]
    assert started["env"]["OLLAMA_API_BASE"] == "http://localhost:11500"
    assert "--api-key" not in started["argv"]
    assert {"--no-check-update", "--no-analytics", "--no-show-release-notes"} <= set(started["argv"])
    assert all(name not in started["env"] for name in ("DEEPSEEK_API_KEY", "OPENROUTER_API_KEY", "OPENAI_API_KEY"))


def test_local_cli_respects_existing_loopback_ollama_base(tmp_path, monkeypatch):
    monkeypatch.setattr(aider_runner, "ensure_aider_installed", lambda: None)
    monkeypatch.setattr(aider_runner, "generate_completion_prompt", lambda **kwargs: "project context")
    monkeypatch.setenv("OLLAMA_API_BASE", "http://localhost:11501")
    started = {}
    monkeypatch.setattr(aider_runner.subprocess, "run", lambda argv, **kwargs:
        started.update(argv=argv, env=kwargs["env"]))
    aider_runner.run_aider_cli([], "complete", "ollama_chat/qwen2.5-coder:1.5b", None,
        project_dir=str(tmp_path))
    assert started["env"]["OLLAMA_API_BASE"] == "http://localhost:11501"
    assert "--api-key" not in started["argv"]
    monkeypatch.setenv("OLLAMA_API_BASE", "https://remote.example")
    with pytest.raises(ValueError, match="loopback"):
        aider_runner.run_aider_cli([], "complete", "ollama_chat/qwen2.5-coder:1.5b", None,
            project_dir=str(tmp_path))


@pytest.mark.parametrize("plugin,runner_module", [
    (CodeRepairPlugin, "code_agent.plugins.code_repair"),
    (CodeSummaryPlugin, "code_agent.plugins.code_summary"),
])
def test_other_aider_plugins_forward_local_base(tmp_path, monkeypatch, plugin, runner_module):
    (tmp_path / "main.c").write_text("int f() {}\n", encoding="utf-8")
    called = []
    monkeypatch.setattr(runner_module + ".run_aider_stream", lambda **kwargs:
        called.append(kwargs) or iter(["done"]))
    context = ExecutionContext(str(tmp_path), ["main.c"], "repair", "ollama_chat/qwen2.5-coder:1.5b",
        None, base_url="http://localhost:11501/v1")
    assert list(plugin().execute(context)) == ["done"]
    assert called[0]["base_url"] == "http://localhost:11501/v1"
