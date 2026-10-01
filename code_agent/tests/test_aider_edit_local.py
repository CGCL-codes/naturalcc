from code_agent.agent_core.contracts import RiskLevel, ToolContext
from code_agent.agent_core.tool_registry import build_default_registry


def test_aider_edit_uses_ollama_run_config_without_cloud_key(tmp_path, monkeypatch):
    (tmp_path / "main.c").write_text("int f() {}\n", encoding="utf-8")
    called = []
    monkeypatch.setattr("code_agent.aider_runner.run_aider_stream", lambda **kwargs:
        called.append(kwargs) or iter(["任务圆满完成"]))
    context = ToolContext("r", tmp_path, tmp_path / "artifacts", approved_risks={RiskLevel.WRITE},
        metadata={"model": "deepseek-chat", "api_key": "cloud-secret", "runtime_model_config": {
            "provider": "ollama", "model": "qwen2.5-coder:1.5b", "base_url": "http://localhost:11501/v1"}})
    result = build_default_registry().execute("aider.edit",
        {"target_files": ["main.c"], "instruction": "complete f"}, context)
    assert result.status == "success"
    assert called[0]["model"] == "ollama_chat/qwen2.5-coder:1.5b"
    assert called[0]["base_url"] == "http://localhost:11501/v1"
    assert called[0]["api_key"] is None


def test_aider_edit_keeps_existing_cloud_model_and_key(tmp_path, monkeypatch):
    (tmp_path / "main.c").write_text("int f() {}\n", encoding="utf-8")
    called = []
    monkeypatch.setattr("code_agent.aider_runner.run_aider_stream", lambda **kwargs:
        called.append(kwargs) or iter(["任务圆满完成"]))
    context = ToolContext("r", tmp_path, tmp_path / "artifacts", approved_risks={RiskLevel.WRITE},
        metadata={"model": "deepseek-chat", "api_key": "cloud-key"})
    result = build_default_registry().execute("aider.edit",
        {"target_files": ["main.c"], "instruction": "complete f"}, context)
    assert result.status == "success"
    assert called[0]["model"] == "deepseek-chat"
    assert called[0]["api_key"] == "cloud-key"
