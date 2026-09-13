from pathlib import Path

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from code_agent.agent_core.contracts import ModelResponse, RiskLevel, ToolCall, ToolContext
from code_agent.agent_core.event_store import EventStore
from code_agent.agent_core.model_gateway import ScriptedModelGateway
from code_agent.agent_core.run_engine import RunEngine
from code_agent.agent_core.tool_registry import build_default_registry
from code_agent.api.agent_routes import create_agent_router
from code_agent.plugins.base import PluginResult
from code_agent.plugins.code_completion import CodeCompletionPlugin
from code_agent.plugins.vulnerability_detection import VulnerabilityDetectionPlugin


def test_pipeline_registry_and_read_only_scan(tmp_path):
    registry = build_default_registry(False)
    assert registry.get("vulnerability_detection") is not None
    assert registry.get("code_completion") is None
    assert registry.get("vulnerability_detection.fix") is None
    (tmp_path / "main.c").write_text("void f(char *s) { strcpy(buf, s); }", encoding="utf-8")
    context = ToolContext("r", tmp_path, tmp_path / "artifacts")
    result = registry.execute("vulnerability_detection", {"target_files": ["main.c"]}, context)
    assert result.status == "success"
    assert result.data["findings"][0]["rule_id"] == "cwe-120"
    assert not result.changed_files
    assert result.data["coverage"][1]["status"] == "not_requested"


@pytest.mark.parametrize("name", ["code_completion", "vulnerability_detection.fix", "vulnerability_detection.analyze"])
def test_plugin_actions_need_approval(tmp_path, name):
    registry = build_default_registry()
    result = registry.execute(name, {}, ToolContext("r", tmp_path, tmp_path / "artifacts"))
    assert result.error["type"] == "ApprovalRequired"


def test_completion_api_approval_dispatch_metadata_and_snapshot(tmp_path, monkeypatch):
    source = tmp_path / "main.c"
    source.write_text("int f() {}\n", encoding="utf-8")
    called = []
    def execute(self, context):
        called.append(context)
        source.write_text("int f() { return 1; }\n", encoding="utf-8")
        yield "\u4efb\u52a1\u5706\u6ee1\u5b8c\u6210"
    monkeypatch.setattr(CodeCompletionPlugin, "execute", execute)
    model = ScriptedModelGateway([
        ModelResponse(tool_calls=[ToolCall("c", "code_completion", {"target_files": ["main.c"], "instruction": "complete f", "symbol": "f"})]),
        ModelResponse(content="completed"),
    ])
    model.model = "deepseek-chat"
    store = EventStore(tmp_path / "runtime.db")
    engine = RunEngine(store, build_default_registry(), model)
    app = FastAPI()
    app.include_router(create_agent_router(engine))
    client = TestClient(app)
    run_id = client.post("/api/agent/runs", json={"workspace": str(tmp_path), "goal": "complete f", "target_files": ["main.c"], "api_key": "test-key"}).json()["run_id"]
    assert client.post(f"/api/agent/runs/{run_id}/step").json()["status"] == "waiting_approval"
    assert called == []
    client.post(f"/api/agent/runs/{run_id}/approve", json={"risk": "write", "tool_call_id": "c"})
    assert client.post(f"/api/agent/runs/{run_id}/run").json()["status"] == "completed"
    assert called[0].api_key == "test-key"
    assert called[0].model == "deepseek/deepseek-chat"
    assert called[0].feature_config["symbol"] == "f"
    result = [e.payload["result"] for e in store.list_events(run_id) if e.type == "tool.finished"][0]
    assert result["changed_files"] == ["main.c"]
    assert "return 1" in result["data"]["diff"]
    context = ToolContext(run_id, tmp_path, tmp_path / ".code-agent" / "runs" / run_id, approved_risks={RiskLevel.WRITE})
    restored = engine.registry.execute("workspace.restore_snapshot", {"path": "main.c"}, context)
    assert restored.status == "success"
    assert source.read_text() == "int f() {}\n"


def test_partial_mutation_is_recorded_on_plugin_exception(tmp_path, monkeypatch):
    source = tmp_path / "main.c"
    source.write_text("before", encoding="utf-8")
    def execute(self, context):
        source.write_text("after", encoding="utf-8")
        raise RuntimeError("backend failed")
        yield
    monkeypatch.setattr(CodeCompletionPlugin, "execute", execute)
    context = ToolContext("r", tmp_path, tmp_path / "artifacts", approved_risks={RiskLevel.WRITE})
    result = build_default_registry().execute("code_completion", {"target_files": ["main.c"], "instruction": "complete"}, context)
    assert result.status == "error"
    assert result.changed_files == ["main.c"]
    assert result.artifacts


def test_plugin_paths_and_authority_are_not_model_controlled(tmp_path):
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    (tmp_path / "outside.c").write_text("secret", encoding="utf-8")
    (workspace / "a.c").write_text("a", encoding="utf-8")
    (workspace / "b.c").write_text("b", encoding="utf-8")
    context = ToolContext("r", workspace, tmp_path / "artifacts", approved_risks={RiskLevel.WRITE}, metadata={"target_files": ["a.c"]})
    registry = build_default_registry()
    for args in [
        {"target_files": ["../outside.c"], "instruction": "complete"},
        {"target_files": ["b.c"], "instruction": "complete"},
        {"target_files": ["a.c"], "instruction": "complete", "api_key": "injected"},
    ]:
        assert registry.execute("code_completion", args, context).status == "error"


def test_vulnerability_fix_uses_plugin_and_records_actual_changes(tmp_path, monkeypatch):
    source = tmp_path / "main.c"
    source.write_text("before", encoding="utf-8")
    def execute(self, context):
        assert context.feature_config["auto_fix"] is True
        assert context.feature_config["scan_scope"] == "targets"
        source.write_text("after", encoding="utf-8")
        yield PluginResult(success=True, message="fixed", report="report")
    monkeypatch.setattr(VulnerabilityDetectionPlugin, "execute", execute)
    context = ToolContext("r", tmp_path, tmp_path / "artifacts", approved_risks={RiskLevel.EXECUTE})
    result = build_default_registry().execute("vulnerability_detection_fix", {"target_files": ["main.c"]}, context)
    assert result.status == "success"
    assert result.changed_files == ["main.c"]
