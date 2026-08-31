from pathlib import Path

import pytest

from code_agent.agent_core.contracts import RiskLevel, ToolContext, ToolResult, ToolSpec
from code_agent.agent_core.tool_registry import ToolRegistry


def test_registry_rejects_unknown_and_invalid_arguments(tmp_path: Path):
    registry = ToolRegistry()
    registry.register(
        ToolSpec(
            name="echo",
            description="echo",
            input_schema={
                "type": "object",
                "properties": {"value": {"type": "string"}},
                "required": ["value"],
                "additionalProperties": False,
            },
            risk_level=RiskLevel.READ,
            execute=lambda _ctx, args: ToolResult.success(args["value"]),
        )
    )
    context = ToolContext("run-1", tmp_path, tmp_path / "artifacts")

    assert registry.execute("echo", {"value": "ok"}, context).summary == "ok"
    assert registry.execute("echo", {"value": 3}, context).status == "error"
    assert registry.execute("echo", {"value": "ok", "project_dir": "/"}, context).status == "error"
    assert registry.execute("missing", {}, context).status == "error"


def test_registry_rejects_duplicate_names():
    registry = ToolRegistry()
    spec = ToolSpec("same", "same", {"type": "object", "properties": {}}, RiskLevel.READ, lambda c, a: ToolResult.success("ok"))
    registry.register(spec)
    with pytest.raises(ValueError, match="already registered"):
        registry.register(spec)


def test_registry_exposes_openai_safe_tool_names_and_resolves_aliases(tmp_path: Path):
    registry = ToolRegistry()
    registry.register(
        ToolSpec(
            "workspace.read",
            "read",
            {"type": "object", "properties": {}},
            RiskLevel.READ,
            lambda c, a: ToolResult.success("read ok"),
        )
    )

    definitions = registry.model_definitions()

    assert definitions[0]["function"]["name"] == "workspace_read"
    assert registry.to_model_name("workspace.read") == "workspace_read"
    assert registry.to_canonical_name("workspace_read") == "workspace.read"
    result = registry.execute("workspace_read", {}, ToolContext("r", tmp_path, tmp_path / "a"))
    assert result.status == "success"
    assert result.summary == "read ok"


def test_registry_truncates_model_facing_output(tmp_path: Path):
    registry = ToolRegistry()
    registry.register(
        ToolSpec(
            "long",
            "long",
            {"type": "object", "properties": {}},
            RiskLevel.READ,
            lambda c, a: ToolResult.success("x" * 20),
            max_output_chars=8,
        )
    )
    result = registry.execute("long", {}, ToolContext("r", tmp_path, tmp_path / "a"))
    assert result.summary == "xxxxxxxx"
    assert result.truncated is True


def test_registry_redacts_credentials_from_tool_results(tmp_path: Path):
    registry = ToolRegistry()
    fake_token = "sk-" + "1234567890abcdefghijklmnop"
    registry.register(
        ToolSpec(
            "secret",
            "secret",
            {"type": "object", "properties": {}},
            RiskLevel.READ,
            lambda c, a: ToolResult.success(
                "API_KEY=super-secret-value",
                data={"nested": {"token": fake_token}},
            ),
        )
    )

    result = registry.execute("secret", {}, ToolContext("r", tmp_path, tmp_path / "a"))

    assert "super-secret-value" not in result.summary
    assert fake_token[:13] not in str(result.data)
    assert "[REDACTED]" in result.summary
