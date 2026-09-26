import asyncio

from code_agent import agent_web_api


def test_bootstrap_exposes_openrouter_runtime_defaults(monkeypatch):
    monkeypatch.setenv("CODE_AGENT_PROVIDER", "openrouter")
    monkeypatch.setenv("CODE_AGENT_MODEL", "anthropic/claude-sonnet-4.5")
    monkeypatch.setenv("CODE_AGENT_API_BASE", "https://openrouter.ai/api/v1")

    payload = asyncio.run(agent_web_api.bootstrap())

    assert payload["default_model"] == "deepseek/deepseek-chat"
    assert payload["runtime_default_model_config"]["provider"] == "openrouter"
    assert payload["runtime_default_model_config"]["model"] == "anthropic/claude-sonnet-4.5"
    assert payload["runtime_provider_defaults"]["openrouter"]["model"] == "deepseek/deepseek-chat"
