from pathlib import Path

import pytest

from code_agent.api.agent_routes import (
    build_default_context_runtime,
    get_default_runtime_model_config,
    get_runtime_provider_defaults,
    normalize_runtime_model_config,
)


def test_default_context_runtime_reads_deepseek_budget_environment(monkeypatch):
    code_agent_root = Path(__file__).resolve().parents[1]
    tokenizer_dir = code_agent_root / "resources" / "deepseek_v3_tokenizer"
    monkeypatch.setenv("CODE_AGENT_TOKENIZER_DIR", str(tokenizer_dir))
    monkeypatch.setenv("CODE_AGENT_CONTEXT_WINDOW_TOKENS", "32768")
    monkeypatch.setenv("CODE_AGENT_OUTPUT_RESERVE_TOKENS", "2048")
    monkeypatch.setenv("CODE_AGENT_CONTEXT_SAFETY_MARGIN_TOKENS", "256")
    monkeypatch.setenv("CODE_AGENT_PROVIDER_FRAMING_TOKENS", "128")
    monkeypatch.setenv("CODE_AGENT_COMPACTION_TRIGGER_RATIO", "0.70")
    monkeypatch.setenv("CODE_AGENT_COMPACTION_TARGET_RATIO", "0.45")

    runtime = build_default_context_runtime(code_agent_root)

    assert runtime.profile.context_window_tokens == 32768
    assert runtime.profile.default_output_reserve_tokens == 2048
    assert runtime.profile.safety_margin_tokens == 256
    assert runtime.profile.provider_framing_tokens == 128
    assert runtime.profile.compaction_trigger_ratio == 0.70
    assert runtime.profile.compaction_target_ratio == 0.45
    assert runtime.counter.count_text("Hello!") == 2


def test_default_context_runtime_rejects_invalid_geometry(monkeypatch):
    code_agent_root = Path(__file__).resolve().parents[1]
    monkeypatch.setenv("CODE_AGENT_COMPACTION_TRIGGER_RATIO", "0.40")
    monkeypatch.setenv("CODE_AGENT_COMPACTION_TARGET_RATIO", "0.70")

    with pytest.raises(ValueError, match="target_ratio"):
        build_default_context_runtime(code_agent_root)


def test_default_context_runtime_uses_openrouter_defaults(monkeypatch):
    code_agent_root = Path(__file__).resolve().parents[1]
    tokenizer_dir = code_agent_root / "resources" / "deepseek_v3_tokenizer"
    monkeypatch.setenv("CODE_AGENT_TOKENIZER_DIR", str(tokenizer_dir))
    monkeypatch.setenv("CODE_AGENT_PROVIDER", "openrouter")
    monkeypatch.setenv("CODE_AGENT_MODEL", "anthropic/claude-sonnet-4.5")
    monkeypatch.delenv("CODE_AGENT_CONTEXT_SAFETY_MARGIN_TOKENS", raising=False)

    runtime = build_default_context_runtime(code_agent_root)

    assert runtime.default_model_config.provider == "openrouter"
    assert runtime.default_model_config.base_url == "https://openrouter.ai/api/v1"
    assert runtime.profile.safety_margin_tokens == 4096


def test_default_runtime_model_config_uses_provider_model_when_not_configured(monkeypatch):
    monkeypatch.setenv("CODE_AGENT_PROVIDER", "openrouter")
    monkeypatch.delenv("CODE_AGENT_MODEL", raising=False)

    config = get_default_runtime_model_config()

    assert config.provider == "openrouter"
    assert config.model == "deepseek/deepseek-chat"


def test_ollama_is_available_in_provider_defaults_without_cloud_settings(monkeypatch):
    monkeypatch.setenv("CODE_AGENT_PROVIDER", "ollama")
    monkeypatch.setenv("CODE_AGENT_API_BASE", "https://api.deepseek.com/v1")
    monkeypatch.delenv("CODE_AGENT_MODEL", raising=False)
    monkeypatch.delenv("CODE_AGENT_CONTEXT_WINDOW_TOKENS", raising=False)
    config = get_default_runtime_model_config()
    assert config.provider == "ollama"
    assert config.base_url == "http://127.0.0.1:11434/v1"
    assert config.context_window_tokens == 8192
    assert get_runtime_provider_defaults()["ollama"]["model"] == "qwen2.5-coder:7b"


def test_switching_thread_provider_to_ollama_uses_local_model_default():
    default = get_default_runtime_model_config()
    selected = normalize_runtime_model_config({"provider": "ollama"}, default,
        legacy_model="deepseek-chat")
    assert selected["model"] == "qwen2.5-coder:7b"
    assert selected["base_url"] == "http://127.0.0.1:11434/v1"
