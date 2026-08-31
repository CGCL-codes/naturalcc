from pathlib import Path

import pytest

from code_agent.api.agent_routes import build_default_context_runtime


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
