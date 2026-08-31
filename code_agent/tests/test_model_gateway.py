import sys
from types import SimpleNamespace

from code_agent.agent_core.contracts import ModelRequest, ModelResponse
from code_agent.agent_core.model_gateway import (
    DeepSeekRequestSerializer,
    OpenAICompatibleGateway,
    ScriptedModelGateway,
)


def test_serializer_preserves_tool_protocol():
    serializer = DeepSeekRequestSerializer()
    messages = [
        {
            "role": "assistant",
            "content": "",
            "tool_calls": [
                {
                    "id": "c1",
                    "name": "workspace.read",
                    "args": {"path": "README.md"},
                }
            ],
        },
        {
            "role": "tool",
            "tool_call_id": "c1",
            "name": "workspace.read",
            "content": "ok",
        },
    ]

    result = serializer.serialize_messages(messages)

    assert result[0]["tool_calls"][0]["function"]["arguments"] == (
        '{"path": "README.md"}'
    )
    assert result[1]["tool_call_id"] == "c1"


def test_scripted_gateway_records_request_purpose_and_output_limit():
    gateway = ScriptedModelGateway([ModelResponse(content='{"ok": true}')])
    request = ModelRequest(
        messages=[{"role": "user", "content": "analyze"}],
        tools=[],
        purpose="compaction_analysis",
        max_output_tokens=1024,
        response_format={"type": "json_object"},
    )

    gateway.generate(request)

    assert gateway.requests[0].purpose == "compaction_analysis"
    assert gateway.requests[0].max_output_tokens == 1024


def test_openai_gateway_records_prompt_cache_usage(monkeypatch):
    usage = SimpleNamespace(
        prompt_tokens=120,
        completion_tokens=15,
        model_extra={
            "prompt_cache_hit_tokens": 90,
            "prompt_cache_miss_tokens": 30,
        },
    )
    response = SimpleNamespace(
        choices=[
            SimpleNamespace(
                message=SimpleNamespace(content="done", tool_calls=[])
            )
        ],
        usage=usage,
    )
    completions = SimpleNamespace(create=lambda **kwargs: response)
    client = SimpleNamespace(chat=SimpleNamespace(completions=completions))
    monkeypatch.setitem(
        sys.modules,
        "openai",
        SimpleNamespace(OpenAI=lambda **kwargs: client),
    )
    gateway = OpenAICompatibleGateway("deepseek-chat", api_key="test-key")

    result = gateway.generate(
        ModelRequest(messages=[{"role": "user", "content": "hello"}])
    )

    assert result.input_tokens == 120
    assert result.prompt_cache_hit_tokens == 90
    assert result.prompt_cache_miss_tokens == 30


def test_openai_gateway_prefers_request_scoped_api_key(monkeypatch):
    captured = {}
    usage = SimpleNamespace(prompt_tokens=1, completion_tokens=1, model_extra={})
    response = SimpleNamespace(
        choices=[
            SimpleNamespace(
                message=SimpleNamespace(content="done", tool_calls=[])
            )
        ],
        usage=usage,
    )
    completions = SimpleNamespace(create=lambda **kwargs: response)
    client = SimpleNamespace(chat=SimpleNamespace(completions=completions))

    def openai_client(**kwargs):
        captured.update(kwargs)
        return client

    monkeypatch.setitem(
        sys.modules,
        "openai",
        SimpleNamespace(OpenAI=openai_client),
    )
    monkeypatch.setenv("DEEPSEEK_API_KEY", "env-key")
    gateway = OpenAICompatibleGateway("deepseek-chat")

    gateway.generate(
        ModelRequest(
            messages=[{"role": "user", "content": "hello"}],
            metadata={"api_key": "request-key"},
        )
    )

    assert captured["api_key"] == "request-key"
