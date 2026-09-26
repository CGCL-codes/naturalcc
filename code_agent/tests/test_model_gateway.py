import sys
from types import SimpleNamespace

from code_agent.agent_core.contracts import ModelRequest, ModelResponse
from code_agent.agent_core.model_gateway import (
    DeepSeekRequestSerializer,
    OpenAICompatibleGateway,
    RoutedModelGateway,
    RuntimeModelConfig,
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


def test_runtime_model_config_uses_openrouter_defaults_and_routes_fallbacks():
    config = RuntimeModelConfig.from_dict(
        {
            "provider": "openrouter",
            "model": "anthropic/claude-sonnet-4.5",
            "fallback_models": ["google/gemini-2.5-pro"],
            "provider_preferences": {"allow_fallbacks": True},
        }
    )

    assert config.base_url == "https://openrouter.ai/api/v1"
    assert config.safety_margin_tokens == 4096
    assert config.openrouter_extra_body() == {
        "models": ["google/gemini-2.5-pro"],
        "provider": {"allow_fallbacks": True},
    }


def test_routed_gateway_uses_explicit_openrouter_config_and_key(monkeypatch):
    captured = {}
    usage = SimpleNamespace(prompt_tokens=10, completion_tokens=2, model_extra={})
    response = SimpleNamespace(
        model="google/gemini-2.5-pro",
        choices=[SimpleNamespace(message=SimpleNamespace(content="done", tool_calls=[]))],
        usage=usage,
    )
    def create_completion(**kwargs):
        captured["request"] = kwargs
        return response

    completions = SimpleNamespace(create=create_completion)
    client = SimpleNamespace(chat=SimpleNamespace(completions=completions))

    def openai_client(**kwargs):
        captured["client"] = kwargs
        return client

    monkeypatch.setitem(sys.modules, "openai", SimpleNamespace(OpenAI=openai_client))
    monkeypatch.setenv("DEEPSEEK_API_KEY", "must-not-be-used")
    gateway = RoutedModelGateway()
    config = RuntimeModelConfig.from_dict(
        {
            "provider": "openrouter",
            "model": "anthropic/claude-sonnet-4.5",
            "fallback_models": ["google/gemini-2.5-pro"],
            "provider_preferences": {"allow_fallbacks": True},
        }
    )

    result = gateway.generate(
        ModelRequest(
            messages=[{"role": "user", "content": "hello"}],
            metadata={
                "api_key": "openrouter-key",
                "runtime_model_config": config.to_dict(),
            },
        )
    )

    assert captured["client"]["api_key"] == "openrouter-key"
    assert captured["client"]["base_url"] == "https://openrouter.ai/api/v1"
    assert captured["request"]["model"] == "anthropic/claude-sonnet-4.5"
    assert captured["request"]["extra_body"] == config.openrouter_extra_body()
    assert result.model == "google/gemini-2.5-pro"


def test_openrouter_stream_reassembles_reasoning_and_tool_calls(monkeypatch):
    captured = {}

    def chunk(delta=None, usage=None):
        return SimpleNamespace(
            model="anthropic/claude-sonnet-4.5",
            choices=[SimpleNamespace(delta=delta)] if delta is not None else [],
            usage=usage,
        )

    chunks = [
        chunk(SimpleNamespace(
            content=None,
            tool_calls=[],
            model_extra={"reasoning_details": [{"index": 0, "type": "reasoning.text", "text": "Check "}]},
        )),
        chunk(SimpleNamespace(
            content=None,
            tool_calls=[SimpleNamespace(index=0, id="call_1", function=SimpleNamespace(name="workspace.read", arguments='{"path":'))],
            model_extra={"reasoning_details": [{"index": 0, "text": "file."}]},
        )),
        chunk(SimpleNamespace(
            content="Reading now.",
            tool_calls=[SimpleNamespace(index=0, id=None, function=SimpleNamespace(name=None, arguments='"x.py"}'))],
            model_extra={},
        )),
        chunk(SimpleNamespace(
            content=None,
            tool_calls=[],
            model_extra={"reasoning_details": [{"index": 1, "type": "reasoning.encrypted", "data": "opaque-block"}]},
        )),
        chunk(usage=SimpleNamespace(prompt_tokens=10, completion_tokens=5, model_extra={})),
    ]

    def create_completion(**kwargs):
        captured.update(kwargs)
        return iter(chunks)

    client = SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(create=create_completion)))
    monkeypatch.setitem(sys.modules, "openai", SimpleNamespace(OpenAI=lambda **kwargs: client))
    gateway = RoutedModelGateway()
    config = RuntimeModelConfig.from_dict({"provider": "openrouter", "model": "anthropic/claude-sonnet-4.5"})
    events = list(gateway.stream(ModelRequest(
        messages=[{"role": "user", "content": "read x.py"}],
        metadata={"api_key": "test-key", "runtime_model_config": config.to_dict()},
    )))

    assert captured["stream"] is True
    assert captured["extra_body"]["reasoning"] == {"enabled": True, "exclude": False}
    assert [event.text for event in events if event.kind == "reasoning"] == ["Check ", "file."]
    response = events[-1].response
    assert response.reasoning == "Check file."
    assert response.reasoning_details[0]["text"] == "Check file."
    assert response.reasoning_details[1]["data"] == "opaque-block"
    assert response.tool_calls[0].args == {"path": "x.py"}
    assert response.input_tokens == 10
    serialized = DeepSeekRequestSerializer().serialize_messages([response.to_message()])
    assert serialized[0]["reasoning_details"] == response.reasoning_details


def test_deepseek_stream_preserves_reasoning_content_for_next_tool_turn(monkeypatch):
    captured = {}

    def create_completion(**kwargs):
        captured.update(kwargs)
        return iter([
            SimpleNamespace(model="deepseek-chat", usage=None, choices=[SimpleNamespace(delta=SimpleNamespace(
                content=None, tool_calls=[], model_extra={"reasoning_content": "Need a file."},
            ))]),
            SimpleNamespace(model="deepseek-chat", usage=None, choices=[SimpleNamespace(delta=SimpleNamespace(
                content="I'll read it.",
                tool_calls=[SimpleNamespace(index=0, id="call_1", function=SimpleNamespace(name="workspace.read", arguments='{"path":"x.py"}'))],
                model_extra={},
            ))]),
            SimpleNamespace(model="deepseek-chat", usage=SimpleNamespace(prompt_tokens=7, completion_tokens=8, model_extra={}), choices=[]),
        ])

    client = SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(create=create_completion)))
    monkeypatch.setitem(sys.modules, "openai", SimpleNamespace(OpenAI=lambda **kwargs: client))
    gateway = RoutedModelGateway()
    config = RuntimeModelConfig.from_dict({"provider": "deepseek", "model": "deepseek-chat"})
    response = list(gateway.stream(ModelRequest(
        messages=[{"role": "user", "content": "read x.py"}],
        metadata={"api_key": "test-key", "runtime_model_config": config.to_dict()},
    )))[-1].response

    assert captured["extra_body"]["thinking"] == {"type": "enabled"}
    assert response.reasoning == "Need a file."
    assert response.tool_calls[0].name == "workspace.read"
    serialized = DeepSeekRequestSerializer().serialize_messages([response.to_message()])
    assert serialized[0]["reasoning_content"] == "Need a file."
