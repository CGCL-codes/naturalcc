import sys
from types import SimpleNamespace

import pytest

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
    assert result.prompt_cache_usage_status == "reported"


@pytest.mark.parametrize(
    ("usage", "expected"),
    [
        (None, (0, 0, 0, "unreported")),
        ({"prompt_tokens": 100}, (100, 0, 0, "unreported")),
        ({"prompt_tokens": 100, "prompt_tokens_details": {"cached_tokens": 0}},
         (100, 0, 100, "reported")),
        ({"prompt_tokens": 100, "prompt_cache_hit_tokens": 0, "prompt_cache_miss_tokens": 100,
          "prompt_tokens_details": {"cached_tokens": 60}}, (100, 0, 100, "reported")),
        ({"prompt_tokens": 100, "prompt_cache_hit_tokens": 100, "prompt_cache_miss_tokens": 0},
         (100, 100, 0, "reported")),
        ({"prompt_tokens": 100, "prompt_tokens_details": {"cached_tokens": 60}},
         (100, 60, 40, "reported")),
        ({"prompt_tokens": 100, "prompt_cache_miss_tokens": 40}, (100, 60, 40, "reported")),
        ({"prompt_cache_hit_tokens": 60, "prompt_cache_miss_tokens": 40}, (100, 60, 40, "reported")),
        ({"prompt_tokens_details": {"cached_tokens": 60}}, (0, 60, 0, "partial")),
        ({"prompt_tokens": 100, "prompt_tokens_details": {"cached_tokens": 110}},
         (100, 0, 0, "invalid")),
        ({"prompt_tokens": 100, "prompt_cache_hit_tokens": 90, "prompt_cache_miss_tokens": 30},
         (100, 0, 0, "invalid")),
        ({"prompt_tokens": 100, "prompt_cache_hit_tokens": -1}, (100, 0, 0, "invalid")),
        ({"prompt_tokens": 100, "prompt_cache_hit_tokens": True}, (100, 0, 0, "invalid")),
    ],
)
def test_gateway_distinguishes_reported_unreported_and_inconsistent_cache_usage(monkeypatch, usage, expected):
    response = SimpleNamespace(
        choices=[SimpleNamespace(message=SimpleNamespace(content="done", tool_calls=[]))],
        usage=usage,
    )
    completions = SimpleNamespace(create=lambda **kwargs: response)
    client = SimpleNamespace(chat=SimpleNamespace(completions=completions))
    monkeypatch.setitem(sys.modules, "openai", SimpleNamespace(OpenAI=lambda **kwargs: client))

    result = OpenAICompatibleGateway("deepseek-chat", api_key="test-key").generate(
        ModelRequest(messages=[{"role": "user", "content": "hello"}])
    )

    assert (
        result.input_tokens,
        result.prompt_cache_hit_tokens,
        result.prompt_cache_miss_tokens,
        result.prompt_cache_usage_status,
    ) == expected
    assert result.served_model is None
    assert result.served_provider is None


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


def test_ollama_runtime_config_uses_local_defaults_and_rejects_remote_urls():
    config = RuntimeModelConfig.from_dict({"provider": "ollama", "model": "qwen2.5-coder:1.5b"})
    assert config.base_url == "http://127.0.0.1:11434/v1"
    assert config.model == "qwen2.5-coder:1.5b"
    assert RuntimeModelConfig.from_dict({"provider": "ollama", "base_url": "http://localhost:11434/v1"}).provider == "ollama"
    for url in ("https://example.com/v1", "http://127.0.0.1.evil.test/v1", "https://127.0.0.1:11434/v1",
                "http://localhost:11434",
                "http://user:password@127.0.0.1:11434/v1", "http://192.168.1.2:11434/v1"):
        with pytest.raises(ValueError, match="loopback"):
            RuntimeModelConfig.from_dict({"provider": "ollama", "base_url": url})


def test_ollama_gateway_ignores_cloud_keys_for_generate_and_stream(monkeypatch):
    clients = []
    requests = []
    closed = []
    def create(**kwargs):
        requests.append(kwargs)
        if kwargs.get("stream"):
            return iter([SimpleNamespace(model="qwen2.5-coder:1.5b", usage=None,
                choices=[SimpleNamespace(delta=SimpleNamespace(content="read", tool_calls=[
                    SimpleNamespace(index=0, id="call_1", function=SimpleNamespace(name="workspace.read", arguments='{"path":"x.py"}'))]))])])
        return SimpleNamespace(model="qwen2.5-coder:1.5b", usage=None,
            choices=[SimpleNamespace(message=SimpleNamespace(content="read", tool_calls=[
                SimpleNamespace(id="call_1", function=SimpleNamespace(name="workspace.read", arguments='{"path":"x.py"}'))]))])
    def openai_client(**kwargs):
        clients.append(kwargs)
        return SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(create=create)),
            close=lambda: closed.append(True))
    transport_options = []
    monkeypatch.setattr("httpx.Client", lambda **kwargs: transport_options.append(kwargs) or object())
    monkeypatch.setitem(sys.modules, "openai", SimpleNamespace(OpenAI=openai_client))
    monkeypatch.setenv("DEEPSEEK_API_KEY", "deepseek-secret")
    monkeypatch.setenv("OPENROUTER_API_KEY", "openrouter-secret")
    monkeypatch.setenv("OPENAI_API_KEY", "openai-secret")
    config = RuntimeModelConfig.from_dict({"provider": "ollama", "model": "qwen2.5-coder:1.5b"})
    request = ModelRequest(messages=[{"role": "user", "content": "read x.py"}],
        tools=[{"type": "function", "function": {"name": "workspace.read", "parameters": {"type": "object"}}}],
        metadata={"api_key": "request-cloud-secret", "runtime_model_config": config.to_dict()})
    gateway = RoutedModelGateway()
    assert gateway.generate(request).tool_calls[0].args == {"path": "x.py"}
    streamed = list(gateway.stream(request))
    assert streamed[-1].response.tool_calls[0].args == {"path": "x.py"}
    assert [item["api_key"] for item in clients] == ["ollama", "ollama"]
    assert all(item["base_url"] == "http://127.0.0.1:11434/v1" for item in clients)
    assert requests[0]["model"] == requests[1]["model"] == "qwen2.5-coder:1.5b"
    assert requests[1]["stream"] is True
    assert transport_options == [{"trust_env": False}, {"trust_env": False}]
    assert closed == [True, True]


def test_routed_gateway_uses_explicit_openrouter_config_and_key(monkeypatch):
    captured = {}
    usage = SimpleNamespace(prompt_tokens=10, completion_tokens=2, model_extra={})
    response = SimpleNamespace(
        model="google/gemini-2.5-pro",
        model_extra={"provider": "Google"},
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
    assert result.served_model == "google/gemini-2.5-pro"
    assert result.served_provider == "Google"


@pytest.mark.parametrize(
    ("config_overrides", "purpose", "tuned"),
    [
        ({}, "memory_recall", True),
        ({"base_url": "https://api.deepseek.com"}, "memory_recall", True),
        ({"model": "deepseek-flash", "base_url": "https://api.deepseek.com/v1/"}, "memory_recall", True),
        ({"model": "deepseek-v4-flash"}, "memory_recall", True),
        ({"model": "deepseek-v4-flash-vision-exp"}, "memory_recall", True),
        ({"model": "deepseek-chat"}, "memory_recall", False),
        ({"base_url": "https://api.deepseek.com.example.test/v1"}, "memory_recall", False),
        ({"base_url": "http://api.deepseek.com/v1"}, "memory_recall", False),
        ({"base_url": "https://api.deepseek.com/beta"}, "memory_recall", False),
        ({"provider": "openrouter", "model": "deepseek/deepseek-v4-pro"}, "memory_recall", False),
        ({}, "agent", False),
        ({}, "memory_proposal", False),
    ],
)
def test_routed_gateway_tunes_only_official_deepseek_recall(monkeypatch, config_overrides, purpose, tuned):
    captured = {}
    response = SimpleNamespace(
        choices=[SimpleNamespace(message=SimpleNamespace(content='{"ids":[]}', tool_calls=[]))],
        usage=None,
    )

    def create_completion(**kwargs):
        captured["request"] = kwargs
        return response

    def openai_client(**kwargs):
        captured["client"] = kwargs
        return SimpleNamespace(
            chat=SimpleNamespace(completions=SimpleNamespace(create=create_completion)),
            close=lambda: None,
        )

    monkeypatch.setitem(sys.modules, "openai", SimpleNamespace(OpenAI=openai_client))
    config = RuntimeModelConfig.from_dict(config_overrides)
    request_extras = {"fixture_option": "kept"}
    if tuned:
        request_extras.update(thinking={"type": "enabled"}, temperature=1)
    result = RoutedModelGateway().generate(ModelRequest(
        messages=[{"role": "user", "content": "Select memory IDs as JSON."}],
        purpose=purpose,
        max_output_tokens=256,
        response_format={"type": "json_object"},
        metadata={
            "runtime_model_config": config.to_dict(),
            "api_key": "test-key",
            "timeout_seconds": 18,
            "extra_body": request_extras,
        },
    ))

    request = captured["request"]
    assert request["response_format"] == {"type": "json_object"}
    assert request["tools"] is None
    assert request["max_tokens"] == 256
    assert captured["client"]["timeout"] == 18
    assert captured["client"]["max_retries"] == 0
    assert result.content == '{"ids":[]}'
    if tuned:
        assert request["temperature"] == 0
        assert request["extra_body"] == {"fixture_option": "kept", "thinking": {"type": "disabled"}}
    else:
        assert "temperature" not in request
        assert request["extra_body"] == request_extras


def test_openrouter_stream_reassembles_reasoning_and_tool_calls(monkeypatch):
    captured = {}

    def chunk(delta=None, usage=None, provider=None):
        return SimpleNamespace(
            model="anthropic/claude-sonnet-4.5",
            choices=[SimpleNamespace(delta=delta)] if delta is not None else [],
            usage=usage,
            model_extra={"provider": provider},
        )

    chunks = [
        chunk(SimpleNamespace(
            content=None,
            tool_calls=[],
            model_extra={"reasoning_details": [{"index": 0, "type": "reasoning.text", "text": "Check "}]},
        ), provider="Anthropic"),
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
        chunk(usage=SimpleNamespace(prompt_tokens=10, completion_tokens=5,
                                    prompt_tokens_details={"cached_tokens": 6}, model_extra={})),
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
    assert response.prompt_cache_hit_tokens == 6
    assert response.prompt_cache_miss_tokens == 4
    assert response.prompt_cache_usage_status == "reported"
    assert response.served_model == "anthropic/claude-sonnet-4.5"
    assert response.served_provider == "Anthropic"
    serialized = DeepSeekRequestSerializer().serialize_messages([response.to_message()])
    assert serialized[0]["reasoning_details"] == response.reasoning_details


def test_bounded_stream_disables_sdk_retries_and_closes_on_deadline(monkeypatch):
    captured = {}
    ticks = iter([10.0, 12.0])
    monkeypatch.setattr(
        "code_agent.agent_core.model_gateway.time",
        SimpleNamespace(monotonic=lambda: next(ticks)),
    )

    def client(**options):
        captured.update(options)
        return SimpleNamespace(
            chat=SimpleNamespace(completions=SimpleNamespace(create=lambda **_: iter([{}]))),
            close=lambda: captured.update(closed=True),
        )

    monkeypatch.setitem(sys.modules, "openai", SimpleNamespace(OpenAI=client))
    gateway = OpenAICompatibleGateway("deepseek-chat", api_key="test-key")
    request = ModelRequest(
        messages=[{"role": "user", "content": "Continue."}],
        metadata={"timeout_seconds": 1.0},
    )

    with pytest.raises(TimeoutError, match="request time budget"):
        list(gateway.stream(request))
    assert captured["timeout"] == 1.0
    assert captured["max_retries"] == 0
    assert captured["closed"] is True


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
