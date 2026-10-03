from __future__ import annotations

import json
import os
import ipaddress
from abc import ABC, abstractmethod
from collections import deque
from dataclasses import dataclass, field
from typing import Any, Iterator
from urllib.parse import urlsplit

from .contracts import ModelRequest, ModelResponse, ModelStreamEvent, ToolCall


class ModelGateway(ABC):
    @abstractmethod
    def generate(self, request: ModelRequest) -> ModelResponse:
        raise NotImplementedError

    def stream(self, request: ModelRequest) -> Iterator[ModelStreamEvent]:
        # Existing gateways and offline tests continue to use the Harness contract.
        yield ModelStreamEvent("completed", response=self.generate(request))


def _field(value: Any, name: str, default: Any = None) -> Any:
    if isinstance(value, dict):
        return value.get(name, default)
    result = getattr(value, name, None)
    if result is None:
        result = (getattr(value, "model_extra", None) or {}).get(name)
    return default if result is None else result


def _reasoning_details(value: Any) -> list[dict[str, Any]]:
    details = _field(value, "reasoning_details", []) or []
    return [
        item if isinstance(item, dict) else item.model_dump(exclude_none=True)
        for item in details
    ]


def _visible_reasoning(value: Any) -> str:
    details = _reasoning_details(value)
    visible = [str(item.get("text") or item.get("summary") or "") for item in details]
    if any(visible):
        return "".join(visible)
    return str(_field(value, "reasoning", "") or _field(value, "reasoning_content", "") or "")


def _usage_numbers(usage: Any) -> tuple[int, int, int, int]:
    def number(name: str) -> int:
        return int(_field(usage, name, 0) or 0)

    prompt = number("prompt_tokens")
    completion = number("completion_tokens")
    details = _field(usage, "prompt_tokens_details", {}) or {}
    hit = number("prompt_cache_hit_tokens") or int(_field(details, "cached_tokens", 0) or 0)
    miss = number("prompt_cache_miss_tokens") or max(0, prompt - hit)
    return prompt, completion, hit, miss


def _parsed_tool_calls(raw_calls: Any) -> list[ToolCall]:
    calls: list[ToolCall] = []
    for index, raw in enumerate(raw_calls or []):
        function = _field(raw, "function", {})
        try:
            args = json.loads(_field(function, "arguments", "") or "{}")
        except json.JSONDecodeError:
            args = {}
        calls.append(ToolCall(_field(raw, "id") or f"call_{index + 1:03d}", _field(function, "name", ""), args))
    return calls


def _merge_detail(target: list[dict[str, Any]], raw: dict[str, Any]) -> None:
    index = raw.get("index")
    if isinstance(index, int) and index >= 0:
        while len(target) <= index:
            target.append({})
        current = target[index]
    else:
        identifier = raw.get("id")
        current = next((item for item in target if identifier and item.get("id") == identifier), None)
        if current is None and target and not identifier and target[-1].get("type") == raw.get("type"):
            current = target[-1]
        if current is None:
            current = {}
            target.append(current)
    for key, value in raw.items():
        if key in {"text", "summary", "data"} and isinstance(value, str):
            current[key] = str(current.get(key) or "") + value
        elif value is not None:
            current[key] = value


_PROVIDER_DEFAULTS = {
    "deepseek": {
        "base_url": "https://api.deepseek.com/v1",
        "model": "deepseek-chat",
        "api_key_env": "DEEPSEEK_API_KEY",
        "context_window_tokens": 65_536,
        "safety_margin_tokens": 512,
    },
    "openrouter": {
        "base_url": "https://openrouter.ai/api/v1",
        "model": "deepseek/deepseek-chat",
        "api_key_env": "OPENROUTER_API_KEY",
        "context_window_tokens": 65_536,
        # The first stage still estimates with DeepSeek's tokenizer. Leave a
        # larger margin for other providers' chat framing and tokenization.
        "safety_margin_tokens": 4_096,
    },
    "ollama": {
        "base_url": "http://127.0.0.1:11434/v1",
        "model": "qwen2.5-coder:7b",
        "api_key_env": "",
        "context_window_tokens": 8_192,
        "safety_margin_tokens": 512,
    },
}


def _validate_ollama_url(base_url: str) -> None:
    try:
        parsed = urlsplit(base_url)
        host = parsed.hostname or ""
        parsed.port
        loopback = host == "localhost" or ipaddress.ip_address(host).is_loopback
    except ValueError as exc:
        raise ValueError("ollama base_url must be a local HTTP loopback URL ending in /v1") from exc
    if (parsed.scheme != "http" or not loopback or parsed.username is not None or parsed.password is not None
            or parsed.query or parsed.fragment or parsed.path.rstrip("/") != "/v1"):
        raise ValueError("ollama base_url must be a local HTTP loopback URL ending in /v1")


@dataclass(frozen=True)
class RuntimeModelConfig:
    """User-selected provider settings persisted on a Thread and snapshotted on a Run.

    API keys deliberately do not belong here: this object is stored in SQLite and
    may be returned by the API, while a request-scoped key remains in memory only.
    """

    provider: str = "deepseek"
    model: str = "deepseek-chat"
    base_url: str = "https://api.deepseek.com/v1"
    context_window_tokens: int = 65_536
    safety_margin_tokens: int = 512
    fallback_models: tuple[str, ...] = ()
    provider_preferences: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.provider not in _PROVIDER_DEFAULTS:
            raise ValueError(f"unsupported runtime model provider: {self.provider}")
        if not self.model.strip():
            raise ValueError("runtime model must not be empty")
        if not self.base_url.strip():
            raise ValueError("runtime model base_url must not be empty")
        if self.provider == "ollama":
            _validate_ollama_url(self.base_url)
        if self.context_window_tokens <= 0:
            raise ValueError("context_window_tokens must be positive")
        if self.safety_margin_tokens < 0:
            raise ValueError("safety_margin_tokens must be non-negative")
        if self.safety_margin_tokens >= self.context_window_tokens:
            raise ValueError("safety_margin_tokens must fit the context window")
        if any(not value.strip() for value in self.fallback_models):
            raise ValueError("fallback model names must not be empty")
        if not isinstance(self.provider_preferences, dict):
            raise ValueError("provider_preferences must be an object")

    @classmethod
    def from_dict(
        cls,
        value: dict[str, Any] | None,
        *,
        default_provider: str = "deepseek",
        default_model: str | None = None,
        default_base_url: str | None = None,
        default_context_window_tokens: int | None = None,
        default_safety_margin_tokens: int | None = None,
    ) -> "RuntimeModelConfig":
        source = dict(value or {})
        provider = str(source.get("provider") or default_provider).strip().lower()
        if provider not in _PROVIDER_DEFAULTS:
            raise ValueError(f"unsupported runtime model provider: {provider}")
        defaults = _PROVIDER_DEFAULTS[provider]
        use_provider_defaults = bool(source.get("provider")) and provider != default_provider
        fallback_base_url = defaults["base_url"] if use_provider_defaults else default_base_url
        fallback_context_window = (
            defaults["context_window_tokens"]
            if use_provider_defaults
            else default_context_window_tokens
        )
        fallback_safety_margin = (
            defaults["safety_margin_tokens"]
            if use_provider_defaults
            else default_safety_margin_tokens
        )
        fallbacks = source.get("fallback_models") or []
        if not isinstance(fallbacks, list) or not all(isinstance(item, str) for item in fallbacks):
            raise ValueError("fallback_models must be a list of strings")
        preferences = source.get("provider_preferences") or {}
        if not isinstance(preferences, dict):
            raise ValueError("provider_preferences must be an object")
        model = str(
            source.get("model")
            or (defaults["model"] if use_provider_defaults else default_model)
            or defaults["model"]
        ).strip()
        # The legacy Aider UI uses LiteLLM's ``deepseek/deepseek-chat`` form;
        # the official DeepSeek endpoint expects the bare model identifier.
        if provider == "deepseek" and model.startswith("deepseek/"):
            model = model.split("/", 1)[1]
        return cls(
            provider=provider,
            model=model,
            base_url=str(
                source.get("base_url")
                or fallback_base_url
                or defaults["base_url"]
            ).strip(),
            context_window_tokens=int(
                source.get("context_window_tokens")
                or fallback_context_window
                or defaults["context_window_tokens"]
            ),
            safety_margin_tokens=int(
                source.get("safety_margin_tokens")
                if source.get("safety_margin_tokens") is not None
                else (
                    fallback_safety_margin
                    if fallback_safety_margin is not None
                    else defaults["safety_margin_tokens"]
                )
            ),
            fallback_models=tuple(item.strip() for item in fallbacks),
            provider_preferences=dict(preferences),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "provider": self.provider,
            "model": self.model,
            "base_url": self.base_url,
            "context_window_tokens": self.context_window_tokens,
            "safety_margin_tokens": self.safety_margin_tokens,
            "fallback_models": list(self.fallback_models),
            "provider_preferences": self.provider_preferences,
        }

    @property
    def api_key_env(self) -> str:
        return str(_PROVIDER_DEFAULTS[self.provider]["api_key_env"])

    def openrouter_extra_body(self) -> dict[str, Any]:
        if self.provider != "openrouter":
            return {}
        body: dict[str, Any] = {}
        if self.fallback_models:
            body["models"] = list(self.fallback_models)
        if self.provider_preferences:
            body["provider"] = self.provider_preferences
        return body


class ScriptedModelGateway(ModelGateway):
    """Deterministic model used by tests, demos, and offline evaluations."""

    def __init__(self, responses: list[ModelResponse]) -> None:
        self._responses = deque(responses)
        self.requests: list[ModelRequest] = []

    def generate(self, request: ModelRequest) -> ModelResponse:
        self.requests.append(request)
        if not self._responses:
            return ModelResponse(content="No scripted response remains.")
        return self._responses.popleft()


class DeepSeekRequestSerializer:
    def serialize_messages(
        self, messages: list[dict[str, Any]]
    ) -> list[dict[str, Any]]:
        result: list[dict[str, Any]] = []
        for message in messages:
            role = message.get("role")
            if role == "assistant" and message.get("tool_calls"):
                assistant = {
                        "role": "assistant",
                        "content": message.get("content") or None,
                        "tool_calls": [
                            {
                                "id": call["id"],
                                "type": "function",
                                "function": {
                                    "name": call["name"],
                                    "arguments": json.dumps(
                                        call.get("args", {}), ensure_ascii=False
                                    ),
                                },
                            }
                            for call in message["tool_calls"]
                        ],
                    }
                if message.get("reasoning_details"):
                    assistant["reasoning_details"] = message["reasoning_details"]
                elif message.get("reasoning_content"):
                    assistant["reasoning_content"] = message["reasoning_content"]
                result.append(assistant)
            elif role == "tool":
                result.append(
                    {
                        "role": "tool",
                        "tool_call_id": message["tool_call_id"],
                        "content": str(message.get("content", "")),
                    }
                )
            else:
                item = {"role": role, "content": str(message.get("content", ""))}
                if role == "assistant":
                    if message.get("reasoning_details"):
                        item["reasoning_details"] = message["reasoning_details"]
                    elif message.get("reasoning_content"):
                        item["reasoning_content"] = message["reasoning_content"]
                result.append(item)
        return result

    def serialize_tools(
        self, tools: list[dict[str, Any]]
    ) -> list[dict[str, Any]]:
        return json.loads(json.dumps(tools, ensure_ascii=False, sort_keys=True))


class OpenAICompatibleGateway(ModelGateway):
    def __init__(
        self,
        model: str,
        *,
        api_key: str | None = None,
        base_url: str | None = None,
        max_tokens: int = 4096,
        serializer: DeepSeekRequestSerializer | None = None,
        api_key_env: str | None = None,
        use_request_api_key: bool = True,
        extra_body: dict[str, Any] | None = None,
        default_headers: dict[str, str] | None = None,
    ) -> None:
        self.model = model
        self.api_key = api_key
        self.base_url = base_url
        self.max_tokens = max_tokens
        self.serializer = serializer or DeepSeekRequestSerializer()
        self.api_key_env = api_key_env
        self.use_request_api_key = use_request_api_key
        self.extra_body = dict(extra_body or {})
        self.default_headers = dict(default_headers or {})

    def generate(self, request: ModelRequest) -> ModelResponse:
        from openai import OpenAI

        key = (
            (str(request.metadata.get("api_key") or "").strip() if self.use_request_api_key else "")
            or self.api_key
            or (os.environ.get(self.api_key_env) if self.api_key_env else None)
            or (
                os.environ.get("DEEPSEEK_API_KEY")
                or os.environ.get("OPENAI_API_KEY")
                if self.api_key_env is None
                else None
            )
        )
        if not key:
            raise ValueError("No API key configured for the model gateway")
        serialized_tools = self.serializer.serialize_tools(request.tools)
        options: dict[str, Any] = {}
        if request.response_format is not None:
            options["response_format"] = request.response_format
        request_extra_body = request.metadata.get("extra_body") or {}
        if request_extra_body and not isinstance(request_extra_body, dict):
            raise ValueError("request extra_body must be an object")
        extra_body = {**self.extra_body, **request_extra_body}
        if extra_body:
            options["extra_body"] = extra_body
        client_options: dict[str, Any] = {"api_key": key, "base_url": self.base_url,
            "default_headers": self.default_headers or None}
        if not self.use_request_api_key:
            import httpx
            client_options["http_client"] = httpx.Client(trust_env=False)
        try:
            client = OpenAI(**client_options)
        except Exception:
            if not self.use_request_api_key:
                client_options["http_client"].close()
            raise
        try:
            response = client.chat.completions.create(
                model=self.model,
                messages=self.serializer.serialize_messages(request.messages),
                tools=serialized_tools or None,
                tool_choice="auto" if serialized_tools else None,
                max_tokens=request.max_output_tokens,
                **options,
            )
        finally:
            if not self.use_request_api_key:
                client.close()
        message = response.choices[0].message
        prompt, completion, hit, miss = _usage_numbers(response.usage)
        return ModelResponse(
            content=message.content or "",
            tool_calls=_parsed_tool_calls(message.tool_calls),
            input_tokens=prompt,
            output_tokens=completion,
            prompt_cache_hit_tokens=hit,
            prompt_cache_miss_tokens=miss,
            model=getattr(response, "model", None) or self.model,
            reasoning=_visible_reasoning(message),
            reasoning_details=_reasoning_details(message),
        )

    def stream(self, request: ModelRequest) -> Iterator[ModelStreamEvent]:
        from openai import OpenAI

        key = (
            (str(request.metadata.get("api_key") or "").strip() if self.use_request_api_key else "")
            or self.api_key
            or (os.environ.get(self.api_key_env) if self.api_key_env else None)
            or (
                os.environ.get("DEEPSEEK_API_KEY") or os.environ.get("OPENAI_API_KEY")
                if self.api_key_env is None else None
            )
        )
        if not key:
            raise ValueError("No API key configured for the model gateway")
        tools = self.serializer.serialize_tools(request.tools)
        options: dict[str, Any] = {}
        if request.response_format is not None:
            options["response_format"] = request.response_format
        request_extra_body = request.metadata.get("extra_body") or {}
        if request_extra_body and not isinstance(request_extra_body, dict):
            raise ValueError("request extra_body must be an object")
        extra_body = {**self.extra_body, **request_extra_body}
        if extra_body:
            options["extra_body"] = extra_body
        client_options: dict[str, Any] = {"api_key": key, "base_url": self.base_url,
            "default_headers": self.default_headers or None}
        if not self.use_request_api_key:
            import httpx
            client_options["http_client"] = httpx.Client(trust_env=False)
        try:
            client = OpenAI(**client_options)
        except Exception:
            if not self.use_request_api_key:
                client_options["http_client"].close()
            raise
        try:
            chunks = client.chat.completions.create(
                model=self.model,
                messages=self.serializer.serialize_messages(request.messages),
                tools=tools or None,
                tool_choice="auto" if tools else None,
                max_tokens=request.max_output_tokens,
                stream=True,
                stream_options={"include_usage": True},
                **options,
            )
        except Exception:
            if not self.use_request_api_key:
                client.close()
            raise
        content: list[str] = []
        reasoning: list[str] = []
        details: list[dict[str, Any]] = []
        tool_parts: dict[int, dict[str, Any]] = {}
        usage = None
        model = self.model
        try:
            for chunk in chunks:
                usage = _field(chunk, "usage") or usage
                model = _field(chunk, "model") or model
                for choice in _field(chunk, "choices", []) or []:
                    delta = _field(choice, "delta")
                    if delta is None:
                        continue
                    raw_details = _reasoning_details(delta)
                    if raw_details:
                        for detail in raw_details:
                            _merge_detail(details, detail)
                        visible = "".join(str(item.get("text") or item.get("summary") or "") for item in raw_details)
                        if not visible:
                            visible = str(_field(delta, "reasoning", "") or _field(delta, "reasoning_content", "") or "")
                    else:
                        visible = str(_field(delta, "reasoning", "") or _field(delta, "reasoning_content", "") or "")
                    if visible:
                        reasoning.append(visible)
                        yield ModelStreamEvent("reasoning", visible)
                    part = _field(delta, "content", "") or ""
                    if part:
                        content.append(str(part))
                        yield ModelStreamEvent("content", str(part))
                    for raw_call in _field(delta, "tool_calls", []) or []:
                        index = int(_field(raw_call, "index", 0) or 0)
                        current = tool_parts.setdefault(index, {"id": "", "name": "", "arguments": ""})
                        current["id"] += str(_field(raw_call, "id", "") or "")
                        function = _field(raw_call, "function", {}) or {}
                        current["name"] += str(_field(function, "name", "") or "")
                        current["arguments"] += str(_field(function, "arguments", "") or "")
        finally:
            if not self.use_request_api_key:
                client.close()
        raw_calls = [
            {"id": part["id"], "function": {"name": part["name"], "arguments": part["arguments"]}}
            for _, part in sorted(tool_parts.items())
        ]
        prompt, completion, hit, miss = _usage_numbers(usage)
        yield ModelStreamEvent("completed", response=ModelResponse(
            content="".join(content),
            reasoning="".join(reasoning),
            reasoning_details=[item for item in details if item],
            tool_calls=_parsed_tool_calls(raw_calls),
            input_tokens=prompt,
            output_tokens=completion,
            prompt_cache_hit_tokens=hit,
            prompt_cache_miss_tokens=miss,
            model=model,
        ))


class RoutedModelGateway(ModelGateway):
    """Dispatch by an explicitly injected RuntimeModelConfig, never by key shape."""

    def __init__(self, *, serializer: DeepSeekRequestSerializer | None = None) -> None:
        self.serializer = serializer or DeepSeekRequestSerializer()

    def generate(self, request: ModelRequest) -> ModelResponse:
        raw_config = request.metadata.get("runtime_model_config")
        if not isinstance(raw_config, dict):
            raise ValueError("ModelRequest is missing runtime_model_config")
        config = RuntimeModelConfig.from_dict(raw_config)
        headers: dict[str, str] = {}
        if config.provider == "openrouter":
            referer = os.environ.get("OPENROUTER_HTTP_REFERER", "").strip()
            title = os.environ.get("OPENROUTER_APP_TITLE", "NaturalCC Agent").strip()
            if referer:
                headers["HTTP-Referer"] = referer
            if title:
                headers["X-Title"] = title
        gateway = OpenAICompatibleGateway(
            config.model,
            api_key="ollama" if config.provider == "ollama" else None,
            base_url=config.base_url,
            serializer=self.serializer,
            api_key_env=config.api_key_env,
            use_request_api_key=config.provider != "ollama",
            extra_body=config.openrouter_extra_body(),
            default_headers=headers,
        )
        response = gateway.generate(request)
        # OpenRouter may resolve a fallback model. Preserve the provider-reported
        # model whenever the SDK exposes it, otherwise retain the configured one.
        return response

    def stream(self, request: ModelRequest) -> Iterator[ModelStreamEvent]:
        raw_config = request.metadata.get("runtime_model_config")
        if not isinstance(raw_config, dict):
            raise ValueError("ModelRequest is missing runtime_model_config")
        config = RuntimeModelConfig.from_dict(raw_config)
        headers: dict[str, str] = {}
        if config.provider == "openrouter":
            referer = os.environ.get("OPENROUTER_HTTP_REFERER", "").strip()
            title = os.environ.get("OPENROUTER_APP_TITLE", "NaturalCC Agent").strip()
            if referer:
                headers["HTTP-Referer"] = referer
            if title:
                headers["X-Title"] = title
        extra_body = config.openrouter_extra_body()
        if config.provider == "openrouter":
            extra_body["reasoning"] = {"enabled": True, "exclude": False}
        elif config.provider == "deepseek" and config.model == "deepseek-chat":
            extra_body["thinking"] = {"type": "enabled"}
        gateway = OpenAICompatibleGateway(
            config.model,
            api_key="ollama" if config.provider == "ollama" else None,
            base_url=config.base_url,
            serializer=self.serializer,
            api_key_env=config.api_key_env,
            use_request_api_key=config.provider != "ollama",
            extra_body=extra_body,
            default_headers=headers,
        )
        yield from gateway.stream(request)


class FallbackModelGateway(ModelGateway):
    def __init__(self, gateways: list[ModelGateway]) -> None:
        if not gateways:
            raise ValueError("at least one model gateway is required")
        self.gateways = gateways

    def generate(self, request: ModelRequest) -> ModelResponse:
        errors = []
        for gateway in self.gateways:
            try:
                return gateway.generate(request)
            except Exception as exc:
                errors.append(f"{type(exc).__name__}: {exc}")
        raise RuntimeError("all model gateways failed: " + " | ".join(errors))
