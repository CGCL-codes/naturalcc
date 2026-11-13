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


# [maint-note:edea092f9973]
# 数据与行为导航：ModelGateway。
# 继承接口：ABC。
# [/maint-note:edea092f9973]
class ModelGateway(ABC):
    # [maint-note:b06887929a03]
    # 维护导航：generate；下列为本函数源码中的接口与路径索引。
    # 输入 request：ModelRequest。
    # 返回契约：ModelResponse。
    # 显式抛出：NotImplementedError。
    # [/maint-note:b06887929a03]
    @abstractmethod
    def generate(self, request: ModelRequest) -> ModelResponse:
        raise NotImplementedError

    # [maint-note:7a3a78ff4231]
    # 维护导航：stream；下列为本函数源码中的接口与路径索引。
    # 输入 request：ModelRequest。
    # 返回契约：Iterator[ModelStreamEvent]。
    # 直接调用：ModelStreamEvent。
    # 直接调用：self.generate。
    # 生成器交付：ModelStreamEvent('completed', response=self.generate(request))。
    # [/maint-note:7a3a78ff4231]
    def stream(self, request: ModelRequest) -> Iterator[ModelStreamEvent]:
        # Existing gateways and offline tests continue to use the Harness contract.
        yield ModelStreamEvent("completed", response=self.generate(request))


# [maint-note:bebf4d24eee3]
# 维护导航：_field；下列为本函数源码中的接口与路径索引。
# 输入 value：Any。
# 输入 name：str。
# 输入 default：Any；缺省表达式 None。
# 返回契约：Any。
# 分支条件：isinstance(value, dict)。
# 分支条件：result is None。
# 返回路径：value.get(name, default)。
# 返回路径：default if result is None else result。
# 直接调用：isinstance。
# 直接调用：value.get。
# 直接调用：getattr。
# 直接调用：(getattr(value, 'model_extra', None) or {}).get。
# 表达式分流：result is None；真值与假值分支分别计算，不同时执行。
# [/maint-note:bebf4d24eee3]
def _field(value: Any, name: str, default: Any = None) -> Any:
    if isinstance(value, dict):
        return value.get(name, default)
    result = getattr(value, name, None)
    if result is None:
        result = (getattr(value, "model_extra", None) or {}).get(name)
    return default if result is None else result


# [maint-note:50fc34ecba1e]
# 维护导航：_reasoning_details；下列为本函数源码中的接口与路径索引。
# 输入 value：Any。
# 返回契约：list[dict[str, Any]]。
# 返回路径：[item if isinstance(item, dict) else item.model_dump(exclude_none=True) for item in details]。
# 直接调用：_field。
# 直接调用：isinstance。
# 直接调用：item.model_dump。
# 表达式分流：isinstance(item, dict)；真值与假值分支分别计算，不同时执行。
# 集合派生：item 来自 details。
# [/maint-note:50fc34ecba1e]
def _reasoning_details(value: Any) -> list[dict[str, Any]]:
    details = _field(value, "reasoning_details", []) or []
    return [
        item if isinstance(item, dict) else item.model_dump(exclude_none=True)
        for item in details
    ]


# [maint-note:7076f788f221]
# 维护导航：_visible_reasoning；下列为本函数源码中的接口与路径索引。
# 输入 value：Any。
# 返回契约：str。
# 分支条件：any(visible)。
# 返回路径：''.join(visible)。
# 返回路径：str(_field(value, 'reasoning', '') or _field(value, 'reasoning_content', '') or '')。
# 直接调用：_reasoning_details。
# 直接调用：str。
# 直接调用：item.get。
# 直接调用：any。
# 直接调用：''.join。
# 直接调用：_field。
# 集合派生：item 来自 details。
# [/maint-note:7076f788f221]
def _visible_reasoning(value: Any) -> str:
    details = _reasoning_details(value)
    visible = [str(item.get("text") or item.get("summary") or "") for item in details]
    if any(visible):
        return "".join(visible)
    return str(_field(value, "reasoning", "") or _field(value, "reasoning_content", "") or "")


# [maint-note:e20ccef43c0f]
# 维护导航：_usage_numbers；下列为本函数源码中的接口与路径索引。
# 输入 usage：Any。
# 返回契约：tuple[int, int, int, int]。
# 返回路径：int(_field(usage, name, 0) or 0)。
# 返回路径：(prompt, completion, hit, miss)。
# 直接调用：int。
# 直接调用：_field。
# 直接调用：number。
# 直接调用：max。
# [/maint-note:e20ccef43c0f]
def _usage_numbers(usage: Any) -> tuple[int, int, int, int]:
    # [maint-note:5c61e4c735f3]
    # 维护导航：number；下列为本函数源码中的接口与路径索引。
    # 输入 name：str。
    # 返回契约：int。
    # 返回路径：int(_field(usage, name, 0) or 0)。
    # 直接调用：int。
    # 直接调用：_field。
    # [/maint-note:5c61e4c735f3]
    def number(name: str) -> int:
        return int(_field(usage, name, 0) or 0)

    prompt = number("prompt_tokens")
    completion = number("completion_tokens")
    details = _field(usage, "prompt_tokens_details", {}) or {}
    hit = number("prompt_cache_hit_tokens") or int(_field(details, "cached_tokens", 0) or 0)
    miss = number("prompt_cache_miss_tokens") or max(0, prompt - hit)
    return prompt, completion, hit, miss


# [maint-note:2d632614dc66]
# 维护导航：_parsed_tool_calls；下列为本函数源码中的接口与路径索引。
# 输入 raw_calls：Any。
# 返回契约：list[ToolCall]。
# 遍历边界：(index, raw) ← enumerate(raw_calls or [])。
# 返回路径：calls。
# 捕获异常：json.JSONDecodeError。
# 直接调用：enumerate。
# 直接调用：_field。
# 直接调用：json.loads。
# 直接调用：calls.append。
# 直接调用：ToolCall。
# [/maint-note:2d632614dc66]
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


# [maint-note:bc6eb8d21d26]
# 维护导航：_merge_detail；下列为本函数源码中的接口与路径索引。
# 输入 target：list[dict[str, Any]]。
# 输入 raw：dict[str, Any]。
# 返回契约：None。
# 分支条件：isinstance(index, int) and index >= 0。
# 分支条件：current is None and target and (not identifier) and (target[-1].get('type') == raw.get('type'))。
# 分支条件：current is None。
# 分支条件：key in {'text', 'summary', 'data'} and isinstance(value, str)。
# 分支条件：value is not None。
# 遍历边界：len(target) <= index。
# 遍历边界：(key, value) ← raw.items()。
# 写入目标：current[key]。
# 直接调用：raw.get。
# 直接调用：isinstance。
# 直接调用：len。
# 直接调用：target.append。
# 直接调用：next。
# 直接调用：item.get。
# 直接调用：target[-1].get。
# 直接调用：raw.items。
# 直接调用：str。
# 直接调用：current.get。
# 集合派生：item 来自 target；过滤 identifier and item.get('id') == identifier。
# [/maint-note:bc6eb8d21d26]
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
        "model": "deepseek-v4-pro",
        "api_key_env": "DEEPSEEK_API_KEY",
        "context_window_tokens": 262_144,
        "safety_margin_tokens": 512,
    },
    "openrouter": {
        "base_url": "https://openrouter.ai/api/v1",
        "model": "deepseek/deepseek-chat",
        "api_key_env": "OPENROUTER_API_KEY",
        "context_window_tokens": 262_144,
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


# [maint-note:359a339d691e]
# 维护导航：_validate_ollama_url；下列为本函数源码中的接口与路径索引。
# 输入 base_url：str。
# 返回契约：None。
# 分支条件：parsed.scheme != 'http' or not loopback or parsed.username is not None or (parsed.password is not None) or parsed.query or parsed.fragment or (parsed.path.rstrip('/') != '/v1')。
# 显式抛出：ValueError('ollama base_url must be a local HTTP loopback URL ending in /v1')。
# 捕获异常：ValueError。
# 直接调用：urlsplit。
# 直接调用：ipaddress.ip_address。
# 直接调用：ValueError。
# 直接调用：parsed.path.rstrip。
# [/maint-note:359a339d691e]
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


# [maint-note:a5f61eb2517e]
# 数据与行为导航：RuntimeModelConfig。
# 字段 provider：str；初始表达式 'deepseek'。
# 字段 model：str；初始表达式 'deepseek-v4-pro'。
# 字段 base_url：str；初始表达式 'https://api.deepseek.com/v1'。
# 字段 context_window_tokens：int；初始表达式 262144。
# 字段 safety_margin_tokens：int；初始表达式 512。
# 字段 fallback_models：tuple[str, ...]；初始表达式 ()。
# 字段 provider_preferences：dict[str, Any]；初始表达式 field(default_factory=dict)。
# [/maint-note:a5f61eb2517e]
@dataclass(frozen=True)
class RuntimeModelConfig:
    """User-selected provider settings persisted on a Thread and snapshotted on a Run.

    API keys deliberately do not belong here: this object is stored in SQLite and
    may be returned by the API, while a request-scoped key remains in memory only.
    """

    provider: str = "deepseek"
    model: str = "deepseek-v4-pro"
    base_url: str = "https://api.deepseek.com/v1"
    context_window_tokens: int = 262_144
    safety_margin_tokens: int = 512
    fallback_models: tuple[str, ...] = ()
    provider_preferences: dict[str, Any] = field(default_factory=dict)

    # [maint-note:5f3cc19c671c]
    # 维护导航：__post_init__；下列为本函数源码中的接口与路径索引。
    # 返回契约：None。
    # 分支条件：self.provider not in _PROVIDER_DEFAULTS。
    # 分支条件：not self.model.strip()。
    # 分支条件：not self.base_url.strip()。
    # 分支条件：self.provider == 'ollama'。
    # 分支条件：self.context_window_tokens <= 0。
    # 分支条件：self.safety_margin_tokens < 0。
    # 分支条件：self.safety_margin_tokens >= self.context_window_tokens。
    # 分支条件：any((not value.strip() for value in self.fallback_models))。
    # 分支条件：not isinstance(self.provider_preferences, dict)。
    # 显式抛出：ValueError(f'unsupported runtime model provider: {self.provider}')。
    # 显式抛出：ValueError('runtime model must not be empty')。
    # 显式抛出：ValueError('runtime model base_url must not be empty')。
    # 显式抛出：ValueError('context_window_tokens must be positive')。
    # 显式抛出：ValueError('safety_margin_tokens must be non-negative')。
    # 显式抛出：ValueError('safety_margin_tokens must fit the context window')。
    # 显式抛出：ValueError('fallback model names must not be empty')。
    # 显式抛出：ValueError('provider_preferences must be an object')。
    # 直接调用：ValueError。
    # 直接调用：self.model.strip。
    # 直接调用：self.base_url.strip。
    # 直接调用：_validate_ollama_url。
    # 直接调用：any。
    # 直接调用：value.strip。
    # 直接调用：isinstance。
    # 集合派生：value 来自 self.fallback_models。
    # [/maint-note:5f3cc19c671c]
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

    # [maint-note:ed11b90cff1f]
    # 维护导航：from_dict；下列为本函数源码中的接口与路径索引。
    # 输入 value：dict[str, Any] | None。
    # 输入 default_provider：str；缺省表达式 'deepseek'。
    # 输入 default_model：str | None；缺省表达式 None。
    # 输入 default_base_url：str | None；缺省表达式 None。
    # 输入 default_context_window_tokens：int | None；缺省表达式 None。
    # 输入 default_safety_margin_tokens：int | None；缺省表达式 None。
    # 返回契约：'RuntimeModelConfig'。
    # 分支条件：provider not in _PROVIDER_DEFAULTS。
    # 分支条件：not isinstance(fallbacks, list) or not all((isinstance(item, str) for item in fallbacks))。
    # 分支条件：not isinstance(preferences, dict)。
    # 分支条件：provider == 'deepseek' and model.startswith('deepseek/')。
    # 分支条件：configured_window is not None。
    # 返回路径：cls(provider=provider, model=model, base_url=str(source.get('base_url') or fallback_base_url or defaults['base_url']).strip(), context_window_tokens=context_window_tokens, safety_margin_tokens=int(source.get('safety_margin_tokens') if sourc。
    # 显式抛出：ValueError(f'unsupported runtime model provider: {provider}')。
    # 显式抛出：ValueError('fallback_models must be a list of strings')。
    # 显式抛出：ValueError('provider_preferences must be an object')。
    # 直接调用：dict。
    # 直接调用：str(source.get('provider') or default_provider).strip().lower。
    # 直接调用：str(source.get('provider') or default_provider).strip。
    # 直接调用：str。
    # 直接调用：source.get。
    # 直接调用：ValueError。
    # 直接调用：bool。
    # 直接调用：isinstance。
    # 直接调用：all。
    # 直接调用：str(source.get('model') or (defaults['model'] if use_provider_defaults else default_model) or defaults['model']).strip。
    # 直接调用：model.startswith。
    # 直接调用：model.split。
    # 直接调用：int。
    # 直接调用：cls。
    # 直接调用：str(source.get('base_url') or fallback_base_url or defaults['base_url']).strip。
    # 直接调用：tuple。
    # 直接调用：item.strip。
    # 表达式分流：use_provider_defaults；真值与假值分支分别计算，不同时执行。
    # 集合派生：item 来自 fallbacks。
    # 表达式分流：fallback_context_window is not None；真值与假值分支分别计算，不同时执行。
    # 表达式分流：source.get('safety_margin_tokens') is not None；真值与假值分支分别计算，不同时执行。
    # 表达式分流：fallback_safety_margin is not None；真值与假值分支分别计算，不同时执行。
    # [/maint-note:ed11b90cff1f]
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
        configured_window = source.get("context_window_tokens")
        if configured_window is not None:
            context_window_tokens = int(configured_window)
        else:
            # A missing per-thread value always inherits the provider/global
            # default, regardless of the selected model ID.
            context_window_tokens = int(
                fallback_context_window
                if fallback_context_window is not None
                else defaults["context_window_tokens"]
            )
        return cls(
            provider=provider,
            model=model,
            base_url=str(
                source.get("base_url")
                or fallback_base_url
                or defaults["base_url"]
            ).strip(),
            context_window_tokens=context_window_tokens,
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

    # [maint-note:1b3c1c390963]
    # 维护导航：to_dict；下列为本函数源码中的接口与路径索引。
    # 返回契约：dict[str, Any]。
    # 返回路径：{'provider': self.provider, 'model': self.model, 'base_url': self.base_url, 'context_window_tokens': self.context_window_tokens, 'safety_margin_tokens': self.safety_margin_tokens, 'fallback_models': list(self.fallback_models), 'provider_pre。
    # 直接调用：list。
    # [/maint-note:1b3c1c390963]
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

    # [maint-note:a2916baffb44]
    # 维护导航：api_key_env；下列为本函数源码中的接口与路径索引。
    # 返回契约：str。
    # 返回路径：str(_PROVIDER_DEFAULTS[self.provider]['api_key_env'])。
    # 直接调用：str。
    # [/maint-note:a2916baffb44]
    @property
    def api_key_env(self) -> str:
        return str(_PROVIDER_DEFAULTS[self.provider]["api_key_env"])

    # [maint-note:f08b7135d5a6]
    # 维护导航：openrouter_extra_body；下列为本函数源码中的接口与路径索引。
    # 返回契约：dict[str, Any]。
    # 分支条件：self.provider != 'openrouter'。
    # 分支条件：self.fallback_models。
    # 分支条件：self.provider_preferences。
    # 返回路径：{}。
    # 返回路径：body。
    # 写入目标：body['models']。
    # 写入目标：body['provider']。
    # 直接调用：list。
    # [/maint-note:f08b7135d5a6]
    def openrouter_extra_body(self) -> dict[str, Any]:
        if self.provider != "openrouter":
            return {}
        body: dict[str, Any] = {}
        if self.fallback_models:
            body["models"] = list(self.fallback_models)
        if self.provider_preferences:
            body["provider"] = self.provider_preferences
        return body


# [maint-note:905fa8b7e84d]
# 数据与行为导航：ScriptedModelGateway。
# 继承接口：ModelGateway。
# [/maint-note:905fa8b7e84d]
class ScriptedModelGateway(ModelGateway):
    """Deterministic model used by tests, demos, and offline evaluations."""

    # [maint-note:d3f9183defb8]
    # 维护导航：__init__；下列为本函数源码中的接口与路径索引。
    # 输入 responses：list[ModelResponse]。
    # 返回契约：None。
    # 写入目标：self._responses。
    # 写入目标：self.requests。
    # 直接调用：deque。
    # [/maint-note:d3f9183defb8]
    def __init__(self, responses: list[ModelResponse]) -> None:
        self._responses = deque(responses)
        self.requests: list[ModelRequest] = []

    # [maint-note:cdc23a9530da]
    # 维护导航：generate；下列为本函数源码中的接口与路径索引。
    # 输入 request：ModelRequest。
    # 返回契约：ModelResponse。
    # 分支条件：not self._responses。
    # 返回路径：ModelResponse(content='No scripted response remains.')。
    # 返回路径：self._responses.popleft()。
    # 直接调用：self.requests.append。
    # 直接调用：ModelResponse。
    # 直接调用：self._responses.popleft。
    # [/maint-note:cdc23a9530da]
    def generate(self, request: ModelRequest) -> ModelResponse:
        self.requests.append(request)
        if not self._responses:
            return ModelResponse(content="No scripted response remains.")
        return self._responses.popleft()


class DeepSeekRequestSerializer:
    # [maint-note:c8f2bc8507d9]
    # 维护导航：serialize_messages；下列为本函数源码中的接口与路径索引。
    # 输入 messages：list[dict[str, Any]]。
    # 返回契约：list[dict[str, Any]]。
    # 分支条件：role == 'assistant' and message.get('tool_calls')。
    # 分支条件：message.get('reasoning_details')。
    # 分支条件：message.get('reasoning_content')。
    # 分支条件：role == 'tool'。
    # 分支条件：role == 'assistant'。
    # 遍历边界：message ← messages。
    # 返回路径：result。
    # 写入目标：assistant['reasoning_details']。
    # 写入目标：assistant['reasoning_content']。
    # 写入目标：item['reasoning_details']。
    # 写入目标：item['reasoning_content']。
    # 直接调用：message.get。
    # 直接调用：json.dumps。
    # 直接调用：call.get。
    # 直接调用：result.append。
    # 直接调用：str。
    # 集合派生：call 来自 message['tool_calls']。
    # [/maint-note:c8f2bc8507d9]
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

    # [maint-note:ed7634003214]
    # 维护导航：serialize_tools；下列为本函数源码中的接口与路径索引。
    # 输入 tools：list[dict[str, Any]]。
    # 返回契约：list[dict[str, Any]]。
    # 返回路径：json.loads(json.dumps(tools, ensure_ascii=False, sort_keys=True))。
    # 直接调用：json.loads。
    # 直接调用：json.dumps。
    # [/maint-note:ed7634003214]
    def serialize_tools(
        self, tools: list[dict[str, Any]]
    ) -> list[dict[str, Any]]:
        return json.loads(json.dumps(tools, ensure_ascii=False, sort_keys=True))


# [maint-note:3bbf238db156]
# 数据与行为导航：OpenAICompatibleGateway。
# 继承接口：ModelGateway。
# [/maint-note:3bbf238db156]
class OpenAICompatibleGateway(ModelGateway):
    # [maint-note:f854b758d104]
    # 维护导航：__init__；下列为本函数源码中的接口与路径索引。
    # 输入 model：str。
    # 输入 api_key：str | None；缺省表达式 None。
    # 输入 base_url：str | None；缺省表达式 None。
    # 输入 max_tokens：int；缺省表达式 4096。
    # 输入 serializer：DeepSeekRequestSerializer | None；缺省表达式 None。
    # 输入 api_key_env：str | None；缺省表达式 None。
    # 输入 use_request_api_key：bool；缺省表达式 True。
    # 输入 extra_body：dict[str, Any] | None；缺省表达式 None。
    # 输入 default_headers：dict[str, str] | None；缺省表达式 None。
    # 返回契约：None。
    # 写入目标：self.model。
    # 写入目标：self.api_key。
    # 写入目标：self.base_url。
    # 写入目标：self.max_tokens。
    # 写入目标：self.serializer。
    # 写入目标：self.api_key_env。
    # 写入目标：self.use_request_api_key。
    # 写入目标：self.extra_body。
    # 写入目标：self.default_headers。
    # 直接调用：DeepSeekRequestSerializer。
    # 直接调用：dict。
    # [/maint-note:f854b758d104]
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

    # [maint-note:d62ffa343317]
    # 维护导航：generate；下列为本函数源码中的接口与路径索引。
    # 输入 request：ModelRequest。
    # 返回契约：ModelResponse。
    # 分支条件：not key。
    # 分支条件：request.response_format is not None。
    # 分支条件：request_extra_body and (not isinstance(request_extra_body, dict))。
    # 分支条件：extra_body。
    # 分支条件：timeout is not None。
    # 分支条件：not self.use_request_api_key。
    # 分支条件：not self.use_request_api_key or timeout is not None。
    # 返回路径：ModelResponse(content=message.content or '', tool_calls=_parsed_tool_calls(message.tool_calls), input_tokens=prompt, output_tokens=completion, prompt_cache_hit_tokens=hit, prompt_cache_miss_tokens=miss, model=getattr(response, 'model', None。
    # 显式抛出：ValueError('No API key configured for the model gateway')。
    # 显式抛出：ValueError('request extra_body must be an object')。
    # 显式抛出：继续抛出当前异常。
    # 捕获异常：Exception。
    # 写入目标：options['response_format']。
    # 写入目标：options['extra_body']。
    # 写入目标：client_options['http_client']。
    # 直接调用：str(request.metadata.get('api_key') or '').strip。
    # 直接调用：str。
    # 直接调用：request.metadata.get。
    # 直接调用：os.environ.get。
    # 直接调用：ValueError。
    # 直接调用：self.serializer.serialize_tools。
    # 直接调用：isinstance。
    # 直接调用：client_options.update。
    # 直接调用：httpx.Client。
    # 直接调用：OpenAI。
    # 直接调用：client_options['http_client'].close。
    # 直接调用：client.chat.completions.create。
    # 直接调用：self.serializer.serialize_messages。
    # 直接调用：client.close。
    # 直接调用：_usage_numbers。
    # 直接调用：ModelResponse。
    # 直接调用：_parsed_tool_calls。
    # 直接调用：getattr。
    # 直接调用：_visible_reasoning。
    # 直接调用：_reasoning_details。
    # 表达式分流：self.use_request_api_key；真值与假值分支分别计算，不同时执行。
    # 表达式分流：self.api_key_env；真值与假值分支分别计算，不同时执行。
    # 表达式分流：self.api_key_env is None；真值与假值分支分别计算，不同时执行。
    # 清理边界：本函数含 finally；正常返回和异常展开都经过清理块。
    # 表达式分流：serialized_tools；真值与假值分支分别计算，不同时执行。
    # [/maint-note:d62ffa343317]
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
        timeout = request.metadata.get("timeout_seconds")
        if timeout is not None:
            client_options.update(timeout=timeout, max_retries=0)
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
            if not self.use_request_api_key or timeout is not None:
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

    # [maint-note:8d726e70b4b1]
    # 维护导航：stream；下列为本函数源码中的接口与路径索引。
    # 输入 request：ModelRequest。
    # 返回契约：Iterator[ModelStreamEvent]。
    # 分支条件：not key。
    # 分支条件：request.response_format is not None。
    # 分支条件：request_extra_body and (not isinstance(request_extra_body, dict))。
    # 分支条件：extra_body。
    # 分支条件：not self.use_request_api_key。
    # 分支条件：delta is None。
    # 分支条件：raw_details。
    # 分支条件：not visible。
    # 分支条件：visible。
    # 分支条件：part。
    # 遍历边界：chunk ← chunks。
    # 遍历边界：choice ← _field(chunk, 'choices', []) or []。
    # 遍历边界：detail ← raw_details。
    # 遍历边界：raw_call ← _field(delta, 'tool_calls', []) or []。
    # 显式抛出：ValueError('No API key configured for the model gateway')。
    # 显式抛出：ValueError('request extra_body must be an object')。
    # 显式抛出：继续抛出当前异常。
    # 捕获异常：Exception。
    # 写入目标：options['response_format']。
    # 写入目标：options['extra_body']。
    # 写入目标：client_options['http_client']。
    # 写入目标：current['id']。
    # 写入目标：current['name']。
    # 写入目标：current['arguments']。
    # 直接调用：str(request.metadata.get('api_key') or '').strip。
    # 直接调用：str。
    # 直接调用：request.metadata.get。
    # 直接调用：os.environ.get。
    # 直接调用：ValueError。
    # 直接调用：self.serializer.serialize_tools。
    # 直接调用：isinstance。
    # 直接调用：httpx.Client。
    # 直接调用：OpenAI。
    # 直接调用：client_options['http_client'].close。
    # 直接调用：client.chat.completions.create。
    # 直接调用：self.serializer.serialize_messages。
    # 直接调用：client.close。
    # 直接调用：_field。
    # 直接调用：_reasoning_details。
    # 直接调用：_merge_detail。
    # 直接调用：''.join。
    # 直接调用：item.get。
    # 直接调用：reasoning.append。
    # 直接调用：ModelStreamEvent。
    # 直接调用：content.append。
    # 直接调用：int。
    # 直接调用：tool_parts.setdefault。
    # 直接调用：sorted。
    # 直接调用：tool_parts.items。
    # 直接调用：_usage_numbers。
    # 直接调用：ModelResponse。
    # 直接调用：_parsed_tool_calls。
    # 表达式分流：self.use_request_api_key；真值与假值分支分别计算，不同时执行。
    # 表达式分流：self.api_key_env；真值与假值分支分别计算，不同时执行。
    # 表达式分流：self.api_key_env is None；真值与假值分支分别计算，不同时执行。
    # 表达式分流：tools；真值与假值分支分别计算，不同时执行。
    # 清理边界：本函数含 finally；正常返回和异常展开都经过清理块。
    # 集合派生：item 来自 raw_details。
    # 生成器交付：ModelStreamEvent('reasoning', visible)。
    # 生成器交付：ModelStreamEvent('content', str(part))。
    # 集合派生：(_, part) 来自 sorted(tool_parts.items())。
    # 生成器交付：ModelStreamEvent('completed', response=ModelResponse(content=''.join(content), reasoning=''.join(reasoning), reasoning_details=[item for item in details if item], tool_calls=_parsed_tool_calls(raw_calls), input_tokens=prompt, output_tokens=。
    # 集合派生：item 来自 details；过滤 item。
    # [/maint-note:8d726e70b4b1]
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


# [maint-note:c124a324d21d]
# 数据与行为导航：RoutedModelGateway。
# 继承接口：ModelGateway。
# [/maint-note:c124a324d21d]
class RoutedModelGateway(ModelGateway):
    """Dispatch by an explicitly injected RuntimeModelConfig, never by key shape."""

    # [maint-note:f431c494354e]
    # 维护导航：__init__；下列为本函数源码中的接口与路径索引。
    # 输入 serializer：DeepSeekRequestSerializer | None；缺省表达式 None。
    # 返回契约：None。
    # 写入目标：self.serializer。
    # 直接调用：DeepSeekRequestSerializer。
    # [/maint-note:f431c494354e]
    def __init__(self, *, serializer: DeepSeekRequestSerializer | None = None) -> None:
        self.serializer = serializer or DeepSeekRequestSerializer()

    # [maint-note:f365a6855f02]
    # 维护导航：generate；下列为本函数源码中的接口与路径索引。
    # 输入 request：ModelRequest。
    # 返回契约：ModelResponse。
    # 分支条件：not isinstance(raw_config, dict)。
    # 分支条件：config.provider == 'openrouter'。
    # 分支条件：referer。
    # 分支条件：title。
    # 返回路径：response。
    # 显式抛出：ValueError('ModelRequest is missing runtime_model_config')。
    # 写入目标：headers['HTTP-Referer']。
    # 写入目标：headers['X-Title']。
    # 直接调用：request.metadata.get。
    # 直接调用：isinstance。
    # 直接调用：ValueError。
    # 直接调用：RuntimeModelConfig.from_dict。
    # 直接调用：os.environ.get('OPENROUTER_HTTP_REFERER', '').strip。
    # 直接调用：os.environ.get。
    # 直接调用：os.environ.get('OPENROUTER_APP_TITLE', 'NaturalCC Agent').strip。
    # 直接调用：OpenAICompatibleGateway。
    # 直接调用：config.openrouter_extra_body。
    # 直接调用：gateway.generate。
    # 表达式分流：config.provider == 'ollama'；真值与假值分支分别计算，不同时执行。
    # [/maint-note:f365a6855f02]
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

    # [maint-note:786e8865dde0]
    # 维护导航：stream；下列为本函数源码中的接口与路径索引。
    # 输入 request：ModelRequest。
    # 返回契约：Iterator[ModelStreamEvent]。
    # 分支条件：not isinstance(raw_config, dict)。
    # 分支条件：config.provider == 'openrouter'。
    # 分支条件：referer。
    # 分支条件：title。
    # 分支条件：config.provider == 'deepseek' and config.model == 'deepseek-chat'。
    # 显式抛出：ValueError('ModelRequest is missing runtime_model_config')。
    # 写入目标：headers['HTTP-Referer']。
    # 写入目标：headers['X-Title']。
    # 写入目标：extra_body['reasoning']。
    # 写入目标：extra_body['thinking']。
    # 直接调用：request.metadata.get。
    # 直接调用：isinstance。
    # 直接调用：ValueError。
    # 直接调用：RuntimeModelConfig.from_dict。
    # 直接调用：os.environ.get('OPENROUTER_HTTP_REFERER', '').strip。
    # 直接调用：os.environ.get。
    # 直接调用：os.environ.get('OPENROUTER_APP_TITLE', 'NaturalCC Agent').strip。
    # 直接调用：config.openrouter_extra_body。
    # 直接调用：OpenAICompatibleGateway。
    # 直接调用：gateway.stream。
    # 表达式分流：config.provider == 'ollama'；真值与假值分支分别计算，不同时执行。
    # 生成器交付：gateway.stream(request)。
    # [/maint-note:786e8865dde0]
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


# [maint-note:802f212efaf5]
# 数据与行为导航：FallbackModelGateway。
# 继承接口：ModelGateway。
# [/maint-note:802f212efaf5]
class FallbackModelGateway(ModelGateway):
    # [maint-note:99c5dec14132]
    # 维护导航：__init__；下列为本函数源码中的接口与路径索引。
    # 输入 gateways：list[ModelGateway]。
    # 返回契约：None。
    # 分支条件：not gateways。
    # 显式抛出：ValueError('at least one model gateway is required')。
    # 写入目标：self.gateways。
    # 直接调用：ValueError。
    # [/maint-note:99c5dec14132]
    def __init__(self, gateways: list[ModelGateway]) -> None:
        if not gateways:
            raise ValueError("at least one model gateway is required")
        self.gateways = gateways

    # [maint-note:c5161b7b5717]
    # 维护导航：generate；下列为本函数源码中的接口与路径索引。
    # 输入 request：ModelRequest。
    # 返回契约：ModelResponse。
    # 遍历边界：gateway ← self.gateways。
    # 返回路径：gateway.generate(request)。
    # 显式抛出：RuntimeError('all model gateways failed: ' + ' | '.join(errors))。
    # 捕获异常：Exception。
    # 直接调用：gateway.generate。
    # 直接调用：errors.append。
    # 直接调用：type。
    # 直接调用：RuntimeError。
    # 直接调用：' | '.join。
    # [/maint-note:c5161b7b5717]
    def generate(self, request: ModelRequest) -> ModelResponse:
        errors = []
        for gateway in self.gateways:
            try:
                return gateway.generate(request)
            except Exception as exc:
                errors.append(f"{type(exc).__name__}: {exc}")
        raise RuntimeError("all model gateways failed: " + " | ".join(errors))
