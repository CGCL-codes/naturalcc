from __future__ import annotations

import json
import os
from abc import ABC, abstractmethod
from collections import deque
from typing import Any

from .contracts import ModelRequest, ModelResponse, ToolCall


class ModelGateway(ABC):
    @abstractmethod
    def generate(self, request: ModelRequest) -> ModelResponse:
        raise NotImplementedError


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
                result.append(
                    {
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
                )
            elif role == "tool":
                result.append(
                    {
                        "role": "tool",
                        "tool_call_id": message["tool_call_id"],
                        "content": str(message.get("content", "")),
                    }
                )
            else:
                result.append(
                    {"role": role, "content": str(message.get("content", ""))}
                )
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
    ) -> None:
        self.model = model
        self.api_key = api_key
        self.base_url = base_url
        self.max_tokens = max_tokens
        self.serializer = serializer or DeepSeekRequestSerializer()

    def generate(self, request: ModelRequest) -> ModelResponse:
        from openai import OpenAI

        key = (
            str(request.metadata.get("api_key") or "").strip()
            or self.api_key
            or os.environ.get("DEEPSEEK_API_KEY")
            or os.environ.get("OPENAI_API_KEY")
        )
        if not key:
            raise ValueError("No API key configured for the model gateway")
        client = OpenAI(api_key=key, base_url=self.base_url)
        serialized_tools = self.serializer.serialize_tools(request.tools)
        options: dict[str, Any] = {}
        if request.response_format is not None:
            options["response_format"] = request.response_format
        response = client.chat.completions.create(
            model=self.model,
            messages=self.serializer.serialize_messages(request.messages),
            tools=serialized_tools or None,
            tool_choice="auto" if serialized_tools else None,
            max_tokens=request.max_output_tokens,
            **options,
        )
        message = response.choices[0].message
        calls: list[ToolCall] = []
        for index, raw in enumerate(message.tool_calls or []):
            try:
                args = json.loads(raw.function.arguments or "{}")
            except json.JSONDecodeError:
                args = {}
            calls.append(ToolCall(raw.id or f"call_{index + 1:03d}", raw.function.name, args))
        usage = response.usage
        usage_extra = getattr(usage, "model_extra", None) or {}

        def usage_value(name: str) -> int:
            value = getattr(usage, name, None) if usage else None
            if value is None and isinstance(usage, dict):
                value = usage.get(name)
            if value is None and isinstance(usage_extra, dict):
                value = usage_extra.get(name)
            return int(value or 0)

        return ModelResponse(
            content=message.content or "",
            tool_calls=calls,
            input_tokens=usage_value("prompt_tokens"),
            output_tokens=usage_value("completion_tokens"),
            prompt_cache_hit_tokens=usage_value("prompt_cache_hit_tokens"),
            prompt_cache_miss_tokens=usage_value("prompt_cache_miss_tokens"),
            model=self.model,
        )


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
