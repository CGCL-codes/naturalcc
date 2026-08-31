from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
from typing import Any, TYPE_CHECKING

if TYPE_CHECKING:
    from .model_gateway import DeepSeekRequestSerializer


class TokenizerUnavailable(RuntimeError):
    """Raised when the pinned local DeepSeek tokenizer cannot be loaded."""


class ContextHardLimitExceeded(RuntimeError):
    def __init__(self, breakdown: "TokenBreakdown") -> None:
        self.breakdown = breakdown
        super().__init__(
            "DeepSeek context hard limit exceeded: "
            f"{breakdown.total_with_reserve} > {breakdown.context_window_tokens}"
        )


@dataclass(frozen=True)
class DeepSeekModelProfile:
    model: str = "deepseek-chat"
    context_window_tokens: int = 65_536
    default_output_reserve_tokens: int = 4_096
    safety_margin_tokens: int = 512
    provider_framing_tokens: int = 256
    compaction_trigger_ratio: float = 0.72
    compaction_target_ratio: float = 0.50
    analyzer_output_tokens: int = 4_096
    summarizer_output_tokens: int = 2_048

    def __post_init__(self) -> None:
        if self.model != "deepseek-chat":
            raise ValueError("first-stage token budgeting only supports deepseek-chat")
        if not 0 < self.compaction_target_ratio < self.compaction_trigger_ratio < 1:
            raise ValueError(
                "target_ratio must be below trigger_ratio and both must be between 0 and 1"
            )
        if self.context_window_tokens <= 0:
            raise ValueError("context_window_tokens must be positive")
        for name in (
            "default_output_reserve_tokens",
            "safety_margin_tokens",
            "provider_framing_tokens",
        ):
            if getattr(self, name) < 0:
                raise ValueError(f"{name} must be non-negative")
        for name in ("analyzer_output_tokens", "summarizer_output_tokens"):
            if getattr(self, name) <= 0:
                raise ValueError(f"{name} must be positive")
        if (
            self.default_output_reserve_tokens + self.safety_margin_tokens
            >= self.context_window_tokens
        ):
            raise ValueError("output reserve and safety margin must fit the context window")


@dataclass(frozen=True)
class TokenBreakdown:
    message_tokens: int
    tool_schema_tokens: int
    provider_framing_tokens: int
    reserved_output_tokens: int
    safety_margin_tokens: int
    context_window_tokens: int

    @property
    def input_tokens(self) -> int:
        return (
            self.message_tokens
            + self.tool_schema_tokens
            + self.provider_framing_tokens
        )

    @property
    def total_with_reserve(self) -> int:
        return (
            self.input_tokens
            + self.reserved_output_tokens
            + self.safety_margin_tokens
        )


class DeepSeekTokenCounter:
    def __init__(
        self,
        tokenizer: Any,
        *,
        chat_template: str,
        tokenizer_config: dict[str, Any] | None = None,
    ) -> None:
        self.tokenizer = tokenizer
        self.chat_template = chat_template
        self.tokenizer_config = {
            key: self._normalize_template_value(value)
            for key, value in dict(tokenizer_config or {}).items()
        }

    @classmethod
    def _normalize_template_value(cls, value: Any) -> Any:
        """Match Transformers' AddedToken-to-string template semantics."""
        if isinstance(value, dict) and value.get("__type") == "AddedToken":
            return value.get("content", "")
        if isinstance(value, dict):
            return {
                key: cls._normalize_template_value(item)
                for key, item in value.items()
            }
        if isinstance(value, list):
            return [cls._normalize_template_value(item) for item in value]
        return value

    @classmethod
    def from_directory(cls, directory: str | Path) -> "DeepSeekTokenCounter":
        root = Path(directory).expanduser().resolve()
        required = [root / "tokenizer.json", root / "tokenizer_config.json"]
        if not all(path.is_file() for path in required):
            raise TokenizerUnavailable(
                f"DeepSeek tokenizer files are missing from {root}"
            )
        try:
            from tokenizers import Tokenizer

            tokenizer = Tokenizer.from_file(str(root / "tokenizer.json"))
            config = json.loads(
                (root / "tokenizer_config.json").read_text(encoding="utf-8")
            )
            chat_template = config.get("chat_template")
            if not isinstance(chat_template, str) or not chat_template.strip():
                raise ValueError("tokenizer_config.json has no chat_template")
        except Exception as exc:
            raise TokenizerUnavailable(
                f"DeepSeek tokenizer could not be loaded from {root}: {exc}"
            ) from exc
        return cls(
            tokenizer,
            chat_template=chat_template,
            tokenizer_config=config,
        )

    def count_text(self, value: str) -> int:
        return len(self.tokenizer.encode(value, add_special_tokens=False))

    def count_request(
        self,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]],
        serializer: "DeepSeekRequestSerializer",
        profile: DeepSeekModelProfile,
        *,
        reserved_output_tokens: int | None = None,
    ) -> TokenBreakdown:
        serialized_messages = serializer.serialize_messages(messages)
        rendered_messages = self._render_chat(serialized_messages)
        serialized_tools = serializer.serialize_tools(tools)
        tool_schema_tokens = 0
        if serialized_tools:
            canonical_tools = json.dumps(
                serialized_tools,
                ensure_ascii=False,
                separators=(",", ":"),
                sort_keys=True,
            )
            tool_schema_tokens = self.count_text(canonical_tools)
        return TokenBreakdown(
            message_tokens=self.count_text(rendered_messages),
            tool_schema_tokens=tool_schema_tokens,
            provider_framing_tokens=profile.provider_framing_tokens,
            reserved_output_tokens=(
                profile.default_output_reserve_tokens
                if reserved_output_tokens is None
                else reserved_output_tokens
            ),
            safety_margin_tokens=profile.safety_margin_tokens,
            context_window_tokens=profile.context_window_tokens,
        )

    def assert_fits(
        self,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]],
        serializer: "DeepSeekRequestSerializer",
        profile: DeepSeekModelProfile,
        *,
        reserved_output_tokens: int | None = None,
    ) -> TokenBreakdown:
        breakdown = self.count_request(
            messages,
            tools,
            serializer,
            profile,
            reserved_output_tokens=reserved_output_tokens,
        )
        if breakdown.total_with_reserve > breakdown.context_window_tokens:
            raise ContextHardLimitExceeded(breakdown)
        return breakdown

    def _render_chat(self, messages: list[dict[str, Any]]) -> str:
        try:
            from jinja2.sandbox import ImmutableSandboxedEnvironment

            template = ImmutableSandboxedEnvironment().from_string(self.chat_template)
            return template.render(
                messages=messages,
                add_generation_prompt=True,
                **self.tokenizer_config,
            )
        except Exception as exc:
            raise TokenizerUnavailable(
                f"DeepSeek chat template could not be rendered: {exc}"
            ) from exc
