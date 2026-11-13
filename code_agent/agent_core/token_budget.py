from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
from typing import Any, TYPE_CHECKING

if TYPE_CHECKING:
    from .model_gateway import DeepSeekRequestSerializer


# [maint-note:1a8df8ba8137]
# 数据与行为导航：TokenizerUnavailable。
# 继承接口：RuntimeError。
# [/maint-note:1a8df8ba8137]
class TokenizerUnavailable(RuntimeError):
    """Raised when the pinned local DeepSeek tokenizer cannot be loaded."""


# [maint-note:152e91381d17]
# 数据与行为导航：ContextHardLimitExceeded。
# 继承接口：RuntimeError。
# [/maint-note:152e91381d17]
class ContextHardLimitExceeded(RuntimeError):
    # [maint-note:ec267a81bca8]
    # 维护导航：__init__；下列为本函数源码中的接口与路径索引。
    # 输入 breakdown：'TokenBreakdown'。
    # 返回契约：None。
    # 写入目标：self.breakdown。
    # 直接调用：super().__init__。
    # 直接调用：super。
    # [/maint-note:ec267a81bca8]
    def __init__(self, breakdown: "TokenBreakdown") -> None:
        self.breakdown = breakdown
        super().__init__(
            "DeepSeek context hard limit exceeded: "
            f"{breakdown.total_with_reserve} > {breakdown.context_window_tokens}"
        )


# [maint-note:e4da7be73399]
# 数据与行为导航：ModelContextProfile。
# 字段 model：str；初始表达式 'deepseek-v4-pro'。
# 字段 context_window_tokens：int；初始表达式 262144。
# 字段 default_output_reserve_tokens：int；初始表达式 4096。
# 字段 safety_margin_tokens：int；初始表达式 512。
# 字段 provider_framing_tokens：int；初始表达式 256。
# 字段 compaction_trigger_ratio：float；初始表达式 0.65。
# 字段 compaction_target_ratio：float；初始表达式 0.45。
# 字段 analyzer_output_tokens：int；初始表达式 4096。
# 字段 summarizer_output_tokens：int；初始表达式 2048。
# [/maint-note:e4da7be73399]
@dataclass(frozen=True)
class ModelContextProfile:
    model: str = "deepseek-v4-pro"
    context_window_tokens: int = 262_144
    default_output_reserve_tokens: int = 4_096
    safety_margin_tokens: int = 512
    provider_framing_tokens: int = 256
    compaction_trigger_ratio: float = 0.65
    compaction_target_ratio: float = 0.45
    analyzer_output_tokens: int = 4_096
    summarizer_output_tokens: int = 2_048

    # [maint-note:156fce88e5ba]
    # 维护导航：__post_init__；下列为本函数源码中的接口与路径索引。
    # 返回契约：None。
    # 分支条件：not 0 < self.compaction_target_ratio < self.compaction_trigger_ratio < 1。
    # 分支条件：self.context_window_tokens <= 0。
    # 分支条件：getattr(self, name) < 0。
    # 分支条件：getattr(self, name) <= 0。
    # 分支条件：self.default_output_reserve_tokens + self.safety_margin_tokens >= self.context_window_tokens。
    # 遍历边界：name ← ('default_output_reserve_tokens', 'safety_margin_tokens', 'provider_framing_tokens')。
    # 遍历边界：name ← ('analyzer_output_tokens', 'summarizer_output_tokens')。
    # 显式抛出：ValueError('target_ratio must be below trigger_ratio and both must be between 0 and 1')。
    # 显式抛出：ValueError('context_window_tokens must be positive')。
    # 显式抛出：ValueError(f'{name} must be non-negative')。
    # 显式抛出：ValueError(f'{name} must be positive')。
    # 显式抛出：ValueError('output reserve and safety margin must fit the context window')。
    # 直接调用：ValueError。
    # 直接调用：getattr。
    # [/maint-note:156fce88e5ba]
    def __post_init__(self) -> None:
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


# The local tokenizer remains DeepSeek-specific in the first rollout, but the
# context geometry belongs to the selected runtime model. Keep this alias so
# existing integrations and tests retain their public import path.
DeepSeekModelProfile = ModelContextProfile


# [maint-note:2cc1eb7cd495]
# 数据与行为导航：TokenBreakdown。
# 字段 message_tokens：int。
# 字段 tool_schema_tokens：int。
# 字段 provider_framing_tokens：int。
# 字段 reserved_output_tokens：int。
# 字段 safety_margin_tokens：int。
# 字段 context_window_tokens：int。
# [/maint-note:2cc1eb7cd495]
@dataclass(frozen=True)
class TokenBreakdown:
    message_tokens: int
    tool_schema_tokens: int
    provider_framing_tokens: int
    reserved_output_tokens: int
    safety_margin_tokens: int
    context_window_tokens: int

    # [maint-note:dd612411a7c6]
    # 维护导航：input_tokens；下列为本函数源码中的接口与路径索引。
    # 返回契约：int。
    # 返回路径：self.message_tokens + self.tool_schema_tokens + self.provider_framing_tokens。
    # [/maint-note:dd612411a7c6]
    @property
    def input_tokens(self) -> int:
        return (
            self.message_tokens
            + self.tool_schema_tokens
            + self.provider_framing_tokens
        )

    # [maint-note:31ded4241e85]
    # 维护导航：total_with_reserve；下列为本函数源码中的接口与路径索引。
    # 返回契约：int。
    # 返回路径：self.input_tokens + self.reserved_output_tokens + self.safety_margin_tokens。
    # [/maint-note:31ded4241e85]
    @property
    def total_with_reserve(self) -> int:
        return (
            self.input_tokens
            + self.reserved_output_tokens
            + self.safety_margin_tokens
        )


class DeepSeekTokenCounter:
    # [maint-note:aea41f50c75a]
    # 维护导航：__init__；下列为本函数源码中的接口与路径索引。
    # 输入 tokenizer：Any。
    # 输入 chat_template：str。
    # 输入 tokenizer_config：dict[str, Any] | None；缺省表达式 None。
    # 返回契约：None。
    # 写入目标：self.tokenizer。
    # 写入目标：self.chat_template。
    # 写入目标：self.tokenizer_config。
    # 直接调用：self._normalize_template_value。
    # 直接调用：dict(tokenizer_config or {}).items。
    # 直接调用：dict。
    # 集合派生：(key, value) 来自 dict(tokenizer_config or {}).items()。
    # [/maint-note:aea41f50c75a]
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

    # [maint-note:369b9221bebc]
    # 维护导航：_normalize_template_value；下列为本函数源码中的接口与路径索引。
    # 输入 value：Any。
    # 返回契约：Any。
    # 分支条件：isinstance(value, dict) and value.get('__type') == 'AddedToken'。
    # 分支条件：isinstance(value, dict)。
    # 分支条件：isinstance(value, list)。
    # 返回路径：value.get('content', '')。
    # 返回路径：{key: cls._normalize_template_value(item) for (key, item) in value.items()}。
    # 返回路径：[cls._normalize_template_value(item) for item in value]。
    # 返回路径：value。
    # 直接调用：isinstance。
    # 直接调用：value.get。
    # 直接调用：cls._normalize_template_value。
    # 直接调用：value.items。
    # 集合派生：(key, item) 来自 value.items()。
    # 集合派生：item 来自 value。
    # [/maint-note:369b9221bebc]
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

    # [maint-note:357966908027]
    # 维护导航：from_directory；下列为本函数源码中的接口与路径索引。
    # 输入 directory：str | Path。
    # 返回契约：'DeepSeekTokenCounter'。
    # 分支条件：not all((path.is_file() for path in required))。
    # 分支条件：not isinstance(chat_template, str) or not chat_template.strip()。
    # 返回路径：cls(tokenizer, chat_template=chat_template, tokenizer_config=config)。
    # 显式抛出：TokenizerUnavailable(f'DeepSeek tokenizer files are missing from {root}')。
    # 显式抛出：ValueError('tokenizer_config.json has no chat_template')。
    # 显式抛出：TokenizerUnavailable(f'DeepSeek tokenizer could not be loaded from {root}: {exc}')。
    # 捕获异常：Exception。
    # 直接调用：Path(directory).expanduser().resolve。
    # 直接调用：Path(directory).expanduser。
    # 直接调用：Path。
    # 直接调用：all。
    # 直接调用：path.is_file。
    # 直接调用：TokenizerUnavailable。
    # 直接调用：Tokenizer.from_file。
    # 直接调用：str。
    # 直接调用：json.loads。
    # 直接调用：(root / 'tokenizer_config.json').read_text。
    # 直接调用：config.get。
    # 直接调用：isinstance。
    # 直接调用：chat_template.strip。
    # 直接调用：ValueError。
    # 直接调用：cls。
    # 集合派生：path 来自 required。
    # [/maint-note:357966908027]
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

    # [maint-note:e8078911c448]
    # 维护导航：count_text；下列为本函数源码中的接口与路径索引。
    # 输入 value：str。
    # 返回契约：int。
    # 返回路径：len(self.tokenizer.encode(value, add_special_tokens=False))。
    # 直接调用：len。
    # 直接调用：self.tokenizer.encode。
    # [/maint-note:e8078911c448]
    def count_text(self, value: str) -> int:
        return len(self.tokenizer.encode(value, add_special_tokens=False))

    # [maint-note:3129c415934a]
    # 维护导航：count_request；下列为本函数源码中的接口与路径索引。
    # 输入 messages：list[dict[str, Any]]。
    # 输入 tools：list[dict[str, Any]]。
    # 输入 serializer：'DeepSeekRequestSerializer'。
    # 输入 profile：ModelContextProfile。
    # 输入 reserved_output_tokens：int | None；缺省表达式 None。
    # 返回契约：TokenBreakdown。
    # 分支条件：serialized_tools。
    # 返回路径：TokenBreakdown(message_tokens=self.count_text(rendered_messages), tool_schema_tokens=tool_schema_tokens, provider_framing_tokens=profile.provider_framing_tokens, reserved_output_tokens=profile.default_output_reserve_tokens if reserved_outpu。
    # 直接调用：serializer.serialize_messages。
    # 直接调用：self._render_chat。
    # 直接调用：serializer.serialize_tools。
    # 直接调用：json.dumps。
    # 直接调用：self.count_text。
    # 直接调用：TokenBreakdown。
    # 表达式分流：reserved_output_tokens is None；真值与假值分支分别计算，不同时执行。
    # [/maint-note:3129c415934a]
    def count_request(
        self,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]],
        serializer: "DeepSeekRequestSerializer",
        profile: ModelContextProfile,
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

    # [maint-note:f5fa09381475]
    # 维护导航：assert_fits；下列为本函数源码中的接口与路径索引。
    # 输入 messages：list[dict[str, Any]]。
    # 输入 tools：list[dict[str, Any]]。
    # 输入 serializer：'DeepSeekRequestSerializer'。
    # 输入 profile：ModelContextProfile。
    # 输入 reserved_output_tokens：int | None；缺省表达式 None。
    # 返回契约：TokenBreakdown。
    # 分支条件：breakdown.total_with_reserve > breakdown.context_window_tokens。
    # 返回路径：breakdown。
    # 显式抛出：ContextHardLimitExceeded(breakdown)。
    # 直接调用：self.count_request。
    # 直接调用：ContextHardLimitExceeded。
    # [/maint-note:f5fa09381475]
    def assert_fits(
        self,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]],
        serializer: "DeepSeekRequestSerializer",
        profile: ModelContextProfile,
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

    # [maint-note:3fbfe73414c9]
    # 维护导航：_render_chat；下列为本函数源码中的接口与路径索引。
    # 输入 messages：list[dict[str, Any]]。
    # 返回契约：str。
    # 返回路径：template.render(messages=messages, add_generation_prompt=True, **self.tokenizer_config)。
    # 显式抛出：TokenizerUnavailable(f'DeepSeek chat template could not be rendered: {exc}')。
    # 捕获异常：Exception。
    # 直接调用：ImmutableSandboxedEnvironment().from_string。
    # 直接调用：ImmutableSandboxedEnvironment。
    # 直接调用：template.render。
    # 直接调用：TokenizerUnavailable。
    # [/maint-note:3fbfe73414c9]
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
