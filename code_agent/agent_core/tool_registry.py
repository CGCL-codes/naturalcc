from __future__ import annotations

import re
from typing import Any

from .contracts import RiskLevel, ToolContext, ToolResult, ToolSpec


_TYPE_MAP = {
    "string": str,
    "integer": int,
    "number": (int, float),
    "boolean": bool,
    "array": list,
    "object": dict,
}
_SECRET_KEY_NAMES = {
    "api_key",
    "apikey",
    "access_token",
    "auth_token",
    "password",
    "passwd",
    "secret",
    "token",
}
_ASSIGNMENT_SECRET = re.compile(
    r"(?i)\b(api[_-]?key|access[_-]?token|auth[_-]?token|password|passwd|secret)"
    r"(\s*[:=]\s*)([^\s,;]+)"
)
_TOKEN_SECRET = re.compile(r"\b(?:sk|ghp|github_pat|xox[baprs])-[A-Za-z0-9_-]{16,}\b")
_MODEL_TOOL_NAME = re.compile(r"^[a-zA-Z0-9_-]+$")
_MODEL_TOOL_NAME_CHARS = re.compile(r"[^a-zA-Z0-9_-]")


# [maint-note:67452bcee150]
# 维护导航：_model_safe_name；下列为本函数源码中的接口与路径索引。
# 输入 name：str。
# 返回契约：str。
# 分支条件：_MODEL_TOOL_NAME.match(name)。
# 返回路径：name。
# 返回路径：_MODEL_TOOL_NAME_CHARS.sub('_', name)。
# 直接调用：_MODEL_TOOL_NAME.match。
# 直接调用：_MODEL_TOOL_NAME_CHARS.sub。
# [/maint-note:67452bcee150]
def _model_safe_name(name: str) -> str:
    if _MODEL_TOOL_NAME.match(name):
        return name
    return _MODEL_TOOL_NAME_CHARS.sub("_", name)


# [maint-note:9a3c270744e1]
# 维护导航：_validate_value；下列为本函数源码中的接口与路径索引。
# 输入 name：str。
# 输入 value：Any。
# 输入 schema：dict[str, Any]。
# 返回契约：None。
# 分支条件：expected is not None。
# 分支条件：expected_name in {'integer', 'number'} and isinstance(value, bool)。
# 分支条件：not isinstance(value, expected)。
# 分支条件：'enum' in schema and value not in schema['enum']。
# 分支条件：expected_name == 'array' and 'items' in schema。
# 遍历边界：(index, item) ← enumerate(value)。
# 显式抛出：ValueError(f'{name} must be {expected_name}')。
# 显式抛出：ValueError(f"{name} must be one of {schema['enum']}")。
# 直接调用：schema.get。
# 直接调用：_TYPE_MAP.get。
# 直接调用：isinstance。
# 直接调用：ValueError。
# 直接调用：enumerate。
# 直接调用：_validate_value。
# [/maint-note:9a3c270744e1]
def _validate_value(name: str, value: Any, schema: dict[str, Any]) -> None:
    expected_name = schema.get("type")
    expected = _TYPE_MAP.get(expected_name)
    if expected is not None:
        if expected_name in {"integer", "number"} and isinstance(value, bool):
            raise ValueError(f"{name} must be {expected_name}")
        if not isinstance(value, expected):
            raise ValueError(f"{name} must be {expected_name}")
    if "enum" in schema and value not in schema["enum"]:
        raise ValueError(f"{name} must be one of {schema['enum']}")
    if expected_name == "array" and "items" in schema:
        for index, item in enumerate(value):
            _validate_value(f"{name}[{index}]", item, schema["items"])


# [maint-note:d0aa2585299d]
# 维护导航：validate_arguments；下列为本函数源码中的接口与路径索引。
# 输入 arguments：dict[str, Any]。
# 输入 schema：dict[str, Any]。
# 返回契约：None。
# 分支条件：not isinstance(arguments, dict)。
# 分支条件：missing。
# 分支条件：schema.get('additionalProperties', True) is False。
# 分支条件：unknown。
# 分支条件：name in properties。
# 遍历边界：(name, value) ← arguments.items()。
# 显式抛出：ValueError('tool arguments must be an object')。
# 显式抛出：ValueError(f"missing required arguments: {', '.join(missing)}")。
# 显式抛出：ValueError(f"unknown arguments: {', '.join(unknown)}")。
# 直接调用：isinstance。
# 直接调用：ValueError。
# 直接调用：schema.get。
# 直接调用：set。
# 直接调用：sorted。
# 直接调用：', '.join。
# 直接调用：arguments.items。
# 直接调用：_validate_value。
# [/maint-note:d0aa2585299d]
def validate_arguments(arguments: dict[str, Any], schema: dict[str, Any]) -> None:
    if not isinstance(arguments, dict):
        raise ValueError("tool arguments must be an object")
    properties = schema.get("properties", {})
    required = set(schema.get("required", []))
    missing = sorted(required - set(arguments))
    if missing:
        raise ValueError(f"missing required arguments: {', '.join(missing)}")
    if schema.get("additionalProperties", True) is False:
        unknown = sorted(set(arguments) - set(properties))
        if unknown:
            raise ValueError(f"unknown arguments: {', '.join(unknown)}")
    for name, value in arguments.items():
        if name in properties:
            _validate_value(name, value, properties[name])


# [maint-note:3c8735f1c699]
# 维护导航：redact_sensitive_text；下列为本函数源码中的接口与路径索引。
# 输入 value：Any。
# 返回契约：str。
# 分支条件：value is None。
# 分支条件：not isinstance(value, str)。
# 返回路径：''。
# 返回路径：_TOKEN_SECRET.sub('[REDACTED]', value)。
# 直接调用：isinstance。
# 直接调用：str。
# 直接调用：_ASSIGNMENT_SECRET.sub。
# 直接调用：_TOKEN_SECRET.sub。
# [/maint-note:3c8735f1c699]
def redact_sensitive_text(value: Any) -> str:
    if value is None:
        return ""
    if not isinstance(value, str):
        value = str(value)
    value = _ASSIGNMENT_SECRET.sub(lambda match: f"{match.group(1)}{match.group(2)}[REDACTED]", value)
    return _TOKEN_SECRET.sub("[REDACTED]", value)


# [maint-note:45150dad4604]
# 维护导航：redact_sensitive_value；下列为本函数源码中的接口与路径索引。
# 输入 value：Any。
# 输入 key：str | None；缺省表达式 None。
# 返回契约：Any。
# 分支条件：key and key.casefold().replace('-', '_') in _SECRET_KEY_NAMES。
# 分支条件：isinstance(value, str)。
# 分支条件：isinstance(value, dict)。
# 分支条件：isinstance(value, list)。
# 返回路径：'[REDACTED]'。
# 返回路径：redact_sensitive_text(value)。
# 返回路径：{item_key: redact_sensitive_value(item, str(item_key)) for (item_key, item) in value.items()}。
# 返回路径：[redact_sensitive_value(item) for item in value]。
# 返回路径：value。
# 直接调用：key.casefold().replace。
# 直接调用：key.casefold。
# 直接调用：isinstance。
# 直接调用：redact_sensitive_text。
# 直接调用：redact_sensitive_value。
# 直接调用：str。
# 直接调用：value.items。
# 集合派生：(item_key, item) 来自 value.items()。
# 集合派生：item 来自 value。
# [/maint-note:45150dad4604]
def redact_sensitive_value(value: Any, key: str | None = None) -> Any:
    if key and key.casefold().replace("-", "_") in _SECRET_KEY_NAMES:
        return "[REDACTED]"
    if isinstance(value, str):
        return redact_sensitive_text(value)
    if isinstance(value, dict):
        return {
            item_key: redact_sensitive_value(item, str(item_key))
            for item_key, item in value.items()
        }
    if isinstance(value, list):
        return [redact_sensitive_value(item) for item in value]
    return value


class ToolRegistry:
    # [maint-note:7b87d005545a]
    # 维护导航：__init__；下列为本函数源码中的接口与路径索引。
    # 返回契约：None。
    # 写入目标：self._specs。
    # 写入目标：self._model_names。
    # 写入目标：self._canonical_names。
    # [/maint-note:7b87d005545a]
    def __init__(self) -> None:
        self._specs: dict[str, ToolSpec] = {}
        self._model_names: dict[str, str] = {}
        self._canonical_names: dict[str, str] = {}

    # [maint-note:50b6bb3040b5]
    # 维护导航：register；下列为本函数源码中的接口与路径索引。
    # 输入 spec：ToolSpec。
    # 返回契约：None。
    # 分支条件：spec.name in self._specs。
    # 分支条件：existing is not None and existing != spec.name。
    # 显式抛出：ValueError(f'tool already registered: {spec.name}')。
    # 显式抛出：ValueError(f'tool model name collision: {spec.name} and {existing} both map to {model_name}')。
    # 写入目标：self._specs[spec.name]。
    # 写入目标：self._model_names[spec.name]。
    # 写入目标：self._canonical_names[model_name]。
    # 直接调用：ValueError。
    # 直接调用：_model_safe_name。
    # 直接调用：self._canonical_names.get。
    # [/maint-note:50b6bb3040b5]
    def register(self, spec: ToolSpec) -> None:
        if spec.name in self._specs:
            raise ValueError(f"tool already registered: {spec.name}")
        model_name = _model_safe_name(spec.name)
        existing = self._canonical_names.get(model_name)
        if existing is not None and existing != spec.name:
            raise ValueError(f"tool model name collision: {spec.name} and {existing} both map to {model_name}")
        self._specs[spec.name] = spec
        self._model_names[spec.name] = model_name
        self._canonical_names[model_name] = spec.name

    # [maint-note:d07f45ef9b13]
    # 维护导航：get；下列为本函数源码中的接口与路径索引。
    # 输入 name：str。
    # 返回契约：ToolSpec | None。
    # 返回路径：self._specs.get(self.to_canonical_name(name))。
    # 直接调用：self._specs.get。
    # 直接调用：self.to_canonical_name。
    # [/maint-note:d07f45ef9b13]
    def get(self, name: str) -> ToolSpec | None:
        return self._specs.get(self.to_canonical_name(name))

    # [maint-note:b64967712927]
    # 维护导航：to_model_name；下列为本函数源码中的接口与路径索引。
    # 输入 name：str。
    # 返回契约：str。
    # 返回路径：self._model_names.get(canonical, _model_safe_name(name))。
    # 直接调用：self.to_canonical_name。
    # 直接调用：self._model_names.get。
    # 直接调用：_model_safe_name。
    # [/maint-note:b64967712927]
    def to_model_name(self, name: str) -> str:
        canonical = self.to_canonical_name(name)
        return self._model_names.get(canonical, _model_safe_name(name))

    # [maint-note:2989f0daf643]
    # 维护导航：to_canonical_name；下列为本函数源码中的接口与路径索引。
    # 输入 name：str。
    # 返回契约：str。
    # 返回路径：self._canonical_names.get(name, name)。
    # 直接调用：self._canonical_names.get。
    # [/maint-note:2989f0daf643]
    def to_canonical_name(self, name: str) -> str:
        return self._canonical_names.get(name, name)

    # [maint-note:1e201e45d77b]
    # 维护导航：list；下列为本函数源码中的接口与路径索引。
    # 返回契约：list[ToolSpec]。
    # 返回路径：list(self._specs.values())。
    # 直接调用：list。
    # 直接调用：self._specs.values。
    # [/maint-note:1e201e45d77b]
    def list(self) -> list[ToolSpec]:
        return list(self._specs.values())

    # [maint-note:2916400f859a]
    # 维护导航：model_definitions；下列为本函数源码中的接口与路径索引。
    # 输入 include_names：set[str] | None；缺省表达式 None。
    # 输入 exclude_names：set[str] | None；缺省表达式 None。
    # 返回契约：list[dict[str, Any]]。
    # 分支条件：include_names is not None and spec.name not in include_names。
    # 分支条件：exclude_names is not None and spec.name in exclude_names。
    # 遍历边界：spec ← self.list()。
    # 返回路径：definitions。
    # 写入目标：definition['function']['name']。
    # 直接调用：self.list。
    # 直接调用：spec.model_definition。
    # 直接调用：self.to_model_name。
    # 直接调用：definitions.append。
    # [/maint-note:2916400f859a]
    def model_definitions(
        self,
        *,
        include_names: set[str] | None = None,
        exclude_names: set[str] | None = None,
    ) -> list[dict[str, Any]]:
        definitions: list[dict[str, Any]] = []
        for spec in self.list():
            if include_names is not None and spec.name not in include_names:
                continue
            if exclude_names is not None and spec.name in exclude_names:
                continue
            definition = spec.model_definition()
            definition["function"]["name"] = self.to_model_name(spec.name)
            definitions.append(definition)
        return definitions

    # [maint-note:9e37a4eca135]
    # 维护导航：execute；下列为本函数源码中的接口与路径索引。
    # 输入 name：str。
    # 输入 arguments：dict[str, Any]。
    # 输入 context：ToolContext。
    # 返回契约：ToolResult。
    # 分支条件：spec is None。
    # 分支条件：spec.approval_required and spec.risk_level not in context.approved_risks。
    # 分支条件：context.cancellation.is_set()。
    # 分支条件：result.error。
    # 分支条件：len(result.summary) > spec.max_output_chars。
    # 返回路径：ToolResult.failure(f'unknown tool: {name}', 'UnknownTool')。
    # 返回路径：ToolResult.failure(f'approval required for {spec.risk_level.value} tool: {canonical_name}', 'ApprovalRequired')。
    # 返回路径：ToolResult('cancelled', 'run cancelled before tool execution')。
    # 返回路径：ToolResult.failure(str(exc), type(exc).__name__)。
    # 返回路径：result。
    # 捕获异常：Exception。
    # 写入目标：result.summary。
    # 写入目标：result.data。
    # 写入目标：result.error。
    # 写入目标：result.truncated。
    # 直接调用：self.to_canonical_name。
    # 直接调用：self._specs.get。
    # 直接调用：ToolResult.failure。
    # 直接调用：context.cancellation.is_set。
    # 直接调用：ToolResult。
    # 直接调用：validate_arguments。
    # 直接调用：spec.execute。
    # 直接调用：str。
    # 直接调用：type。
    # 直接调用：redact_sensitive_text。
    # 直接调用：redact_sensitive_value。
    # 直接调用：len。
    # [/maint-note:9e37a4eca135]
    def execute(self, name: str, arguments: dict[str, Any], context: ToolContext) -> ToolResult:
        canonical_name = self.to_canonical_name(name)
        spec = self._specs.get(canonical_name)
        if spec is None:
            return ToolResult.failure(f"unknown tool: {name}", "UnknownTool")
        if spec.approval_required and spec.risk_level not in context.approved_risks:
            return ToolResult.failure(
                f"approval required for {spec.risk_level.value} tool: {canonical_name}",
                "ApprovalRequired",
            )
        if context.cancellation.is_set():
            return ToolResult("cancelled", "run cancelled before tool execution")
        try:
            validate_arguments(arguments, spec.input_schema)
            result = spec.execute(context, arguments)
        except Exception as exc:
            return ToolResult.failure(str(exc), type(exc).__name__)
        result.summary = redact_sensitive_text(result.summary)
        result.data = redact_sensitive_value(result.data)
        if result.error:
            result.error = redact_sensitive_value(result.error)
        if len(result.summary) > spec.max_output_chars:
            result.summary = result.summary[: spec.max_output_chars]
            result.truncated = True
        return result


# [maint-note:9f7fbd2e3b0b]
# 维护导航：build_default_registry；下列为本函数源码中的接口与路径索引。
# 输入 include_mutating：bool；缺省表达式 True。
# 返回契约：ToolRegistry。
# 分支条件：include_mutating。
# 遍历边界：spec ← [*workspace_tool_specs(), *naturalcc_tool_specs(), *codegraph_tool_specs(), *pipeline_tool_specs(include_mutating)]。
# 遍历边界：spec ← editing_tool_specs()。
# 遍历边界：spec ← verification_tool_specs()。
# 返回路径：registry。
# 直接调用：ToolRegistry。
# 直接调用：workspace_tool_specs。
# 直接调用：naturalcc_tool_specs。
# 直接调用：codegraph_tool_specs。
# 直接调用：pipeline_tool_specs。
# 直接调用：registry.register。
# 直接调用：editing_tool_specs。
# 直接调用：command_tool_spec。
# 直接调用：verification_tool_specs。
# [/maint-note:9f7fbd2e3b0b]
def build_default_registry(include_mutating: bool = True) -> ToolRegistry:
    from .tools.codegraph import codegraph_tool_specs
    from .tools.naturalcc import naturalcc_tool_specs
    from .tools.workspace import workspace_tool_specs
    from .tools.pipeline import pipeline_tool_specs

    registry = ToolRegistry()
    for spec in [
        *workspace_tool_specs(),
        *naturalcc_tool_specs(),
        *codegraph_tool_specs(),
        *pipeline_tool_specs(include_mutating),
    ]:
        registry.register(spec)
    if include_mutating:
        from .tools.command import command_tool_spec
        from .tools.editing import editing_tool_specs
        from .tools.verification import verification_tool_specs

        for spec in editing_tool_specs():
            registry.register(spec)
        registry.register(command_tool_spec())
        for spec in verification_tool_specs():
            registry.register(spec)
    return registry
