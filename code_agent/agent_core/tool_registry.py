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


def _model_safe_name(name: str) -> str:
    if _MODEL_TOOL_NAME.match(name):
        return name
    return _MODEL_TOOL_NAME_CHARS.sub("_", name)


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


def redact_sensitive_text(value: str) -> str:
    value = _ASSIGNMENT_SECRET.sub(lambda match: f"{match.group(1)}{match.group(2)}[REDACTED]", value)
    return _TOKEN_SECRET.sub("[REDACTED]", value)


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
    def __init__(self) -> None:
        self._specs: dict[str, ToolSpec] = {}
        self._model_names: dict[str, str] = {}
        self._canonical_names: dict[str, str] = {}

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

    def get(self, name: str) -> ToolSpec | None:
        return self._specs.get(self.to_canonical_name(name))

    def to_model_name(self, name: str) -> str:
        canonical = self.to_canonical_name(name)
        return self._model_names.get(canonical, _model_safe_name(name))

    def to_canonical_name(self, name: str) -> str:
        return self._canonical_names.get(name, name)

    def list(self) -> list[ToolSpec]:
        return list(self._specs.values())

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


def build_default_registry(include_mutating: bool = True) -> ToolRegistry:
    from .tools.codegraph import codegraph_tool_specs
    from .tools.naturalcc import naturalcc_tool_specs
    from .tools.workspace import workspace_tool_specs

    registry = ToolRegistry()
    for spec in [
        *workspace_tool_specs(),
        *naturalcc_tool_specs(),
        *codegraph_tool_specs(),
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
