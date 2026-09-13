"""Explicit adapters for Pipeline plugins; authority comes from ToolContext."""
from __future__ import annotations

import json
from uuid import uuid4

from ..contracts import RiskLevel, ToolContext, ToolResult, ToolSpec
from .editing import _unified_diff, _write_snapshot_once
from .workspace import resolve_workspace_path


def _execute(context: ToolContext, args: dict, feature: str, *, fix=False, deep=False) -> ToolResult:
    from code_agent.plugins.base import ExecutionContext, PluginResult
    from code_agent.plugins.code_completion import CodeCompletionPlugin
    from code_agent.plugins.vulnerability_detection import VulnerabilityDetectionPlugin

    plugin = CodeCompletionPlugin() if feature == "code_completion" else VulnerabilityDetectionPlugin()
    targets = []
    for value in args.get("target_files", []):
        path = resolve_workspace_path(context, value, must_exist=True)
        # Pipeline scanners assume a single project root, including in reports.
        relative = path.relative_to(context.workspace).as_posix()
        if not path.is_file():
            raise ValueError(f"target must be a file: {value}")
        if relative not in targets:
            targets.append(relative)
    mutating = feature == "code_completion" or fix
    if mutating and not targets:
        raise ValueError("target_files must contain at least one existing file")
    allowed = context.metadata.get("target_files") or []
    if mutating and allowed:
        permitted = {resolve_workspace_path(context, value) for value in allowed}
        if any(context.workspace / value not in permitted for value in targets):
            raise ValueError("plugin targets exceed the run's target_files")
    config = {key: value for key, value in args.items() if key not in {"instruction", "target_files"}}
    if feature == "vulnerability_detection":
        config.update(auto_fix=fix, analyzer="cppcheck" if deep else "auto" if fix else "builtin")
        config.setdefault("scan_scope", "targets" if targets else "project")
        if fix:
            config["scan_scope"] = "targets"
        if config.get("sanitizer_report"):
            report = resolve_workspace_path(context, config["sanitizer_report"], must_exist=True)
            config["sanitizer_report"] = report.relative_to(context.workspace).as_posix()
    error = plugin.validate(config)
    if error:
        raise ValueError(error)
    model = context.metadata.get("model", "deepseek/deepseek-chat")
    if model in {"deepseek-chat", "deepseek-reasoner"}:
        model = f"deepseek/{model}"
    execution = ExecutionContext(
        project_dir=str(context.workspace), target_files=targets,
        instruction=args.get("instruction", ""),
        model=model,
        api_key=context.metadata.get("api_key"), feature_config=config,
    )
    before, snapshots = {}, []
    if mutating:
        for relative in targets:
            before[relative] = (context.workspace / relative).read_text(encoding="utf-8")
            snapshots.append(str(_write_snapshot_once(context, relative, before[relative])))
    logs, final, failure = [], None, None
    try:
        for item in plugin.execute(execution):
            if context.cancellation.is_set():
                failure = "Plugin execution cancelled"
                break
            if isinstance(item, PluginResult):
                final = item
            elif isinstance(item, str):
                # Aider emits cumulative output, other plugin phases emit deltas.
                if logs and item.startswith(logs[-1]):
                    logs[-1] = item
                else:
                    logs.append(item)
    except Exception as exc:
        failure = str(exc)
    text = "\n".join(logs)
    success = final.success if final is not None else "任务圆满完成" in text and "❌" not in text
    if failure:
        success = False
    diffs, changed = [], []
    for relative, original in before.items():
        path = context.workspace / relative
        current = path.read_text(encoding="utf-8", errors="replace") if path.is_file() else ""
        if current != original or not path.exists():
            changed.append(relative)
            diffs.append(_unified_diff(relative, original, current))
    # Use the same redaction boundary for persisted artifacts as tool results.
    from ..tool_registry import redact_sensitive_text
    artifact = context.artifact_root / f"{feature}-{uuid4().hex}.json"
    data = {"report": final.report, **final.artifacts} if final else {}
    data["diff"] = "\n".join(diffs)
    artifact.write_text(redact_sensitive_text(json.dumps({"log": text, **data}, ensure_ascii=False)), encoding="utf-8")
    message = failure or (final.message if final else ("Code completion finished" if success else text[-2000:] or "Plugin did not report completion"))
    return ToolResult(
        "success" if success else "error", message, data=data,
        artifacts=[*snapshots, str(artifact)], changed_files=changed,
        error=None if success else {"type": "PipelinePluginFailed", "message": message},
    )


def pipeline_tool_specs(include_mutating=True):
    targets = {"type": "array", "items": {"type": "string"}, "minItems": 1}
    scan = {
        "target_files": targets,
        "scan_scope": {"type": "string", "enum": ["targets", "project"]},
        "severity_threshold": {"type": "string", "enum": ["low", "medium", "high", "critical"]},
        "rule_profile": {"type": "string", "enum": ["default", "c_cpp", "web"]},
        "max_findings": {"type": "integer", "minimum": 1, "maximum": 1000},
        "incremental": {"type": "boolean"},
        "sanitizer_report": {"type": "string"},
    }
    def schema(properties, required=()):
        return {"type": "object", "properties": properties, "required": list(required), "additionalProperties": False}
    specs = [ToolSpec(
        "vulnerability_detection", "Run the Pipeline vulnerability scanner without editing source. Returns findings and coverage limits. For array bounds, null pointers and leaks use vulnerability_detection.analyze.",
        schema(scan), RiskLevel.READ,
        lambda c, a: _execute(c, a, "vulnerability_detection"), parallel_safe=False,
    )]
    if include_mutating:
        specs.extend([
            ToolSpec("code_completion", "Run the Pipeline NaturalCC code-completion plugin and Aider on selected existing files. Requires write approval; returns snapshots and actual diffs.",
                schema({"target_files": targets, "instruction": {"type": "string", "minLength": 1},
                    "symbol": {"type": "string"}, "prefix": {"type": "string"},
                    "completion_type": {"type": "string", "enum": ["", "member", "variable", "function", "function_body", "type"]}}, ("target_files", "instruction")),
                RiskLevel.WRITE, lambda c, a: _execute(c, a, "code_completion"), idempotent=False, parallel_safe=False, default_timeout_seconds=900),
            ToolSpec("vulnerability_detection.analyze", "Run the Pipeline scanner plus installed Cppcheck for C/C++ bounds, null dereferences and leaks. Requires execute approval; never runs the target program.",
                schema(scan), RiskLevel.EXECUTE, lambda c, a: _execute(c, a, "vulnerability_detection", deep=True), parallel_safe=False, default_timeout_seconds=180),
            ToolSpec("vulnerability_detection.fix", "Scan selected files (including Cppcheck if available), then edit them with Pipeline Aider remediation. Requires execute approval for analysis and edits. Findings are pre-repair; re-scan and test afterwards.",
                schema({**scan, "instruction": {"type": "string"}, "extra_instruction": {"type": "string"}}, ("target_files",)),
                RiskLevel.EXECUTE, lambda c, a: _execute(c, a, "vulnerability_detection", fix=True), idempotent=False, parallel_safe=False, default_timeout_seconds=900),
        ])
    return specs
