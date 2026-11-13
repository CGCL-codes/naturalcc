"""Explicit adapters for Pipeline plugins; authority comes from ToolContext."""
from __future__ import annotations

import json
from uuid import uuid4

from ..contracts import RiskLevel, ToolContext, ToolResult, ToolSpec
from .editing import _local_aider_runtime, _unified_diff, _write_snapshot_once
from .workspace import resolve_workspace_path


# [maint-note:f28c525d32bd]
# 维护导航：_execute；下列为本函数源码中的接口与路径索引。
# 输入 context：ToolContext。
# 输入 args：dict。
# 输入 feature：str。
# 输入 fix：类型由调用方与函数体约束；缺省表达式 False。
# 输入 deep：类型由调用方与函数体约束；缺省表达式 False。
# 返回契约：ToolResult。
# 分支条件：not path.is_file()。
# 分支条件：relative not in targets。
# 分支条件：mutating and (not targets)。
# 分支条件：mutating and allowed。
# 分支条件：any((context.workspace / value not in permitted for value in targets))。
# 分支条件：feature == 'vulnerability_detection'。
# 分支条件：fix。
# 分支条件：config.get(field)。
# 分支条件：error。
# 分支条件：local_runtime。
# 分支条件：model in {'deepseek-chat', 'deepseek-reasoner'}。
# 分支条件：mutating。
# 分支条件：context.cancellation.is_set()。
# 分支条件：isinstance(item, PluginResult)。
# 分支条件：isinstance(item, str)。
# 分支条件：logs and item.startswith(logs[-1])。
# 分支条件：failure。
# 分支条件：current != original or not path.exists()。
# 遍历边界：value ← args.get('target_files', [])。
# 遍历边界：field ← ('sanitizer_report', 'ground_truth_file')。
# 遍历边界：relative ← targets。
# 遍历边界：item ← plugin.execute(execution)。
# 遍历边界：(relative, original) ← before.items()。
# 返回路径：ToolResult('success' if success else 'error', message, data=data, artifacts=[*snapshots, str(artifact)], changed_files=changed, error=None if success else {'type': 'PipelinePluginFailed', 'message': message})。
# 显式抛出：ValueError(f'target must be a file: {value}')。
# 显式抛出：ValueError('target_files must contain at least one existing file')。
# 显式抛出：ValueError("plugin targets exceed the run's target_files")。
# 显式抛出：ValueError(error)。
# 捕获异常：Exception。
# 写入目标：config['scan_scope']。
# 写入目标：config[field]。
# 写入目标：before[relative]。
# 写入目标：logs[-1]。
# 写入目标：data['diff']。
# 直接调用：CodeCompletionPlugin。
# 直接调用：VulnerabilityDetectionPlugin。
# 直接调用：args.get。
# 直接调用：resolve_workspace_path。
# 直接调用：path.relative_to(context.workspace).as_posix。
# 直接调用：path.relative_to。
# 直接调用：path.is_file。
# 直接调用：ValueError。
# 直接调用：targets.append。
# 直接调用：context.metadata.get。
# 直接调用：any。
# 直接调用：args.items。
# 直接调用：config.update。
# 直接调用：config.setdefault。
# 直接调用：config.get。
# 直接调用：report.relative_to(context.workspace).as_posix。
# 直接调用：report.relative_to。
# 直接调用：plugin.validate。
# 直接调用：_local_aider_runtime。
# 直接调用：ExecutionContext。
# 直接调用：str。
# 直接调用：(context.workspace / relative).read_text。
# 直接调用：snapshots.append。
# 直接调用：_write_snapshot_once。
# 直接调用：plugin.execute。
# 直接调用：context.cancellation.is_set。
# 直接调用：isinstance。
# 直接调用：item.startswith。
# 直接调用：logs.append。
# 直接调用：'\n'.join。
# 直接调用：before.items。
# 直接调用：path.read_text。
# 直接调用：path.exists。
# 直接调用：changed.append。
# 直接调用：diffs.append。
# 直接调用：_unified_diff。
# 直接调用：uuid4。
# 直接调用：artifact.write_text。
# 直接调用：redact_sensitive_text。
# 直接调用：json.dumps。
# 直接调用：ToolResult。
# 表达式分流：feature == 'code_completion'；真值与假值分支分别计算，不同时执行。
# 集合派生：value 来自 allowed。
# 集合派生：value 来自 targets。
# 集合派生：(key, value) 来自 args.items()；过滤 key not in {'instruction', 'target_files'}。
# 表达式分流：deep；真值与假值分支分别计算，不同时执行。
# 表达式分流：fix；真值与假值分支分别计算，不同时执行。
# 表达式分流：targets；真值与假值分支分别计算，不同时执行。
# 表达式分流：local_runtime；真值与假值分支分别计算，不同时执行。
# 表达式分流：base_url is not None；真值与假值分支分别计算，不同时执行。
# 表达式分流：final is not None；真值与假值分支分别计算，不同时执行。
# 表达式分流：path.is_file()；真值与假值分支分别计算，不同时执行。
# 表达式分流：final；真值与假值分支分别计算，不同时执行。
# 表达式分流：success；真值与假值分支分别计算，不同时执行。
# [/maint-note:f28c525d32bd]
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
        config.update(auto_fix=fix, analyzer=args.get("analyzer", "cppcheck") if deep else "auto" if fix else "builtin")
        config.setdefault("scan_scope", "targets" if targets else "project")
        if fix:
            config["scan_scope"] = "targets"
        for field in ("sanitizer_report", "ground_truth_file"):
            if config.get(field):
                report = resolve_workspace_path(context, config[field], must_exist=True)
                config[field] = report.relative_to(context.workspace).as_posix()
    error = plugin.validate(config)
    if error:
        raise ValueError(error)
    model = context.metadata.get("model", "deepseek/deepseek-chat")
    local_runtime = _local_aider_runtime(context.metadata)
    if local_runtime:
        model = local_runtime[0]
    base_url = local_runtime[1] if local_runtime else None
    if model in {"deepseek-chat", "deepseek-reasoner"}:
        model = f"deepseek/{model}"
    execution = ExecutionContext(
        project_dir=str(context.workspace), target_files=targets,
        instruction=args.get("instruction", ""),
        model=model,
        api_key=None if base_url is not None else context.metadata.get("api_key"),
        feature_config=config, base_url=base_url,
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


# [maint-note:713be7f4094e]
# 维护导航：pipeline_tool_specs；下列为本函数源码中的接口与路径索引。
# 输入 include_mutating：类型由调用方与函数体约束；缺省表达式 True。
# 分支条件：include_mutating。
# 返回路径：{'type': 'object', 'properties': properties, 'required': list(required), 'additionalProperties': False}。
# 返回路径：specs。
# 直接调用：list。
# 直接调用：ToolSpec。
# 直接调用：schema。
# 直接调用：specs.extend。
# [/maint-note:713be7f4094e]
def pipeline_tool_specs(include_mutating=True):
    targets = {"type": "array", "items": {"type": "string"}, "minItems": 1}
    scan = {
        "target_files": targets,
        "scan_scope": {"type": "string", "enum": ["targets", "project"]},
        "scan_type": {"type": "string", "enum": ["frequent_defects", "high_risk"]},
        "severity_threshold": {"type": "string", "enum": ["low", "medium", "high", "critical"]},
        "rule_profile": {"type": "string", "enum": ["default", "c_cpp", "web"]},
        "max_findings": {"type": "integer", "minimum": 1, "maximum": 1000},
        "incremental": {"type": "boolean"},
        "sanitizer_report": {"type": "string"},
        "ground_truth_file": {"type": "string", "description": "Workspace-relative labeled test manifest; requires scan_type. Detection never reads labels."},
    }
    # [maint-note:a117cdb34142]
    # 维护导航：schema；下列为本函数源码中的接口与路径索引。
    # 输入 properties：类型由调用方与函数体约束。
    # 输入 required：类型由调用方与函数体约束；缺省表达式 ()。
    # 返回路径：{'type': 'object', 'properties': properties, 'required': list(required), 'additionalProperties': False}。
    # 直接调用：list。
    # [/maint-note:a117cdb34142]
    def schema(properties, required=()):
        return {"type": "object", "properties": properties, "required": list(required), "additionalProperties": False}
    specs = [ToolSpec(
        "vulnerability_detection", "Start a vulnerability audit with this read-only Pipeline scan. Returns candidate findings, coverage limits, and a C/C++ semantic review checklist. Read source and apply the checklist before your final verdict; patterns are not proven vulnerabilities. For deeper bounds/null/leak evidence use vulnerability_detection.analyze with approval.",
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
                schema({**scan, "analyzer": {"type": "string", "enum": ["cppcheck", "comprehensive"]}}), RiskLevel.EXECUTE, lambda c, a: _execute(c, a, "vulnerability_detection", deep=True), parallel_safe=False, default_timeout_seconds=300),
            ToolSpec("vulnerability_detection.fix", "Scan selected files (including Cppcheck if available), then edit them with Pipeline Aider remediation. Requires execute approval for analysis and edits. Findings are pre-repair; re-scan and test afterwards.",
                schema({**scan, "instruction": {"type": "string"}, "extra_instruction": {"type": "string"}}, ("target_files",)),
                RiskLevel.EXECUTE, lambda c, a: _execute(c, a, "vulnerability_detection", fix=True), idempotent=False, parallel_safe=False, default_timeout_seconds=900),
        ])
    return specs
