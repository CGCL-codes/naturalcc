import json
from typing import Any, Dict, Generator

from .base import ExecutionMode, ExecutionContext, PluginResult
from .registry import registry


# [maint-note:f671af2ebd87]
# 维护导航：ndjson_event；下列为本函数源码中的接口与路径索引。
# 输入 payload：Dict[str, Any]。
# 返回契约：str。
# 返回路径：json.dumps(payload, ensure_ascii=False) + '\n'。
# 直接调用：json.dumps。
# [/maint-note:f671af2ebd87]
def ndjson_event(payload: Dict[str, Any]) -> str:
    return json.dumps(payload, ensure_ascii=False) + "\n"


class ExecutionDispatcher:
    # [maint-note:dd3298ce2812]
    # 维护导航：dispatch；下列为本函数源码中的接口与路径索引。
    # 输入 context：ExecutionContext。
    # 返回契约：Generator[str, None, None]。
    # 分支条件：plugin is None。
    # 分支条件：error。
    # 分支条件：mode == ExecutionMode.AIDER。
    # 分支条件：mode == ExecutionMode.DIRECT。
    # 分支条件：mode == ExecutionMode.HYBRID。
    # 返回路径：无返回值。
    # 直接调用：context.feature_config.get。
    # 直接调用：registry.get。
    # 直接调用：ndjson_event。
    # 直接调用：context.feature_config.items。
    # 直接调用：plugin.validate。
    # 直接调用：self._execute_aider。
    # 直接调用：self._execute_direct。
    # 直接调用：self._execute_hybrid。
    # 生成器交付：ndjson_event({'type': 'error', 'status': 'error', 'log': f'Unknown feature: {feature_name}'})。
    # 集合派生：(k, v) 来自 context.feature_config.items()；过滤 k != 'feature'。
    # 生成器交付：ndjson_event({'type': 'error', 'status': 'error', 'log': f'Validation error: {error}'})。
    # 生成器交付：self._execute_aider(plugin, context)。
    # 生成器交付：self._execute_direct(plugin, context)。
    # 生成器交付：self._execute_hybrid(plugin, context)。
    # [/maint-note:dd3298ce2812]
    def dispatch(
        self,
        context: ExecutionContext,
    ) -> Generator[str, None, None]:
        feature_name = context.feature_config.get("feature", "code_completion")
        plugin = registry.get(feature_name)
        if plugin is None:
            yield ndjson_event({
                "type": "error",
                "status": "error",
                "log": f"Unknown feature: {feature_name}",
            })
            return

        pure_config = {k: v for k, v in context.feature_config.items() if k != "feature"}
        error = plugin.validate(pure_config, context.uploaded_files)
        if error:
            yield ndjson_event({
                "type": "error",
                "status": "error",
                "log": f"Validation error: {error}",
            })
            return

        mode = plugin.metadata.execution_mode

        if mode == ExecutionMode.AIDER:
            yield from self._execute_aider(plugin, context)
        elif mode == ExecutionMode.DIRECT:
            yield from self._execute_direct(plugin, context)
        elif mode == ExecutionMode.HYBRID:
            yield from self._execute_hybrid(plugin, context)

    # [maint-note:86248eae21ec]
    # 维护导航：_execute_aider；下列为本函数源码中的接口与路径索引。
    # 输入 plugin：类型由调用方与函数体约束。
    # 输入 context：类型由调用方与函数体约束。
    # 分支条件：final_status in {'idle', 'running'}。
    # 遍历边界：log ← plugin.execute(context)。
    # 直接调用：ndjson_event。
    # 直接调用：plugin.execute。
    # 直接调用：self._infer_status。
    # 生成器交付：ndjson_event({'type': 'start', 'status': 'running', 'log': 'Preparing execution...\n', 'mode': 'aider'})。
    # 生成器交付：ndjson_event({'type': 'log', 'status': self._infer_status(log), 'log': log, 'mode': 'aider'})。
    # 生成器交付：ndjson_event({'type': 'done', 'status': final_status, 'log': log or 'Execution completed.', 'mode': 'aider'})。
    # [/maint-note:86248eae21ec]
    def _execute_aider(self, plugin, context):
        yield ndjson_event({
            "type": "start",
            "status": "running",
            "log": "Preparing execution...\n",
            "mode": "aider",
        })

        log = ""
        for log in plugin.execute(context):
            yield ndjson_event({
                "type": "log",
                "status": self._infer_status(log),
                "log": log,
                "mode": "aider",
            })

        final_status = self._infer_status(log)
        if final_status in {"idle", "running"}:
            final_status = "success"

        yield ndjson_event({
            "type": "done",
            "status": final_status,
            "log": log or "Execution completed.",
            "mode": "aider",
        })

    # [maint-note:c915ccc64350]
    # 维护导航：_execute_direct；下列为本函数源码中的接口与路径索引。
    # 输入 plugin：类型由调用方与函数体约束。
    # 输入 context：类型由调用方与函数体约束。
    # 分支条件：isinstance(item, PluginResult)。
    # 分支条件：result。
    # 遍历边界：item ← plugin.execute(context)。
    # 直接调用：ndjson_event。
    # 直接调用：plugin.execute。
    # 直接调用：isinstance。
    # 直接调用：str。
    # 生成器交付：ndjson_event({'type': 'start', 'status': 'running', 'log': 'Preparing execution...\n', 'mode': 'direct'})。
    # 生成器交付：ndjson_event({'type': 'log', 'status': 'running', 'log': str(item), 'mode': 'direct'})。
    # 生成器交付：ndjson_event({'type': 'done', 'status': 'success' if result.success else 'error', 'log': result.log or result.message, 'report': result.report, 'files_modified': result.files_modified, 'artifacts': result.artifacts, 'mode': 'direct'})。
    # 表达式分流：result.success；真值与假值分支分别计算，不同时执行。
    # 生成器交付：ndjson_event({'type': 'done', 'status': 'success', 'log': 'Execution completed.', 'mode': 'direct'})。
    # [/maint-note:c915ccc64350]
    def _execute_direct(self, plugin, context):
        yield ndjson_event({
            "type": "start",
            "status": "running",
            "log": "Preparing execution...\n",
            "mode": "direct",
        })

        result: PluginResult = None
        for item in plugin.execute(context):
            if isinstance(item, PluginResult):
                result = item
            else:
                yield ndjson_event({
                    "type": "log",
                    "status": "running",
                    "log": str(item),
                    "mode": "direct",
                })

        if result:
            yield ndjson_event({
                "type": "done",
                "status": "success" if result.success else "error",
                "log": result.log or result.message,
                "report": result.report,
                "files_modified": result.files_modified,
                "artifacts": result.artifacts,
                "mode": "direct",
            })
        else:
            yield ndjson_event({
                "type": "done",
                "status": "success",
                "log": "Execution completed.",
                "mode": "direct",
            })

    # [maint-note:f2eeb03f1aba]
    # 维护导航：_execute_hybrid；下列为本函数源码中的接口与路径索引。
    # 输入 plugin：类型由调用方与函数体约束。
    # 输入 context：类型由调用方与函数体约束。
    # 分支条件：isinstance(item, dict) and item.get('phase') == 'analysis_done'。
    # 分支条件：isinstance(item, PluginResult)。
    # 遍历边界：item ← plugin.execute(context)。
    # 直接调用：ndjson_event。
    # 直接调用：plugin.execute。
    # 直接调用：isinstance。
    # 直接调用：item.get。
    # 直接调用：str。
    # 生成器交付：ndjson_event({'type': 'start', 'status': 'running', 'log': 'Phase 1: Analysis...\n', 'mode': 'hybrid'})。
    # 生成器交付：ndjson_event({'type': 'log', 'status': 'running', 'log': 'Analysis complete. Starting repair...\n', 'report': item.get('report'), 'mode': 'hybrid'})。
    # 生成器交付：ndjson_event({'type': 'done', 'status': 'success' if item.success else 'error', 'log': item.log or item.message, 'report': item.report, 'files_modified': item.files_modified, 'artifacts': item.artifacts, 'mode': 'hybrid'})。
    # 表达式分流：item.success；真值与假值分支分别计算，不同时执行。
    # 生成器交付：ndjson_event({'type': 'log', 'status': 'running', 'log': str(item), 'mode': 'hybrid'})。
    # [/maint-note:f2eeb03f1aba]
    def _execute_hybrid(self, plugin, context):
        yield ndjson_event({
            "type": "start",
            "status": "running",
            "log": "Phase 1: Analysis...\n",
            "mode": "hybrid",
        })

        for item in plugin.execute(context):
            if isinstance(item, dict) and item.get("phase") == "analysis_done":
                yield ndjson_event({
                    "type": "log",
                    "status": "running",
                    "log": "Analysis complete. Starting repair...\n",
                    "report": item.get("report"),
                    "mode": "hybrid",
                })
            elif isinstance(item, PluginResult):
                yield ndjson_event({
                    "type": "done",
                    "status": "success" if item.success else "error",
                    "log": item.log or item.message,
                    "report": item.report,
                    "files_modified": item.files_modified,
                    "artifacts": item.artifacts,
                    "mode": "hybrid",
                })
            else:
                yield ndjson_event({
                    "type": "log",
                    "status": "running",
                    "log": str(item),
                    "mode": "hybrid",
                })

    # [maint-note:9c188acfe17c]
    # 维护导航：_infer_status；下列为本函数源码中的接口与路径索引。
    # 输入 log_text：str。
    # 返回契约：str。
    # 分支条件：'✅ [NaturalCC Agent]' in log_text or '任务圆满完成' in log_text。
    # 分支条件：'❌' in log_text or '[错误]' in log_text or '失败' in log_text。
    # 分支条件：log_text.strip()。
    # 返回路径：'success'。
    # 返回路径：'error'。
    # 返回路径：'running'。
    # 返回路径：'idle'。
    # 直接调用：log_text.strip。
    # [/maint-note:9c188acfe17c]
    def _infer_status(self, log_text: str) -> str:
        if "✅ [NaturalCC Agent]" in log_text or "任务圆满完成" in log_text:
            return "success"
        if "❌" in log_text or "[错误]" in log_text or "失败" in log_text:
            return "error"
        if log_text.strip():
            return "running"
        return "idle"


dispatcher = ExecutionDispatcher()
