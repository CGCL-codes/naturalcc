import json
from typing import Any, Dict, Generator

from .base import ExecutionMode, ExecutionContext, PluginResult
from .registry import registry


def ndjson_event(payload: Dict[str, Any]) -> str:
    return json.dumps(payload, ensure_ascii=False) + "\n"


class ExecutionDispatcher:
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

    def _execute_aider(self, plugin, context):
        yield ndjson_event({
            "type": "start",
            "status": "running",
            "log": "Preparing execution...\n",
            "mode": "aider",
        })

        for log in plugin.execute(context):
            yield ndjson_event({
                "type": "log",
                "status": self._infer_status(log),
                "log": log,
                "mode": "aider",
            })

        yield ndjson_event({
            "type": "done",
            "status": "success",
            "log": log,
            "mode": "aider",
        })

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

    def _infer_status(self, log_text: str) -> str:
        if "✅ [NaturalCC Agent]" in log_text or "任务圆满完成" in log_text:
            return "success"
        if "❌" in log_text or "[错误]" in log_text or "失败" in log_text:
            return "error"
        if log_text.strip():
            return "running"
        return "idle"


dispatcher = ExecutionDispatcher()
