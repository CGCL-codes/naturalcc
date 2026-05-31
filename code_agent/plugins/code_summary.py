import os
from pathlib import Path
from typing import Any, Dict, Generator, List, Optional

from code_agent.aider_runner import preview_prompt, run_aider_stream
from code_agent.plugins.base import (
    ConfigField,
    ConfigFieldType,
    ExecutionContext,
    ExecutionMode,
    FeatureMetadata,
    FeaturePlugin,
)
from code_agent.plugins.registry import register_plugin


SOURCE_EXTENSIONS = {
    ".c",
    ".cc",
    ".cpp",
    ".cxx",
    ".h",
    ".hpp",
    ".java",
    ".py",
    ".js",
    ".jsx",
    ".ts",
    ".tsx",
    ".go",
    ".rs",
    ".php",
}

IGNORE_DIRS = {
    ".git",
    ".idea",
    ".vscode",
    ".venv",
    "__pycache__",
    "build",
    "dist",
    "node_modules",
    "out",
    "venv",
}


@register_plugin
class CodeSummaryPlugin(FeaturePlugin):
    @property
    def metadata(self) -> FeatureMetadata:
        return FeatureMetadata(
            name="code_summary",
            label="Code Summarization",
            description="Use NaturalCC semantic context and Aider dry-run mode to produce a deeper code summarization",
            execution_mode=ExecutionMode.AIDER,
        )

    @property
    def config_schema(self) -> List[ConfigField]:
        return [
            ConfigField(
                name="summary_scope",
                label="Summarization Scope / 总结范围",
                type=ConfigFieldType.SELECT,
                required=False,
                default="targets",
                options=[
                    {"value": "targets", "label": "Selected target files / 仅选中文件"},
                    {"value": "project", "label": "Whole project (source files) / 全项目源码"},
                ],
                help_text="Selected files are more focused; project mode sends source files through the NaturalCC+Aider summarization path. / 选中文件更聚焦；全项目会通过 NaturalCC+Aider 总结路径处理源码文件。",
            ),
            ConfigField(
                name="detail_level",
                label="Detail Level / 详细程度",
                type=ConfigFieldType.SELECT,
                required=False,
                default="standard",
                options=[
                    {"value": "brief", "label": "brief / 简要"},
                    {"value": "standard", "label": "standard / 标准"},
                    {"value": "detailed", "label": "detailed / 详细"},
                ],
                help_text="Controls how much explanation Aider should include. / 控制 Aider 输出的解释细节。",
            ),
            ConfigField(
                name="include_symbols",
                label="Include Symbols / 包含符号",
                type=ConfigFieldType.SWITCH,
                required=False,
                default=True,
                help_text="Ask Aider to include important functions, types, and data flow. / 要求 Aider 总结关键函数、类型和数据流。",
            ),
            ConfigField(
                name="max_files",
                label="Max Files / 最大文件数",
                type=ConfigFieldType.TEXT,
                required=False,
                default="20",
                placeholder="20",
                help_text="Upper bound of files sent to NaturalCC and Aider. / 发送给 NaturalCC 和 Aider 的文件数量上限。",
            ),
        ]

    def validate(self, config: Dict[str, Any], files=None) -> Optional[str]:
        if str(config.get("summary_scope", "targets")).strip() not in {"targets", "project"}:
            return "summary_scope must be one of: targets, project"
        if str(config.get("detail_level", "standard")).strip() not in {"brief", "standard", "detailed"}:
            return "detail_level must be one of: brief, standard, detailed"
        max_files_raw = str(config.get("max_files", "20")).strip()
        if not max_files_raw.isdigit() or int(max_files_raw) <= 0:
            return "max_files must be a positive integer"
        return None

    def preview(self, context: ExecutionContext) -> str:
        files = self._collect_files(context)
        if not files:
            return "❌ Code Summarization requires at least one readable source file."
        return preview_prompt(
            target_files=files,
            user_instruction=self._build_summary_instruction(context, files),
            model=context.model,
            api_key=context.api_key,
            project_dir=context.project_dir,
            dry_run=True,
        )

    def execute(self, context: ExecutionContext) -> Generator[str, None, None]:
        files = self._collect_files(context)
        if not files:
            yield "⚠️ [错误]: Code Summarization requires at least one readable source file.\n"
            return

        for log in run_aider_stream(
            target_files=files,
            user_instruction=self._build_summary_instruction(context, files),
            model=context.model,
            api_key=context.api_key,
            project_dir=context.project_dir,
            dry_run=True,
        ):
            yield log

    def _collect_files(self, context: ExecutionContext) -> List[str]:
        config = context.feature_config or {}
        summary_scope = str(config.get("summary_scope", "targets")).strip()
        max_files = self._max_files(config)
        project_root = Path(context.project_dir).expanduser().resolve()

        if summary_scope == "targets":
            files = []
            for item in context.target_files:
                path = Path(item).expanduser()
                full_path = path.resolve() if path.is_absolute() else (project_root / item).resolve()
                if full_path.is_file():
                    files.append(str(full_path))
            return files[:max_files]

        files = []
        for root, dirs, names in os.walk(project_root):
            dirs[:] = sorted(
                [d for d in dirs if d not in IGNORE_DIRS and not d.startswith(".")],
                key=str.casefold,
            )
            for name in sorted(names, key=str.casefold):
                path = Path(root) / name
                if path.suffix.lower() in SOURCE_EXTENSIONS and path.is_file():
                    files.append(str(path.resolve()))
                    if len(files) >= max_files:
                        return files
        return files

    def _build_summary_instruction(self, context: ExecutionContext, files: List[str]) -> str:
        config = context.feature_config or {}
        detail_level = str(config.get("detail_level", "standard")).strip()
        include_symbols = self._bool_config(config.get("include_symbols", True))
        focus = context.instruction.strip()

        detail_guidance = {
            "brief": "输出简洁摘要，突出用途、入口和主要风险。",
            "standard": "输出标准代码总结，覆盖职责、关键流程、重要符号和依赖关系。",
            "detailed": "输出较详细的代码总结，包含模块职责、调用/数据流、关键符号、边界条件和潜在问题。",
        }.get(detail_level, "输出标准代码总结。")

        relative_files = [self._relative_file(context.project_dir, file_path) for file_path in files]
        parts = [
            "请对 dry-run 上下文中的代码做代码总结，不要修改任何文件。",
            "",
            "【总结要求】",
            detail_guidance,
            "用中文输出，结构清晰，避免泛泛而谈，结论必须基于实际代码。",
            "说明每个文件的职责，以及这些文件之间如何协作。",
        ]
        if include_symbols:
            parts.append("列出关键函数、类型、全局变量或重要成员，并说明它们的作用。")
        if focus:
            parts.extend(["", "【用户关注点】", focus])
        parts.extend(["", "【目标文件】", "\n".join(f"- {item}" for item in relative_files)])
        return "\n".join(parts)

    def _relative_file(self, project_dir: str, file_path: str) -> str:
        try:
            return str(Path(file_path).resolve().relative_to(Path(project_dir).resolve())).replace("\\", "/")
        except ValueError:
            return file_path

    def _bool_config(self, value: Any) -> bool:
        if isinstance(value, str):
            return value.lower() in {"1", "true", "yes", "on"}
        return bool(value)

    def _max_files(self, config: Dict[str, Any]) -> int:
        raw = str(config.get("max_files", "20")).strip()
        if raw.isdigit() and int(raw) > 0:
            return int(raw)
        return 20
