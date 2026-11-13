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


# [maint-note:62d9033c0f5c]
# 数据与行为导航：CodeSummaryPlugin。
# 继承接口：FeaturePlugin。
# [/maint-note:62d9033c0f5c]
@register_plugin
class CodeSummaryPlugin(FeaturePlugin):
    # [maint-note:c034157b7895]
    # 维护导航：metadata；下列为本函数源码中的接口与路径索引。
    # 返回契约：FeatureMetadata。
    # 返回路径：FeatureMetadata(name='code_summary', label='Code Summarization', description='Use NaturalCC semantic context and Aider dry-run mode to produce a deeper code summarization', execution_mode=ExecutionMode.AIDER)。
    # 直接调用：FeatureMetadata。
    # [/maint-note:c034157b7895]
    @property
    def metadata(self) -> FeatureMetadata:
        return FeatureMetadata(
            name="code_summary",
            label="Code Summarization",
            description="Use NaturalCC semantic context and Aider dry-run mode to produce a deeper code summarization",
            execution_mode=ExecutionMode.AIDER,
        )

    # [maint-note:cc8dedbb4667]
    # 维护导航：config_schema；下列为本函数源码中的接口与路径索引。
    # 返回契约：List[ConfigField]。
    # 返回路径：[ConfigField(name='summary_scope', label='Summarization Scope / 总结范围', type=ConfigFieldType.SELECT, required=False, default='targets', options=[{'value': 'targets', 'label': 'Selected target files / 仅选中文件'}, {'value': 'project', 'label': 'W。
    # 直接调用：ConfigField。
    # [/maint-note:cc8dedbb4667]
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

    # [maint-note:a9157067b36c]
    # 维护导航：validate；下列为本函数源码中的接口与路径索引。
    # 输入 config：Dict[str, Any]。
    # 输入 files：类型由调用方与函数体约束；缺省表达式 None。
    # 返回契约：Optional[str]。
    # 分支条件：str(config.get('summary_scope', 'targets')).strip() not in {'targets', 'project'}。
    # 分支条件：str(config.get('detail_level', 'standard')).strip() not in {'brief', 'standard', 'detailed'}。
    # 分支条件：not max_files_raw.isdigit() or int(max_files_raw) <= 0。
    # 返回路径：'summary_scope must be one of: targets, project'。
    # 返回路径：'detail_level must be one of: brief, standard, detailed'。
    # 返回路径：'max_files must be a positive integer'。
    # 返回路径：None。
    # 直接调用：str(config.get('summary_scope', 'targets')).strip。
    # 直接调用：str。
    # 直接调用：config.get。
    # 直接调用：str(config.get('detail_level', 'standard')).strip。
    # 直接调用：str(config.get('max_files', '20')).strip。
    # 直接调用：max_files_raw.isdigit。
    # 直接调用：int。
    # [/maint-note:a9157067b36c]
    def validate(self, config: Dict[str, Any], files=None) -> Optional[str]:
        if str(config.get("summary_scope", "targets")).strip() not in {"targets", "project"}:
            return "summary_scope must be one of: targets, project"
        if str(config.get("detail_level", "standard")).strip() not in {"brief", "standard", "detailed"}:
            return "detail_level must be one of: brief, standard, detailed"
        max_files_raw = str(config.get("max_files", "20")).strip()
        if not max_files_raw.isdigit() or int(max_files_raw) <= 0:
            return "max_files must be a positive integer"
        return None

    # [maint-note:5a08351a382d]
    # 维护导航：preview；下列为本函数源码中的接口与路径索引。
    # 输入 context：ExecutionContext。
    # 返回契约：str。
    # 分支条件：not files。
    # 返回路径：'❌ Code Summarization requires at least one readable source file.'。
    # 返回路径：preview_prompt(target_files=files, user_instruction=self._build_summary_instruction(context, files), model=context.model, api_key=context.api_key, project_dir=context.project_dir, dry_run=True)。
    # 直接调用：self._collect_files。
    # 直接调用：preview_prompt。
    # 直接调用：self._build_summary_instruction。
    # [/maint-note:5a08351a382d]
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

    # [maint-note:7b8f6f29141b]
    # 维护导航：execute；下列为本函数源码中的接口与路径索引。
    # 输入 context：ExecutionContext。
    # 返回契约：Generator[str, None, None]。
    # 分支条件：not files。
    # 遍历边界：log ← run_aider_stream(target_files=files, user_instruction=self._build_summary_instruction(context, files), model=context.model, api_key=context.api_key, base_url=context.base_url, project_dir=context.project_dir, dry_run=True)。
    # 返回路径：无返回值。
    # 直接调用：self._collect_files。
    # 直接调用：run_aider_stream。
    # 直接调用：self._build_summary_instruction。
    # 生成器交付：'⚠️ [错误]: Code Summarization requires at least one readable source file.\n'。
    # 生成器交付：log。
    # [/maint-note:7b8f6f29141b]
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
            base_url=context.base_url,
            project_dir=context.project_dir,
            dry_run=True,
        ):
            yield log

    # [maint-note:75e7541b7508]
    # 维护导航：_collect_files；下列为本函数源码中的接口与路径索引。
    # 输入 context：ExecutionContext。
    # 返回契约：List[str]。
    # 分支条件：summary_scope == 'targets'。
    # 分支条件：full_path.is_file()。
    # 分支条件：path.suffix.lower() in SOURCE_EXTENSIONS and path.is_file()。
    # 分支条件：len(files) >= max_files。
    # 遍历边界：item ← context.target_files。
    # 遍历边界：(root, dirs, names) ← os.walk(project_root)。
    # 遍历边界：name ← sorted(names, key=str.casefold)。
    # 返回路径：files[:max_files]。
    # 返回路径：files。
    # 写入目标：dirs[:]。
    # 直接调用：str(config.get('summary_scope', 'targets')).strip。
    # 直接调用：str。
    # 直接调用：config.get。
    # 直接调用：self._max_files。
    # 直接调用：Path(context.project_dir).expanduser().resolve。
    # 直接调用：Path(context.project_dir).expanduser。
    # 直接调用：Path。
    # 直接调用：Path(item).expanduser。
    # 直接调用：path.is_absolute。
    # 直接调用：path.resolve。
    # 直接调用：(project_root / item).resolve。
    # 直接调用：full_path.is_file。
    # 直接调用：files.append。
    # 直接调用：os.walk。
    # 直接调用：sorted。
    # 直接调用：d.startswith。
    # 直接调用：path.suffix.lower。
    # 直接调用：path.is_file。
    # 直接调用：len。
    # 表达式分流：path.is_absolute()；真值与假值分支分别计算，不同时执行。
    # 集合派生：d 来自 dirs；过滤 d not in IGNORE_DIRS and (not d.startswith('.'))。
    # [/maint-note:75e7541b7508]
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

    # [maint-note:6cf14797bacd]
    # 维护导航：_build_summary_instruction；下列为本函数源码中的接口与路径索引。
    # 输入 context：ExecutionContext。
    # 输入 files：List[str]。
    # 返回契约：str。
    # 分支条件：include_symbols。
    # 分支条件：focus。
    # 返回路径：'\n'.join(parts)。
    # 直接调用：str(config.get('detail_level', 'standard')).strip。
    # 直接调用：str。
    # 直接调用：config.get。
    # 直接调用：self._bool_config。
    # 直接调用：context.instruction.strip。
    # 直接调用：{'brief': '输出简洁摘要，突出用途、入口和主要风险。', 'standard': '输出标准代码总结，覆盖职责、关键流程、重要符号和依赖关系。', 'detailed': '输出较详细的代码总结，包含模块职责、调用/数据流、关键符号、边界条件和潜在问题。'}.get。
    # 直接调用：self._relative_file。
    # 直接调用：parts.append。
    # 直接调用：parts.extend。
    # 直接调用：'\n'.join。
    # 集合派生：file_path 来自 files。
    # 集合派生：item 来自 relative_files。
    # [/maint-note:6cf14797bacd]
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

    # [maint-note:fc87159c960f]
    # 维护导航：_relative_file；下列为本函数源码中的接口与路径索引。
    # 输入 project_dir：str。
    # 输入 file_path：str。
    # 返回契约：str。
    # 返回路径：str(Path(file_path).resolve().relative_to(Path(project_dir).resolve())).replace('\\', '/')。
    # 返回路径：file_path。
    # 捕获异常：ValueError。
    # 直接调用：str(Path(file_path).resolve().relative_to(Path(project_dir).resolve())).replace。
    # 直接调用：str。
    # 直接调用：Path(file_path).resolve().relative_to。
    # 直接调用：Path(file_path).resolve。
    # 直接调用：Path。
    # 直接调用：Path(project_dir).resolve。
    # [/maint-note:fc87159c960f]
    def _relative_file(self, project_dir: str, file_path: str) -> str:
        try:
            return str(Path(file_path).resolve().relative_to(Path(project_dir).resolve())).replace("\\", "/")
        except ValueError:
            return file_path

    # [maint-note:e5169ef532b0]
    # 维护导航：_bool_config；下列为本函数源码中的接口与路径索引。
    # 输入 value：Any。
    # 返回契约：bool。
    # 分支条件：isinstance(value, str)。
    # 返回路径：value.lower() in {'1', 'true', 'yes', 'on'}。
    # 返回路径：bool(value)。
    # 直接调用：isinstance。
    # 直接调用：value.lower。
    # 直接调用：bool。
    # [/maint-note:e5169ef532b0]
    def _bool_config(self, value: Any) -> bool:
        if isinstance(value, str):
            return value.lower() in {"1", "true", "yes", "on"}
        return bool(value)

    # [maint-note:c32f5f1ec411]
    # 维护导航：_max_files；下列为本函数源码中的接口与路径索引。
    # 输入 config：Dict[str, Any]。
    # 返回契约：int。
    # 分支条件：raw.isdigit() and int(raw) > 0。
    # 返回路径：int(raw)。
    # 返回路径：20。
    # 直接调用：str(config.get('max_files', '20')).strip。
    # 直接调用：str。
    # 直接调用：config.get。
    # 直接调用：raw.isdigit。
    # 直接调用：int。
    # [/maint-note:c32f5f1ec411]
    def _max_files(self, config: Dict[str, Any]) -> int:
        raw = str(config.get("max_files", "20")).strip()
        if raw.isdigit() and int(raw) > 0:
            return int(raw)
        return 20
