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


REPAIR_TYPES = {
    "bug_fix": "修复运行时 bug 或逻辑错误",
    "compile_error": "修复编译、类型检查或 lint 错误",
    "test_failure": "修复测试失败",
    "safe_refactor": "在保持行为不变的前提下做小范围修复性重构",
}


# [maint-note:a5e9c307353b]
# 数据与行为导航：CodeRepairPlugin。
# 继承接口：FeaturePlugin。
# [/maint-note:a5e9c307353b]
@register_plugin
class CodeRepairPlugin(FeaturePlugin):
    # [maint-note:d941cca23e72]
    # 维护导航：metadata；下列为本函数源码中的接口与路径索引。
    # 返回契约：FeatureMetadata。
    # 返回路径：FeatureMetadata(name='code_repair', label='Code Repair', description='Build a focused repair prompt and delegate minimal code fixes to Aider', execution_mode=ExecutionMode.AIDER)。
    # 直接调用：FeatureMetadata。
    # [/maint-note:d941cca23e72]
    @property
    def metadata(self) -> FeatureMetadata:
        return FeatureMetadata(
            name="code_repair",
            label="Code Repair",
            description="Build a focused repair prompt and delegate minimal code fixes to Aider",
            execution_mode=ExecutionMode.AIDER,
        )

    # [maint-note:c47df86eba2b]
    # 维护导航：config_schema；下列为本函数源码中的接口与路径索引。
    # 返回契约：List[ConfigField]。
    # 返回路径：[ConfigField(name='repair_type', label='Repair Type / 修复类型', type=ConfigFieldType.SELECT, required=False, default='bug_fix', options=[{'value': 'bug_fix', 'label': 'bug_fix / 逻辑或运行时错误'}, {'value': 'compile_error', 'label': 'compile_error / 。
    # 直接调用：ConfigField。
    # [/maint-note:c47df86eba2b]
    @property
    def config_schema(self) -> List[ConfigField]:
        return [
            ConfigField(
                name="repair_type",
                label="Repair Type / 修复类型",
                type=ConfigFieldType.SELECT,
                required=False,
                default="bug_fix",
                options=[
                    {"value": "bug_fix", "label": "bug_fix / 逻辑或运行时错误"},
                    {"value": "compile_error", "label": "compile_error / 编译或类型错误"},
                    {"value": "test_failure", "label": "test_failure / 测试失败"},
                    {"value": "safe_refactor", "label": "safe_refactor / 安全重构修复"},
                ],
                help_text="Choose the main repair intent. / 选择本次修复的主要目标。",
            ),
            ConfigField(
                name="failure_log",
                label="Failure Log / 错误日志",
                type=ConfigFieldType.TEXTAREA,
                required=False,
                default="",
                placeholder="Paste compiler, test, stack trace, or runtime error output",
                help_text="Optional concrete failure evidence. / 可粘贴编译错误、测试失败、堆栈或运行时报错。",
            ),
            ConfigField(
                name="extra_context",
                label="Extra Context / 额外上下文",
                type=ConfigFieldType.TEXTAREA,
                required=False,
                default="",
                placeholder="Any constraints, expected behavior, or reproduction steps",
                help_text="Additional constraints or expected behavior. / 补充约束、期望行为或复现步骤。",
            ),
            ConfigField(
                name="allow_refactor",
                label="Allow Refactor / 允许重构",
                type=ConfigFieldType.SWITCH,
                required=False,
                default=False,
                help_text="Allow small supporting refactors only when they reduce repair risk. / 仅在降低修复风险时允许小范围辅助重构。",
            ),
        ]

    # [maint-note:c237ee07fc39]
    # 维护导航：validate；下列为本函数源码中的接口与路径索引。
    # 输入 config：Dict[str, Any]。
    # 输入 files：类型由调用方与函数体约束；缺省表达式 None。
    # 返回契约：Optional[str]。
    # 分支条件：repair_type not in REPAIR_TYPES。
    # 返回路径：'repair_type must be one of: bug_fix, compile_error, test_failure, safe_refactor'。
    # 返回路径：None。
    # 直接调用：str(config.get('repair_type', 'bug_fix')).strip。
    # 直接调用：str。
    # 直接调用：config.get。
    # [/maint-note:c237ee07fc39]
    def validate(self, config: Dict[str, Any], files=None) -> Optional[str]:
        repair_type = str(config.get("repair_type", "bug_fix")).strip()
        if repair_type not in REPAIR_TYPES:
            return "repair_type must be one of: bug_fix, compile_error, test_failure, safe_refactor"
        return None

    # [maint-note:911c596dac2f]
    # 维护导航：preview；下列为本函数源码中的接口与路径索引。
    # 输入 context：ExecutionContext。
    # 返回契约：str。
    # 分支条件：not context.target_files。
    # 返回路径：'❌ Code Repair requires at least one target file.'。
    # 返回路径：preview_prompt(target_files=context.target_files, user_instruction=self._build_repair_instruction(context), model=context.model, api_key=context.api_key, project_dir=context.project_dir, symbol=None, completion_type=None, prefix='')。
    # 直接调用：preview_prompt。
    # 直接调用：self._build_repair_instruction。
    # [/maint-note:911c596dac2f]
    def preview(self, context: ExecutionContext) -> str:
        if not context.target_files:
            return "❌ Code Repair requires at least one target file."

        return preview_prompt(
            target_files=context.target_files,
            user_instruction=self._build_repair_instruction(context),
            model=context.model,
            api_key=context.api_key,
            project_dir=context.project_dir,
            symbol=None,
            completion_type=None,
            prefix="",
        )

    # [maint-note:66986fad3ec8]
    # 维护导航：execute；下列为本函数源码中的接口与路径索引。
    # 输入 context：ExecutionContext。
    # 返回契约：Generator[str, None, None]。
    # 分支条件：not context.target_files。
    # 遍历边界：log ← run_aider_stream(target_files=context.target_files, user_instruction=repair_instruction, model=context.model, api_key=context.api_key, base_url=context.base_url, project_dir=context.project_dir, symbol=None, completion_type=None, prefix='')。
    # 返回路径：无返回值。
    # 直接调用：self._build_repair_instruction。
    # 直接调用：run_aider_stream。
    # 生成器交付：'⚠️ [错误]: Code Repair requires at least one target file.\n'。
    # 生成器交付：log。
    # [/maint-note:66986fad3ec8]
    def execute(self, context: ExecutionContext) -> Generator[str, None, None]:
        if not context.target_files:
            yield "⚠️ [错误]: Code Repair requires at least one target file.\n"
            return

        repair_instruction = self._build_repair_instruction(context)
        for log in run_aider_stream(
            target_files=context.target_files,
            user_instruction=repair_instruction,
            model=context.model,
            api_key=context.api_key,
            base_url=context.base_url,
            project_dir=context.project_dir,
            symbol=None,
            completion_type=None,
            prefix="",
        ):
            yield log

    # [maint-note:da8abdb11c00]
    # 维护导航：_build_repair_instruction；下列为本函数源码中的接口与路径索引。
    # 输入 context：ExecutionContext。
    # 返回契约：str。
    # 分支条件：allow_refactor。
    # 分支条件：user_instruction。
    # 分支条件：failure_log。
    # 分支条件：extra_context。
    # 返回路径：'\n'.join(parts)。
    # 直接调用：str(config.get('repair_type', 'bug_fix')).strip。
    # 直接调用：str。
    # 直接调用：config.get。
    # 直接调用：str(config.get('failure_log', '')).strip。
    # 直接调用：str(config.get('extra_context', '')).strip。
    # 直接调用：self._bool_config。
    # 直接调用：context.instruction.strip。
    # 直接调用：REPAIR_TYPES.get。
    # 直接调用：parts.append。
    # 直接调用：parts.extend。
    # 直接调用：'\n'.join。
    # [/maint-note:da8abdb11c00]
    def _build_repair_instruction(self, context: ExecutionContext) -> str:
        config = context.feature_config or {}
        repair_type = str(config.get("repair_type", "bug_fix")).strip()
        failure_log = str(config.get("failure_log", "")).strip()
        extra_context = str(config.get("extra_context", "")).strip()
        allow_refactor = self._bool_config(config, "allow_refactor", False)
        user_instruction = context.instruction.strip()

        parts = [
            "你是一个代码修复助手。请根据项目上下文和目标文件，完成一次最小必要代码修复。",
            "",
            "【修复目标】",
            REPAIR_TYPES.get(repair_type, REPAIR_TYPES["bug_fix"]),
            "",
            "【硬性要求】",
            "1. 优先定位根因，不要只掩盖症状。",
            "2. 保持现有公开接口、文件结构和调用约定，除非修复必须改变。",
            "3. 只修改与本次问题直接相关的代码。",
            "4. 不要引入新的第三方依赖，除非用户明确要求。",
            "5. 修改后保持项目现有代码风格。",
        ]

        if allow_refactor:
            parts.append("6. 允许小范围辅助重构，但必须服务于本次修复。")
        else:
            parts.append("6. 不要做无关重构或格式化。")

        if user_instruction:
            parts.extend(["", "【用户修复需求】", user_instruction])
        if failure_log:
            parts.extend(["", "【失败日志或错误输出】", failure_log])
        if extra_context:
            parts.extend(["", "【额外上下文】", extra_context])

        parts.extend(
            [
                "",
                "【交付标准】",
                "- 修复目标文件中的问题。",
                "- 若能从日志或上下文推断验证方式，请按该验证目标调整实现。",
                "- 如果信息不足，请做最保守的修复，并避免扩大修改范围。",
            ]
        )
        return "\n".join(parts)

    # [maint-note:3a1e3cc00bd5]
    # 维护导航：_bool_config；下列为本函数源码中的接口与路径索引。
    # 输入 config：Dict[str, Any]。
    # 输入 name：str。
    # 输入 default：bool。
    # 返回契约：bool。
    # 分支条件：isinstance(value, str)。
    # 返回路径：value.lower() in {'1', 'true', 'yes', 'on'}。
    # 返回路径：bool(value)。
    # 直接调用：config.get。
    # 直接调用：isinstance。
    # 直接调用：value.lower。
    # 直接调用：bool。
    # [/maint-note:3a1e3cc00bd5]
    def _bool_config(self, config: Dict[str, Any], name: str, default: bool) -> bool:
        value = config.get(name, default)
        if isinstance(value, str):
            return value.lower() in {"1", "true", "yes", "on"}
        return bool(value)
