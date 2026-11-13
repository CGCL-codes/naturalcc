import sys
from pathlib import Path
from typing import Any, Dict, Generator, List, Optional

package_root = Path(__file__).resolve().parent.parent
if str(package_root) not in sys.path:
    sys.path.insert(0, str(package_root))

from code_agent.aider_runner import preview_prompt, run_aider_stream
from code_agent.plugins.base import (
    ConfigField,
    ConfigFieldType,
    ExecutionContext,
    ExecutionMode,
    FeatureMetadata,
    FeaturePlugin,
    PluginResult,
)
from code_agent.plugins.registry import register_plugin


# [maint-note:37730c6ac6c2]
# 数据与行为导航：CodeCompletionPlugin。
# 继承接口：FeaturePlugin。
# [/maint-note:37730c6ac6c2]
@register_plugin
class CodeCompletionPlugin(FeaturePlugin):
    """
    代码补全插件 —— 将原有的硬编码 symbol / completion_type / prefix 逻辑
    封装为 FeaturePlugin，完全不改变原有行为。
    """

    # [maint-note:d1b275e1bce2]
    # 维护导航：metadata；下列为本函数源码中的接口与路径索引。
    # 返回契约：FeatureMetadata。
    # 返回路径：FeatureMetadata(name='code_completion', label='Code Completion', description='基于项目语义图谱的智能代码补全', execution_mode=ExecutionMode.AIDER)。
    # 直接调用：FeatureMetadata。
    # [/maint-note:d1b275e1bce2]
    @property
    def metadata(self) -> FeatureMetadata:
        return FeatureMetadata(
            name="code_completion",
            label="Code Completion",
            description="基于项目语义图谱的智能代码补全",
            execution_mode=ExecutionMode.AIDER,
        )

    # [maint-note:994424d416ad]
    # 维护导航：config_schema；下列为本函数源码中的接口与路径索引。
    # 返回契约：List[ConfigField]。
    # 返回路径：[ConfigField(name='symbol', label='Symbol', type=ConfigFieldType.TEXT, required=False, default='', placeholder='Auto inferred', help_text='目标符号名称，留空则自动推断'), ConfigField(name='completion_type', label='Completion Type', type=ConfigFieldType.S。
    # 直接调用：ConfigField。
    # [/maint-note:994424d416ad]
    @property
    def config_schema(self) -> List[ConfigField]:
        return [
            ConfigField(
                name="symbol",
                label="Symbol",
                type=ConfigFieldType.TEXT,
                required=False,
                default="",
                placeholder="Auto inferred",
                help_text="目标符号名称，留空则自动推断",
            ),
            ConfigField(
                name="completion_type",
                label="Completion Type",
                type=ConfigFieldType.SELECT,
                required=False,
                default="",
                options=[
                    {"value": "", "label": "auto"},
                    {"value": "member", "label": "member"},
                    {"value": "variable", "label": "variable"},
                    {"value": "function", "label": "function"},
                    {"value": "function_body", "label": "function_body"},
                    {"value": "type", "label": "type"},
                ],
                help_text="补全类型",
            ),
            ConfigField(
                name="prefix",
                label="Prefix",
                type=ConfigFieldType.TEXT,
                required=False,
                default="",
                placeholder="Optional filter",
                help_text="前缀过滤",
            ),
        ]

    # [maint-note:38fddda98e77]
    # 维护导航：validate；下列为本函数源码中的接口与路径索引。
    # 输入 config：Dict[str, Any]。
    # 输入 files：类型由调用方与函数体约束；缺省表达式 None。
    # 返回契约：Optional[str]。
    # 分支条件：completion_type not in valid_types。
    # 返回路径：f'Invalid completion_type: {completion_type}'。
    # 返回路径：None。
    # 直接调用：config.get。
    # [/maint-note:38fddda98e77]
    def validate(self, config: Dict[str, Any], files=None) -> Optional[str]:
        completion_type = config.get("completion_type", "")
        valid_types = ["", "member", "variable", "function", "function_body", "type"]
        if completion_type not in valid_types:
            return f"Invalid completion_type: {completion_type}"
        return None

    # [maint-note:1bc4fe0e2b46]
    # 维护导航：preview；下列为本函数源码中的接口与路径索引。
    # 输入 context：ExecutionContext。
    # 返回契约：str。
    # 返回路径：preview_prompt(target_files=context.target_files, user_instruction=context.instruction, model=context.model, api_key=context.api_key, project_dir=context.project_dir, symbol=config.get('symbol') or None, completion_type=config.get('completi。
    # 直接调用：preview_prompt。
    # 直接调用：config.get。
    # [/maint-note:1bc4fe0e2b46]
    def preview(self, context: ExecutionContext) -> str:
        config = context.feature_config
        return preview_prompt(
            target_files=context.target_files,
            user_instruction=context.instruction,
            model=context.model,
            api_key=context.api_key,
            project_dir=context.project_dir,
            symbol=config.get("symbol") or None,
            completion_type=config.get("completion_type") or None,
            prefix=config.get("prefix", ""),
        )

    # [maint-note:ba44ddc2a424]
    # 维护导航：execute；下列为本函数源码中的接口与路径索引。
    # 输入 context：ExecutionContext。
    # 返回契约：Generator[str, None, None]。
    # 遍历边界：log ← run_aider_stream(target_files=context.target_files, user_instruction=context.instruction, model=context.model, api_key=context.api_key, base_url=context.base_url, project_dir=context.project_dir, symbol=config.get('symbol') or None, complet。
    # 直接调用：run_aider_stream。
    # 直接调用：config.get。
    # 生成器交付：log。
    # [/maint-note:ba44ddc2a424]
    def execute(self, context: ExecutionContext) -> Generator[str, None, None]:
        config = context.feature_config
        for log in run_aider_stream(
            target_files=context.target_files,
            user_instruction=context.instruction,
            model=context.model,
            api_key=context.api_key,
            base_url=context.base_url,
            project_dir=context.project_dir,
            symbol=config.get("symbol") or None,
            completion_type=config.get("completion_type") or None,
            prefix=config.get("prefix", ""),
        ):
            yield log
