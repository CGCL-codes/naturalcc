"""CLI bridge for the feature plugin system."""

import sys
from collections.abc import Iterable
from copy import deepcopy
from pathlib import Path
from typing import Any, Dict, Optional


if __package__ in (None, ""):
    package_root = Path(__file__).resolve().parent.parent
    if str(package_root) not in sys.path:
        sys.path.insert(0, str(package_root))

import code_agent.plugins  # noqa: F401 - importing registers bundled plugins
from code_agent.aider_runner import normalize_project_dir, normalize_target_files
from code_agent.plugins.base import ConfigFieldType, ExecutionContext
from code_agent.plugins.dispatcher import dispatcher, ndjson_event
from code_agent.plugins.registry import registry


# [maint-note:1fb4c045986b]
# 维护导航：_serialize_field；下列为本函数源码中的接口与路径索引。
# 输入 field：类型由调用方与函数体约束。
# 返回契约：Dict[str, Any]。
# 返回路径：{'name': field.name, 'label': field.label, 'type': field.type.value, 'required': field.required, 'default': field.default, 'placeholder': field.placeholder, 'help_text': field.help_text, 'options': field.options, 'accept': field.accept, 'mu。
# [/maint-note:1fb4c045986b]
def _serialize_field(field) -> Dict[str, Any]:
    return {
        "name": field.name,
        "label": field.label,
        "type": field.type.value,
        "required": field.required,
        "default": field.default,
        "placeholder": field.placeholder,
        "help_text": field.help_text,
        "options": field.options,
        "accept": field.accept,
        "multiple": field.multiple,
    }


# [maint-note:aa4a5fef616e]
# 维护导航：_serialize_metadata；下列为本函数源码中的接口与路径索引。
# 输入 metadata：类型由调用方与函数体约束。
# 返回契约：Dict[str, Any]。
# 返回路径：{'name': metadata.name, 'label': metadata.label, 'description': metadata.description, 'icon': metadata.icon, 'execution_mode': metadata.execution_mode.value}。
# [/maint-note:aa4a5fef616e]
def _serialize_metadata(metadata) -> Dict[str, Any]:
    return {
        "name": metadata.name,
        "label": metadata.label,
        "description": metadata.description,
        "icon": metadata.icon,
        "execution_mode": metadata.execution_mode.value,
    }


# [maint-note:2e86aa928a08]
# 维护导航：_serialize_plugin；下列为本函数源码中的接口与路径索引。
# 输入 plugin：类型由调用方与函数体约束。
# 输入 include_schema：bool；缺省表达式 False。
# 返回契约：Dict[str, Any]。
# 分支条件：include_schema。
# 返回路径：payload。
# 写入目标：payload['config_schema']。
# 直接调用：_serialize_metadata。
# 直接调用：_serialize_field。
# 集合派生：field 来自 plugin.config_schema。
# [/maint-note:2e86aa928a08]
def _serialize_plugin(plugin, include_schema: bool = False) -> Dict[str, Any]:
    payload = _serialize_metadata(plugin.metadata)
    if include_schema:
        payload["config_schema"] = [
            _serialize_field(field)
            for field in plugin.config_schema
        ]
    return payload


# [maint-note:ebc2b8c91916]
# 维护导航：list_features；下列为本函数源码中的接口与路径索引。
# 返回路径：sorted([_serialize_metadata(metadata) for metadata in registry.list_plugins()], key=lambda item: item['name'])。
# 直接调用：sorted。
# 直接调用：_serialize_metadata。
# 直接调用：registry.list_plugins。
# 集合派生：metadata 来自 registry.list_plugins()。
# [/maint-note:ebc2b8c91916]
def list_features():
    """Return metadata for all registered feature plugins."""

    return sorted(
        [
            _serialize_metadata(metadata)
            for metadata in registry.list_plugins()
        ],
        key=lambda item: item["name"],
    )


# [maint-note:e2a5772f0e60]
# 维护导航：describe_feature；下列为本函数源码中的接口与路径索引。
# 输入 name：Optional[str]。
# 返回契约：Dict[str, Any]。
# 分支条件：not feature_name。
# 分支条件：plugin is None。
# 返回路径：{'error': 'feature name is required'}。
# 返回路径：{'error': f'Unknown feature: {feature_name}'}。
# 返回路径：_serialize_plugin(plugin, include_schema=True)。
# 直接调用：(name or '').strip。
# 直接调用：registry.get。
# 直接调用：_serialize_plugin。
# [/maint-note:e2a5772f0e60]
def describe_feature(name: Optional[str]) -> Dict[str, Any]:
    """Return full metadata and config schema for one registered feature plugin."""

    feature_name = (name or "").strip()
    if not feature_name:
        return {"error": "feature name is required"}

    plugin = registry.get(feature_name)
    if plugin is None:
        return {"error": f"Unknown feature: {feature_name}"}

    return _serialize_plugin(plugin, include_schema=True)


# [maint-note:7efa59953b0b]
# 维护导航：_apply_config_defaults；下列为本函数源码中的接口与路径索引。
# 输入 feature：str。
# 输入 config：Dict[str, Any]。
# 返回契约：None。
# 分支条件：plugin is None。
# 分支条件：field.name in config or field.default is None。
# 遍历边界：field ← plugin.config_schema。
# 返回路径：无返回值。
# 写入目标：config[field.name]。
# 直接调用：registry.get。
# 直接调用：deepcopy。
# [/maint-note:7efa59953b0b]
def _apply_config_defaults(feature: str, config: Dict[str, Any]) -> None:
    plugin = registry.get(feature)
    if plugin is None:
        return

    for field in plugin.config_schema:
        if field.name in config or field.default is None:
            continue
        config[field.name] = deepcopy(field.default)


class _PathUpload:
    """Small UploadFile-compatible wrapper for CLI-provided file paths."""

    # [maint-note:1c27acd8ea2a]
    # 维护导航：__init__；下列为本函数源码中的接口与路径索引。
    # 输入 path：Path。
    # 写入目标：self.path。
    # 写入目标：self.filename。
    # 写入目标：self.file。
    # 直接调用：path.open。
    # [/maint-note:1c27acd8ea2a]
    def __init__(self, path: Path):
        self.path = path
        self.filename = path.name
        self.file = path.open("rb")

    # [maint-note:d241b63ec6f2]
    # 维护导航：close；下列为本函数源码中的接口与路径索引。
    # 返回契约：None。
    # 直接调用：self.file.close。
    # [/maint-note:d241b63ec6f2]
    def close(self) -> None:
        self.file.close()


# [maint-note:cdc963963269]
# 维护导航：_merge_feature_config；下列为本函数源码中的接口与路径索引。
# 输入 feature：Optional[str]。
# 输入 feature_config：Optional[Dict[str, Any]]。
# 输入 symbol：Optional[str]。
# 输入 completion_type：Optional[str]。
# 输入 prefix：str。
# 返回契约：Dict[str, Any]。
# 分支条件：symbol is not None。
# 分支条件：completion_type is not None。
# 分支条件：prefix。
# 返回路径：config。
# 写入目标：config['feature']。
# 直接调用：dict。
# 直接调用：config.setdefault。
# 直接调用：config.get。
# 直接调用：isinstance。
# 直接调用：_apply_config_defaults。
# 表达式分流：isinstance(configured_feature, str)；真值与假值分支分别计算，不同时执行。
# [/maint-note:cdc963963269]
def _merge_feature_config(
    feature: Optional[str],
    feature_config: Optional[Dict[str, Any]],
    symbol: Optional[str],
    completion_type: Optional[str],
    prefix: str,
) -> Dict[str, Any]:
    config = dict(feature_config or {})
    if symbol is not None:
        config.setdefault("symbol", symbol)
    if completion_type is not None:
        config.setdefault("completion_type", completion_type)
    if prefix:
        config.setdefault("prefix", prefix)
    configured_feature = config.get("feature")
    config["feature"] = (
        feature
        or (configured_feature if isinstance(configured_feature, str) else None)
        or "code_completion"
    )
    _apply_config_defaults(config["feature"], config)
    return config


# [maint-note:a9cdda19d29c]
# 维护导航：_resolve_path；下列为本函数源码中的接口与路径索引。
# 输入 value：Any。
# 输入 project_dir：str。
# 返回契约：Optional[Path]。
# 分支条件：value is None。
# 分支条件：not raw。
# 分支条件：candidate.is_absolute()。
# 分支条件：project_candidate.exists()。
# 分支条件：cwd_candidate.exists()。
# 返回路径：None。
# 返回路径：candidate if candidate.exists() else None。
# 返回路径：project_candidate。
# 返回路径：cwd_candidate。
# 直接调用：str(value).strip。
# 直接调用：str。
# 直接调用：Path(raw).expanduser。
# 直接调用：Path。
# 直接调用：candidate.is_absolute。
# 直接调用：candidate.exists。
# 直接调用：(Path(project_dir) / candidate).resolve。
# 直接调用：project_candidate.exists。
# 直接调用：candidate.resolve。
# 直接调用：cwd_candidate.exists。
# 表达式分流：candidate.exists()；真值与假值分支分别计算，不同时执行。
# [/maint-note:a9cdda19d29c]
def _resolve_path(value: Any, project_dir: str) -> Optional[Path]:
    if value is None:
        return None
    raw = str(value).strip()
    if not raw:
        return None

    candidate = Path(raw).expanduser()
    if candidate.is_absolute():
        return candidate if candidate.exists() else None

    project_candidate = (Path(project_dir) / candidate).resolve()
    if project_candidate.exists():
        return project_candidate

    cwd_candidate = candidate.resolve()
    if cwd_candidate.exists():
        return cwd_candidate

    return None


# [maint-note:f036b69c7914]
# 维护导航：_make_upload；下列为本函数源码中的接口与路径索引。
# 输入 path：Path。
# 返回契约：Optional[_PathUpload]。
# 返回路径：_PathUpload(path)。
# 返回路径：None。
# 捕获异常：OSError。
# 直接调用：_PathUpload。
# [/maint-note:f036b69c7914]
def _make_upload(path: Path) -> Optional[_PathUpload]:
    try:
        return _PathUpload(path)
    except OSError:
        return None


# [maint-note:423660bace24]
# 维护导航：_to_path_uploads；下列为本函数源码中的接口与路径索引。
# 输入 value：Any。
# 输入 project_dir：str。
# 返回契约：Any。
# 分支条件：isinstance(value, (list, tuple))。
# 分支条件：path is not None。
# 分支条件：upload is not None。
# 分支条件：path is None。
# 遍历边界：item ← value。
# 返回路径：uploads。
# 返回路径：None。
# 返回路径：_make_upload(path)。
# 直接调用：isinstance。
# 直接调用：_resolve_path。
# 直接调用：_make_upload。
# 直接调用：uploads.append。
# [/maint-note:423660bace24]
def _to_path_uploads(value: Any, project_dir: str) -> Any:
    if isinstance(value, (list, tuple)):
        uploads = []
        for item in value:
            path = _resolve_path(item, project_dir)
            if path is not None:
                upload = _make_upload(path)
                if upload is not None:
                    uploads.append(upload)
        return uploads

    path = _resolve_path(value, project_dir)
    if path is None:
        return None
    return _make_upload(path)


# [maint-note:ab6d2b0d385e]
# 维护导航：_build_uploaded_files；下列为本函数源码中的接口与路径索引。
# 输入 feature：str。
# 输入 feature_config：Dict[str, Any]。
# 输入 project_dir：str。
# 输入 uploaded_files：Optional[Dict[str, Any]]。
# 返回契约：Dict[str, Any]。
# 分支条件：plugin is None。
# 分支条件：field.type != ConfigFieldType.FILE。
# 分支条件：field.name in files。
# 分支条件：upload。
# 遍历边界：field ← plugin.config_schema。
# 返回路径：files。
# 写入目标：files[field.name]。
# 直接调用：dict。
# 直接调用：registry.get。
# 直接调用：_to_path_uploads。
# 直接调用：feature_config.get。
# [/maint-note:ab6d2b0d385e]
def _build_uploaded_files(
    feature: str,
    feature_config: Dict[str, Any],
    project_dir: str,
    uploaded_files: Optional[Dict[str, Any]],
) -> Dict[str, Any]:
    files = dict(uploaded_files or {})
    plugin = registry.get(feature)
    if plugin is None:
        return files

    for field in plugin.config_schema:
        if field.type != ConfigFieldType.FILE:
            continue
        if field.name in files:
            continue
        upload = _to_path_uploads(feature_config.get(field.name), project_dir)
        if upload:
            files[field.name] = upload

    return files


# [maint-note:c26046b06c39]
# 维护导航：_close_uploads；下列为本函数源码中的接口与路径索引。
# 输入 uploaded_files：Dict[str, Any]。
# 返回契约：None。
# 分支条件：hasattr(item, 'close')。
# 分支条件：isinstance(value, Iterable) and (not isinstance(value, (str, bytes, bytearray, dict)))。
# 遍历边界：value ← uploaded_files.values()。
# 遍历边界：item ← value。
# 直接调用：hasattr。
# 直接调用：item.close。
# 直接调用：uploaded_files.values。
# 直接调用：isinstance。
# 直接调用：close_item。
# [/maint-note:c26046b06c39]
def _close_uploads(uploaded_files: Dict[str, Any]) -> None:
    # [maint-note:80340f58629a]
    # 维护导航：close_item；下列为本函数源码中的接口与路径索引。
    # 输入 item：Any。
    # 返回契约：None。
    # 分支条件：hasattr(item, 'close')。
    # 直接调用：hasattr。
    # 直接调用：item.close。
    # [/maint-note:80340f58629a]
    def close_item(item: Any) -> None:
        if hasattr(item, "close"):
            item.close()

    for value in uploaded_files.values():
        if isinstance(value, Iterable) and not isinstance(value, (str, bytes, bytearray, dict)):
            for item in value:
                close_item(item)
        else:
            close_item(value)


# [maint-note:27c5b88bbd1a]
# 维护导航：_build_context；下列为本函数源码中的接口与路径索引。
# 输入 target_files：类型由调用方与函数体约束。
# 输入 user_instruction：str。
# 输入 model：str。
# 输入 api_key：Optional[str]。
# 输入 project_dir：Optional[str]。
# 输入 symbol：Optional[str]。
# 输入 completion_type：Optional[str]。
# 输入 prefix：str。
# 输入 feature：Optional[str]。
# 输入 feature_config：Optional[Dict[str, Any]]。
# 输入 uploaded_files：Optional[Dict[str, Any]]；缺省表达式 None。
# 返回契约：ExecutionContext。
# 返回路径：ExecutionContext(project_dir=normalized_project_dir, target_files=normalized_targets, instruction=user_instruction or '', model=model, api_key=api_key, feature_config=merged_config, uploaded_files=files, symbol=symbol, completion_type=compl。
# 直接调用：normalize_project_dir。
# 直接调用：normalize_target_files。
# 直接调用：_merge_feature_config。
# 直接调用：str。
# 直接调用：merged_config.get。
# 直接调用：_build_uploaded_files。
# 直接调用：ExecutionContext。
# [/maint-note:27c5b88bbd1a]
def _build_context(
    target_files,
    user_instruction: str,
    model: str,
    api_key: Optional[str],
    project_dir: Optional[str],
    symbol: Optional[str],
    completion_type: Optional[str],
    prefix: str,
    feature: Optional[str],
    feature_config: Optional[Dict[str, Any]],
    uploaded_files: Optional[Dict[str, Any]] = None,
) -> ExecutionContext:
    normalized_project_dir = normalize_project_dir(project_dir)
    normalized_targets = normalize_target_files(target_files, project_dir=normalized_project_dir)
    merged_config = _merge_feature_config(
        feature=feature,
        feature_config=feature_config,
        symbol=symbol,
        completion_type=completion_type,
        prefix=prefix,
    )
    feature_name = str(merged_config.get("feature") or "code_completion")
    files = _build_uploaded_files(feature_name, merged_config, normalized_project_dir, uploaded_files)

    return ExecutionContext(
        project_dir=normalized_project_dir,
        target_files=normalized_targets,
        instruction=user_instruction or "",
        model=model,
        api_key=api_key,
        feature_config=merged_config,
        uploaded_files=files,
        symbol=symbol,
        completion_type=completion_type,
        prefix=prefix or "",
    )


# [maint-note:f535fa941fab]
# 维护导航：run_feature_stream；下列为本函数源码中的接口与路径索引。
# 输入 target_files：类型由调用方与函数体约束。
# 输入 user_instruction：str。
# 输入 model：str。
# 输入 api_key：Optional[str]；缺省表达式 None。
# 输入 project_dir：Optional[str]；缺省表达式 None。
# 输入 symbol：Optional[str]；缺省表达式 None。
# 输入 completion_type：Optional[str]；缺省表达式 None。
# 输入 prefix：str；缺省表达式 ''。
# 输入 feature：Optional[str]；缺省表达式 None。
# 输入 feature_config：Optional[Dict[str, Any]]；缺省表达式 None。
# 输入 uploaded_files：Optional[Dict[str, Any]]；缺省表达式 None。
# 分支条件：context is not None。
# 捕获异常：Exception。
# 直接调用：_build_context。
# 直接调用：dispatcher.dispatch。
# 直接调用：ndjson_event。
# 直接调用：_close_uploads。
# 清理边界：本函数含 finally；正常返回和异常展开都经过清理块。
# 生成器交付：dispatcher.dispatch(context)。
# 生成器交付：ndjson_event({'type': 'error', 'status': 'error', 'log': f'Feature execution failed: {exc}'})。
# [/maint-note:f535fa941fab]
def run_feature_stream(
    target_files,
    user_instruction: str,
    model: str,
    api_key: Optional[str] = None,
    project_dir: Optional[str] = None,
    symbol: Optional[str] = None,
    completion_type: Optional[str] = None,
    prefix: str = "",
    feature: Optional[str] = None,
    feature_config: Optional[Dict[str, Any]] = None,
    uploaded_files: Optional[Dict[str, Any]] = None,
):
    """Run a feature plugin and yield dispatcher-compatible NDJSON events."""

    context = None
    try:
        context = _build_context(
            target_files=target_files,
            user_instruction=user_instruction,
            model=model,
            api_key=api_key,
            project_dir=project_dir,
            symbol=symbol,
            completion_type=completion_type,
            prefix=prefix,
            feature=feature,
            feature_config=feature_config,
            uploaded_files=uploaded_files,
        )
        yield from dispatcher.dispatch(context)
    except Exception as exc:
        yield ndjson_event({
            "type": "error",
            "status": "error",
            "log": f"Feature execution failed: {exc}",
        })
    finally:
        if context is not None:
            _close_uploads(context.uploaded_files)


# [maint-note:d68c4d3ed36e]
# 维护导航：preview_feature；下列为本函数源码中的接口与路径索引。
# 输入 target_files：类型由调用方与函数体约束。
# 输入 user_instruction：str。
# 输入 model：str。
# 输入 api_key：Optional[str]；缺省表达式 None。
# 输入 project_dir：Optional[str]；缺省表达式 None。
# 输入 symbol：Optional[str]；缺省表达式 None。
# 输入 completion_type：Optional[str]；缺省表达式 None。
# 输入 prefix：str；缺省表达式 ''。
# 输入 feature：Optional[str]；缺省表达式 None。
# 输入 feature_config：Optional[Dict[str, Any]]；缺省表达式 None。
# 输入 uploaded_files：Optional[Dict[str, Any]]；缺省表达式 None。
# 返回契约：str。
# 分支条件：plugin is None。
# 分支条件：context is not None。
# 返回路径：f'Unknown feature: {feature_name}'。
# 返回路径：plugin.preview(context)。
# 返回路径：f'Feature preview failed: {exc}'。
# 捕获异常：Exception。
# 直接调用：_build_context。
# 直接调用：context.feature_config.get。
# 直接调用：registry.get。
# 直接调用：plugin.preview。
# 直接调用：_close_uploads。
# 清理边界：本函数含 finally；正常返回和异常展开都经过清理块。
# [/maint-note:d68c4d3ed36e]
def preview_feature(
    target_files,
    user_instruction: str,
    model: str,
    api_key: Optional[str] = None,
    project_dir: Optional[str] = None,
    symbol: Optional[str] = None,
    completion_type: Optional[str] = None,
    prefix: str = "",
    feature: Optional[str] = None,
    feature_config: Optional[Dict[str, Any]] = None,
    uploaded_files: Optional[Dict[str, Any]] = None,
) -> str:
    """Return the selected feature plugin's preview text."""

    context = None
    try:
        context = _build_context(
            target_files=target_files,
            user_instruction=user_instruction,
            model=model,
            api_key=api_key,
            project_dir=project_dir,
            symbol=symbol,
            completion_type=completion_type,
            prefix=prefix,
            feature=feature,
            feature_config=feature_config,
            uploaded_files=uploaded_files,
        )
        feature_name = context.feature_config.get("feature", "code_completion")
        plugin = registry.get(feature_name)
        if plugin is None:
            return f"Unknown feature: {feature_name}"
        return plugin.preview(context)
    except Exception as exc:
        return f"Feature preview failed: {exc}"
    finally:
        if context is not None:
            _close_uploads(context.uploaded_files)
