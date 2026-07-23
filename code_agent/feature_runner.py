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


def _serialize_metadata(metadata) -> Dict[str, Any]:
    return {
        "name": metadata.name,
        "label": metadata.label,
        "description": metadata.description,
        "icon": metadata.icon,
        "execution_mode": metadata.execution_mode.value,
    }


def _serialize_plugin(plugin, include_schema: bool = False) -> Dict[str, Any]:
    payload = _serialize_metadata(plugin.metadata)
    if include_schema:
        payload["config_schema"] = [
            _serialize_field(field)
            for field in plugin.config_schema
        ]
    return payload


def list_features():
    """Return metadata for all registered feature plugins."""

    return sorted(
        [
            _serialize_metadata(metadata)
            for metadata in registry.list_plugins()
        ],
        key=lambda item: item["name"],
    )


def describe_feature(name: Optional[str]) -> Dict[str, Any]:
    """Return full metadata and config schema for one registered feature plugin."""

    feature_name = (name or "").strip()
    if not feature_name:
        return {"error": "feature name is required"}

    plugin = registry.get(feature_name)
    if plugin is None:
        return {"error": f"Unknown feature: {feature_name}"}

    return _serialize_plugin(plugin, include_schema=True)


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

    def __init__(self, path: Path):
        self.path = path
        self.filename = path.name
        self.file = path.open("rb")

    def close(self) -> None:
        self.file.close()


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


def _make_upload(path: Path) -> Optional[_PathUpload]:
    try:
        return _PathUpload(path)
    except OSError:
        return None


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


def _close_uploads(uploaded_files: Dict[str, Any]) -> None:
    def close_item(item: Any) -> None:
        if hasattr(item, "close"):
            item.close()

    for value in uploaded_files.values():
        if isinstance(value, Iterable) and not isinstance(value, (str, bytes, bytearray, dict)):
            for item in value:
                close_item(item)
        else:
            close_item(value)


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
