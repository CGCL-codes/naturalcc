import argparse
import asyncio
import json
import os
import shlex
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, AsyncIterable, Dict, Iterable, List, Optional, Sequence, Tuple

from fastapi import FastAPI, File, Form, HTTPException, Query, Request, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

if __package__ in (None, ""):
    package_root = Path(__file__).resolve().parent.parent
    if str(package_root) not in sys.path:
        sys.path.insert(0, str(package_root))
    from code_agent.aider_runner import (
        DEFAULT_PROJECT_DIR,
        detect_provider,
        normalize_project_dir,
        preview_prompt,
        run_aider_stream,
    )
else:
    from .aider_runner import (
        DEFAULT_PROJECT_DIR,
        detect_provider,
        normalize_project_dir,
        preview_prompt,
        run_aider_stream,
    )

if __package__ in (None, ""):
    import code_agent.plugins
    from code_agent.plugins.base import ExecutionContext
    from code_agent.plugins.dispatcher import dispatcher, ndjson_event
    from code_agent.plugins.registry import registry
else:
    from . import plugins
    from .plugins.base import ExecutionContext
    from .plugins.dispatcher import dispatcher, ndjson_event
    from .plugins.registry import registry


MODELS = [
    "deepseek/deepseek-chat",
    "deepseek/deepseek-coder",
]
DEFAULT_MODEL = "deepseek/deepseek-chat"
STREAM_HEADERS = {
    "Cache-Control": "no-cache, no-transform",
    "X-Accel-Buffering": "no",
    "Connection": "keep-alive",
}

IGNORE_DIRS = {
    ".git",
    "__pycache__",
    "venv",
    ".venv",
    ".aider.tags.cache.v3",
    ".aider.tags.cache.v4",
    "datasets",
    "node_modules",
    ".idea",
    ".vscode",
}
IGNORE_FILE_EXTENSIONS = {".log", ".ps"}
PARSABLE_SOURCE_EXTENSIONS = {".c", ".cpp", ".h", ".hpp", ".java"}
COMPLETION_TYPES = ["", "member", "variable", "function", "function_body", "type"]
WINDOWS_DEVICE_NAMES = {
    "CON",
    "PRN",
    "AUX",
    "NUL",
    *(f"COM{index}" for index in range(1, 10)),
    *(f"LPT{index}" for index in range(1, 10)),
}


# [maint-note:f23f69b19c4f]
# 数据与行为导航：AgentRequest。
# 继承接口：BaseModel。
# 字段 project_dir：str；初始表达式 Field(default=DEFAULT_PROJECT_DIR)。
# 字段 target_files：List[str]；初始表达式 Field(default_factory=list)。
# 字段 instruction：str；初始表达式 ''。
# 字段 model：str；初始表达式 DEFAULT_MODEL。
# 字段 api_key：Optional[str]；初始表达式 None。
# 字段 symbol：Optional[str]；初始表达式 None。
# 字段 completion_type：Optional[str]；初始表达式 None。
# 字段 prefix：str；初始表达式 ''。
# 字段 feature：str；初始表达式 Field(default='code_completion')。
# 字段 feature_config：Dict[str, Any]；初始表达式 Field(default_factory=dict)。
# [/maint-note:f23f69b19c4f]
class AgentRequest(BaseModel):
    project_dir: str = Field(default=DEFAULT_PROJECT_DIR)
    target_files: List[str] = Field(default_factory=list)
    instruction: str = ""
    model: str = DEFAULT_MODEL
    api_key: Optional[str] = None
    symbol: Optional[str] = None
    completion_type: Optional[str] = None
    prefix: str = ""
    feature: str = Field(default="code_completion")
    feature_config: Dict[str, Any] = Field(default_factory=dict)


# [maint-note:6299a04844cb]
# 数据与行为导航：ScanRequest。
# 继承接口：BaseModel。
# 字段 project_dir：str；初始表达式 Field(default=DEFAULT_PROJECT_DIR)。
# 字段 target_files：List[str]；初始表达式 Field(default_factory=list)。
# [/maint-note:6299a04844cb]
class ScanRequest(BaseModel):
    project_dir: str = Field(default=DEFAULT_PROJECT_DIR)
    target_files: List[str] = Field(default_factory=list)


# [maint-note:87b23febcf5a]
# 数据与行为导航：DirectoryPickerRequest。
# 继承接口：BaseModel。
# 字段 initial_dir：Optional[str]；初始表达式 None。
# [/maint-note:87b23febcf5a]
class DirectoryPickerRequest(BaseModel):
    initial_dir: Optional[str] = None


# [maint-note:e1b2df2ec7a3]
# 维护导航：is_windows_device_name；下列为本函数源码中的接口与路径索引。
# 输入 filename：str。
# 返回契约：bool。
# 返回路径：stem in WINDOWS_DEVICE_NAMES。
# 直接调用：Path(filename).name.strip().rstrip。
# 直接调用：Path(filename).name.strip。
# 直接调用：Path。
# 直接调用：name.split('.', 1)[0].upper。
# 直接调用：name.split。
# [/maint-note:e1b2df2ec7a3]
def is_windows_device_name(filename: str) -> bool:
    name = Path(filename).name.strip().rstrip(" .")
    stem = name.split(".", 1)[0].upper()
    return stem in WINDOWS_DEVICE_NAMES


# [maint-note:b953ed070583]
# 维护导航：should_ignore_file；下列为本函数源码中的接口与路径索引。
# 输入 filename：str。
# 返回契约：bool。
# 返回路径：filename.startswith('.') or is_windows_device_name(filename) or Path(filename).suffix.lower() in IGNORE_FILE_EXTENSIONS。
# 直接调用：filename.startswith。
# 直接调用：is_windows_device_name。
# 直接调用：Path(filename).suffix.lower。
# 直接调用：Path。
# [/maint-note:b953ed070583]
def should_ignore_file(filename: str) -> bool:
    return (
        filename.startswith(".")
        or is_windows_device_name(filename)
        or Path(filename).suffix.lower() in IGNORE_FILE_EXTENSIONS
    )


# [maint-note:594dbc40d442]
# 维护导航：sanitize_target_files；下列为本函数源码中的接口与路径索引。
# 输入 target_files：Optional[Iterable[str]]。
# 返回契约：List[str]。
# 分支条件：target_files is None。
# 分支条件：not normalized_item。
# 分支条件：Path(normalized_item).suffix.lower() in IGNORE_FILE_EXTENSIONS。
# 分支条件：normalized_item in seen。
# 遍历边界：item ← target_files。
# 返回路径：[]。
# 返回路径：result。
# 直接调用：set。
# 直接调用：str(item or '').strip().replace。
# 直接调用：str(item or '').strip。
# 直接调用：str。
# 直接调用：Path(normalized_item).suffix.lower。
# 直接调用：Path。
# 直接调用：result.append。
# 直接调用：seen.add。
# [/maint-note:594dbc40d442]
def sanitize_target_files(target_files: Optional[Iterable[str]]) -> List[str]:
    if target_files is None:
        return []

    result: List[str] = []
    seen = set()
    for item in target_files:
        normalized_item = str(item or "").strip().replace("\\", "/")
        if not normalized_item:
            continue
        if Path(normalized_item).suffix.lower() in IGNORE_FILE_EXTENSIONS:
            continue
        if normalized_item in seen:
            continue
        result.append(normalized_item)
        seen.add(normalized_item)
    return result


# [maint-note:ccdce24ab13e]
# 维护导航：get_local_files；下列为本函数源码中的接口与路径索引。
# 输入 root_dir：str；缺省表达式 DEFAULT_PROJECT_DIR。
# 返回契约：List[str]。
# 分支条件：not os.path.isdir(normalized_root)。
# 分支条件：should_ignore_file(filename)。
# 遍历边界：(dirpath, dirnames, filenames) ← os.walk(normalized_root)。
# 遍历边界：filename ← filenames。
# 返回路径：[]。
# 返回路径：sorted(file_list, key=str.casefold)。
# 捕获异常：ValueError。
# 写入目标：dirnames[:]。
# 直接调用：normalize_project_dir。
# 直接调用：os.path.isdir。
# 直接调用：os.walk。
# 直接调用：should_ignore_file。
# 直接调用：os.path.relpath。
# 直接调用：os.path.join。
# 直接调用：file_list.append。
# 直接调用：full_path.replace。
# 直接调用：sorted。
# 集合派生：dirname 来自 dirnames；过滤 dirname not in IGNORE_DIRS。
# [/maint-note:ccdce24ab13e]
def get_local_files(root_dir: str = DEFAULT_PROJECT_DIR) -> List[str]:
    normalized_root = normalize_project_dir(root_dir)
    if not os.path.isdir(normalized_root):
        return []

    file_list: List[str] = []
    for dirpath, dirnames, filenames in os.walk(normalized_root):
        dirnames[:] = [dirname for dirname in dirnames if dirname not in IGNORE_DIRS]
        for filename in filenames:
            if should_ignore_file(filename):
                continue
            try:
                full_path = os.path.relpath(os.path.join(dirpath, filename), normalized_root)
            except ValueError:
                continue
            file_list.append(full_path.replace("\\", "/"))
    return sorted(file_list, key=str.casefold)


# [maint-note:e0e93677c7a3]
# 维护导航：merge_file_choices；下列为本函数源码中的接口与路径索引。
# 输入 file_choices：Sequence[str]。
# 输入 target_files：Sequence[str]。
# 返回契约：Tuple[List[str], List[str]]。
# 分支条件：item in seen。
# 遍历边界：item ← selected_files。
# 返回路径：(merged_choices, selected_files)。
# 直接调用：sanitize_target_files。
# 直接调用：list。
# 直接调用：set。
# 直接调用：merged_choices.append。
# 直接调用：seen.add。
# [/maint-note:e0e93677c7a3]
def merge_file_choices(file_choices: Sequence[str], target_files: Sequence[str]) -> Tuple[List[str], List[str]]:
    selected_files = sanitize_target_files(target_files)
    merged_choices = list(file_choices)
    seen = set(merged_choices)
    for item in selected_files:
        if item in seen:
            continue
        merged_choices.append(item)
        seen.add(item)
    return merged_choices, selected_files


# [maint-note:fd8bfdcbed68]
# 维护导航：mask_secret；下列为本函数源码中的接口与路径索引。
# 输入 secret：Optional[str]。
# 返回契约：str。
# 分支条件：not secret。
# 分支条件：len(secret) <= 8。
# 返回路径：'not-set'。
# 返回路径：f'{secret[:2]}***'。
# 返回路径：f'{secret[:4]}...{secret[-4:]}'。
# 直接调用：(secret or '').strip。
# 直接调用：len。
# [/maint-note:fd8bfdcbed68]
def mask_secret(secret: Optional[str]) -> str:
    secret = (secret or "").strip()
    if not secret:
        return "not-set"
    if len(secret) <= 8:
        return f"{secret[:2]}***"
    return f"{secret[:4]}...{secret[-4:]}"


# [maint-note:ad30f85196a1]
# 维护导航：build_cli_command_preview；下列为本函数源码中的接口与路径索引。
# 输入 request：AgentRequest。
# 输入 preview_only：bool。
# 返回契约：str。
# 分支条件：selected_files。
# 分支条件：model。
# 分支条件：api_key。
# 分支条件：symbol。
# 分支条件：completion_type。
# 分支条件：prefix。
# 分支条件：preview_only。
# 分支条件：not api_key。
# 返回路径：'\n'.join(header + [' \\\n '.join((shlex.quote(part) for part in command))])。
# 直接调用：sanitize_target_files。
# 直接调用：(request.instruction or '').strip。
# 直接调用：(request.model or '').strip。
# 直接调用：(request.api_key or '').strip。
# 直接调用：dict。
# 直接调用：(request.symbol or feature_config.get('symbol') or '').strip。
# 直接调用：feature_config.get。
# 直接调用：(request.completion_type or feature_config.get('completion_type') or '').strip。
# 直接调用：(request.prefix or feature_config.get('prefix') or '').strip。
# 直接调用：normalize_project_dir。
# 直接调用：command.extend。
# 直接调用：mask_secret。
# 直接调用：command.append。
# 直接调用：header.append。
# 直接调用：'\n'.join。
# 直接调用：' \\\n '.join。
# 直接调用：shlex.quote。
# 集合派生：part 来自 command。
# [/maint-note:ad30f85196a1]
def build_cli_command_preview(request: AgentRequest, preview_only: bool) -> str:
    selected_files = sanitize_target_files(request.target_files)
    instruction = (request.instruction or "").strip()
    model = (request.model or "").strip()
    api_key = (request.api_key or "").strip()
    # Prefer feature_config for backward/forward compatibility
    feature_config = dict(request.feature_config or {})
    symbol = (request.symbol or feature_config.get("symbol") or "").strip()
    completion_type = (request.completion_type or feature_config.get("completion_type") or "").strip()
    prefix = (request.prefix or feature_config.get("prefix") or "").strip()
    project_dir = normalize_project_dir(request.project_dir or DEFAULT_PROJECT_DIR)

    command = ["python", "aider_runner.py", "-dir", project_dir]
    if selected_files:
        command.extend(["-f", *selected_files])
    command.extend(["-i", instruction or "<fill-instruction>"])
    if model:
        command.extend(["-m", model])
    if api_key:
        command.extend(["--api-key", f"<masked:{mask_secret(api_key)}>"])
    if symbol:
        command.extend(["-s", symbol])
    if completion_type:
        command.extend(["-t", completion_type])
    if prefix:
        command.extend(["--prefix", prefix])
    if preview_only:
        command.append("--preview")

    header = [
        "# Equivalent CLI",
        "# The first target file is the NaturalCC primary parse file.",
    ]
    if not api_key:
        header.append("# Empty API key falls back to OPENROUTER_API_KEY / OPENAI_API_KEY.")

    return "\n".join(header + [" \\\n  ".join(shlex.quote(part) for part in command)])


# [maint-note:1b58aa2a7986]
# 维护导航：build_workspace_snapshot；下列为本函数源码中的接口与路径索引。
# 输入 project_dir：str。
# 输入 target_files：Optional[Sequence[str]]；缺省表达式 None。
# 返回契约：Dict[str, Any]。
# 返回路径：{'normalized_root': normalized_root, 'exists': os.path.isdir(normalized_root), 'files': merged_choices, 'scanned_files': file_choices, 'selected_files': selected_files, 'counts': {'visible_files': len(file_choices), 'parsable_files': len(pa。
# 直接调用：normalize_project_dir。
# 直接调用：os.path.isdir。
# 直接调用：get_local_files。
# 直接调用：merge_file_choices。
# 直接调用：Path(path).suffix.lower。
# 直接调用：Path。
# 直接调用：set。
# 直接调用：len。
# 直接调用：sorted。
# 表达式分流：os.path.isdir(normalized_root)；真值与假值分支分别计算，不同时执行。
# 集合派生：path 来自 file_choices；过滤 Path(path).suffix.lower() in PARSABLE_SOURCE_EXTENSIONS。
# 集合派生：path 来自 selected_files；过滤 path not in set(file_choices)。
# 表达式分流：selected_files；真值与假值分支分别计算，不同时执行。
# [/maint-note:1b58aa2a7986]
def build_workspace_snapshot(project_dir: str, target_files: Optional[Sequence[str]] = None) -> Dict[str, Any]:
    normalized_root = normalize_project_dir(project_dir or DEFAULT_PROJECT_DIR)
    file_choices = get_local_files(normalized_root) if os.path.isdir(normalized_root) else []
    merged_choices, selected_files = merge_file_choices(file_choices, target_files or [])
    parsable_files = [
        path for path in file_choices
        if Path(path).suffix.lower() in PARSABLE_SOURCE_EXTENSIONS
    ]
    custom_targets = [path for path in selected_files if path not in set(file_choices)]

    return {
        "normalized_root": normalized_root,
        "exists": os.path.isdir(normalized_root),
        "files": merged_choices,
        "scanned_files": file_choices,
        "selected_files": selected_files,
        "counts": {
            "visible_files": len(file_choices),
            "parsable_files": len(parsable_files),
            "selected_files": len(selected_files),
            "custom_targets": len(custom_targets),
        },
        "primary_target": selected_files[0] if selected_files else "",
        "ignore_dirs": sorted(IGNORE_DIRS),
        "parsable_extensions": sorted(PARSABLE_SOURCE_EXTENSIONS),
    }


# [maint-note:a2f8f532cf75]
# 维护导航：browse_directory；下列为本函数源码中的接口与路径索引。
# 输入 path_value：Optional[str]。
# 返回契约：Dict[str, Any]。
# 分支条件：not path.is_absolute()。
# 分支条件：path.is_file()。
# 分支条件：not path.exists() or not path.is_dir()。
# 分支条件：entry.name in IGNORE_DIRS or entry.name.startswith('.')。
# 分支条件：entry.is_dir()。
# 分支条件：not should_ignore_file(entry.name)。
# 遍历边界：entry ← entries。
# 返回路径：{'path': str(path), 'parent': str(path.parent) if path.parent != path else '', 'directories': directories, 'files': files}。
# 显式抛出：HTTPException(status_code=404, detail=f'Directory not found: {path}')。
# 显式抛出：HTTPException(status_code=403, detail=f'Permission denied: {path}')。
# 捕获异常：PermissionError。
# 直接调用：(path_value or DEFAULT_PROJECT_DIR).strip。
# 直接调用：Path(raw_path).expanduser。
# 直接调用：Path。
# 直接调用：path.is_absolute。
# 直接调用：Path.cwd。
# 直接调用：path.resolve。
# 直接调用：path.is_file。
# 直接调用：path.exists。
# 直接调用：path.is_dir。
# 直接调用：HTTPException。
# 直接调用：sorted。
# 直接调用：path.iterdir。
# 直接调用：entry.name.startswith。
# 直接调用：str。
# 直接调用：entry.is_dir。
# 直接调用：directories.append。
# 直接调用：should_ignore_file。
# 直接调用：files.append。
# 表达式分流：path.parent != path；真值与假值分支分别计算，不同时执行。
# [/maint-note:a2f8f532cf75]
def browse_directory(path_value: Optional[str]) -> Dict[str, Any]:
    raw_path = (path_value or DEFAULT_PROJECT_DIR).strip()
    path = Path(raw_path).expanduser()
    if not path.is_absolute():
        path = Path.cwd() / path
    path = path.resolve()

    if path.is_file():
        path = path.parent
    if not path.exists() or not path.is_dir():
        raise HTTPException(status_code=404, detail=f"Directory not found: {path}")

    directories = []
    files = []
    try:
        entries = sorted(path.iterdir(), key=lambda item: (not item.is_dir(), item.name.lower()))
        for entry in entries:
            if entry.name in IGNORE_DIRS or entry.name.startswith("."):
                continue
            item = {
                "name": entry.name,
                "path": str(entry),
                "is_dir": entry.is_dir(),
            }
            if entry.is_dir():
                directories.append(item)
            elif not should_ignore_file(entry.name):
                files.append(item)
    except PermissionError as exc:
        raise HTTPException(status_code=403, detail=f"Permission denied: {path}") from exc

    return {
        "path": str(path),
        "parent": str(path.parent) if path.parent != path else "",
        "directories": directories,
        "files": files,
    }


# [maint-note:848e9d238067]
# 维护导航：select_local_directory；下列为本函数源码中的接口与路径索引。
# 输入 initial_dir：Optional[str]。
# 返回契约：Dict[str, Any]。
# 分支条件：initial_path.is_file()。
# 分支条件：not initial_path.exists()。
# 分支条件：not selected。
# 分支条件：not selected_path.is_dir()。
# 遍历边界：not initial_path.exists() and initial_path.parent != initial_path。
# 返回路径：{'selected': False, 'path': ''}。
# 返回路径：{'selected': True, 'path': str(selected_path)}。
# 显式抛出：RuntimeError('Local folder picker is unavailable in this Python environment.')。
# 显式抛出：RuntimeError('Local folder picker could not open in this desktop session.')。
# 显式抛出：RuntimeError(f'Selected path is not a directory: {selected_path}')。
# 捕获异常：Exception。
# 直接调用：RuntimeError。
# 直接调用：Path。
# 直接调用：normalize_project_dir。
# 直接调用：initial_path.is_file。
# 直接调用：initial_path.exists。
# 直接调用：tk.Tk。
# 直接调用：root.withdraw。
# 直接调用：root.attributes。
# 直接调用：root.update。
# 直接调用：filedialog.askdirectory。
# 直接调用：str。
# 直接调用：root.destroy。
# 直接调用：Path(selected).expanduser().resolve。
# 直接调用：Path(selected).expanduser。
# 直接调用：selected_path.is_dir。
# 清理边界：本函数含 finally；正常返回和异常展开都经过清理块。
# [/maint-note:848e9d238067]
def select_local_directory(initial_dir: Optional[str]) -> Dict[str, Any]:
    try:
        import tkinter as tk
        from tkinter import filedialog
    except Exception as exc:  # pragma: no cover - depends on local Python build.
        raise RuntimeError("Local folder picker is unavailable in this Python environment.") from exc

    initial_path = Path(normalize_project_dir(initial_dir or DEFAULT_PROJECT_DIR))
    if initial_path.is_file():
        initial_path = initial_path.parent
    while not initial_path.exists() and initial_path.parent != initial_path:
        initial_path = initial_path.parent
    if not initial_path.exists():
        initial_path = Path(DEFAULT_PROJECT_DIR)

    try:
        root = tk.Tk()
    except Exception as exc:  # pragma: no cover - depends on local desktop session.
        raise RuntimeError("Local folder picker could not open in this desktop session.") from exc

    root.withdraw()
    try:
        try:
            root.attributes("-topmost", True)
        except Exception:
            pass
        root.update()
        selected = filedialog.askdirectory(
            parent=root,
            initialdir=str(initial_path),
            title="Select agent workspace",
            mustexist=True,
        )
    finally:
        root.destroy()

    if not selected:
        return {"selected": False, "path": ""}

    selected_path = Path(selected).expanduser().resolve()
    if not selected_path.is_dir():
        raise RuntimeError(f"Selected path is not a directory: {selected_path}")
    return {"selected": True, "path": str(selected_path)}


# [maint-note:880a42b3b7c1]
# 维护导航：infer_status；下列为本函数源码中的接口与路径索引。
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
# [/maint-note:880a42b3b7c1]
def infer_status(log_text: str) -> str:
    if "✅ [NaturalCC Agent]" in log_text or "任务圆满完成" in log_text:
        return "success"
    if "❌" in log_text or "[错误]" in log_text or "失败" in log_text:
        return "error"
    if log_text.strip():
        return "running"
    return "idle"


# [maint-note:5522d810ef61]
# 维护导航：ndjson_event；下列为本函数源码中的接口与路径索引。
# 输入 payload：Dict[str, Any]。
# 返回契约：str。
# 返回路径：json.dumps(payload, ensure_ascii=False) + '\n'。
# 直接调用：json.dumps。
# [/maint-note:5522d810ef61]
def ndjson_event(payload: Dict[str, Any]) -> str:
    return json.dumps(payload, ensure_ascii=False) + "\n"


app = FastAPI(title="NaturalCC Agent Web API")
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://127.0.0.1:5173",
        "http://localhost:5173",
        "http://127.0.0.1:7860",
        "http://localhost:7860",
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# The durable Agent Runtime is mounted separately from the legacy Pipeline API.
# Construction is lazy with respect to model clients; no API key is persisted.
if __package__ in (None, ""):
    from code_agent.api.agent_routes import (
        build_default_agent_router,
        get_default_runtime_model_config,
        get_runtime_provider_defaults,
    )
else:
    from .api.agent_routes import (
        build_default_agent_router,
        get_default_runtime_model_config,
        get_runtime_provider_defaults,
    )

app.include_router(build_default_agent_router())


# [maint-note:b72e2db1cc02]
# 维护导航：health；下列为本函数源码中的接口与路径索引。
# 返回契约：Dict[str, str]。
# 异步边界：调用方需等待协程；异常沿 await 链传播或被上层捕获。
# 返回路径：{'status': 'ok', 'time': datetime.now().isoformat(timespec='seconds')}。
# 直接调用：datetime.now().isoformat。
# 直接调用：datetime.now。
# [/maint-note:b72e2db1cc02]
@app.get("/api/health")
async def health() -> Dict[str, str]:
    return {"status": "ok", "time": datetime.now().isoformat(timespec="seconds")}


# [maint-note:014878b3994b]
# 维护导航：bootstrap；下列为本函数源码中的接口与路径索引。
# 返回契约：Dict[str, Any]。
# 异步边界：调用方需等待协程；异常沿 await 链传播或被上层捕获。
# 返回路径：{'default_project_dir': normalize_project_dir(DEFAULT_PROJECT_DIR), 'models': MODELS, 'default_model': DEFAULT_MODEL, 'runtime_default_model_config': runtime_default_model_config, 'runtime_provider_defaults': get_runtime_provider_defaults()。
# 直接调用：get_default_runtime_model_config().to_dict。
# 直接调用：get_default_runtime_model_config。
# 直接调用：normalize_project_dir。
# 直接调用：get_runtime_provider_defaults。
# 直接调用：registry.list_plugins。
# 直接调用：registry.get_schemas。
# 集合派生：m 来自 registry.list_plugins()。
# [/maint-note:014878b3994b]
@app.get("/api/bootstrap")
async def bootstrap() -> Dict[str, Any]:
    runtime_default_model_config = get_default_runtime_model_config().to_dict()
    return {
        "default_project_dir": normalize_project_dir(DEFAULT_PROJECT_DIR),
        "models": MODELS,
        "default_model": DEFAULT_MODEL,
        # ``default_model`` is the legacy Pipeline/Aider default.  The Agent
        # Runtime reads this separate, persisted configuration instead.
        "runtime_default_model_config": runtime_default_model_config,
        "runtime_provider_defaults": get_runtime_provider_defaults(),
        "completion_types": COMPLETION_TYPES,
        "features": [
            {
                "name": m.name,
                "label": m.label,
                "description": m.description,
                "execution_mode": m.execution_mode.value,
            }
            for m in registry.list_plugins()
        ],
        "schemas": registry.get_schemas(),
        "default_feature": "code_completion",
    }


# [maint-note:270bf4c9e1de]
# 维护导航：models；下列为本函数源码中的接口与路径索引。
# 返回契约：Dict[str, Any]。
# 异步边界：调用方需等待协程；异常沿 await 链传播或被上层捕获。
# 返回路径：{'models': MODELS, 'default_model': DEFAULT_MODEL}。
# [/maint-note:270bf4c9e1de]
@app.get("/api/models")
async def models() -> Dict[str, Any]:
    return {"models": MODELS, "default_model": DEFAULT_MODEL}


# [maint-note:4400a36f9013]
# 维护导航：scan_workspace_get；下列为本函数源码中的接口与路径索引。
# 输入 project_dir：str；缺省表达式 Query(default=DEFAULT_PROJECT_DIR)。
# 输入 target_files：Optional[List[str]]；缺省表达式 Query(default=None)。
# 返回契约：Dict[str, Any]。
# 异步边界：调用方需等待协程；异常沿 await 链传播或被上层捕获。
# 返回路径：build_workspace_snapshot(project_dir, target_files)。
# 直接调用：build_workspace_snapshot。
# [/maint-note:4400a36f9013]
@app.get("/api/workspace/scan")
async def scan_workspace_get(
    project_dir: str = Query(default=DEFAULT_PROJECT_DIR),
    target_files: Optional[List[str]] = Query(default=None),
) -> Dict[str, Any]:
    return build_workspace_snapshot(project_dir, target_files)


# [maint-note:b27e607c139b]
# 维护导航：scan_workspace_post；下列为本函数源码中的接口与路径索引。
# 输入 request：ScanRequest。
# 返回契约：Dict[str, Any]。
# 异步边界：调用方需等待协程；异常沿 await 链传播或被上层捕获。
# 返回路径：build_workspace_snapshot(request.project_dir, request.target_files)。
# 直接调用：build_workspace_snapshot。
# [/maint-note:b27e607c139b]
@app.post("/api/workspace/scan")
async def scan_workspace_post(request: ScanRequest) -> Dict[str, Any]:
    return build_workspace_snapshot(request.project_dir, request.target_files)


# [maint-note:bc451acdc981]
# 维护导航：browse；下列为本函数源码中的接口与路径索引。
# 输入 path：Optional[str]；缺省表达式 Query(default=None)。
# 返回契约：Dict[str, Any]。
# 异步边界：调用方需等待协程；异常沿 await 链传播或被上层捕获。
# 返回路径：browse_directory(path)。
# 直接调用：browse_directory。
# [/maint-note:bc451acdc981]
@app.get("/api/browse")
async def browse(path: Optional[str] = Query(default=None)) -> Dict[str, Any]:
    return browse_directory(path)


# [maint-note:b7b8ad594ef7]
# 维护导航：select_workspace_directory；下列为本函数源码中的接口与路径索引。
# 输入 request：DirectoryPickerRequest。
# 返回契约：Dict[str, Any]。
# 异步边界：调用方需等待协程；异常沿 await 链传播或被上层捕获。
# 返回路径：await asyncio.to_thread(select_local_directory, request.initial_dir)。
# 显式抛出：HTTPException(status_code=503, detail=str(exc))。
# 捕获异常：RuntimeError。
# 直接调用：asyncio.to_thread。
# 直接调用：HTTPException。
# 直接调用：str。
# 等待边界：asyncio.to_thread(select_local_directory, request.initial_dir)；异常在此 await 返回给当前协程。
# [/maint-note:b7b8ad594ef7]
@app.post("/api/workspace/select-directory")
async def select_workspace_directory(request: DirectoryPickerRequest) -> Dict[str, Any]:
    try:
        return await asyncio.to_thread(select_local_directory, request.initial_dir)
    except RuntimeError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc


# [maint-note:654daf2c5066]
# 维护导航：command_preview；下列为本函数源码中的接口与路径索引。
# 输入 request：AgentRequest。
# 返回契约：Dict[str, str]。
# 异步边界：调用方需等待协程；异常沿 await 链传播或被上层捕获。
# 返回路径：{'preview_command': build_cli_command_preview(request, preview_only=True), 'run_command': build_cli_command_preview(request, preview_only=False), 'provider': detect_provider(request.model or DEFAULT_MODEL), 'api_key_label': mask_secret(requ。
# 直接调用：build_cli_command_preview。
# 直接调用：detect_provider。
# 直接调用：mask_secret。
# [/maint-note:654daf2c5066]
@app.post("/api/command-preview")
async def command_preview(request: AgentRequest) -> Dict[str, str]:
    return {
        "preview_command": build_cli_command_preview(request, preview_only=True),
        "run_command": build_cli_command_preview(request, preview_only=False),
        "provider": detect_provider(request.model or DEFAULT_MODEL),
        "api_key_label": mask_secret(request.api_key),
    }


# [maint-note:0187d63f8a7c]
# 维护导航：prompt_preview；下列为本函数源码中的接口与路径索引。
# 输入 request：AgentRequest。
# 返回契约：Dict[str, Any]。
# 异步边界：调用方需等待协程；异常沿 await 链传播或被上层捕获。
# 分支条件：request.symbol is not None。
# 分支条件：request.completion_type is not None。
# 分支条件：request.prefix。
# 分支条件：plugin is not None。
# 返回路径：{'status': status, 'log': preview_text, 'command': build_cli_command_preview(request, preview_only=True), 'primary_target': selected_files[0] if selected_files else ''}。
# 直接调用：sanitize_target_files。
# 直接调用：normalize_project_dir。
# 直接调用：dict。
# 直接调用：feature_config.setdefault。
# 直接调用：registry.get。
# 直接调用：ExecutionContext。
# 直接调用：plugin.preview。
# 直接调用：preview_prompt。
# 直接调用：preview_text.startswith。
# 直接调用：build_cli_command_preview。
# 表达式分流：preview_text.startswith('❌')；真值与假值分支分别计算，不同时执行。
# 表达式分流：selected_files；真值与假值分支分别计算，不同时执行。
# [/maint-note:0187d63f8a7c]
@app.post("/api/prompt/preview")
async def prompt_preview(request: AgentRequest) -> Dict[str, Any]:
    selected_files = sanitize_target_files(request.target_files)
    project_dir = normalize_project_dir(request.project_dir or DEFAULT_PROJECT_DIR)

    feature = request.feature or "code_completion"
    feature_config = dict(request.feature_config or {})
    if request.symbol is not None:
        feature_config.setdefault("symbol", request.symbol)
    if request.completion_type is not None:
        feature_config.setdefault("completion_type", request.completion_type)
    if request.prefix:
        feature_config.setdefault("prefix", request.prefix)

    plugin = registry.get(feature)
    if plugin is not None:
        context = ExecutionContext(
            project_dir=project_dir,
            target_files=selected_files,
            instruction=request.instruction or "",
            model=request.model or DEFAULT_MODEL,
            api_key=request.api_key,
            feature_config={"feature": feature, **feature_config},
            uploaded_files={},
            symbol=request.symbol,
            completion_type=request.completion_type,
            prefix=request.prefix,
        )
        preview_text = plugin.preview(context)
    else:
        preview_text = preview_prompt(
            target_files=selected_files,
            user_instruction=request.instruction or "",
            model=request.model or DEFAULT_MODEL,
            api_key=request.api_key or None,
            project_dir=project_dir,
            symbol=(request.symbol or None),
            completion_type=(request.completion_type or None),
            prefix=request.prefix or "",
        )
    status = "error" if preview_text.startswith("❌") else "success"
    return {
        "status": status,
        "log": preview_text,
        "command": build_cli_command_preview(request, preview_only=True),
        "primary_target": selected_files[0] if selected_files else "",
    }


# [maint-note:62c1484caaec]
# 维护导航：_stream_with_context；下列为本函数源码中的接口与路径索引。
# 输入 context：ExecutionContext。
# 返回契约：AsyncIterable[str]。
# 异步边界：调用方需等待协程；异常沿 await 链传播或被上层捕获。
# 遍历边界：event ← dispatcher.dispatch(context)。
# 直接调用：dispatcher.dispatch。
# 直接调用：asyncio.sleep。
# 生成器交付：event。
# 等待边界：asyncio.sleep(0)；异常在此 await 返回给当前协程。
# [/maint-note:62c1484caaec]
async def _stream_with_context(context: ExecutionContext) -> AsyncIterable[str]:
    for event in dispatcher.dispatch(context):
        yield event
        await asyncio.sleep(0)


# [maint-note:5a973a581059]
# 维护导航：run_agent；下列为本函数源码中的接口与路径索引。
# 输入 request：Request。
# 返回契约：StreamingResponse。
# 异步边界：调用方需等待协程；异常沿 await 链传播或被上层捕获。
# 分支条件：'multipart/form-data' in content_type or 'application/x-www-form-urlencoded' in content_type。
# 分支条件：hasattr(value, 'filename') and hasattr(value, 'file')。
# 分支条件：agent_request.symbol is not None。
# 分支条件：agent_request.completion_type is not None。
# 分支条件：agent_request.prefix。
# 遍历边界：key ← form。
# 返回路径：StreamingResponse(_stream_with_context(context), media_type='application/x-ndjson', headers=STREAM_HEADERS)。
# 写入目标：uploaded_files[key]。
# 直接调用：request.headers.get。
# 直接调用：request.form。
# 直接调用：str。
# 直接调用：form.get。
# 直接调用：json.loads。
# 直接调用：hasattr。
# 直接调用：ExecutionContext。
# 直接调用：normalize_project_dir。
# 直接调用：sanitize_target_files。
# 直接调用：StreamingResponse。
# 直接调用：_stream_with_context。
# 直接调用：request.json。
# 直接调用：AgentRequest。
# 直接调用：dict。
# 直接调用：feature_config.setdefault。
# 等待边界：request.form()；异常在此 await 返回给当前协程。
# 表达式分流：target_files_raw；真值与假值分支分别计算，不同时执行。
# 表达式分流：feature_config_raw；真值与假值分支分别计算，不同时执行。
# 等待边界：request.json()；异常在此 await 返回给当前协程。
# [/maint-note:5a973a581059]
@app.post("/api/run")
async def run_agent(request: Request) -> StreamingResponse:
    content_type = request.headers.get("content-type", "")

    if "multipart/form-data" in content_type or "application/x-www-form-urlencoded" in content_type:
        form = await request.form()
        project_dir = str(form.get("project_dir", DEFAULT_PROJECT_DIR))
        target_files_raw = str(form.get("target_files", "[]"))
        instruction = str(form.get("instruction", ""))
        model = str(form.get("model", DEFAULT_MODEL))
        api_key = form.get("api_key")
        feature = str(form.get("feature", "code_completion"))
        feature_config_raw = str(form.get("feature_config", "{}"))

        target_files_list = json.loads(target_files_raw) if target_files_raw else []
        feature_config_dict = json.loads(feature_config_raw) if feature_config_raw else {}

        uploaded_files = {}
        for key in form:
            value = form[key]
            if hasattr(value, "filename") and hasattr(value, "file"):
                uploaded_files[key] = value

        context = ExecutionContext(
            project_dir=normalize_project_dir(project_dir),
            target_files=sanitize_target_files(target_files_list),
            instruction=instruction,
            model=model,
            api_key=api_key,
            feature_config={"feature": feature, **feature_config_dict},
            uploaded_files=uploaded_files,
        )

        return StreamingResponse(
            _stream_with_context(context),
            media_type="application/x-ndjson",
            headers=STREAM_HEADERS,
        )

    # JSON fallback (backward compatible)
    body = await request.json()
    agent_request = AgentRequest(**body)
    selected_files = sanitize_target_files(agent_request.target_files)

    feature = agent_request.feature or "code_completion"
    feature_config = dict(agent_request.feature_config or {})
    if agent_request.symbol is not None:
        feature_config.setdefault("symbol", agent_request.symbol)
    if agent_request.completion_type is not None:
        feature_config.setdefault("completion_type", agent_request.completion_type)
    if agent_request.prefix:
        feature_config.setdefault("prefix", agent_request.prefix)

    context = ExecutionContext(
        project_dir=normalize_project_dir(agent_request.project_dir or DEFAULT_PROJECT_DIR),
        target_files=selected_files,
        instruction=agent_request.instruction or "",
        model=agent_request.model or DEFAULT_MODEL,
        api_key=agent_request.api_key,
        feature_config={"feature": feature, **feature_config},
        uploaded_files={},
        symbol=agent_request.symbol,
        completion_type=agent_request.completion_type,
        prefix=agent_request.prefix,
    )

    return StreamingResponse(
        _stream_with_context(context),
        media_type="application/x-ndjson",
        headers=STREAM_HEADERS,
    )


DIST_DIR = Path(__file__).resolve().parent / "webui" / "dist"

if (DIST_DIR / "assets").is_dir():
    app.mount("/assets", StaticFiles(directory=str(DIST_DIR / "assets")), name="assets")


# [maint-note:1ea07ac2988d]
# 维护导航：index；下列为本函数源码中的接口与路径索引。
# 返回契约：FileResponse。
# 异步边界：调用方需等待协程；异常沿 await 链传播或被上层捕获。
# 分支条件：not index_file.exists()。
# 返回路径：FileResponse(index_file)。
# 显式抛出：HTTPException(status_code=404, detail='webui/dist not found. Run `cd webui && npm run build` or use the Vite dev server.')。
# 直接调用：index_file.exists。
# 直接调用：HTTPException。
# 直接调用：FileResponse。
# [/maint-note:1ea07ac2988d]
@app.get("/", include_in_schema=False)
async def index() -> FileResponse:
    index_file = DIST_DIR / "index.html"
    if not index_file.exists():
        raise HTTPException(
            status_code=404,
            detail="webui/dist not found. Run `cd webui && npm run build` or use the Vite dev server.",
        )
    return FileResponse(index_file)


# [maint-note:d1a3aca187c4]
# 维护导航：spa_fallback；下列为本函数源码中的接口与路径索引。
# 输入 full_path：str。
# 返回契约：FileResponse。
# 异步边界：调用方需等待协程；异常沿 await 链传播或被上层捕获。
# 分支条件：candidate.is_file()。
# 返回路径：FileResponse(candidate)。
# 返回路径：await index()。
# 直接调用：candidate.is_file。
# 直接调用：FileResponse。
# 直接调用：index。
# 等待边界：index()；异常在此 await 返回给当前协程。
# [/maint-note:d1a3aca187c4]
@app.get("/{full_path:path}", include_in_schema=False)
async def spa_fallback(full_path: str) -> FileResponse:
    candidate = DIST_DIR / full_path
    if candidate.is_file():
        return FileResponse(candidate)
    return await index()


# [maint-note:bf00c5105199]
# 维护导航：main；下列为本函数源码中的接口与路径索引。
# 返回契约：None。
# 直接调用：argparse.ArgumentParser。
# 直接调用：parser.add_argument。
# 直接调用：parser.parse_args。
# 直接调用：uvicorn.run。
# [/maint-note:bf00c5105199]
def main() -> None:
    parser = argparse.ArgumentParser(description="NaturalCC Agent web app")
    parser.add_argument("--host", default="127.0.0.1", help="Bind host")
    parser.add_argument("--port", default=7860, type=int, help="Bind port")
    args = parser.parse_args()

    import uvicorn

    uvicorn.run(app, host=args.host, port=args.port)


if __name__ == "__main__":
    main()
